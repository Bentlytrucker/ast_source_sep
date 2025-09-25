#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Separator for Raspberry Pi 5
- Implement AST-based audio source separation pipeline in a single file
- Use the same separation logic as separator.py
- Optimized for Raspberry Pi 5
- Sound type filtering when sending to backend (only send danger, help, warning)
- Include LED control functionality
"""

import os
import sys
import time
import warnings
import json
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Environment configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

try:
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. Sound separation will be disabled.")
    ASTFeatureExtractor = None
    ASTForAudioClassification = None
    TRANSFORMERS_AVAILABLE = False

# ==================== Constants Definition ====================
class AudioConfig:
    """Audio processing configuration"""
    SR = 16000  # Sampling rate
    INPUT_SEC = 4.096  # Actual input audio length
    MODEL_SEC = 10.24  # AST model processing length
    L_INPUT = int(round(INPUT_SEC * SR))  # 65,536 samples
    L_MODEL = int(round(MODEL_SEC * SR))  # 163,840 samples

class STFTConfig:
    """STFT configuration"""
    N_FFT = 400
    HOP = 160
    WINLEN = 400
    N_MELS = 128

class ASTConfig:
    """AST model configuration"""
    MAX_LENGTH = 1024
    FREQ_PATCHES = 12  # (128 - 16) / 10 + 1
    TIME_PATCHES = 101  # (1024 - 16) / 10 + 1

class SeparationConfig:
    """Separation parameters"""
    SIMILARITY_THRESHOLD = 0.6
    WEAK_MASK_FACTOR = 0.1
    MIN_ANCHOR_FRAMES = 8
    MAX_ANCHOR_FRAMES = 128
    CONTINUITY_GAP = 1
    ATTENTION_PERCENTILE = 80
    MEDIUM_MASK_FACTOR = 0.6

class WienerConfig:
    """Wiener filtering configuration"""
    TANH_SCALE = 3.0
    SIGMOID_SCALE = 10.0
    SIGMOID_THRESHOLD = 0.6
    ANCHOR_BOOST = 1.5

# 사운드 타입 분류
DANGER_IDS = {396, 397, 398, 399, 400, 426, 436}
HELP_IDS = {23, 14, 354, 355, 356, 359}
WARNING_IDS = {0, 288, 364, 388, 389, 390, 439, 391, 392, 393, 395, 440, 441, 443, 456, 469, 470, 478, 479}

# 편의를 위한 별칭
SR = AudioConfig.SR
INPUT_SEC = AudioConfig.INPUT_SEC
MODEL_SEC = AudioConfig.MODEL_SEC
L_INPUT = AudioConfig.L_INPUT
L_MODEL = AudioConfig.L_MODEL
N_FFT = STFTConfig.N_FFT
HOP = STFTConfig.HOP
WINLEN = STFTConfig.WINLEN
N_MELS = STFTConfig.N_MELS

# ==================== Data Classes ====================
@dataclass
class SeparationResult:
    """Store separation results"""
    separated_audio: np.ndarray
    residual_audio: np.ndarray
    mask: np.ndarray
    anchor_frames: Tuple[int, int]
    energy_ratio: float
    classification: Dict[str, Any]
    attention_matrix: np.ndarray
    template_similarity: np.ndarray

# ==================== Utility Functions ====================
def get_sound_type(class_id: int) -> str:
    """Convert class ID to sound type"""
    if class_id in DANGER_IDS:
        return "danger"
    elif class_id in HELP_IDS:
        return "help"
    elif class_id in WARNING_IDS:
        return "warning"
    else:
        return "other"

def should_send_to_backend(sound_type: str, class_name: str) -> bool:
    """Function to decide whether to send to backend"""
    # other 타입이거나 silence 클래스는 전송하지 않음
    if sound_type == "other":
        return False
    if class_name.lower() == "silence":
        return False
    # danger, help, warning 타입만 전송
    return sound_type in ["danger", "help", "warning"]

def should_activate_led(sound_type: str, class_name: str) -> bool:
    """Function to decide whether to activate LED"""
    # danger, help, warning 타입에만 LED 활성화
    return sound_type in ["danger", "help", "warning"]

def calculate_sound_occurrence_time(separation_mask: np.ndarray, 
                                  inference_start_time: datetime, 
                                  audio_duration: float = 4.096) -> str:
    """Calculate sound occurrence time based on separation mask"""
    try:
        if separation_mask is None or len(separation_mask.shape) < 2:
            return inference_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 시간 축에서 높은 유사도를 가진 구간 찾기
        time_axis_mask = np.mean(separation_mask, axis=0)
        
        # 임계값 이상인 구간 찾기
        threshold = np.percentile(time_axis_mask, 70)
        active_frames = np.where(time_axis_mask > threshold)[0]
        
        if len(active_frames) == 0:
            return inference_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 첫 번째 활성화 프레임의 시간 계산
        first_active_frame = active_frames[0]
        frame_duration = HOP / SR  # 프레임당 시간
        sound_offset_in_audio = first_active_frame * frame_duration
        
        # 실제 소리 발생 시간 = 추론 시작 시간 - 오디오 길이 + 소리 오프셋
        sound_occurrence_time = inference_start_time - timedelta(seconds=audio_duration) + timedelta(seconds=sound_offset_in_audio)
        
        return sound_occurrence_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
    except Exception as e:
        print(f"Sound occurrence time calculation failed: {e}")
        return inference_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

# ==================== AST Model Processing ====================
class ASTProcessor:
    """AST model processing - Raspberry Pi 5 optimized"""
    
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        # CPU 강제 설정 (Raspberry Pi 5 최적화)
        self.device = torch.device("cpu")
        torch.set_num_threads(2)  # Raspberry Pi 5에 적합한 스레드 수
        print(f"Device: CPU (threads: 2) - Raspberry Pi 5 optimized")
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers not available - Cannot load AST model!")
            self.is_available = False
            return
        
        try:
            # 모델 로드
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
            
            try:
                self.model = ASTForAudioClassification.from_pretrained(
                    model_name,
                    attn_implementation="eager",
                    output_attentions=True,
                    output_hidden_states=True
                )
            except:
                self.model = ASTForAudioClassification.from_pretrained(
                    model_name,
                    output_attentions=True,
                    output_hidden_states=True
                )
            
            # Raspberry Pi 5 최적화: 동적 양자화
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("Quantization applied: Success (Raspberry Pi 5 optimized)")
            except:
                print("Quantization applied: Failed (normal mode)")
            
            self.model.eval()
            self.is_available = True
            print("AST model loaded successfully - Raspberry Pi 5 optimized")
            
        except Exception as e:
            print(f"AST model load failed: {e}")
            self.is_available = False
    
    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """Process with 10.24 second padding"""
        if not self.is_available:
            # 모델이 없을 때 더미 데이터 반환
            dummy_attention = np.ones((ASTConfig.FREQ_PATCHES, ASTConfig.TIME_PATCHES)) * 0.5
            dummy_classification = {
                'predicted_class': "Unknown",
                'confidence': 0.0,
                'top5_classes': ["Unknown"] * 5,
                'top5_probs': [0.0] * 5
            }
            dummy_spectrogram = np.zeros((N_MELS, 1024))
            return dummy_attention, dummy_classification, dummy_spectrogram
        
        # 패딩
        padded = np.pad(audio, (0, max(0, L_MODEL - len(audio))), mode='constant')[:L_MODEL]
        
        # 특징 추출
        inputs = self.feature_extractor(padded, sampling_rate=SR, return_tensors="pt")
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 어텐션 추출
        if outputs.attentions:
            attention = outputs.attentions[-1].squeeze(0).mean(dim=0)
            attention_to_patches = attention[0, 1:]
        else:
            attention_to_patches = torch.ones(ASTConfig.FREQ_PATCHES * ASTConfig.TIME_PATCHES) / (ASTConfig.FREQ_PATCHES * ASTConfig.TIME_PATCHES)
        
        attention_matrix = attention_to_patches[:ASTConfig.FREQ_PATCHES * ASTConfig.TIME_PATCHES].reshape(
            ASTConfig.FREQ_PATCHES, ASTConfig.TIME_PATCHES
        ).cpu().numpy()
        
        # CLS/Distill 토큰 영향 제거
        attention_matrix[0, 0] *= 0.1
        if attention_matrix.shape[0] > 1:
            attention_matrix[1, 0] *= 0.3
        if attention_matrix.shape[1] > 1:
            attention_matrix[0, 1] *= 0.3
        
        # 분류 결과
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1)
        
        classification = {
            'predicted_class': self.model.config.id2label.get(top5_indices[0, 0].item(), "unknown"),
            'confidence': top5_probs[0, 0].item(),
            'top5_classes': [self.model.config.id2label.get(idx.item(), f"class_{idx}") 
                           for idx in top5_indices[0]],
            'top5_probs': top5_probs[0].cpu().numpy().tolist()
        }
        
        spectrogram = inputs['input_values'].squeeze(0).cpu().numpy()
        
        return attention_matrix, classification, spectrogram

# ==================== Audio Processing Functions ====================
def load_audio(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Audio loading and padding"""
    audio, _ = librosa.load(file_path, sr=SR, mono=True)
    
    # 4.096초 버전
    audio_4sec = audio[:L_INPUT] if len(audio) > L_INPUT else np.pad(audio, (0, L_INPUT - len(audio)))
    
    # 10.24초 버전
    audio_10sec = audio[:L_MODEL] if len(audio) > L_MODEL else np.pad(audio, (0, L_MODEL - len(audio)))
    
    return audio_4sec, audio_10sec

def compute_stft(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT calculation"""
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN)
    mag = np.abs(stft)
    phase = np.angle(stft)
    return stft, mag, phase

# ==================== Template-based Distribution Similarity ====================
def calculate_distribution_similarity(template: np.ndarray, current: np.ndarray) -> float:
    """Calculate frequency-amplitude distribution similarity"""
    # 양수 값 보장
    template_positive = np.maximum(template, 1e-8)
    current_positive = np.maximum(current, 1e-8)
    
    # 정규화
    template_norm = template_positive / (np.sum(template_positive) + 1e-8)
    current_norm = current_positive / (np.sum(current_positive) + 1e-8)
    
    template_norm = np.maximum(template_norm, 1e-8)
    current_norm = np.maximum(current_norm, 1e-8)
    
    # 코사인 유사도
    cosine_sim = np.dot(template_norm, current_norm) / (
        np.linalg.norm(template_norm) * np.linalg.norm(current_norm) + 1e-8
    )
    cosine_sim = np.clip(cosine_sim, 0, 1)
    
    # 스펙트럴 중심 비교
    freqs = np.arange(len(template))
    template_centroid = np.sum(freqs * template_norm)
    current_centroid = np.sum(freqs * current_norm)
    centroid_diff = abs(template_centroid - current_centroid) / len(template)
    centroid_sim = np.exp(-centroid_diff * 5)
    centroid_sim = np.clip(centroid_sim, 0, 1)
    
    # 종합 유사도
    total_sim = 0.7 * cosine_sim + 0.3 * centroid_sim
    total_sim = np.clip(total_sim, 0, 1)
    
    return total_sim

# ==================== Anchor Selection ====================
def find_attention_anchor(
    attention_matrix: np.ndarray,
    suppressed_regions: List[Tuple[int, int]],
    previous_peaks: List[Tuple[int, int]],
    audio_length: int,
    is_duplicate_detected: bool = False,
    previous_anchors: List[Tuple[int, int]] = None
) -> Tuple[int, int, bool]:
    """Select anchor from attention matrix"""
    num_freq_patches, num_time_patches = attention_matrix.shape
    max_frames = audio_length // HOP
    valid_patches = int(INPUT_SEC / MODEL_SEC * num_time_patches)
    
    # 억제된 영역 마스킹
    att_masked = attention_matrix.copy()
    for start, end in suppressed_regions:
        start_patch = int(start * num_time_patches / (L_MODEL // HOP))
        end_patch = int(end * num_time_patches / (L_MODEL // HOP))
        att_masked[:, max(0, start_patch):min(num_time_patches, end_patch)] *= 0.05
    
    # 중복 분류 감지 시 이전 앵커 영역 강하게 억제
    if is_duplicate_detected and previous_anchors:
        print(f"  🔄 Duplicate classification detected - restricting previous anchor regions")
        for anchor_start, anchor_end in previous_anchors:
            # 앵커를 패치 좌표로 변환
            start_patch = int(anchor_start * num_time_patches / max_frames)
            end_patch = int(anchor_end * num_time_patches / max_frames)
            
            # 이전 앵커 영역을 강하게 억제 (0.01배로 감소)
            att_masked[:, max(0, start_patch):min(num_time_patches, end_patch)] *= 0.01
            print(f"    Suppressed anchor region: frames {anchor_start}-{anchor_end} (patches {start_patch}-{end_patch})")
    
    # 이전 패스의 상위 어텐션 영역 약화
    if previous_peaks:
        for start, end in previous_peaks:
            att_masked[:, max(0, start):min(num_time_patches, end)] *= 0.5
    
    # 유효 범위 제한
    att_masked[:, valid_patches:] = 0
    
    # 연속적인 활성화 구간 찾기
    attention_threshold = np.percentile(att_masked, SeparationConfig.ATTENTION_PERCENTILE)
    
    # 주파수별 활성화 마스크 생성
    frequency_segments = []
    for freq_idx in range(num_freq_patches):
        freq_attention = att_masked[freq_idx, :valid_patches]
        high_attention_mask = freq_attention >= attention_threshold
        
        if not np.any(high_attention_mask):
            continue
            
        high_indices = np.where(high_attention_mask)[0]
        
        # 연속 구간 그룹화
        continuous_segments = []
        current_segment = [high_indices[0]]
        
        for i in range(1, len(high_indices)):
            if high_indices[i] - high_indices[i-1] <= SeparationConfig.CONTINUITY_GAP:
                current_segment.append(high_indices[i])
            else:
                if len(current_segment) >= 2:
                    continuous_segments.append((freq_idx, current_segment))
                current_segment = [high_indices[i]]
        
        if len(current_segment) >= 2:
            continuous_segments.append((freq_idx, current_segment))
        
        frequency_segments.extend(continuous_segments)
    
    # 최적 세그먼트 선택
    best_segment = None
    best_score = -1
    
    for freq_idx, segment in frequency_segments:
        segment_attention = np.mean([att_masked[freq_idx, i] for i in segment])
        
        # 주파수 가중치
        freq_weight = 1.0
        if freq_idx <= 3:  # 저주파
            freq_weight = 1.2
        elif freq_idx >= 8:  # 고주파
            freq_weight = 1.1
        
        # 중복 감지 시 새로운 앵커 선택 가중치 강화
        if is_duplicate_detected:
            # 연속성 가중치 - 더 긴 연속 구간을 선호
            continuity_weight = 1.0 + (len(segment) / 10.0)
            
            # 어텐션 강도 가중치 - 높은 어텐션을 더 선호
            attention_weight = 1.0 + segment_attention
            
            # 위치 다양성 가중치 - 이전과 다른 위치를 선호
            position_weight = 1.0
            if previous_anchors:
                min_distance = float('inf')
                segment_center = np.mean(segment)
                for prev_start, prev_end in previous_anchors:
                    prev_center = (prev_start + prev_end) / 2
                    prev_center_patch = prev_center * num_time_patches / max_frames
                    distance = abs(segment_center - prev_center_patch)
                    min_distance = min(min_distance, distance)
                
                # 거리가 멀수록 높은 가중치
                position_weight = 1.0 + min(2.0, min_distance / 10.0)
            
            score = len(segment) * segment_attention * freq_weight * continuity_weight * attention_weight * position_weight
            print(f"    Segment at freq {freq_idx}, pos {segment[0]}-{segment[-1]}: "
                  f"attention={segment_attention:.3f}, continuity={continuity_weight:.2f}, "
                  f"position={position_weight:.2f}, score={score:.3f}")
        else:
            score = len(segment) * segment_attention * freq_weight
        
        if score > best_score:
            best_score = score
            best_segment = segment
    
    if best_segment is None:
        # 폴백: 최대 어텐션 위치
        max_freq_idx, max_time_idx = np.unravel_index(np.argmax(att_masked), att_masked.shape)
        anchor_size = min(5, valid_patches // 8)
        start_patch = max(0, max_time_idx - anchor_size // 2)
        end_patch = min(valid_patches, start_patch + anchor_size)
        best_segment = list(range(start_patch, end_patch))
    
    # 앵커 구간 설정
    anchor_start_patch = best_segment[0]
    anchor_end_patch = best_segment[-1] + 1
    
    # 패치를 프레임으로 변환
    anchor_start = int(anchor_start_patch * max_frames / valid_patches)
    anchor_end = int(anchor_end_patch * max_frames / valid_patches)
    
    # 경계 검증
    anchor_start = max(0, min(anchor_start, max_frames - 1))
    anchor_end = max(anchor_start + 1, min(anchor_end, max_frames))
    
    # 크기 제한
    actual_anchor_size = anchor_end - anchor_start
    if actual_anchor_size < SeparationConfig.MIN_ANCHOR_FRAMES:
        center = (anchor_start + anchor_end) // 2
        anchor_start = max(0, center - SeparationConfig.MIN_ANCHOR_FRAMES // 2)
        anchor_end = min(max_frames, anchor_start + SeparationConfig.MIN_ANCHOR_FRAMES)
    elif actual_anchor_size > SeparationConfig.MAX_ANCHOR_FRAMES:
        center = (anchor_start + anchor_end) // 2
        anchor_start = max(0, center - SeparationConfig.MAX_ANCHOR_FRAMES // 2)
        anchor_end = min(max_frames, anchor_start + SeparationConfig.MAX_ANCHOR_FRAMES)
    
    return anchor_start, anchor_end, True

# ==================== Template-based Masking ====================
def create_template_mask(
    stft_mag: np.ndarray,
    anchor_frames: Tuple[int, int],
    attention_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Template-based distribution similarity masking"""
    anchor_start, anchor_end = anchor_frames
    
    # 앵커 범위 검증
    anchor_start = min(anchor_start, stft_mag.shape[1] - 1)
    anchor_end = min(anchor_end, stft_mag.shape[1])
    
    if anchor_start >= anchor_end:
        return np.zeros_like(stft_mag), np.zeros(stft_mag.shape[1])
    
    # 앵커 스펙트로그램에서 템플릿 생성
    anchor_spec = stft_mag[:, anchor_start:anchor_end]
    template = np.mean(anchor_spec, axis=1)
    
    # 어텐션 가중치 적용
    if attention_weights is not None and len(attention_weights.shape) > 1:
        # 주파수별 어텐션 평균
        freq_attention = np.mean(attention_weights, axis=1)
        
        # STFT 주파수 빈으로 매핑
        stft_freq_bins = len(template)
        attention_freq_patches = len(freq_attention)
        
        mapped_attention = np.zeros(stft_freq_bins)
        for i in range(stft_freq_bins):
            attention_idx = int(i * attention_freq_patches / stft_freq_bins)
            attention_idx = min(attention_idx, attention_freq_patches - 1)
            mapped_attention[i] = freq_attention[attention_idx]
        
        # 어텐션 가중치 적용
        attention_factor = 0.3 + 0.7 * mapped_attention
        template *= attention_factor
    
    # 템플릿 정규화
    template = np.maximum(template, 1e-8)
    template_norm = template / (np.sum(template) + 1e-8)
    
    # 유사도 계산
    num_frames = stft_mag.shape[1]
    similarity = np.zeros(num_frames)
    
    for t in range(num_frames):
        current = stft_mag[:, t]
        similarity[t] = calculate_distribution_similarity(template_norm, current)
    
    # 시간적 스무딩
    from scipy.ndimage import gaussian_filter1d
    similarity = gaussian_filter1d(similarity, sigma=1.0)
    
    # 2D 마스크 생성
    mask = np.zeros_like(stft_mag)
    
    for t in range(num_frames):
        if similarity[t] >= SeparationConfig.SIMILARITY_THRESHOLD:
            mask[:, t] = similarity[t] * 1.5
        elif similarity[t] >= SeparationConfig.SIMILARITY_THRESHOLD * 0.7:
            mask[:, t] = similarity[t] * 1.2
        elif similarity[t] >= SeparationConfig.SIMILARITY_THRESHOLD * 0.4:
            mask[:, t] = similarity[t] * 0.8
        else:
            mask[:, t] = similarity[t] * 0.3
    
    # 마스크 정규화 및 스무딩
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=(0.8, 0.6))
    mask = np.clip(mask, 0, 1)
    
    return mask, similarity

# ==================== Wiener Filtering ====================
def apply_adaptive_wiener(
    stft: np.ndarray,
    mask: np.ndarray,
    anchor_frames: Tuple[int, int]
) -> np.ndarray:
    """Adaptive Wiener filtering"""
    anchor_start, anchor_end = anchor_frames
    
    # 노이즈 추정
    noise_mask = 1.0 - mask
    noise_power = np.abs(stft) ** 2 * noise_mask
    noise_estimate = np.mean(noise_power, axis=1, keepdims=True) + 1e-8
    
    # 신호 파워
    signal_power = np.abs(stft) ** 2 * mask
    
    # Wiener gain 계산
    wiener_gain = signal_power / (signal_power + noise_estimate)
    wiener_gain = np.clip(wiener_gain, 0, 1)
    
    # 강도 조절
    intensity_factor = 0.5 + 0.5 * np.tanh(WienerConfig.TANH_SCALE * (mask - 0.5))
    sigmoid_factor = 1 / (1 + np.exp(-WienerConfig.SIGMOID_SCALE * (mask - WienerConfig.SIGMOID_THRESHOLD)))
    final_intensity = intensity_factor * (0.7 + 0.3 * sigmoid_factor)
    
    enhanced_wiener_gain = wiener_gain * final_intensity
    
    # 앵커 구간 부스트
    enhanced_wiener_gain[:, anchor_start:anchor_end] *= WienerConfig.ANCHOR_BOOST
    
    enhanced_wiener_gain = np.clip(enhanced_wiener_gain, 0, 1)
    
    return stft * enhanced_wiener_gain

# ==================== Single Pass Processing ====================
def process_single_pass(
    audio_4sec: np.ndarray,
    audio_10sec: np.ndarray,
    ast_processor: ASTProcessor,
    suppressed_regions: List[Tuple[int, int]],
    previous_peaks: List[Tuple[int, int]],
    pass_idx: int,
    previous_classifications: List[str] = None,
    is_duplicate_detected: bool = False,
    previous_anchors: List[Tuple[int, int]] = None
) -> Optional[SeparationResult]:
    """Single pass audio source separation"""
    
    print(f"  Processing pass {pass_idx + 1}...")
    
    # AST 처리
    attention_matrix, classification, ast_spec = ast_processor.process(audio_10sec)
    
    # STFT
    stft, mag, phase = compute_stft(audio_4sec)
    
    # 앵커 선정
    anchor_start, anchor_end, is_valid = find_attention_anchor(
        attention_matrix, suppressed_regions, previous_peaks, len(audio_4sec),
        is_duplicate_detected, previous_anchors
    )
    
    if not is_valid:
        return None
    
    # 템플릿 마스킹
    mask, similarity = create_template_mask(mag, (anchor_start, anchor_end), attention_matrix)
    
    # Wiener 필터링
    filtered_stft = apply_adaptive_wiener(stft, mask, (anchor_start, anchor_end))
    
    # 역변환
    separated = librosa.istft(filtered_stft, hop_length=HOP, win_length=WINLEN)
    
    # 길이 조정
    if len(separated) < L_INPUT:
        separated = np.pad(separated, (0, L_INPUT - len(separated)))
    else:
        separated = separated[:L_INPUT]
    
    # 잔여 계산
    residual = audio_4sec - separated
    
    # 에너지 비율
    energy_ratio = np.sum(separated ** 2) / (np.sum(audio_4sec ** 2) + 1e-8)
    
    return SeparationResult(
        separated_audio=separated,
        residual_audio=residual,
        mask=mask,
        anchor_frames=(anchor_start, anchor_end),
        energy_ratio=energy_ratio,
        classification=classification,
        attention_matrix=attention_matrix,
        template_similarity=similarity
    )

# ==================== Multi-pass Separation ====================
def multi_pass_separation(
    audio_4sec: np.ndarray,
    audio_10sec: np.ndarray,
    ast_processor: ASTProcessor,
    max_passes: int = 2,
    backend_url: str = None,
    led_controller = None,
    angle: int = 0
) -> List[SeparationResult]:
    """Multi-pass audio source separation with per-pass processing"""
    results = []
    suppressed_regions = []
    previous_peaks = []
    previous_classifications = []  # Track previous pass classifications
    
    current_4sec = audio_4sec.copy()
    current_10sec = audio_10sec.copy()
    
    for i in range(max_passes):
        print(f"\n=== Pass {i + 1}/{max_passes} ===")
        
        # 중복 분류 감지 (pass 2 이상에서)
        is_duplicate_detected = False
        previous_anchors = []
        
        if i >= 1 and len(previous_classifications) > 0:
            # 이전 패스의 분류 결과들과 비교할 수 있도록 임시로 AST 처리
            temp_attention_matrix, temp_classification, _ = ast_processor.process(current_10sec)
            current_class = temp_classification['predicted_class']
            
            # 이전 분류와 비교
            if current_class in previous_classifications:
                is_duplicate_detected = True
                # 이전 결과들의 앵커 정보 수집
                for prev_result in results:
                    previous_anchors.append(prev_result.anchor_frames)
                print(f"  🔄 Duplicate classification detected: '{current_class}' (same as previous pass)")
        
        result = process_single_pass(
            current_4sec, current_10sec, ast_processor,
            suppressed_regions, previous_peaks, i,
            previous_classifications, is_duplicate_detected, previous_anchors
        )
        
        if result is None or result.energy_ratio < 0.0001:
            print(f"  Termination: Insufficient energy")
            break
        
        # Extract classification info for this pass
        class_name = result.classification['predicted_class']
        confidence = result.classification['confidence']
        
        # Check for silence detection - break immediately if silence detected
        if class_name.lower() == "silence":
            print(f"  Silence detected in pass {i + 1}, breaking separation")
            break
        
        # Get class ID and sound type
        class_id = -1
        if hasattr(ast_processor, 'model') and hasattr(ast_processor.model, 'config'):
            for id_val, label in ast_processor.model.config.id2label.items():
                if label == class_name:
                    class_id = id_val
                    break
        
        sound_type = get_sound_type(class_id)
        
        # Calculate dB
        rms = np.sqrt(np.mean(result.separated_audio ** 2))
        db_mean = 20 * np.log10(rms + 1e-10) if rms > 0 else -np.inf
        
        # Calculate sound occurrence time
        recording_end_time = datetime.utcnow()
        occurred_at = calculate_sound_occurrence_time(
            result.mask, 
            recording_end_time, 
            audio_duration=len(audio_4sec)/SR
        )
        
        # Per-pass backend sending and LED control
        backend_sent = False
        led_activated = False
        
        if sound_type == "other":
            print(f"  Pass {i + 1}: {class_name} (other) - No backend, No LED")
        else:
            # Backend sending for danger/help/warning types
            if should_send_to_backend(sound_type, class_name) and backend_url:
                backend_sent = send_to_backend(
                    sound_type, class_name, angle, db_mean, 
                    occurred_at, backend_url
                )
                if backend_sent:
                    print(f"  Pass {i + 1}: {class_name} ({sound_type}) - Backend sent")
                else:
                    print(f"  Pass {i + 1}: {class_name} ({sound_type}) - Backend failed")
            
            # LED activation for danger/help/warning types
            if should_activate_led(sound_type, class_name) and led_controller:
                try:
                    led_activated = led_controller.activate_led(angle, class_name, sound_type)
                    if led_activated:
                        print(f"  Pass {i + 1}: {class_name} ({sound_type}) - LED activated")
                    else:
                        print(f"  Pass {i + 1}: {class_name} ({sound_type}) - LED failed")
                except Exception as e:
                    print(f"  Pass {i + 1}: LED activation error: {e}")
        
        # Store backend and LED status in result
        result.backend_sent = backend_sent
        result.led_activated = led_activated
        result.sound_type = sound_type
        result.class_id = class_id
        result.db_mean = db_mean
        result.occurred_at = occurred_at
        result.angle = angle
        
        # Store classification for duplicate detection
        previous_classifications.append(class_name)
        
        # Store attention peaks for next pass
        att_mean = np.mean(result.attention_matrix, axis=0)
        threshold = np.percentile(att_mean, SeparationConfig.ATTENTION_PERCENTILE)
        peaks = np.where(att_mean > threshold)[0]
        if len(peaks) > 0:
            previous_peaks.append((peaks[0], peaks[-1]))
        
        results.append(result)
        suppressed_regions.append(result.anchor_frames)
        
        # Update residual audio for next pass
        current_4sec = result.residual_audio
        current_10sec = np.pad(result.residual_audio, (0, max(0, L_MODEL - len(result.residual_audio))))[:L_MODEL]
        
        print(f"  Class: {class_name}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Energy: {result.energy_ratio:.3f}")
    
    return results

# ==================== Backend Transmission ====================
def send_to_backend(sound_type: str, sound_detail: str, angle: int, decibel: float, 
                   occurred_at: str, backend_url: str, user_id: int = 6) -> bool:
    """Send results to backend"""
    try:
        data = {
            "user_id": user_id,
            "sound_type": sound_type,
            "sound_detail": sound_detail,
            "angle": angle,
            "occurred_at": occurred_at,
            "sound_icon": "string",
            "location_image_url": "string",
            "decibel": float(decibel)
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'SingleSeparator/1.0'
        }
        
        print(f"🔄 Sending to backend: {backend_url}")
        print(f"📤 Data: {data}")
        
        response = requests.post(
            backend_url, 
            json=data, 
            headers=headers,
            timeout=10.0,
            verify=False
        )
        
        if response.status_code == 200:
            print(f"✅ Backend send successful: {sound_detail} ({sound_type})")
            return True
        else:
            print(f"❌ Backend error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Backend send failed: {e}")
        return False

# ==================== Main Separator Class ====================
class SingleSeparator:
    """Single audio source separator for Raspberry Pi 5"""
    
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 backend_url: str = "http://13.238.200.232:8000/sound-events/",
                 led_controller=None):
        """Initialization
        
        Args:
            model_name: AST model name
            backend_url: Backend API URL
            led_controller: LED controller (optional)
        """
        self.backend_url = backend_url
        self.led_controller = led_controller
        self.ast_processor = ASTProcessor(model_name)
        
        print("🎵 Single Separator for Raspberry Pi 5 initialization completed")
        print(f"Model available: {self.ast_processor.is_available}")
    
    def separate_and_process(self, audio_file: str, angle: int = 0, 
                           max_passes: int = 2, output_dir: str = None) -> List[Dict[str, Any]]:
        """Audio file separation and processing
        
        Args:
            audio_file: Audio file path
            angle: Direction angle (0-359)
            max_passes: Maximum number of passes
            output_dir: Output directory (optional)
            
        Returns:
            List of separated source information
        """
        print(f"\n🔍 Starting audio separation: {audio_file}")
        print(f"Angle: {angle}°, Max passes: {max_passes}")
        
        try:
            # Load audio
            audio_4sec, audio_10sec = load_audio(audio_file)
            print(f"Audio loaded: {INPUT_SEC} seconds")
            
            # Execute separation with per-pass processing
            results = multi_pass_separation(
                audio_4sec, audio_10sec, self.ast_processor, max_passes,
                backend_url=self.backend_url, led_controller=self.led_controller, angle=angle
            )
            
            # Process results
            processed_sources = []
            
            for idx, result in enumerate(results):
                # Extract class information
                class_name = result.classification['predicted_class']
                confidence = result.classification['confidence']
                
                source_info = {
                    'pass': idx + 1,
                    'class_name': class_name,
                    'sound_type': result.sound_type,
                    'confidence': confidence,
                    'class_id': result.class_id,
                    'db_mean': result.db_mean,
                    'angle': result.angle,
                    'occurred_at': result.occurred_at,
                    'energy_ratio': result.energy_ratio,
                    'audio': result.separated_audio,
                    'anchor_frames': result.anchor_frames,
                    'separation_mask': result.mask,
                    'backend_sent': result.backend_sent,
                    'led_activated': result.led_activated
                }
                
                # Save file (optional)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = int(time.time())
                    safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_class_name = safe_class_name.replace(' ', '_')
                    filename = f"separated_{timestamp}_{safe_class_name}_{result.sound_type}_pass{idx+1}.wav"
                    filepath = os.path.join(output_dir, filename)
                    sf.write(filepath, result.separated_audio, SR)
                    source_info['file_path'] = filepath
                
                processed_sources.append(source_info)
                
                print(f"🎵 Pass {idx+1}: {class_name} ({result.sound_type}) - Confidence: {confidence:.3f}")
            
            print(f"✅ Separation completed: {len(processed_sources)} sources")
            return processed_sources
            
        except Exception as e:
            print(f"❌ Separation error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def cleanup(self):
        """Resource cleanup"""
        if self.led_controller:
            try:
                self.led_controller.turn_off()
            except:
                pass

# ==================== Factory Function ====================
def create_single_separator(model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                           backend_url: str = "http://13.238.200.232:8000/sound-events/",
                           led_controller=None) -> SingleSeparator:
    """Create SingleSeparator instance"""
    return SingleSeparator(model_name, backend_url, led_controller)

# ==================== Main Execution ====================
def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Single Separator for Raspberry Pi 5')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input audio file path')
    parser.add_argument('--output', '-o', type=str, default='separated_output', help='Output directory')
    parser.add_argument('--angle', '-a', type=int, default=0, help='Direction angle (0-359)')
    parser.add_argument('--max-passes', type=int, default=2, help='Maximum number of passes')
    parser.add_argument('--backend-url', type=str, default="http://13.238.200.232:8000/sound-events/", 
                       help='Backend API URL')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    print("🎵 Single Separator for Raspberry Pi 5")
    print("=" * 50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Angle: {args.angle}°")
    print(f"Max passes: {args.max_passes}")
    print(f"Backend URL: {args.backend_url}")
    print("=" * 50)
    
    # Create and run separator
    separator = create_single_separator(backend_url=args.backend_url)
    
    try:
        results = separator.separate_and_process(
            args.input, 
            angle=args.angle, 
            max_passes=args.max_passes,
            output_dir=args.output
        )
        
        # Result summary
        print(f"\n📊 Separation result summary:")
        print(f"Total {len(results)} sources separated")
        
        for i, result in enumerate(results, 1):
            backend_status = "✅" if result['backend_sent'] else "❌"
            led_status = "💡" if result['led_activated'] else "⭕"
            print(f"{i}. {result['class_name']} ({result['sound_type']}) - "
                  f"Confidence: {result['confidence']:.3f}, "
                  f"Backend: {backend_status}, LED: {led_status}")
        
        print("\n✅ Processing completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        separator.cleanup()

if __name__ == "__main__":
    main()