#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST-guided Source Separator (Final Integrated Version)
A unified pipeline combining the best features from separator.py and test.py:
- Enhanced Frequency Attention with AST model integration
- Adaptive masking strategy with conservative and aggressive modes
- Energy conservation with automatic adjustment mechanisms
- Comprehensive classification and analysis
- Multi-pass separation with intelligent anchor selection
- Energy suppression for previously separated regions
- Separation skipping for high confidence/purity sources
- Robust error handling and dimension management
"""

import os
import time
import warnings
import argparse
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import requests
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# =========================
# Global Constants
# =========================
SR = 16000
WIN_SEC = 4.096
ANCHOR_SEC = 0.512
L_FIXED = int(round(WIN_SEC * SR))

N_FFT, HOP, WINLEN = 400, 160, 400
WINDOW = torch.hann_window(WINLEN)
N_MELS = 128
EPS = 1e-10

# Processing parameters
SMOOTH_T = 19
ALPHA_ATT = 0.80
BETA_PUR = 1.20
W_E = 0.30
TOP_PCT_CORE_IN_ANCHOR = 0.50

# Masking parameters
MASK_SIGMOID_CENTER = 0.6
MASK_SIGMOID_SLOPE = 20.0

# Strategy parameters
OMEGA_Q_CONSERVATIVE = 0.2
OMEGA_MIN_BINS = 5
AST_FREQ_QUANTILE_CONSERVATIVE = 0.4

# Classification thresholds
CONFIDENCE_THRESHOLD = 0.8
PURITY_THRESHOLD = 0.7
RESIDUAL_CONFIDENCE_THRESHOLD = 0.7

# Processing limits
MAX_PASSES = 2
MIN_ERATIO = 0.001  # 0.01에서 0.001로 대폭 감소

# Audio amplification parameters
MIN_ANCHOR_ENERGY = 0.001  # 앵커 에너지 최소 임계값
AMPLIFICATION_FACTOR = 10.0  # 증폭 배수
MAX_AMPLIFICATION = 100.0  # 최대 증폭 제한

# Sound classification mappings
DANGER_IDS = {396, 397, 398, 399, 400, 426, 436}
HELP_IDS = {23, 14, 354, 355, 356, 359}
WARNING_IDS = {288, 364, 388, 389, 390, 439, 391, 392, 393, 395, 440, 441, 443, 456, 469, 470, 478, 479}

# =========================
# AST Caching System
# =========================
class ASTCache:
    def __init__(self):
        self.attention_cache: Dict[str, torch.Tensor] = {}
        self.cls_head_cache: Dict[str, torch.Tensor] = {}
        self.classification_cache: Dict[str, Tuple[str, str, int, float]] = {}
    
    def get_cache_key(self, audio: np.ndarray) -> str:
        return str(hash(audio.tobytes()))
    
    def cache_ast_results(self, audio: np.ndarray, attention_map: torch.Tensor, 
                         cls_features: torch.Tensor, classification_result: Tuple[str, str, int, float]):
        key = self.get_cache_key(audio)
        self.attention_cache[key] = attention_map
        self.cls_head_cache[key] = cls_features
        self.classification_cache[key] = classification_result
    
    def get_attention(self, audio: np.ndarray) -> Optional[torch.Tensor]:
        key = self.get_cache_key(audio)
        return self.attention_cache.get(key)
    
    def get_cls_features(self, audio: np.ndarray) -> Optional[torch.Tensor]:
        key = self.get_cache_key(audio)
        return self.cls_head_cache.get(key)
    
    def get_classification(self, audio: np.ndarray) -> Optional[Tuple[str, str, int, float]]:
        key = self.get_cache_key(audio)
        return self.classification_cache.get(key)

# 전역 캐시 인스턴스
ast_cache = ASTCache()

# 전역 변수: 모델 추론 시작 시간 (하나의 음성 파일에 대해 모든 패스에서 동일하게 사용)
inference_start_time = None

# =========================
# Utility Functions
# =========================
def norm01(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def smooth1d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    ker = torch.ones(k, device=x.device) / k
    return F.conv1d(x.view(1,1,-1), ker.view(1,1,-1), padding=k//2).view(-1)

def align_len_1d(x: torch.Tensor, target_len: int, device: torch.device, mode: str = "linear") -> torch.Tensor:
    """1D 텐서의 길이를 목표 길이에 맞춤"""
    if x.shape[0] == target_len:
        return x
    
    if mode == "linear":
        return F.interpolate(x.view(1, 1, -1), size=target_len, mode="linear", align_corners=False).view(-1)
    elif mode == "nearest":
        return F.interpolate(x.view(1, 1, -1), size=target_len, mode="nearest").view(-1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def get_sound_type(class_id: int) -> str:
    if class_id in DANGER_IDS:
        return "danger"
    elif class_id in HELP_IDS:
        return "help"
    elif class_id in WARNING_IDS:
        return "warning"
    else:
        return "other"

def calculate_global_purity(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor) -> float:
    """전체 오디오에 대한 순수도 계산"""
    if w_bar is None or omega is None:
        return 0.5  # 기본값
    g_pres = presence_from_energy(Xmel, omega)
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)
    global_purity = cos_t_raw.mean().item()
    return global_purity

def should_skip_separation(confidence: float, purity: float, class_id: int) -> bool:
    """분리를 건너뛸지 결정하는 함수"""
    # 신뢰도 임계값 (0.8 이상)
    confidence_threshold = 0.8
    
    # 순수도 임계값 (0.7 이상)
    purity_threshold = 0.7
    
    # "other" 클래스는 분리 건너뛰지 않음
    if get_sound_type(class_id) == "other":
        return False
    
    # 신뢰도와 순수도가 모두 임계값 이상이면 분리 건너뛰기
    return confidence >= confidence_threshold and purity >= purity_threshold

def adaptive_strategy_selection(prev_energy_ratio: float, pass_idx: int) -> str:
    """
    이전 에너지 비율과 패스 인덱스에 따라 적응적 전략 선택
    """
    if pass_idx == 0:
        return "conservative"  # 첫 번째 패스는 항상 보수적
    
    if prev_energy_ratio < 0.3:
        return "aggressive"  # 이전 분리가 효과적이면 공격적
    else:
        return "conservative"  # 이전 분리가 비효과적이면 보수적

def calculate_decibel(audio: np.ndarray) -> Tuple[float, float, float]:
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return -np.inf, -np.inf, -np.inf
    db = 20 * np.log10(rms)
    return db, db, db


# =========================
# Audio Processing Functions
# =========================
def load_fixed_audio(file_path: str) -> np.ndarray:
    """Load audio with 10-second limit"""
    wav, sr = torchaudio.load(file_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    
    audio = wav[0].numpy()
    
    # 10초 이하 오디오만 처리
    max_length = int(10.0 * SR)  # 10 seconds max
    if len(audio) > max_length:
        audio = audio[:max_length]
    
    return audio.astype(np.float32)

@torch.no_grad()
def stft_all(audio: np.ndarray, mel_fb_m2f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform STFT analysis and mel conversion"""
    wav = torch.from_numpy(audio)
    st = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                   window=WINDOW, return_complex=True, center=True)
    
    mag = st.abs()
    P = (mag * mag).clamp_min(EPS)
    phase = torch.angle(st)
    
    # Ensure correct dimensions for mel filterbank
    if mel_fb_m2f.shape[0] != N_MELS:
        mel_fb_m2f = mel_fb_m2f.T.contiguous()
    
    mel_fb_m2f = mel_fb_m2f.to(P.dtype).to(P.device)
    mel_pow = (mel_fb_m2f @ P).clamp_min(EPS)
    
    return st, mag, P, phase, mel_pow

@torch.no_grad()
def ast_attention_freq_time_cached(audio: np.ndarray, extractor, ast_model, T_out: int, F_out: int, mel_fb_m2f: torch.Tensor = None, anchor_region: Tuple[int, int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AST 어텐션에서 시간과 주파수 정보를 모두 추출 (캐싱 지원)
    Returns: (time_attention, freq_attention)
    """
    # 캐시 확인
    cached_attention = ast_cache.get_attention(audio)
    cached_cls = ast_cache.get_cls_features(audio)
    
    if cached_attention is not None and cached_cls is not None:
        # 캐시된 시간 어텐션을 T_out 길이로 보간
        time_attn_interp = F.interpolate(cached_attention.view(1,1,-1), size=T_out, mode="linear", align_corners=False).view(-1)
        time_attn_smooth = smooth1d(time_attn_interp, SMOOTH_T)
        time_attn_norm = norm01(time_attn_smooth)
        
        # 주파수 어텐션은 기본값 사용 (캐시에서 추출 불가)
        freq_attn_norm = torch.ones(F_out) * 0.5
        
        return time_attn_norm, freq_attn_norm
    # 10초로 패딩
    target_len = int(10.0 * SR)
    if len(audio) < target_len:
        audio_padded = np.zeros(target_len, dtype=np.float32)
        audio_padded[:len(audio)] = audio
    else:
        audio_padded = audio[:target_len]
    
    feat = extractor(audio_padded, sampling_rate=SR, return_tensors="pt")
    feat["input_values"] = feat["input_values"].cpu()
    outputs = ast_model(input_values=feat["input_values"], output_attentions=True, return_dict=True)
    
    # Attention map 추출
    attns = outputs.attentions
    if not attns or len(attns) == 0:
        time_attention = torch.ones(101) * 0.5
    else:
        A = attns[-1]
        cls_to_patches = A[0, :, 0, 2:].mean(dim=0)
        
        Fp, Tp = 12, 101
        expected_len = Fp * Tp
        
        if cls_to_patches.numel() != expected_len:
            actual_len = cls_to_patches.numel()
            if actual_len < expected_len:
                cls_to_patches = F.pad(cls_to_patches, (0, expected_len - actual_len))
            else:
                cls_to_patches = cls_to_patches[:expected_len]
        
        full_map = cls_to_patches.reshape(Fp, Tp)
        time_attention = full_map.mean(dim=0)
    
    # 시간 어텐션을 T_out 길이로 보간
    time_attn_interp = F.interpolate(time_attention.view(1,1,-1), size=T_out, mode="linear", align_corners=False).view(-1)
    time_attn_smooth = smooth1d(time_attn_interp, SMOOTH_T)
    time_attn_norm = norm01(time_attn_smooth)
    
    # 주파수 어텐션은 기본값 사용
    freq_attn_norm = torch.ones(F_out) * 0.5
    
    # CLS features 추출 및 캐싱
    if hasattr(outputs, 'last_hidden_state'):
        cls_features = outputs.last_hidden_state[:, 0, :]  # CLS token features
    else:
        cls_features = outputs.logits
    
    # 결과 캐싱
    ast_cache.cache_ast_results(audio, time_attention, cls_features, None)
    
    return time_attn_norm, freq_attn_norm

@torch.no_grad()
def classify_from_cached_attention(audio: np.ndarray, ast_model, anchor_start: int, anchor_end: int) -> Tuple[str, str, int, float]:
    """캐싱된 attention map의 앵커 구간을 사용하여 분류"""
    cls_features = ast_cache.get_cls_features(audio)
    
    if cls_features is None:
        return "Unknown", "other", 0, 0.0
    
    # CLS features를 CPU로 이동 (AST 모델이 CPU에서 실행되므로)
    cls_features = cls_features.cpu()
    
    # CLS features가 이미 logits인 경우와 hidden state인 경우를 구분
    if cls_features.shape[-1] == ast_model.config.num_labels:
        # 이미 logits인 경우
        logits = cls_features
    else:
        # hidden state인 경우 classifier 통과
        logits = ast_model.classifier(cls_features)
    
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = logits.argmax(dim=-1).item()
    confidence = probabilities[0, predicted_class_id].item()
    
    class_name = ast_model.config.id2label[predicted_class_id]
    sound_type = get_sound_type(predicted_class_id)
    
    return class_name, sound_type, predicted_class_id, confidence

@torch.no_grad()
def classify_audio_segment(audio: np.ndarray, extractor, ast_model) -> Tuple[str, str, int, float, List[Tuple[str, float, int]]]:
    """Residual audio classification with confidence threshold"""
    # 10초로 패딩
    target_len = int(10.0 * SR)
    if len(audio) < target_len:
        audio_padded = np.zeros(target_len, dtype=np.float32)
        audio_padded[:len(audio)] = audio
    else:
        audio_padded = audio[:target_len]
    
    feat = extractor(audio_padded, sampling_rate=SR, return_tensors="pt")
    out = ast_model(input_values=feat["input_values"], return_dict=True)
    
    logits = out.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = logits.argmax(dim=-1).item()
    confidence = probabilities[0, predicted_class_id].item()
    
    class_name = ast_model.config.id2label[predicted_class_id]
    sound_type = get_sound_type(predicted_class_id)
    
    # Top 5 클래스 추출
    top1_probs, top1_indices = torch.topk(probabilities[0], 1)
    top1_classes = []
    
    class_id = top1_indices[0].item()
    class_name_top1 = ast_model.config.id2label[class_id]
    prob = top1_probs[0].item()
    top1_classes.append((class_name_top1, prob, class_id))
    
    return class_name, sound_type, predicted_class_id, confidence, top1_classes

@torch.no_grad()
def ast_attention_and_classify(audio: np.ndarray, extractor, ast_model, T_out: int, F_out: int, mel_fb_m2f: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, int, float, List[Tuple[str, float, int]], torch.Tensor]:
    """
    AST 모델을 한 번만 호출하여 어텐션 맵과 분류 결과를 모두 추출
    
    Returns:
        (time_attention, freq_attention, attention_matrix, class_name, sound_type, class_id, confidence, top5_classes, ast_spectrogram)
    """
    # 10초로 패딩
    target_len = int(10.0 * SR)
    if len(audio) < target_len:
        audio_padded = np.zeros(target_len, dtype=np.float32)
        audio_padded[:len(audio)] = audio
    else:
        audio_padded = audio[:target_len]
    
    feat = extractor(audio_padded, sampling_rate=SR, return_tensors="pt")
    
    # AST 모델이 사용하는 실제 스펙트로그램 추출
    ast_spectrogram = feat["input_values"]  # [1, 128, 1024] 형태
    
    outputs = ast_model(input_values=feat["input_values"], output_attentions=True, return_dict=True)
    
    # Attention map 추출
    attns = outputs.attentions
    if not attns or len(attns) == 0:
        time_attention = torch.ones(101) * 0.5
        freq_attention = torch.ones(12) * 0.5
        attention_matrix = torch.ones(12, 101) * 0.5
    else:
        A = attns[-1]
        cls_to_patches = A[0, :, 0, 2:].mean(dim=0)
        
        Fp, Tp = 12, 101
        expected_len = Fp * Tp
        
        if cls_to_patches.numel() != expected_len:
            actual_len = cls_to_patches.numel()
            if actual_len < expected_len:
                cls_to_patches = F.pad(cls_to_patches, (0, expected_len - actual_len))
            else:
                cls_to_patches = cls_to_patches[:expected_len]
        
        full_map = cls_to_patches.reshape(Fp, Tp)
        
        # 시간 어텐션 (주파수 차원으로 평균)
        time_attention = full_map.mean(dim=0)  # [101]
        time_attn_interp = F.interpolate(time_attention.view(1,1,-1), size=T_out, mode="linear", align_corners=False).view(-1)
        time_attn_smooth = smooth1d(time_attn_interp, SMOOTH_T)
        time_attn_norm = norm01(time_attn_smooth)
        
        # 주파수 어텐션 (시간 차원으로 평균)
        freq_attention = full_map.mean(dim=1)  # [12]
        freq_attn_interp = F.interpolate(freq_attention.view(1,1,-1), size=F_out, mode="linear", align_corners=False).view(-1)
        freq_attn_norm = norm01(freq_attn_interp)
        
        # 어텐션 매트릭스 저장
        attention_matrix = full_map
        
        # 시간/주파수 어텐션을 정규화된 버전으로 업데이트
        time_attention = time_attn_norm
        freq_attention = freq_attn_norm
    
    # 분류 결과 추출
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = logits.argmax(dim=-1).item()
    confidence = probabilities[0, predicted_class_id].item()
    
    class_name = ast_model.config.id2label[predicted_class_id]
    sound_type = get_sound_type(predicted_class_id)
    
    # Top 5 클래스 추출
    top5_probs, top5_indices = torch.topk(probabilities[0], 5)
    top5_classes = []
    for i in range(5):
        class_id = top5_indices[i].item()
        class_name_top5 = ast_model.config.id2label[class_id]
        prob = top5_probs[i].item()
        top5_classes.append((class_name_top5, prob, class_id))
    
    return time_attention, freq_attention, attention_matrix, class_name, sound_type, predicted_class_id, confidence, top5_classes, ast_spectrogram

# =========================
# Attention-based Anchor Selection
# =========================
def find_attention_based_anchor(attention_matrix: torch.Tensor, La: int, T: int) -> Tuple[int, int, int, int]:
    """
    어텐션 매트릭스에서 최댓값의 주파수에서 상위 20% 연속 구간을 앵커로 선택하는 함수
    
    Args:
        attention_matrix: [Fp, Tp] 형태의 어텐션 매트릭스 (12, 101)
        La: 앵커 길이 (프레임 수) - 참고용, 실제로는 동적 크기 사용
        T: 전체 시간 프레임 수
    
    Returns:
        (anchor_start, anchor_end, core_start_relative, core_end_relative)
    """
    Fp, Tp = attention_matrix.shape
    
    # 1. 전체 매트릭스에서 최대 어텐션 위치 찾기
    max_attention_value = attention_matrix.max().item()
    max_indices = torch.where(attention_matrix == attention_matrix.max())
    
    if len(max_indices[0]) > 0:
        # 첫 번째 최대값 위치 사용
        max_freq_patch = max_indices[0][0].item()
        max_time_patch = max_indices[1][0].item()
    else:
        # 폴백: 시간축 평균으로 최대값 찾기
        time_attention = attention_matrix.mean(dim=0)
        max_time_patch = torch.argmax(time_attention).item()
        max_attention_value = time_attention[max_time_patch].item()
        max_freq_patch = 0  # 기본값
    
    # 2. 최댓값의 주파수에서 시간축 어텐션 추출
    freq_attention = attention_matrix[max_freq_patch, :]  # [Tp] - 해당 주파수의 시간축 어텐션
    
    # 3. 상위 20% 임계값 계산
    sorted_values, _ = torch.sort(freq_attention, descending=True)
    top_20_percent_idx = int(len(sorted_values) * 0.2)
    threshold = sorted_values[top_20_percent_idx].item()
    
    # 4. 상위 20% 이상인 시간 패치들 찾기
    active_time_mask = freq_attention >= threshold
    active_time_indices = torch.where(active_time_mask)[0]
    
    if len(active_time_indices) == 0:
        # 활성화된 시간 패치가 없으면 최대값 주변만 사용
        active_time_indices = torch.tensor([max_time_patch])
    
    # 5. 연속된 활성화 구간 찾기
    if len(active_time_indices) > 1:
        # 연속된 구간의 시작과 끝 찾기
        consecutive_groups = []
        current_group = [active_time_indices[0].item()]
        
        for i in range(1, len(active_time_indices)):
            if active_time_indices[i].item() - active_time_indices[i-1].item() == 1:
                current_group.append(active_time_indices[i].item())
            else:
                consecutive_groups.append(current_group)
                current_group = [active_time_indices[i].item()]
        consecutive_groups.append(current_group)
        
        # 가장 긴 연속 구간 선택
        longest_group = max(consecutive_groups, key=len)
        core_start_time_patch = longest_group[0]
        core_end_time_patch = longest_group[-1] + 1
    else:
        # 단일 활성화 패치
        core_start_time_patch = active_time_indices[0].item()
        core_end_time_patch = core_start_time_patch + 1
    
    # 5. 시간 패치를 STFT 프레임으로 변환
    if Tp != T:
        core_start_frame = int((core_start_time_patch / Tp) * T)
        core_end_frame = int((core_end_time_patch / Tp) * T)
    else:
        core_start_frame = core_start_time_patch
        core_end_frame = core_end_time_patch
    
    # 6. 동적 앵커 크기 설정 - 활성화된 구간만 사용 (패딩 최소화)
    active_duration = core_end_frame - core_start_frame
    
    # 최소 패딩만 추가 (활성화 구간의 10% 또는 최소 2프레임)
    padding = max(active_duration // 10, 2)
    
    anchor_start = max(0, core_start_frame - padding)
    anchor_end = min(T, core_end_frame + padding)
    
    # 최소 앵커 크기 보장 (너무 작으면 약간만 확장)
    min_anchor_size = max(5, La // 8)  # 최소 5프레임 또는 La의 1/8
    if anchor_end - anchor_start < min_anchor_size:
        center = (anchor_start + anchor_end) // 2
        half_size = min_anchor_size // 2
        anchor_start = max(0, center - half_size)
        anchor_end = min(T, center + half_size)
    
    # 7. 코어 구간을 앵커 내 상대 좌표로 변환
    core_start_relative = max(0, core_start_frame - anchor_start)
    core_end_relative = min(anchor_end - anchor_start, core_end_frame - anchor_start)
    
    return anchor_start, anchor_end, core_start_relative, core_end_relative

# =========================
# Core Separation Logic
# =========================
def calculate_purity(P: torch.Tensor) -> torch.Tensor:
    """Calculate spectral purity (simplified for attention-based anchor selection)"""
    fbins, T = P.shape
    e = P.sum(dim=0)
    
    # Silence 감지: 에너지가 전체 평균의 5% 미만인 구간
    energy_threshold = e.mean() * 0.05
    silence_mask = e < energy_threshold
    
    # 단순화된 순수도: 에너지 기반만 사용
    e_n = e / (e.max() + EPS)
    pur = e_n  # 에너지 정규화만 사용
    
    # Silence 구간의 순수도를 0으로 설정
    pur[silence_mask] = 0.0
    
    return norm01(smooth1d(pur, SMOOTH_T))

def anchor_score(A_t: torch.Tensor, Pur: torch.Tensor) -> torch.Tensor:
    return norm01(smooth1d((A_t.clamp(0,1)**ALPHA_ATT) * (Pur.clamp(0,1)**BETA_PUR), SMOOTH_T))

def anchor_score_with_exclusion(A_t: torch.Tensor, Pur: torch.Tensor, previous_anchors: List[Tuple[int, int]], 
                               attention_matrix: torch.Tensor = None) -> torch.Tensor:
    """이전 앵커 영역만 제외한 앵커 스코어 계산 (어텐션 상위 30% 제외 로직 제거)"""
    Sc = norm01(smooth1d((A_t.clamp(0,1)**ALPHA_ATT) * (Pur.clamp(0,1)**BETA_PUR), SMOOTH_T))
    
    # 이전 앵커 영역들에만 페널티 적용
    for prev_s, prev_e in previous_anchors:
        # 이전 앵커 영역에 작은 페널티 적용 (완전히 0으로 만들지 않음)
        penalty_factor = 0.1  # 10%로 감소
        Sc[prev_s:prev_e] = Sc[prev_s:prev_e] * penalty_factor
        print(f"    🚫 Applied penalty to previous anchor region [{prev_s}:{prev_e}] (factor: {penalty_factor})")
    
    return Sc

# pick_anchor_region 함수는 어텐션 기반 앵커 선정으로 대체됨

def create_frequency_support(Ablk: torch.Tensor, ast_freq_attn: torch.Tensor) -> torch.Tensor:
    """Create frequency support mask from anchor block and AST attention"""
    fbins = Ablk.shape[0]
    
    # Calculate energy per frequency bin
    energy_per_freq = Ablk.sum(dim=1)
    energy_threshold = torch.quantile(energy_per_freq, OMEGA_Q_CONSERVATIVE)
    
    # Create smooth mask using sigmoid instead of binary
    # 시그모이드 함수로 부드러운 전환 생성
    sigmoid_slope = 10.0  # 시그모이드 기울기 (가파를수록 이진에 가까움)
    energy_sigmoid = torch.sigmoid(sigmoid_slope * (energy_per_freq - energy_threshold))
    
    # Ensure minimum number of active bins (부드러운 버전)
    if energy_sigmoid.sum() < OMEGA_MIN_BINS:
        _, top_indices = torch.topk(energy_per_freq, OMEGA_MIN_BINS)
        # 상위 빈들에 더 높은 가중치 부여
        omega = torch.zeros_like(energy_sigmoid)
        omega[top_indices] = 1.0
        # 나머지는 시그모이드 값 유지
        omega = torch.maximum(omega, energy_sigmoid * 0.3)  # 최소 30% 가중치
    else:
        omega = energy_sigmoid
    
    # Apply AST frequency attention weighting (부드러운 버전)
    ast_threshold = torch.quantile(ast_freq_attn, AST_FREQ_QUANTILE_CONSERVATIVE)
    ast_sigmoid = torch.sigmoid(sigmoid_slope * (ast_freq_attn - ast_threshold))
    
    # Combine energy and attention masks (부드러운 곱셈)
    omega = omega * ast_sigmoid
    
    return omega

def create_template(Ablk: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Create spectral template from anchor block using frequency support (separator.py style)"""
    # separator.py 방식: 시간 평균 + 스무딩 + 주파수 가중치 재적용
    om = omega.view(-1, 1)
    w = (Ablk * om).mean(dim=1) * omega  # 1단계: 시간 평균 후 주파수 가중치 재적용
    w = w / (w.sum() + EPS)
    w_sm = F.avg_pool1d(w.view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1)  # 2단계: 3점 평균 필터
    w = (w_sm * omega)  # 3단계: 스무딩 후 다시 주파수 가중치 적용
    w = w / (w.sum() + EPS)  # 최종 정규화
    
    return w

def presence_from_energy(Xmel: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Calculate presence from energy using frequency support (smooth version)"""
    om = omega.view(-1, 1)
    e_omega = (Xmel * om).sum(dim=0)
    e_omega = smooth1d(e_omega, 9)
    thr = torch.quantile(e_omega, 0.20)
    
    # 시그모이드 함수로 부드러운 presence 계산
    sigmoid_slope = 5.0  # presence용 시그모이드 기울기
    g_pres = torch.sigmoid(sigmoid_slope * (e_omega - torch.clamp(thr, min=1e-10)))
    
    return g_pres

def cos_similarity_over_omega(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, g_pres: torch.Tensor) -> torch.Tensor:
    """Calculate cosine similarity over frequency support"""
    Xo = Xmel * omega.view(-1, 1)
    wn = (w_bar * omega) / ((w_bar * omega).norm(p=2) + 1e-8)
    Xn = Xo / (Xo.norm(p=2, dim=0, keepdim=True) + 1e-8)
    cos_t_raw = (wn.view(-1,1) * Xn).sum(dim=0).clamp(0,1) * g_pres
    return cos_t_raw

def improved_adaptive_masking_strategy(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, 
                                     ast_freq_attn: torch.Tensor, P: torch.Tensor, mel_fb_m2f: torch.Tensor,
                                     s: int, e: int, Ablk: torch.Tensor, confidence: float, strategy: str = "conservative") -> torch.Tensor:
    """
    개선된 적응적 마스킹 전략:
    1. 템플릿 주파수 가중치를 유사도 임계치 이상인 모든 구간에 일관성 있게 적용
    2. 마스크 모양이 일관성 있게 적용되도록 개선
    3. 소리가 겹쳐도 일관된 분리 결과 보장
    """
    fbins, T = P.shape
    
    # 1. 기본 코사인 유사도 계산
    g_pres = presence_from_energy(Xmel, omega)
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)
    
    # 2. 동적 유사도 임계값 설정
    similarity_threshold = confidence
    
    
    # 3. 유사도 기반 마스크 생성 (전체 시간 구간에 일관성 있게 적용)
    high_similarity_mask = (cos_t_raw >= similarity_threshold).float()
    low_similarity_mask = (cos_t_raw < similarity_threshold).float()
    
    
    # 4. 주파수 가중치 계산
    # Linear 도메인에서 직접 계산
    if mel_fb_m2f.shape[0] == 128 and mel_fb_m2f.shape[1] == 201:
        # mel_fb_m2f가 [128, 201] 형태인 경우
        omega_lin = ((mel_fb_m2f.T @ omega).clamp_min(0.0) > 1e-12).float()
    elif mel_fb_m2f.shape[0] == 201 and mel_fb_m2f.shape[1] == 128:
        # mel_fb_m2f가 [201, 128] 형태인 경우
        omega_lin = ((mel_fb_m2f @ omega).clamp_min(0.0) > 1e-12).float()
    else:
        # 차원이 맞지 않는 경우 기본값 사용
        print(f"    ⚠️ mel_fb_m2f shape mismatch for omega: {mel_fb_m2f.shape}, using default mask")
        omega_lin = torch.ones(201, device=omega.device)  # N_FFT//2 + 1 = 201
    
    # 앵커 영역의 상위 20% 진폭 주파수 선택 (부드러운 버전)
    anchor_spec = P[:, s:e]
    anchor_max_amp = anchor_spec.max(dim=1).values
    amp_threshold = torch.quantile(anchor_max_amp, 0.8)
    # 시그모이드 함수로 부드러운 진폭 마스크 생성
    sigmoid_slope = 6.0  # 진폭용 시그모이드 기울기
    high_amp_mask_lin = torch.sigmoid(sigmoid_slope * (anchor_max_amp - amp_threshold))
    
    # AST 주파수 어텐션을 Linear 도메인으로 변환 (부드러운 버전)
    ast_freq_threshold = torch.quantile(ast_freq_attn, 0.4 if strategy == "conservative" else 0.2)
    # 시그모이드 함수로 부드러운 마스크 생성
    sigmoid_slope = 8.0  # AST 어텐션용 시그모이드 기울기
    ast_active_mask_mel = torch.sigmoid(sigmoid_slope * (ast_freq_attn - ast_freq_threshold))
    
    # mel_fb_m2f 차원 확인 및 조정 (부드러운 버전)
    if mel_fb_m2f.shape[0] == 128 and mel_fb_m2f.shape[1] == 201:
        # mel_fb_m2f가 [128, 201] 형태인 경우 - 부드러운 변환
        ast_active_mask_lin = (mel_fb_m2f.T @ ast_active_mask_mel).clamp_min(0.0)
    elif mel_fb_m2f.shape[0] == 201 and mel_fb_m2f.shape[1] == 128:
        # mel_fb_m2f가 [201, 128] 형태인 경우 - 부드러운 변환
        ast_active_mask_lin = (mel_fb_m2f @ ast_active_mask_mel).clamp_min(0.0)
    else:
        # 차원이 맞지 않는 경우 기본값 사용
        print(f"    ⚠️ mel_fb_m2f shape mismatch: {mel_fb_m2f.shape}, using default mask")
        ast_active_mask_lin = torch.ones(omega_lin.shape[0], device=omega.device)
    
    # 주파수 가중치 결합
    freq_boost_mask = torch.maximum(high_amp_mask_lin, ast_active_mask_lin)
    
    # 5. 전략에 따른 가중치 적용 (원본을 넘지 않는 선에서 조정)
    if strategy == "conservative":
        # 보수적 방식: 템플릿 가중 주파수에 1.5배 가중치 (원본 제한 내)
        freq_weight = 1.0 + 0.5 * freq_boost_mask  # [1.0, 1.5]
    else:  # aggressive
        # 공격적 방식: 템플릿 가중 주파수에 1.2배 가중치 (원본 제한 내)
        freq_weight = 1.0 + 0.2 * freq_boost_mask  # [1.0, 1.2]
    
    # 6. 개선된 마스크 계산 (일관성 있는 적용)
    M_base = omega_lin.view(-1, 1) * cos_t_raw.view(1, -1)
    
    # 7. 유사도 임계치 기반 일관성 있는 마스킹
    # 유사도가 높은 구간: 템플릿 주파수 가중치 적용
    # 유사도가 낮은 구간: 최소 분리 (0.05)로 일관성 유지
    M_high_sim = M_base * freq_weight.view(-1, 1) * high_similarity_mask.view(1, -1)
    M_low_sim = M_base * 0.05 * low_similarity_mask.view(1, -1)  # 최소 분리
    
    # 8. 최종 마스크 결합
    M_combined = M_high_sim + M_low_sim
    
    # 9. 에너지 보존을 위한 제한
    spec_magnitude = P.sqrt()
    M_lin = torch.minimum(M_combined, spec_magnitude)
    M_lin = torch.clamp(M_lin, 0.0, 1.0)
    
    # print(f"    🎯 Template frequency weights applied to {high_similarity_mask.sum().item()}/{len(cos_t_raw)} time frames")
    # print(f"    🎯 Frequency boost mask active on {freq_boost_mask.sum().item()}/{len(freq_boost_mask)} frequency bins")
    
    # 8.5. 추가 에너지 보존 검증 (마스크 단계에서)
    # 각 시간 프레임별로 마스크가 원본을 초과하지 않도록 검증
    for t in range(T):
        frame_magnitude = spec_magnitude[:, t]
        frame_mask = M_lin[:, t]
        masked_magnitude = frame_mask * frame_magnitude
        
        # 해당 프레임에서 마스크된 에너지가 원본 에너지를 초과하는지 확인
        original_frame_energy = torch.sum(frame_magnitude**2).item()
        masked_frame_energy = torch.sum(masked_magnitude**2).item()
        
        if masked_frame_energy > original_frame_energy * 1.01:  # 1% 허용 오차
            # 마스크를 스케일링하여 에너지 보존
            scale_factor = torch.sqrt(torch.tensor(original_frame_energy / (masked_frame_energy + 1e-8), device=M_lin.device))
            M_lin[:, t] = M_lin[:, t] * scale_factor
    
    # 9. 동적 임계값 미만인 부분에 대한 특별 처리
    # 앵커 부분 에너지의 1%에 해당하는 수치만 추출
    if low_similarity_mask.sum() > 0:  # 동적 임계값 미만인 시간대가 있는 경우
        # 앵커 영역의 평균 에너지 계산
        anchor_energy = Ablk.mean().item()
        target_energy = anchor_energy * 0.01  # 앵커 에너지의 1% (더 강한 분리)
        
        scaled_count = 0
        for t in range(T):
            if low_similarity_mask[t] > 0:  # 유사도 0.6 미만인 시간대
                # 해당 시간대의 현재 마스크 값들
                current_mask_values = M_lin[:, t]
                
                # 현재 마스크로 추출되는 에너지 계산
                current_energy = (current_mask_values * P[:, t]).sum().item()
                
                if current_energy > target_energy:
                    # 목표 에너지에 맞춰 마스크 스케일링
                    scale_factor = target_energy / (current_energy + 1e-8)
                    M_lin[:, t] = M_lin[:, t] * scale_factor
                    scaled_count += 1
        
        #if scaled_count > 0:
            #print(f"  📊 Low similarity regions: {scaled_count} time frames scaled to 1% anchor energy (threshold: {similarity_threshold:.3f})")
    
    # 마스크를 1.0으로 제한 (원본을 초과하지 않도록)
    M_lin = torch.clamp(M_lin, 0.0, 1.0)
    
    # 간단한 통계 출력
    #print(f"  📊 Mask ({strategy}): mean={M_lin.mean().item():.3f}, conf={confidence:.3f}, boost={freq_weight.max().item():.1f}x")
    
    return M_lin, cos_t_raw, freq_weight

# =========================
# Main Processing Pipeline
# =========================
def single_pass(audio: np.ndarray, extractor, ast_model,
                mel_fb_m2f: torch.Tensor,
                used_mask_prev: Optional[torch.Tensor],
                prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor,torch.Tensor]],
                pass_idx: int, out_dir: Optional[str], prev_energy_ratio: float = 1.0,
                separated_time_regions: List[dict] = None,
                previous_anchors: List[Tuple[int, int]] = None,
                original_audio: np.ndarray = None):

    t0 = time.time()
    
    # 1. 전체 오디오 에너지 체크
    overall_energy = np.mean(audio**2)
    
    # 에너지가 너무 낮으면 증폭하지 않고 패스 건너뛰기
    if overall_energy < MIN_ANCHOR_ENERGY * 0.5:  # 더 엄격한 임계값
        print(f"  ⚠️ Audio energy too low ({overall_energy:.6f}), skipping separation")
        return np.zeros_like(audio), audio, 0.0, None, {
            "src_amp": np.zeros_like(audio),
            "res": audio,
            "er": 0.0,
            "class_name": "Silence",
            "sound_type": "other",
            "confidence": 0.0,
            "elapsed": time.time() - time.time(),
            "separation_skipped": True
        }
    # 증폭 제거 - 에너지가 낮아도 원본 그대로 사용
    
    # 2. 오디오로 STFT 계산
    st, mag, P, phase, Xmel = stft_all(audio, mel_fb_m2f)
    fbins, T = P.shape
    La = int(round(ANCHOR_SEC * SR / HOP))

    # 이전에 분리된 시간대의 에너지 억제 (AST 추론 전에 적용)
    audio_for_ast = audio  # AST용 오디오 (기본값: 원본)
    if separated_time_regions and len(separated_time_regions) > 0:
        print(f"  🔇 Suppressing energy in {len(separated_time_regions)} previously separated time regions")
        for region in separated_time_regions:
            time_mask = region['time_mask']
            class_name_prev = region['class_name']
            confidence_prev = region['confidence']
            
            # 시간 마스크 크기 조정
            if time_mask.shape[0] != T:
                time_mask = align_len_1d(time_mask, T, device=P.device, mode="linear")
            
            # 에너지 억제 (0.1%만 남기기) - 훨씬 더 강한 억제
            suppression_factor = 0.999  # 99.9% 억제하여 0.1%만 남김
            P_suppressed = P * (1.0 - time_mask * suppression_factor)
            P = P_suppressed
            
            print(f"    📉 Suppressed {class_name_prev} (conf: {confidence_prev:.3f}) to 0.1% (factor: {suppression_factor:.3f})")
        
        # 억제된 스펙트로그램을 오디오로 변환하여 AST 모델에 전달
        print(f"  🔄 Converting suppressed spectrogram back to audio for AST inference")
        mag_suppressed = torch.sqrt(P)  # Power에서 Magnitude로 변환
        stft_suppressed = mag_suppressed * torch.exp(1j * phase)  # 복소수 STFT 재구성
        audio_for_ast = torch.istft(stft_suppressed, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, 
                                   window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()

    # AST 모델 추론 (패스마다 한 번만 호출)
    A_t, ast_freq_attn, attention_matrix, class_name, sound_type, class_id, confidence, top5_classes, ast_spectrogram = ast_attention_and_classify(audio, extractor, ast_model, T, N_MELS, mel_fb_m2f)
    Pur = calculate_purity(P)
    
    # 앵커 스코어 계산 (이전 앵커 영역과 어텐션 상위 30% 패치 제외)
    Sc = anchor_score(A_t, Pur)

    print(f"  🎯 Pass {pass_idx + 1}: {class_name} (Confidence: {confidence:.3f})")
    
    # Silence 감지 시 즉시 패스 종료 (강화된 감지)
    silence_keywords = ['silence', 'silent', 'quiet', 'no sound', 'mute', 'hush', 'stillness']
    is_silence = (class_name.lower() in silence_keywords or 
                  confidence < 0.05 or 
                  'silence' in class_name.lower() or
                  'quiet' in class_name.lower())
    
    if is_silence:
        print(f"  ⚠️ Silence detected: '{class_name}' (confidence: {confidence:.3f}), stopping separation")
        return np.zeros_like(audio), audio, 0.0, None, {
            "src_amp": np.zeros_like(audio),
            "res": audio,
            "er": 0.0,
            "class_name": "Silence",
            "sound_type": "other",
            "confidence": confidence,
            "elapsed": time.time() - t0,
            "separation_skipped": True,
            "silence_detected": True
        }
    
    # 순수도 계산
    global_purity = calculate_global_purity(Xmel, None, None)  # 임시로 None 전달
    
    # 분리 건너뛰기 확인
    if should_skip_separation(confidence, global_purity, class_id):
        print(f"  ⚡ High confidence & purity detected! Skipping separation...")
        src_amp = audio.copy()
        res = np.zeros_like(audio)
        er = 1.0
        info = {
            "src_amp": src_amp,
            "res": res,
            "er": er,
            "class_name": class_name,
            "sound_type": sound_type,
            "confidence": confidence,
            "elapsed": time.time() - t0,
            "separation_skipped": True
        }
        return src_amp, res, er, None, info

    # Suppress used frames
    if used_mask_prev is not None:
        if used_mask_prev.shape[0] != T:
            used_mask_prev = align_len_1d(used_mask_prev, T, device=Sc.device, mode="linear")
        Sc = Sc * (1.0 - used_mask_prev)

    # Suppress previous anchors (강화된 억제)
    for prev_s, prev_e, prev_mask, prev_weight, prev_anchor_score in prev_anchors:
        if prev_mask.shape[0] != T:
            prev_mask = align_len_1d(prev_mask, T, device=Sc.device, mode="linear")
        ca = (prev_s + prev_e) // 2
        sigma = (prev_e - prev_s) / 4.0  # 더 좁은 시그마로 강한 억제
        idx = torch.arange(T, device=Sc.device) - ca
        # 이전 앵커 영역에 더 강한 억제 적용
        Sc = Sc * (1 - 0.8 * torch.exp(-(idx**2)/(2*(sigma**2)+1e-8)))
        core_s = max(0, ca - La//2); core_e = min(T, ca + La//2)
        Sc[core_s:core_e] *= 0.05  # 핵심 영역은 거의 0으로
        print(f"    🚫 Suppressed previous anchor region [{prev_s}:{prev_e}] (center: {ca})")
    
    # Pick anchor and core regions using AST attention (simplified)
    s, e, core_s_rel, core_e_rel = find_attention_based_anchor(attention_matrix, La, T)
    
    # 무효한 앵커 검사 (silence 구간으로 인한 무효 앵커)
    if s == -1 or e == -1 or core_s_rel == -1 or core_e_rel == -1:
        print(f"  ❌ Invalid anchor detected (likely silence region), stopping separation")
        src_amp = np.zeros_like(audio)
        res = audio.copy()
        er = 0.0
        info = {
            "src_amp": src_amp,
            "res": res,
            "er": er,
            "class_name": class_name,
            "sound_type": sound_type,
            "confidence": confidence,
            "elapsed": time.time() - t0,
            "invalid_anchor": True
        }
        return src_amp, res, er, None, info
    
    # Create anchor block (Ablk) based on the core indices
    Ablk = Xmel[:, s:e].clone()
    if core_s_rel > 0:  Ablk[:, :core_s_rel] = 0
    if core_e_rel < La: Ablk[:, core_e_rel:] = 0

    # Generate frequency support and template
    omega = create_frequency_support(Ablk, ast_freq_attn)
    w_bar = create_template(Ablk, omega)
    
    # 적응적 전략 선택
    strategy = adaptive_strategy_selection(prev_energy_ratio, pass_idx)
    
    # Create separation mask with improved adaptive strategy
    M_lin, cos_t_raw, freq_weight = improved_adaptive_masking_strategy(Xmel, w_bar, omega, ast_freq_attn, P, mel_fb_m2f, s, e, Ablk, confidence, strategy)
    
    # Subtraction in the complex STFT domain for precision
    stft_full = st
    
    # 마스크를 진폭에만 적용하고 위상은 그대로 유지
    # 차원 맞추기
    if M_lin.shape[0] != mag.shape[0]:
        min_freq = min(M_lin.shape[0], mag.shape[0])
        M_lin = M_lin[:min_freq, :]
        mag = mag[:min_freq, :]
        phase = phase[:min_freq, :]
    
    # 마스크를 1.0으로 제한하여 원본을 초과하지 않도록 함
    M_lin = torch.clamp(M_lin, 0.0, 1.0)
    
    # === 에너지 보존 마스킹 전략 ===
    # 디버깅: 차원 확인
    # print(f"  🔍 Debug - M_lin shape: {M_lin.shape}, mag shape: {mag.shape}")
    # print(f"  🔍 Debug - M_lin device: {M_lin.device}, mag device: {mag.device}")
    # print(f"  🔍 Debug - M_lin dtype: {M_lin.dtype}, mag dtype: {mag.dtype}")
    
    # 차원 맞추기
    if M_lin.shape != mag.shape:
        print(f"  ⚠️ Shape mismatch detected! Adjusting M_lin from {M_lin.shape} to {mag.shape}")
        # 더 작은 차원으로 맞추기
        min_freq = min(M_lin.shape[0], mag.shape[0])
        min_time = min(M_lin.shape[1], mag.shape[1])
        M_lin = M_lin[:min_freq, :min_time]
        mag = mag[:min_freq, :min_time]
        phase = phase[:min_freq, :min_time]
        print(f"  ✅ Adjusted shapes - M_lin: {M_lin.shape}, mag: {mag.shape}")
    
    # 1. 분리 결과용: 마스크를 1.0으로 제한하여 원본을 초과하지 않도록 함
    M_lin = torch.clamp(M_lin, 0.0, 1.0)
    mag_masked = M_lin * mag
    
    # 2. 잔여물 계산: 정확한 에너지 보존을 위한 뺄셈
    mag_residual = torch.maximum(mag - mag_masked, torch.zeros_like(mag))
    
    # 3. 에너지 보존 검증 및 조정
    original_energy = torch.sum(mag**2).item()
    masked_energy = torch.sum(mag_masked**2).item()
    residual_energy = torch.sum(mag_residual**2).item()
    total_energy = masked_energy + residual_energy
    energy_ratio = total_energy / (original_energy + 1e-8)
    
    print(f"  📊 Energy: Original={original_energy:.0f}, Masked={masked_energy:.0f}, Residual={residual_energy:.0f}, Ratio={energy_ratio:.3f}")
    
    # 4. 에너지 보존 검증 및 자동 조정
    if energy_ratio > 1.02:  # 2% 허용 오차
        print(f"  ⚠️ Energy conservation issue detected (ratio: {energy_ratio:.3f}), adjusting...")
        # 전체 에너지를 원본에 맞춰 스케일링
        scale_factor = original_energy / (total_energy + 1e-8)
        mag_masked = mag_masked * torch.sqrt(torch.tensor(scale_factor, device=mag.device))
        mag_residual = mag_residual * torch.sqrt(torch.tensor(scale_factor, device=mag.device))
        
        # 재계산
        masked_energy = torch.sum(mag_masked**2).item()
        residual_energy = torch.sum(mag_residual**2).item()
        total_energy = masked_energy + residual_energy
        energy_ratio = total_energy / (original_energy + 1e-8)
        print(f"  🔧 Energy scaled by factor: {scale_factor:.3f}, new ratio: {energy_ratio:.3f}")
    
    # 5. 최종 에너지 보존 검증
    if energy_ratio < 0.98 or energy_ratio > 1.02:
        print(f"  ⚠️ Final energy ratio still problematic: {energy_ratio:.3f}")
    else:
        print(f"  ✅ Energy conservation achieved: {energy_ratio:.3f}")
    
    stft_src = mag_masked * torch.exp(1j * phase)
    stft_res = mag_residual * torch.exp(1j * phase)
    
    # Restore dimensions if needed
    if stft_src.shape[0] != N_FFT//2 + 1:
        target_freq = N_FFT//2 + 1
        if stft_src.shape[0] < target_freq:
            pad_size = target_freq - stft_src.shape[0]
            stft_src = F.pad(stft_src, (0, 0, 0, pad_size))
            stft_res = F.pad(stft_res, (0, 0, 0, pad_size))
        else:
            stft_src = stft_src[:target_freq, :]
            stft_res = stft_res[:target_freq, :]
    
    # Inverse STFT
    src_amp = torch.istft(stft_src, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                         window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()
    
    res = torch.istft(stft_res, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                     window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()
    
    # 증폭된 경우 최종 결과도 증폭된 상태로 유지
    # if amplification_factor > 1.0:
    #     print(f"  🔊 Amplified result (factor: {amplification_factor:.1f})")
    # else:
    #     print(f"  📊 No amplification applied")
    
    # 진폭 검증 및 정규화 (클리핑 방지)
    src_max = np.max(np.abs(src_amp))
    res_max = np.max(np.abs(res))
    
    if src_max > 1.0 or res_max > 1.0:
        print(f"  ⚠️ Clipping prevented (src: {src_max:.3f}, res: {res_max:.3f})")
        if src_max > 1.0:
            src_amp = src_amp / (src_max + 1e-8)
        if res_max > 1.0:
            res = res / (res_max + 1e-8)
    
    # Calculate energy ratio
    e_src = float(np.sum(src_amp**2))
    e_res = float(np.sum(res**2))
    er = e_src / (e_src + e_res + 1e-12)
    
    # 시간 마스크 생성 (다음 패스를 위한 에너지 억제용)
    src_time_mask = torch.zeros(T, device=P.device)
    src_time_indices = torch.where(M_lin.mean(dim=0) > 0.1)[0]  # 마스크가 0.1 이상인 시간대
    if len(src_time_indices) > 0:
        src_time_mask[src_time_indices] = 1.0
    
    # Debug visualization
    if out_dir is not None:
        # Calculate similarity for debug visualization
        g_pres = presence_from_energy(Xmel, omega)
        cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)
        debug_plot(pass_idx, audio, src_amp, res, Sc, P, M_lin, A_t, ast_freq_attn, 
                  s, e, core_s_rel, core_e_rel, class_name, confidence, out_dir, 
                  original_audio=audio, global_confidence=confidence, global_purity=global_purity,
                  similarity_scores=cos_t_raw, amplification_factor=1.0, attention_map=A_t,
                  attention_matrix=attention_matrix, purity_scores=Pur, ast_spectrogram=ast_spectrogram)
    
    # Decibel analysis
    db_min, db_max, db_mean = calculate_decibel(src_amp)
    
    info = {
        "src_amp": src_amp,
        "res": res,
        "er": er,
        "class_name": class_name,
        "sound_type": sound_type,
        "class_id": class_id,
        "confidence": confidence,
        "elapsed": time.time() - t0,
        "separation_skipped": False,
        "src_time_mask": src_time_mask,
        "src_time_indices": src_time_indices,
        "anchor_region": (s, e),  # 앵커 구간 정보 추가
        "anchor_score": Sc,  # 현재 패스의 anchor score 추가
        "db_min": db_min,
        "db_max": db_max,
        "db_mean": db_mean,
        "separation_mask": M_lin  # 분리 마스크 추가
    }
    
    return src_amp, res, er, None, info

# =========================
# Debug Visualization
# =========================
def debug_plot(pass_idx: int, audio: np.ndarray, src_amp: np.ndarray, res: np.ndarray,
               anchor_score: torch.Tensor, P: torch.Tensor, M_lin: torch.Tensor,
               A_t: torch.Tensor, ast_freq_attn: torch.Tensor,
               s: int, e: int, core_s_rel: int, core_e_rel: int,
               class_name: str, confidence: float, out_dir: str,
               original_audio: np.ndarray = None, global_confidence: float = None, global_purity: float = None,
               similarity_scores: torch.Tensor = None, amplification_factor: float = 1.0, attention_map: torch.Tensor = None,
               attention_matrix: torch.Tensor = None, purity_scores: torch.Tensor = None, ast_spectrogram: torch.Tensor = None):
    """Create comprehensive debug visualization (9 plots)"""
    try:
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Pass {pass_idx + 1}: {class_name} (Conf: {confidence:.3f})', fontsize=16, fontweight='bold')
        
        # Calculate consistent amplitude range for all waveforms
        all_amplitudes = []
        if len(audio) > 0:
            all_amplitudes.extend(audio)
        if len(src_amp) > 0:
            all_amplitudes.extend(src_amp)
        if len(res) > 0:
            all_amplitudes.extend(res)
        
        if all_amplitudes:
            max_amp = max(abs(min(all_amplitudes)), abs(max(all_amplitudes)))
            waveform_range = (-max_amp * 1.1, max_amp * 1.1)
        else:
            waveform_range = (-1, 1)
        
        # Time axis for all waveforms
        if len(audio) > 0:
            time_axis = np.linspace(0, len(audio) / SR, len(audio))
        else:
            time_axis = np.array([])
        
        # 1. Original Audio Waveform
        axes[0, 0].plot(time_axis, audio, 'b-', alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title('Original Audio', fontweight='bold')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(waveform_range)
        
        # 2. Separated Source Waveform
        if len(src_amp) > 0:
            src_time_axis = np.linspace(0, len(src_amp) / SR, len(src_amp))
            axes[0, 1].plot(src_time_axis, src_amp, 'g-', alpha=0.7, linewidth=0.5)
            axes[0, 1].set_title('Separated Source', fontweight='bold')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(waveform_range)
        else:
            axes[0, 1].text(0.5, 0.5, 'No separated audio', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Separated Source', fontweight='bold')
            axes[0, 1].set_ylim(waveform_range)
        
        # 3. Residual Audio Waveform
        if len(res) > 0:
            res_time_axis = np.linspace(0, len(res) / SR, len(res))
            axes[0, 2].plot(res_time_axis, res, 'r-', alpha=0.7, linewidth=0.5)
            axes[0, 2].set_title('Residual Audio', fontweight='bold')
            axes[0, 2].set_ylabel('Amplitude')
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_ylim(waveform_range)
        else:
            axes[0, 2].text(0.5, 0.5, 'No residual audio', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Residual Audio', fontweight='bold')
            axes[0, 2].set_ylim(waveform_range)
        
        # STFT 시간축 계산 (HOP 기반) - 모든 시각화에서 공통으로 사용
        n_frames = P.shape[1]
        time_axis_stft = np.arange(n_frames) * HOP / SR
        
        # 4. AST Spectrogram (AST 모델이 실제 사용한 스펙트로그램)
        if ast_spectrogram is not None:
            # AST 스펙트로그램: [1, 128, 1024] -> [128, 1024]
            ast_spec = ast_spectrogram.squeeze(0).cpu().numpy()  # [128, 1024]
            
            # AST 모델의 시간축 계산 (16x16 패치, stride=10, 1024 프레임)
            # AST는 10초 오디오를 1024 프레임으로 처리
            ast_time_axis = np.linspace(0, 10.0, ast_spec.shape[1])  # 0~10초, 1024 프레임
            
            im1 = axes[1, 0].imshow(ast_spec, aspect='auto', origin='lower', 
                                   cmap='viridis', 
                                   extent=[ast_time_axis[0], ast_time_axis[-1], 0, ast_spec.shape[0]])
            axes[1, 0].set_title('AST Spectrogram (Model Input)', fontweight='bold')
            axes[1, 0].set_ylabel('Mel Frequency Bin')
            axes[1, 0].set_xlabel('Time (s)')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        else:
            # Fallback to original power spectrogram
            P_log = torch.log10(P + 1e-10)
            vmin, vmax = torch.quantile(P_log, torch.tensor([0.05, 0.95]))
            
            im1 = axes[1, 0].imshow(P_log.cpu().numpy(), aspect='auto', origin='lower', 
                                   cmap='viridis', vmin=vmin.item(), vmax=vmax.item(),
                                   extent=[time_axis_stft[0], time_axis_stft[-1], 0, P.shape[0]])
            axes[1, 0].set_title('Power Spectrogram (log scale)', fontweight='bold')
            axes[1, 0].set_ylabel('Frequency Bin')
            axes[1, 0].set_xlabel('Time (s)')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 5. Separation Mask
        im2 = axes[1, 1].imshow(M_lin.cpu().numpy(), aspect='auto', origin='lower', 
                               cmap='hot', vmin=0, vmax=1,
                               extent=[time_axis_stft[0], time_axis_stft[-1], 0, M_lin.shape[0]])
        axes[1, 1].set_title('Generated Separation Mask', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency Bin')
        axes[1, 1].set_xlabel('Time (s)')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 6. AST Time Attention vs Anchor Score 비교 (정확한 시간 매핑)
        if A_t is not None and len(A_t) > 0:
            time_frames = np.arange(len(A_t))
            # 시간 축을 실제 초 단위로 변환 (HOP 기반)
            time_seconds = time_frames * HOP / SR
            
            axes[1, 2].plot(time_seconds, A_t.cpu().numpy(), 'purple', linewidth=2, label='AST Time Attention', alpha=0.7)
            
            # anchor_score가 A_t와 같은 길이인지 확인
            if anchor_score is not None and len(anchor_score) == len(A_t):
                axes[1, 2].plot(time_seconds, anchor_score.cpu().numpy(), 'orange', linewidth=2, label='Anchor Score (A+P)', alpha=0.8)
            else:
                # 길이가 다르면 별도로 처리
                if anchor_score is not None and len(anchor_score) > 0:
                    anchor_time_frames = np.arange(len(anchor_score))
                    anchor_time_seconds = anchor_time_frames * HOP / SR
                    axes[1, 2].plot(anchor_time_seconds, anchor_score.cpu().numpy(), 'orange', linewidth=2, label='Anchor Score (A+P)', alpha=0.8)
        else:
            time_seconds = np.array([])
        
        # 최고점 표시 (시간으로 변환) - 인덱스 경계 확인
        if A_t is not None and len(A_t) > 0 and len(time_seconds) > 0:
            max_attn_idx = torch.argmax(A_t).item()
            max_attn_idx_safe = min(max_attn_idx, len(time_seconds) - 1)
            max_attn_time = time_seconds[max_attn_idx_safe]
        else:
            max_attn_time = 0
            
        if anchor_score is not None and len(anchor_score) > 0 and len(time_seconds) > 0:
            max_anchor_idx = torch.argmax(anchor_score).item()
            max_anchor_idx_safe = min(max_anchor_idx, len(time_seconds) - 1)
            max_anchor_time = time_seconds[max_anchor_idx_safe]
        else:
            max_anchor_time = 0
        axes[1, 2].axvline(max_attn_time, color='purple', linestyle='--', alpha=0.5, label=f'Max AST Attn ({max_attn_time:.2f}s)')
        axes[1, 2].axvline(max_anchor_time, color='orange', linestyle='--', alpha=0.5, label=f'Max Anchor Score ({max_anchor_time:.2f}s)')
        
        # 선택된 앵커 영역 표시 (시간으로 변환) - 인덱스 경계 확인
        if len(time_seconds) > 0:
            s_safe = min(s, len(time_seconds) - 1) if s is not None else 0
            e_safe = min(e, len(time_seconds) - 1) if e is not None else len(time_seconds) - 1
            core_s_safe = min(s_safe + core_s_rel, len(time_seconds) - 1) if core_s_rel is not None else s_safe
            core_e_safe = min(s_safe + core_e_rel, len(time_seconds) - 1) if core_e_rel is not None else e_safe
            
            anchor_start_time = time_seconds[s_safe]
            anchor_end_time = time_seconds[e_safe]
            core_start_time = time_seconds[core_s_safe]
            core_end_time = time_seconds[core_e_safe]
        else:
            anchor_start_time = 0
            anchor_end_time = 0
            core_start_time = 0
            core_end_time = 0
        axes[1, 2].axvspan(anchor_start_time, anchor_end_time, alpha=0.3, color='yellow', label=f'Selected Anchor ({anchor_start_time:.2f}-{anchor_end_time:.2f}s)')
        axes[1, 2].axvspan(core_start_time, core_end_time, alpha=0.5, color='red', label=f'Core Region ({core_start_time:.2f}-{core_end_time:.2f}s)')
        
        axes[1, 2].set_title('AST Attention vs Anchor Score (Time Mapped)', fontweight='bold')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_xlabel('Time (seconds)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Time-based Cosine Similarity (정확한 시간 매핑)
        if similarity_scores is not None:
            time_frames = np.arange(len(similarity_scores))
            time_seconds = time_frames * HOP / SR  # HOP 기반 시간축
            
            axes[2, 0].plot(time_seconds, similarity_scores.cpu().numpy(), 'orange', linewidth=2, label='Cosine Similarity')
            
            # 동적 임계값 표시 (AST 신뢰도 기반)
            dynamic_threshold = global_confidence if global_confidence is not None else 0.6
            axes[2, 0].axhline(y=dynamic_threshold, color='red', linestyle='--', alpha=0.7, 
                              label=f'Dynamic Threshold ({dynamic_threshold:.3f})')
            
            # 선택된 앵커 영역 표시 (시간으로 변환) - 인덱스 경계 확인
            s_safe = min(s, len(time_seconds) - 1) if s is not None else 0
            e_safe = min(e, len(time_seconds) - 1) if e is not None else len(time_seconds) - 1
            anchor_start_time = time_seconds[s_safe]
            anchor_end_time = time_seconds[e_safe]
            axes[2, 0].axvspan(anchor_start_time, anchor_end_time, alpha=0.3, color='yellow', 
                             label=f'Selected Anchor ({anchor_start_time:.2f}-{anchor_end_time:.2f}s)')
            
            axes[2, 0].set_title('Time-based Cosine Similarity (Time Mapped)', fontweight='bold')
            axes[2, 0].set_ylabel('Similarity Score')
            axes[2, 0].set_xlabel('Time (seconds)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].set_ylim(0, 1)
        else:
            # Fallback to AST Frequency Attention if similarity not available
            axes[2, 0].plot(ast_freq_attn.cpu().numpy(), 'orange', linewidth=2)
            axes[2, 0].set_title('AST Frequency Attention', fontweight='bold')
            axes[2, 0].set_ylabel('Attention Weight')
            axes[2, 0].set_xlabel('Mel Frequency Bin')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Purity Score 분석 (정확한 시간 매핑)
        if purity_scores is not None:
            time_frames = np.arange(len(purity_scores))
            time_seconds = time_frames * HOP / SR  # HOP 기반 시간축
            
            axes[2, 1].plot(time_seconds, purity_scores.cpu().numpy(), 'purple', linewidth=2, label='Purity Score')
            
            # 최고 순수도와 선택된 앵커 비교 (시간으로 변환) - 인덱스 경계 확인
            max_purity_idx = torch.argmax(purity_scores).item()
            max_purity_idx_safe = min(max_purity_idx, len(time_seconds) - 1)
            max_purity_time = time_seconds[max_purity_idx_safe]
            
            s_safe = min(s, len(time_seconds) - 1) if s is not None else 0
            e_safe = min(e, len(time_seconds) - 1) if e is not None else len(time_seconds) - 1
            core_s_safe = min(s_safe + core_s_rel, len(time_seconds) - 1) if core_s_rel is not None else s_safe
            core_e_safe = min(s_safe + core_e_rel, len(time_seconds) - 1) if core_e_rel is not None else e_safe
            
            anchor_start_time = time_seconds[s_safe]
            anchor_end_time = time_seconds[e_safe]
            core_start_time = time_seconds[core_s_safe]
            core_end_time = time_seconds[core_e_safe]
            
            axes[2, 1].axvline(max_purity_time, color='purple', linestyle='--', alpha=0.7, label=f'Max Purity ({max_purity_time:.2f}s)')
            axes[2, 1].axvspan(anchor_start_time, anchor_end_time, alpha=0.3, color='yellow', label=f'Selected Anchor ({anchor_start_time:.2f}-{anchor_end_time:.2f}s)')
            axes[2, 1].axvspan(core_start_time, core_end_time, alpha=0.5, color='red', label=f'Core Region ({core_start_time:.2f}-{core_end_time:.2f}s)')
            
            # 순수도 통계 정보
            max_purity = purity_scores.max().item()
            mean_purity = purity_scores.mean().item()
            selected_purity = purity_scores[s + (e-s)//2].item() if s + (e-s)//2 < len(purity_scores) else 0.0
            
            axes[2, 1].text(0.02, 0.98, f'Max: {max_purity:.3f} at {max_purity_time:.2f}s\nMean: {mean_purity:.3f}\nSelected: {selected_purity:.3f}', 
                           transform=axes[2, 1].transAxes, fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes[2, 1].set_title('Purity Score Analysis (Time Mapped)', fontweight='bold')
            axes[2, 1].set_ylabel('Purity Score')
            axes[2, 1].set_xlabel('Time (seconds)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].set_ylim(0, 1)
        else:
            # Fallback to Anchor Score if purity not available
            if anchor_score is not None and len(anchor_score) > 0:
                time_frames = np.arange(len(anchor_score))
                time_seconds = time_frames * HOP / SR  # HOP 기반 시간축
            else:
                time_frames = np.array([])
                time_seconds = np.array([])
            
            if len(time_seconds) > 0 and anchor_score is not None:
                axes[2, 1].plot(time_seconds, anchor_score.cpu().numpy(), 'cyan', linewidth=2, label='Final Anchor Score')
                
                max_score_idx = torch.argmax(anchor_score).item()
                max_score_idx_safe = min(max_score_idx, len(time_seconds) - 1)
                max_score_time = time_seconds[max_score_idx_safe]
            else:
                axes[2, 1].text(0.5, 0.5, 'No anchor score data', ha='center', va='center', transform=axes[2, 1].transAxes)
                max_score_time = 0  # 기본값 설정
            
            axes[2, 1].set_title('Anchor Score Analysis (Time Mapped)', fontweight='bold')
            axes[2, 1].set_ylabel('Score')
            axes[2, 1].set_xlabel('Time (seconds)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        # Add similarity threshold line if available
        if similarity_scores is not None:
            axes[2, 1].axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Similarity Threshold (0.6)')
            axes[2, 1].legend()
        
        # 9. Energy Analysis with Similarity Statistics
        if anchor_score is not None and len(anchor_score) > 0:
            time_frames = np.arange(len(anchor_score))
        else:
            time_frames = np.array([])
        original_energy = np.sum(audio**2)
        src_energy = np.sum(src_amp**2)
        res_energy = np.sum(res**2)
        
        energy_data = [original_energy, src_energy, res_energy]
        energy_labels = ['Original', 'Source', 'Residual']
        colors = ['blue', 'green', 'red']
        
        # AST Attention Matrix (2D) - AST 스펙트로그램과 정확히 매칭
        if attention_matrix is not None:
            # Enhanced AST Attention Matrix Visualization
            attn_matrix_np = attention_matrix.cpu().numpy()
            
            # AST 모델의 시간축 계산 (16x16 패치, stride=10, 1024 프레임)
            # 128 mel bins, 1024 time frames
            # 패치 계산: (128-16)/10 + 1 = 12.2 → 12개 (주파수), (1024-16)/10 + 1 = 101.8 → 101개 (시간)
            # 총 패치: 12 × 101 = 1212개 + CLS(1) + distill(1) = 1214개
            # 실제 어텐션 매트릭스는 12x101 형태이므로 101 패치 사용
            ast_time_axis = np.linspace(0, 10.0, attn_matrix_np.shape[1])  # 0~10초, 101 패치
            
            # 어텐션 매트릭스 시각화 (AST 시간축과 정확히 매칭)
            im = axes[2, 2].imshow(attn_matrix_np, aspect='auto', cmap='plasma', origin='lower',
                                  extent=[ast_time_axis[0], ast_time_axis[-1], 0, attn_matrix_np.shape[0]])
            axes[2, 2].set_title(f'AST Attention Matrix\n({attn_matrix_np.shape[0]} freq × {attn_matrix_np.shape[1]} time patches)', 
                                fontweight='bold', fontsize=10)
            axes[2, 2].set_xlabel('Time (s)')
            axes[2, 2].set_ylabel('Frequency Patches')
            
            # Colorbar 추가
            cbar = plt.colorbar(im, ax=axes[2, 2], fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', rotation=270, labelpad=15)
            
            # 앵커 영역 하이라이트 (실제 활성화된 패치 표시)
            if s is not None and e is not None:
                # STFT 프레임을 AST 시간축으로 변환
                stft_time_start = s * HOP / SR
                stft_time_end = e * HOP / SR
                
                # AST 시간축에서 해당 시간 범위 찾기
                ast_start_idx = np.argmin(np.abs(ast_time_axis - stft_time_start))
                ast_end_idx = np.argmin(np.abs(ast_time_axis - stft_time_end))
                
                # 앵커 영역 하이라이트 (AST 시간축 기준)
                axes[2, 2].axvspan(ast_time_axis[ast_start_idx], ast_time_axis[ast_end_idx], 
                                 alpha=0.3, color='red', 
                                 label=f'Anchor ({stft_time_start:.2f}-{stft_time_end:.2f}s)')
                
                # 실제 활성화된 패치 표시 (어텐션 매트릭스에서 상위 20% 패치)
                if attention_matrix is not None:
                    # 최댓값의 주파수에서 상위 20% 패치 찾기
                    max_attention_value = attention_matrix.max().item()
                    max_indices = torch.where(attention_matrix == attention_matrix.max())
                    
                    if len(max_indices[0]) > 0:
                        max_freq_patch = max_indices[0][0].item()
                        freq_attention = attention_matrix[max_freq_patch, :]
                        
                        # 상위 20% 임계값 계산
                        sorted_values, _ = torch.sort(freq_attention, descending=True)
                        top_20_percent_idx = int(len(sorted_values) * 0.2)
                        threshold = sorted_values[top_20_percent_idx].item()
                        
                        # 상위 20% 이상인 패치들 찾기
                        active_patches = torch.where(freq_attention >= threshold)[0]
                        
                        # 활성화된 패치들을 시각화에 표시
                        for patch_idx in active_patches:
                            if patch_idx < len(ast_time_axis):
                                patch_time = ast_time_axis[patch_idx.item()]
                                # 활성화된 패치를 세로선으로 표시
                                axes[2, 2].axvline(patch_time, color='yellow', alpha=0.7, linewidth=1)
                
                axes[2, 2].legend(fontsize=8)
                
                # Y축을 주파수로 표시
                freq_ticks = np.linspace(0, attn_matrix_np.shape[0]-1, 4)
                freq_labels = [f'{int(i * (SR/2) / attn_matrix_np.shape[0])}Hz' for i in freq_ticks]
                axes[2, 2].set_yticks(freq_ticks)
                axes[2, 2].set_yticklabels(freq_labels, fontsize=8)
            
            # 어텐션 통계 정보 추가
            max_attn = attn_matrix_np.max()
            mean_attn = attn_matrix_np.mean()
            axes[2, 2].text(0.02, 0.98, f'Max: {max_attn:.3f}\nMean: {mean_attn:.3f}', 
                           transform=axes[2, 2].transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Fallback: show energy ratio as text
            axes[2, 2].text(0.5, 0.5, f'Energy Ratio: {energy_ratio:.3f}', 
                           ha='center', va='center', transform=axes[2, 2].transAxes, 
                           fontsize=14, fontweight='bold')
            axes[2, 2].set_title('Energy Ratio', fontweight='bold')
            axes[2, 2].axis('off')
        
        # Add similarity statistics and energy ratio info as text
        if similarity_scores is not None:
            low_sim_count = (similarity_scores < 0.6).sum().item()
            high_sim_count = (similarity_scores >= 0.6).sum().item()
            total_frames = len(similarity_scores)
            low_sim_pct = (low_sim_count / total_frames) * 100
            high_sim_pct = (high_sim_count / total_frames) * 100
            
            # Energy ratio calculation
            energy_ratio = src_energy / (src_energy + res_energy + 1e-8)
            
            # Amplification info (제거됨)
            amp_info = ''
            
            stats_text = f'Similarity Stats:\nLow (<0.6): {low_sim_count} ({low_sim_pct:.1f}%)\nHigh (≥0.6): {high_sim_count} ({high_sim_pct:.1f}%)\n\nEnergy Ratio: {energy_ratio:.3f}{amp_info}'
            axes[2, 2].text(0.02, 0.98, stats_text, transform=axes[2, 2].transAxes, 
                           verticalalignment='top', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        debug_path = os.path.join(out_dir, f'debug_pass_{pass_idx + 1}.png')
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        #print(f"  📊 Debug plot saved: {debug_path}")
        
    except Exception as e:
        import traceback
        print(f"  ❌ Debug plot failed: {e}")
        print(f"  📍 Error location: {traceback.format_exc()}")

# =========================
# Backend Integration
# =========================
def send_to_backend(sound_type: str, sound_detail: str, decibel: float, angle: int = 0, 
                   backend_url: str = "http://13.238.200.232:8000/sound-events/", 
                   user_id: int = 6, occurred_at: str = None) -> bool:
    """
    백엔드로 분리된 소리 정보 전송
    
    Args:
        sound_type: 소리 타입 (danger/warning/help)
        sound_detail: 구체적인 소리 클래스명
        decibel: 데시벨 값
        angle: 방향각 (기본값: 0)
        backend_url: 백엔드 API URL
        user_id: 사용자 ID
        occurred_at: 소리 발생 시간 (ISO format, None이면 현재 시간 사용)
    
    Returns:
        bool: 전송 성공 여부
    """
    try:
        # 백엔드 전송 데이터 구성
        data = {
            "user_id": user_id,
            "sound_type": sound_type,
            "sound_detail": sound_detail,
            "angle": angle,
            "occurred_at": occurred_at if occurred_at else datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sound_icon": "string",
            "location_image_url": "string",
            "decibel": float(decibel),
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AST-Separator/1.0'
        }
        
        print(f"🔄 Sending to backend: {backend_url}")
       # print(f"📤 Data: {data}")
        
        # SSL 경고 비활성화 (테스트용)
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response = requests.post(
            backend_url, 
            json=data, 
            headers=headers,
            timeout=2.0,
            verify=False
        )
        
        if response.status_code == 200:
            print(f"✅ Sent to backend: {sound_detail} ({sound_type}) at {angle}°")
            return True
        else:
            print(f"❌ Backend error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        print(f"❌ Backend connection timeout: {backend_url}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Backend connection error: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Backend send error: {e}")
        return False

def calculate_sound_occurrence_time(mask: torch.Tensor, inference_start_time: datetime, 
                                   audio_duration: float = 4.096) -> str:
    """
    마스크에서 가장 빠른 시간을 계산하여 실제 소리 발생 시간을 반환
    
    Args:
        mask: 분리 마스크 [freq_bins, time_frames]
        inference_start_time: 모델 추론 시작 시간
        audio_duration: 오디오 파일 길이 (초, 기본값: 4.0초)
    
    Returns:
        str: ISO format의 실제 소리 발생 시간
    """
    try:
        # 마스크에서 시간축으로 평균을 내어 활성화된 시간대 찾기
        time_activity = mask.mean(dim=0)  # [time_frames]
        
        # 활성화 임계값 (마스크 평균의 10% 이상)
        threshold = time_activity.max() * 0.1
        
        # 활성화된 시간 프레임들 찾기
        active_frames = torch.where(time_activity >= threshold)[0]
        
        if len(active_frames) > 0:
            # 가장 빠른 활성화 시간 (프레임 인덱스)
            earliest_frame = active_frames[0].item()
            
            # 프레임을 시간으로 변환 (HOP 기반)
            earliest_time_in_audio = earliest_frame * HOP / SR
            
            # 실제 발생 시간 = 추론 시작 시간 - 4초 + 타겟 소리 처음 생긴 구간
            # (4초 녹음의 시작 시점에서 타겟 소리가 처음 생긴 시간을 더함)
            actual_occurrence_time = inference_start_time - timedelta(seconds=audio_duration) + timedelta(seconds=earliest_time_in_audio)
            
            return actual_occurrence_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            # 활성화된 시간이 없으면 추론 시작 시간 사용
            return inference_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            
    except Exception as e:
        print(f"❌ Sound occurrence time calculation error: {e}")
        # 에러 시 추론 시작 시간 사용
        return inference_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

def calculate_decibel_from_audio(audio: np.ndarray) -> float:
    """
    오디오에서 데시벨 값 계산
    
    Args:
        audio: 오디오 데이터 (numpy array)
    
    Returns:
        float: 데시벨 값
    """
    try:
        # RMS 계산
        rms = np.sqrt(np.mean(audio**2))
        
        # 데시벨 변환 (20 * log10(rms))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -60.0  # 최소값
        
        # 합리적인 범위로 제한 (0-100 dB)
        db = max(0, min(100, db))
        
        return db
    except Exception as e:
        print(f"❌ Decibel calculation error: {e}")
        return 60.0  # 기본값

# =========================
# Main Function
# =========================
def main():
    ap = argparse.ArgumentParser(description="AST-guided Source Separator (Final Integrated Version)")
    ap.add_argument("--input", required=True, help="Input audio file")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--passes", type=int, default=MAX_PASSES, help="Number of separation passes")
    ap.add_argument("--no-debug", action="store_true", help="Disable debug visualization")
    ap.add_argument("--strategy", choices=["conservative", "aggressive", "adaptive"], default="adaptive", help="Masking strategy")
    ap.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    ap.add_argument("--backend-url", default="http://13.238.200.232:8000/sound-events/", help="Backend API URL")
    ap.add_argument("--user-id", type=int, default=6, help="User ID for backend")
    ap.add_argument("--angle", type=int, default=0, help="Sound direction angle (0-360)")
    ap.add_argument("--no-backend", action="store_true", help="Disable backend transmission")
    
    args = ap.parse_args()
    
    # Device setup
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load audio with dynamic length
    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 AST-guided Source Separator (Final Integrated Version)\n{'='*64}")
    print(f"Input: {args.input} ({len(audio0)/SR:.3f}s)")
    print(f"Strategy: {args.strategy}")
    print(f"Features: Adaptive Masking, Energy Conservation, Classification, Energy Suppression")
    print(f"Debug visualization: {'OFF' if args.no_debug else 'ON'}")
    
    # Dynamic pass calculation (only for standalone separator.py, not when called from class_eval.py)
    audio_duration = len(audio0) / SR
    
    # Check if this is being called from class_eval.py by looking for specific arguments or environment
    is_evaluation_mode = hasattr(args, 'evaluation_mode') and args.evaluation_mode
    
    if not is_evaluation_mode and args.passes == 3:  # Only adjust for standalone use
        # For standalone separator.py, use duration-based calculation
        dynamic_passes = max(1, min(10, int(audio_duration / 2.0)))
        if dynamic_passes != args.passes:
            print(f"🔄 Adjusted passes from {args.passes} to {dynamic_passes} based on audio length ({audio_duration:.1f}s)")
            args.passes = dynamic_passes
    elif is_evaluation_mode:
        print(f"🎯 Evaluation mode: Using {args.passes} passes (set by class_eval.py)")
    
    # Mel filterbank setup
    fbins = N_FFT//2 + 1
    mel_fb_f2m = torchaudio.functional.melscale_fbanks(
        n_freqs=fbins, f_min=0.0, f_max=SR/2, n_mels=N_MELS,
        sample_rate=SR, norm="slaney"
    )
    mel_fb_m2f = mel_fb_f2m.T.contiguous()
    
    # AST model setup
    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    # Load AST model with quantization support
    try:
        ast_model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            attn_implementation="eager"
        ).to(device).eval()
        
        # Apply dynamic quantization for better performance
        print("🔍 Applying model quantization...")
        try:
            # Dynamic quantization for CPU optimization
            ast_model = torch.quantization.quantize_dynamic(
                ast_model, 
                {torch.nn.Linear, torch.nn.Conv1d}, 
                dtype=torch.qint8
            )
            print("✅ Model quantization completed")
        except Exception as e:
            print(f"⚠️ Quantization failed, using original model: {e}")
            
    except Exception as e:
        print(f"❌ AST model loading failed: {e}")
        return
    
    # Processing variables
    cur = audio0.copy()
    used_mask_prev = None
    prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor]] = []
    total_t0 = time.time()
    saved = 0
    prev_energy_ratio = 1.0
    separated_time_regions = []  # 이전에 분리된 시간대 정보 저장
    previous_anchors = []  # 이전 패스에서 사용된 앵커 구간 정보
    
    # 모델 추론 시작 시간 설정 (전역 변수)
    global inference_start_time
    inference_start_time = datetime.utcnow()
    print(f"🕐 Model inference started at: {inference_start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    
    # Main processing loop - 패스마다 AST 추론 수행
    total_ast_calls = 0
    for i in range(max(1, args.passes)):
        print(f"\n--- Pass {i + 1} ---")
        
        # 패스마다 AST 모델 호출 (현재 오디오로)
        total_ast_calls += 1
        
        src_amp, res, er, used_mask, info = single_pass(
            cur, extractor, ast_model, mel_fb_m2f, used_mask_prev, prev_anchors, 
            i, args.output if not args.no_debug else None, prev_energy_ratio,
            separated_time_regions, previous_anchors, original_audio=cur
        )
        
        # 잔여 오디오 에너지 검사 - 의미없는 수준이면 패스 중단
        residual_energy = np.sum(res ** 2)
        original_energy = np.sum(cur ** 2)
        residual_ratio = residual_energy / (original_energy + 1e-10)
        
        # 에너지가 너무 낮거나 신뢰도가 너무 낮으면 중단
        if residual_ratio < 0.02 or info['confidence'] < 0.10:  # 잔여 에너지 2% 미만 또는 신뢰도 10% 미만
            print(f"  ⚠️ Stopping separation - Residual energy: {residual_ratio:.3f}, Confidence: {info['confidence']:.3f}")
            break
        
        # Save separated source
        if info.get("separation_skipped", False):
            print(f"  ⚡ Separation skipped - using original audio")
            src_path = os.path.join(args.output, f"{i:02d}_{info['class_name']}.wav")
        else:
            src_path = os.path.join(args.output, f"{i:02d}_{info['class_name']}.wav")
        
        torchaudio.save(src_path, torch.from_numpy(src_amp).unsqueeze(0), SR)
        saved += 1
        
        # 소리 발생 시간 계산 (마스크 기반)
        if 'separation_mask' in info and info['separation_mask'] is not None:
            occurred_at = calculate_sound_occurrence_time(
                info['separation_mask'], 
                inference_start_time, 
                audio_duration=len(audio0)/SR
            )
        else:
            occurred_at = inference_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        print(f"  🕐 Sound occurrence time: {occurred_at}")
        
        # 백엔드로 분리된 소리 정보 전송
        if not args.no_backend:
            try:
                # 데시벨 계산
                decibel = calculate_decibel_from_audio(src_amp)
                
                # 백엔드 전송
                backend_success = send_to_backend(
                    sound_type=info['sound_type'],
                    sound_detail=info['class_name'],
                    decibel=decibel,
                    angle=args.angle,
                    backend_url=args.backend_url,
                    user_id=args.user_id,
                    occurred_at=occurred_at
                )
                
                if backend_success:
                    print(f"  📤 Backend sent")
                else:
                    print(f"  ❌ Backend failed")
                    
            except Exception as e:
                print(f"  ❌ Backend error: {e}")
        else:
            print(f"  🚫 Backend disabled")
        
        # 분리된 시간대 정보 수집 (분리 건너뛰기가 아닌 경우만)
        if not info.get("separation_skipped", False):
            separated_time_regions.append({
                'time_mask': info['src_time_mask'],
                'class_name': info['class_name'],
                'confidence': info['confidence'],
                'pass_idx': i
            })
            
            # 앵커 구간 정보 수집 (anchor score 포함)
            if 'anchor_region' in info and 'anchor_score' in info:
                prev_anchors.append((
                    info['anchor_region'][0],  # prev_s
                    info['anchor_region'][1],  # prev_e
                    info['src_time_mask'],     # prev_mask
                    torch.ones_like(info['src_time_mask']),  # prev_weight (기본값)
                    info['anchor_score']       # prev_anchor_score
                ))
        
        # Update for next pass
        prev_energy_ratio = er
        cur = res
        used_mask_prev = used_mask
        
        # Early termination if energy ratio is too low AND residual energy is also low
        residual_energy = float(np.sum(res**2))
        if er < MIN_ERATIO and residual_energy < 0.001:  # 잔여물 에너지도 낮을 때만 종료
            print(f"  ⚠️ Energy ratio {er:.3f} below threshold {MIN_ERATIO} and residual energy {residual_energy:.6f} too low, stopping...")
            break
        elif er < MIN_ERATIO:
            print(f"  ⚠️ Energy ratio {er:.3f} below threshold {MIN_ERATIO}, but residual energy {residual_energy:.6f} is sufficient, continuing...")
    
    # Final residual classification
    if len(cur) > 0 and np.max(np.abs(cur)) > 1e-6:
        print(f"\n--- Final Residual Classification ---")
        # 실제 AST 모델 호출 (residual 오디오로)
        class_name, sound_type, class_id, confidence, top5_classes = classify_audio_segment(cur, extractor, ast_model)
        print(f"  🎯 Residual: {class_name}")
        
        if confidence >= 0.1:
            print(f"  ✅ High confidence residual detected, saving...")
            residual_path = os.path.join(args.output, f"{saved:02d}_{class_name}.wav")
            torchaudio.save(residual_path, torch.from_numpy(cur).unsqueeze(0), SR)
            saved += 1
            
            # 백엔드로 잔여물 소리 정보 전송
            if not args.no_backend:
                try:
                    # 데시벨 계산
                    decibel = calculate_decibel_from_audio(cur)
                    
                    # 잔여물의 경우 전체 오디오 길이를 사용하여 발생 시간 계산
                    # (잔여물은 전체 시간에 걸쳐 있을 수 있으므로 0초로 설정)
                    occurred_at = (inference_start_time - timedelta(seconds=len(audio0)/SR)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    print(f"  🕐 Residual sound occurrence time: {occurred_at}")
                    
                    # 백엔드 전송
                    backend_success = send_to_backend(
                        sound_type=sound_type,
                        sound_detail=class_name,
                        decibel=decibel,
                        angle=args.angle,
                        backend_url=args.backend_url,
                        user_id=args.user_id,
                        occurred_at=occurred_at
                    )
                    
                    if backend_success:
                        print(f"  📤 Residual backend sent")
                    else:
                        print(f"  ❌ Residual backend failed")
                        
                except Exception as e:
                    print(f"  ❌ Residual backend error: {e}")
            else:
                print(f"  🚫 Residual backend disabled")
        else:
            print(f"  📝 Low confidence residual, saving as generic...")
            residual_path = os.path.join(args.output, f"{saved:02d}_residual.wav")
            torchaudio.save(residual_path, torch.from_numpy(cur).unsqueeze(0), SR)
    
    total_time = time.time() - total_t0
    print(f"\n{'='*50}")
    print(f"✅ Completed in {total_time:.2f}s - {saved} files saved")
    print(f"🧠 AST calls: {total_ast_calls}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
