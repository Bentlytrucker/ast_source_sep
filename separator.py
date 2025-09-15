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
from datetime import datetime

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
MAX_PASSES = 3
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
    """Load and fix audio to WIN_SEC length"""
    wav, sr = torchaudio.load(file_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    
    audio = wav[0].numpy()
    
    # Fix length to WIN_SEC
    if len(audio) < L_FIXED:
        audio = np.pad(audio, (0, L_FIXED - len(audio)), mode='constant')
    else:
        audio = audio[:L_FIXED]
    
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
def ast_attention_freq_time(audio: np.ndarray, extractor, ast_model, T_out: int, F_out: int, mel_fb_m2f: torch.Tensor = None, anchor_region: Tuple[int, int] = None) -> Tuple[torch.Tensor, torch.Tensor, str, str, int, float, List[Tuple[str, float, int]]]:
    """
    AST 어텐션에서 시간과 주파수 정보를 모두 추출하고 분류 결과도 함께 반환
    anchor_region이 제공되면 해당 영역만 남기고 나머지는 0으로 만들어서 더 정확한 분류 수행
    Returns: (time_attention, freq_attention, class_name, sound_type, class_id, confidence, top5_classes)
    """
    # 10초로 패딩
    target_len = int(10.0 * SR)
    if len(audio) < target_len:
        audio_padded = np.zeros(target_len, dtype=np.float32)
        audio_padded[:len(audio)] = audio
    else:
        audio_padded = audio[:target_len]
    
    # 앵커 영역이 제공된 경우 해당 영역만 남기고 나머지는 0으로 설정
    if anchor_region is not None:
        anchor_start, anchor_end = anchor_region
        # STFT 프레임을 오디오 샘플로 변환
        frame_to_sample = HOP  # STFT hop length
        audio_start = anchor_start * frame_to_sample
        audio_end = anchor_end * frame_to_sample
        
        # 오디오 길이 내에서 클리핑
        audio_start = max(0, min(audio_start, len(audio_padded)))
        audio_end = max(0, min(audio_end, len(audio_padded)))
        
        # 앵커 영역만 남기고 나머지는 0으로 설정
        audio_anchored = np.zeros_like(audio_padded)
        audio_anchored[audio_start:audio_end] = audio_padded[audio_start:audio_end]
        
        print(f"  🎯 Using anchor-focused classification: frames {anchor_start}-{anchor_end} (samples {audio_start}-{audio_end})")
        audio_for_classification = audio_anchored
    else:
        audio_for_classification = audio_padded
    
    feat = extractor(audio_for_classification, sampling_rate=SR, return_tensors="pt")
    out = ast_model(input_values=feat["input_values"], output_attentions=True, return_dict=True)
    attns = out.attentions
    
    # 분류 결과 추출
    logits = out.logits
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
    
    if not attns or len(attns) == 0:
        return torch.ones(T_out) * 0.5, torch.ones(F_out) * 0.5, class_name, sound_type, predicted_class_id, confidence, top5_classes
    
    # 마지막 레이어의 어텐션 사용
    A = attns[-1]  # [batch, heads, seq, seq]
    
    # CLS 토큰(0번)에서 패치들(2번부터)로의 어텐션
    cls_to_patches = A[0, :, 0, 2:].mean(dim=0)  # 헤드들 평균
    
    # AST는 12(freq) x 101(time) 패치 구조
    Fp, Tp = 12, 101
    expected_len = Fp * Tp
    
    if cls_to_patches.numel() != expected_len:
        actual_len = cls_to_patches.numel()
        if actual_len < expected_len:
            cls_to_patches = F.pad(cls_to_patches, (0, expected_len - actual_len))
        else:
            cls_to_patches = cls_to_patches[:expected_len]
    
    # 2D 맵으로 재구성
    full_map = cls_to_patches.reshape(Fp, Tp)  # [12, 101]
    
    # 실제 오디오 길이에 해당하는 부분만 추출 (정확한 시간 매핑)
    original_audio_duration = len(audio) / SR  # 실제 오디오 길이 (초)
    target_duration = target_len / SR  # 패딩된 오디오 길이 (초)
    original_audio_ratio = original_audio_duration / target_duration  # 정확한 비율
    
    # AST 패치 구조: 101개 패치가 10초를 커버
    time_patches_to_use = int(Tp * original_audio_ratio)  # 사용할 시간 패치 수
    
    print(f"  🔍 Audio duration: {original_audio_duration:.3f}s, Target: {target_duration:.3f}s, Ratio: {original_audio_ratio:.3f}")
    print(f"  🔍 Using {time_patches_to_use}/{Tp} time patches for {original_audio_duration:.3f}s audio")
    
    # 실제 오디오에 해당하는 어텐션 맵만 추출
    full_map_cropped = full_map[:, :time_patches_to_use]  # [12, time_patches_to_use]
    
    # 시간 어텐션 (주파수 차원으로 평균) - 크롭된 맵 사용
    time_attn = full_map_cropped.mean(dim=0)  # [time_patches_to_use]
    time_attn_interp = F.interpolate(time_attn.view(1,1,-1), size=T_out, mode="linear", align_corners=False).view(-1)
    time_attn_smooth = smooth1d(time_attn_interp, SMOOTH_T)
    time_attn_norm = norm01(time_attn_smooth)
    
    # 주파수 어텐션 (시간 차원으로 평균) - 크롭된 맵 사용
    freq_attn_mel = full_map_cropped.mean(dim=1)  # [12] - Mel 스케일
    
    # Mel → Linear 변환 (mel_fb_m2f 사용)
    if mel_fb_m2f is not None and mel_fb_m2f.shape[0] == freq_attn_mel.shape[0]:
        print(f"  🔍 Debug - Converting Mel attention {freq_attn_mel.shape} to Linear using mel_fb_m2f {mel_fb_m2f.shape}")
        freq_attn_linear = torch.matmul(mel_fb_m2f, freq_attn_mel)  # [F_out]
        print(f"  🔍 Debug - Converted to Linear attention: {freq_attn_linear.shape}")
        freq_attn_norm = norm01(freq_attn_linear)
    else:
        # Fallback: 단순 보간 (차원이 맞지 않거나 mel_fb_m2f가 없는 경우)
        if mel_fb_m2f is not None:
            print(f"  ⚠️ Dimension mismatch: AST Mel bins {freq_attn_mel.shape[0]} vs mel_fb_m2f {mel_fb_m2f.shape[0]}, using interpolation")
        else:
            print(f"  ⚠️ No mel_fb_m2f provided, using simple interpolation")
        freq_attn_interp = F.interpolate(freq_attn_mel.view(1,1,-1), size=F_out, mode="linear", align_corners=False).view(-1)
        freq_attn_norm = norm01(freq_attn_interp)
    
    return time_attn_norm, freq_attn_norm, class_name, sound_type, predicted_class_id, confidence, top5_classes, full_map_cropped

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
    top5_probs, top5_indices = torch.topk(probabilities[0], 5)
    top5_classes = []
    for i in range(5):
        class_id = top5_indices[i].item()
        class_name_top5 = ast_model.config.id2label[class_id]
        prob = top5_probs[i].item()
        top5_classes.append((class_name_top5, prob, class_id))
    
    return class_name, sound_type, predicted_class_id, confidence, top5_classes

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

def pick_anchor_region(score: torch.Tensor, La: int, core_pct: float, 
                      previous_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor,torch.Tensor]] = None,
                      attention_matrix: torch.Tensor = None, purity_scores: torch.Tensor = None) -> Tuple[int, int, int, int]:
    """
    앵커 선정: 어텐션 70% + 순수도 30% 조합 방식
    - 어텐션 매트릭스와 순수도를 조합하여 앵커 점수 계산
    - 다음 패스에서 선정된 앵커 구간은 제외
    """
    T = score.numel()
    device = score.device

    # 1단계: 어텐션 매트릭스에서 가장 높은 점수 찾기
    if attention_matrix is not None:
        print(f"    🎯 Using highest attention score for anchor selection")
        
        # 어텐션 매트릭스에서 전체 최대값의 위치 찾기
        max_attention = attention_matrix.max()
        max_indices = torch.where(attention_matrix == max_attention)
        
        if len(max_indices[0]) > 0:
            # 첫 번째 최대값 위치 사용
            freq_patch_idx = max_indices[0][0].item()
            time_patch_idx = max_indices[1][0].item()
            
            print(f"    🔍 Found max attention at freq_patch={freq_patch_idx}, time_patch={time_patch_idx} (value: {max_attention:.3f})")
            
            # 시간 패치를 STFT 프레임으로 변환
            time_patches = attention_matrix.shape[1]
            
            # 해당 시간 패치에 해당하는 STFT 프레임 범위 계산
            frame_start = int((time_patch_idx / time_patches) * T)
            frame_end = int(((time_patch_idx + 1) / time_patches) * T)
            
            # 해당 범위의 중앙을 peak로 설정
            peak_idx = (frame_start + frame_end) // 2
            
            print(f"    ✅ Selected anchor from highest attention: frame {peak_idx} (patch {time_patch_idx}, attention: {max_attention:.3f})")
            selection_method = "highest_attention"
        else:
            # 최대값을 찾지 못한 경우 에너지 기반 폴백
            print(f"    ⚠️ Could not find max attention, using energy fallback")
            peak_idx = int(score.argmax().item())
            selection_method = "energy_fallback"
        
        # 어텐션 기반 앵커 점수 생성 (선택된 위치에만 높은 값)
        attention_score = torch.zeros(T, device=device)
        if selection_method == "highest_attention":
            # 선택된 프레임 주변에 높은 점수 부여
            window_size = La // 4  # 앵커 길이의 1/4
            start_idx = max(0, peak_idx - window_size)
            end_idx = min(T, peak_idx + window_size)
            attention_score[start_idx:end_idx] = 1.0
        
        norm_attention = attention_score
        
    else:
        print(f"    ⚠️ No attention matrix available, using energy-based selection")
        peak_idx = int(score.argmax().item())
        norm_attention = (score - score.min()) / (score.max() - score.min() + 1e-8)
        selection_method = "energy_only"

    # 2단계: 순수도 계산 (purity_scores가 제공된 경우 사용, 아니면 score 기반으로 계산)
    if purity_scores is not None:
        print(f"    🧮 Using provided purity scores")
        norm_purity = purity_scores
    else:
        print(f"    🧮 Calculating purity from energy profile")
        # score를 기반으로 단순한 순수도 계산
        energy_smoothed = smooth1d(score, SMOOTH_T)
        norm_purity = (energy_smoothed - energy_smoothed.min()) / (energy_smoothed.max() - energy_smoothed.min() + 1e-8)

    # 3단계: 앵커 점수 계산 (어텐션 우선, 순수도는 보조)
    if selection_method == "highest_attention":
        # 어텐션 기반 선택의 경우 순수도는 보조적으로만 사용
        anchor_score = norm_attention * 0.9 + norm_purity * 0.1
        print(f"    📊 Anchor score calculated: attention 90% + purity 10% (highest attention method)")
    else:
        # 에너지 기반 선택의 경우 순수도 비중 증가
        anchor_score = norm_attention * 0.6 + norm_purity * 0.4
        print(f"    📊 Anchor score calculated: attention 60% + purity 40% (energy fallback)")
    
    # 앵커 점수 통계 출력
    max_anchor_score = anchor_score.max().item()
    mean_anchor_score = anchor_score.mean().item()
    print(f"    📈 Anchor score stats: max={max_anchor_score:.3f}, mean={mean_anchor_score:.3f}")

    # 4단계: 이전 앵커 구간 제외
    if previous_anchors:
        print(f"    🚫 Excluding {len(previous_anchors)} previous anchor regions")
        
        # 이전 앵커 구간들을 완전히 제외 (0으로 설정)
        for prev_s, prev_e, prev_mask, prev_weight, prev_anchor_score in previous_anchors:
            # 이전 앵커 영역 전체를 제외 (20% 버퍼 추가)
            buffer = int(La * 0.2)  # 앵커 길이의 20% 버퍼
            avoid_start = max(0, prev_s - buffer)
            avoid_end = min(T, prev_e + buffer)
            anchor_score[avoid_start:avoid_end] = 0.0
            
            print(f"    🚫 Excluded previous anchor region: {prev_s}-{prev_e} (extended: {avoid_start}-{avoid_end})")
        
        # 모든 후보가 제외되었는지 확인
        if anchor_score.max() <= 0:
            print(f"    ❌ All anchor candidates excluded, using energy fallback")
            # 폴백: 에너지 기반 선택 (이전 앵커 제외 없이)
            fallback_score = (score - score.min()) / (score.max() - score.min() + 1e-8)
            peak_idx = int(fallback_score.argmax().item())
            selection_method = "energy_fallback"
        else:
            # 어텐션 기반 선택이 이미 peak_idx를 설정했으므로 그대로 사용
            if selection_method != "highest_attention":
                peak_idx = int(anchor_score.argmax().item())
            selection_method += "_with_exclusion"
    else:
        # 어텐션 기반 선택이 이미 peak_idx를 설정했으므로 그대로 사용
        if selection_method != "highest_attention":
            peak_idx = int(anchor_score.argmax().item())
        print(f"    🎯 No previous anchors to exclude")

    # 5단계: 앵커 윈도우 계산
    anchor_s = max(0, min(peak_idx - (La // 2), T - La))
    anchor_e = anchor_s + La

    # 6단계: 코어 영역 계산
    local_score = score[anchor_s:anchor_e]
    peak_idx_rel = int(torch.argmax(local_score).item())
    
    # 코어 영역을 peak 주변으로 설정 (앵커 길이의 1/8)
    core_half_width = max(1, La // 8)
    core_s_rel = max(0, peak_idx_rel - core_half_width)
    core_e_rel = min(La - 1, peak_idx_rel + core_half_width)
    
    # 7단계: 앵커 에너지 검증
    anchor_energy = local_score.mean().item()
    if anchor_energy < 0.005:  # 매우 낮은 에너지 임계값
        print(f"    ❌ Anchor energy too low ({anchor_energy:.6f}), marking as invalid")
        return -1, -1, -1, -1
    elif anchor_energy < 0.05:  # 낮은 에너지
        print(f"    ⚠️ Selected anchor has low energy ({anchor_energy:.6f}), separation may be weak")
    else:
        print(f"    ✅ Selected anchor has good energy ({anchor_energy:.6f})")
    
    # 8단계: 앵커 점수 정보 출력
    final_anchor_score = anchor_score[peak_idx].item()
    
    if selection_method == "highest_attention":
        attention_contribution = norm_attention[peak_idx].item() * 0.9
        purity_contribution = norm_purity[peak_idx].item() * 0.1
    else:
        attention_contribution = norm_attention[peak_idx].item() * 0.6
        purity_contribution = norm_purity[peak_idx].item() * 0.4
    
    print(f"    📍 Final anchor: {anchor_s}-{anchor_e}, Core: {core_s_rel}-{core_e_rel}")
    print(f"    📊 Anchor score: {final_anchor_score:.3f} (attention: {attention_contribution:.3f}, purity: {purity_contribution:.3f})")
    print(f"    🎯 Selection method: {selection_method}")
    
    # 어텐션 매트릭스 정보 출력 (어텐션 기반 선택인 경우)
    if attention_matrix is not None and "highest_attention" in selection_method:
        # 선택된 시간 패치의 주파수 활성화 정보
        time_patches = attention_matrix.shape[1]
        selected_time_patch = int((peak_idx / T) * time_patches)
        selected_time_patch = min(selected_time_patch, time_patches - 1)
        
        freq_attentions = attention_matrix[:, selected_time_patch]
        max_freq_attention = freq_attentions.max().item()
        
        print(f"    🎵 Selected time patch {selected_time_patch}: max attention = {max_freq_attention:.3f}")
    
    return anchor_s, anchor_e, core_s_rel, core_e_rel

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
    print(f"    🎯 Dynamic similarity threshold: {similarity_threshold:.3f}")
    
    # 3. 유사도 기반 마스크 생성 (전체 시간 구간에 일관성 있게 적용)
    high_similarity_mask = (cos_t_raw >= similarity_threshold).float()
    low_similarity_mask = (cos_t_raw < similarity_threshold).float()
    
    print(f"    🎯 High similarity frames: {high_similarity_mask.sum().item()}/{len(cos_t_raw)}")
    
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
    
    print(f"    🎯 Template frequency weights applied to {high_similarity_mask.sum().item()}/{len(cos_t_raw)} time frames")
    print(f"    🎯 Frequency boost mask active on {freq_boost_mask.sum().item()}/{len(freq_boost_mask)} frequency bins")
    
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
        
        if scaled_count > 0:
            print(f"  📊 Low similarity regions: {scaled_count} time frames scaled to 1% anchor energy (threshold: {similarity_threshold:.3f})")
    
    # 마스크를 1.0으로 제한 (원본을 초과하지 않도록)
    M_lin = torch.clamp(M_lin, 0.0, 1.0)
    
    # 간단한 통계 출력
    print(f"  📊 Mask ({strategy}): mean={M_lin.mean().item():.3f}, conf={confidence:.3f}, boost={freq_weight.max().item():.1f}x")
    
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
                previous_anchors: List[Tuple[int, int]] = None):

    t0 = time.time()
    
    # 1. 전체 오디오 에너지 체크 및 증폭
    overall_energy = np.mean(audio**2)
    amplification_factor = 1.0
    
    if overall_energy < MIN_ANCHOR_ENERGY:
        # 전체 오디오가 작으면 증폭
        energy_ratio = MIN_ANCHOR_ENERGY / (overall_energy + 1e-8)
        amplification_factor = min(AMPLIFICATION_FACTOR * np.sqrt(energy_ratio), MAX_AMPLIFICATION)
        
        print(f"  🔊 Overall audio energy too low ({overall_energy:.6f}), amplifying by factor: {amplification_factor:.1f}")
        audio = audio * amplification_factor
        
        # 클리핑 방지
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            print(f"  ⚠️ Clipping prevented, scaled by {1.0/max_val:.3f}")
    
    # 2. 증폭된 오디오로 STFT 계산
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

    # AST 어텐션 맵 추출 (매 패스마다 새로 계산)
    print(f"  🧠 Computing AST attention maps (pass {pass_idx + 1})")
    A_t, ast_freq_attn, class_name, sound_type, class_id, confidence, top5_classes, attention_matrix = ast_attention_freq_time(audio_for_ast, extractor, ast_model, T, N_MELS, mel_fb_m2f)
    Pur = calculate_purity(P)
    
    # 앵커 스코어 계산 (이전 앵커 영역과 어텐션 상위 30% 패치 제외)
    # 어텐션 기반 앵커 선택을 위해 기본 스코어 계산 (순수도는 참고용으로만 사용)
    print(f"  🎯 Computing basic anchor score for reference")
    Sc = anchor_score(A_t, Pur)

    print(f"  🎯 Detected: {class_name} ({sound_type}) - Confidence: {confidence:.3f}")
    
    # Top 5 클래스 출력
    print(f"  📊 Top 5 predictions:")
    for i, (cls_name, prob, cls_id) in enumerate(top5_classes):
        cls_type = get_sound_type(cls_id)
        marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  " if i == 3 else "  "
        print(f"    {marker} {i+1}. {cls_name} ({cls_type}) - {prob:.3f}")
    
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

    # Suppress previous anchors
    for prev_s, prev_e, prev_mask, prev_weight, prev_anchor_score in prev_anchors:
        if prev_mask.shape[0] != T:
            prev_mask = align_len_1d(prev_mask, T, device=Sc.device, mode="linear")
        ca = (prev_s + prev_e) // 2
        sigma = (prev_e - prev_s) / 6.0
        idx = torch.arange(T, device=Sc.device) - ca
        Sc = Sc * (1 - 0.3 * torch.exp(-(idx**2)/(2*(sigma**2)+1e-8)))
        core_s = max(0, ca - La//2); core_e = min(T, ca + La//2)
        Sc[core_s:core_e] *= 0.2
    
    # Pick anchor and core regions using AST attention (simplified)
    # previous_anchors에 anchor score 정보를 포함하여 전달
    prev_anchors_with_score = []
    for prev_s, prev_e, prev_mask, prev_weight, prev_anchor_score in prev_anchors:
        prev_anchors_with_score.append((prev_s, prev_e, prev_mask, prev_weight, prev_anchor_score))
    
    s, e, core_s_rel, core_e_rel = pick_anchor_region(Sc, La, TOP_PCT_CORE_IN_ANCHOR, prev_anchors_with_score, attention_matrix, Pur)
    
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
    print(f"  🎯 Strategy: {strategy}")
    
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
    print(f"  🔍 Debug - M_lin shape: {M_lin.shape}, mag shape: {mag.shape}")
    print(f"  🔍 Debug - M_lin device: {M_lin.device}, mag device: {mag.device}")
    print(f"  🔍 Debug - M_lin dtype: {M_lin.dtype}, mag dtype: {mag.dtype}")
    
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
    if amplification_factor > 1.0:
        print(f"  🔊 Amplified result (factor: {amplification_factor:.1f})")
    else:
        print(f"  📊 No amplification applied")
    
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
                  similarity_scores=cos_t_raw, amplification_factor=amplification_factor, attention_map=A_t,
                  attention_matrix=attention_matrix, purity_scores=Pur)
    
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
        "db_mean": db_mean
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
               attention_matrix: torch.Tensor = None, purity_scores: torch.Tensor = None):
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
        time_axis = np.linspace(0, len(audio) / SR, len(audio))
        
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
        
        # 4. Power Spectrogram (log scale with adaptive range)
        P_log = torch.log10(P + 1e-10)
        vmin, vmax = torch.quantile(P_log, torch.tensor([0.05, 0.95]))
        im1 = axes[1, 0].imshow(P_log.cpu().numpy(), aspect='auto', origin='lower', 
                               cmap='viridis', vmin=vmin.item(), vmax=vmax.item())
        axes[1, 0].set_title('Power Spectrogram (log scale)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency Bin')
        axes[1, 0].set_xlabel('Time Frame')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 5. Separation Mask
        im2 = axes[1, 1].imshow(M_lin.cpu().numpy(), aspect='auto', origin='lower', 
                               cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('Generated Separation Mask', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency Bin')
        axes[1, 1].set_xlabel('Time Frame')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 6. AST Time Attention vs Anchor Score 비교 (정확한 시간 매핑)
        time_frames = np.arange(len(A_t))
        # 시간 축을 실제 초 단위로 변환
        time_seconds = time_frames * (len(audio) / SR) / len(A_t)  # STFT 프레임 → 실제 시간(초)
        
        axes[1, 2].plot(time_seconds, A_t.cpu().numpy(), 'purple', linewidth=2, label='AST Time Attention', alpha=0.7)
        axes[1, 2].plot(time_seconds, anchor_score.cpu().numpy(), 'orange', linewidth=2, label='Anchor Score (A+P)', alpha=0.8)
        
        # 최고점 표시 (시간으로 변환)
        max_attn_idx = torch.argmax(A_t).item()
        max_anchor_idx = torch.argmax(anchor_score).item()
        max_attn_time = time_seconds[max_attn_idx]
        max_anchor_time = time_seconds[max_anchor_idx]
        axes[1, 2].axvline(max_attn_time, color='purple', linestyle='--', alpha=0.5, label=f'Max AST Attn ({max_attn_time:.2f}s)')
        axes[1, 2].axvline(max_anchor_time, color='orange', linestyle='--', alpha=0.5, label=f'Max Anchor Score ({max_anchor_time:.2f}s)')
        
        # 선택된 앵커 영역 표시 (시간으로 변환)
        anchor_start_time = time_seconds[s]
        anchor_end_time = time_seconds[e]
        core_start_time = time_seconds[s + core_s_rel]
        core_end_time = time_seconds[s + core_e_rel]
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
            time_seconds = time_frames * (len(audio) / SR) / len(similarity_scores)  # STFT 프레임 → 실제 시간(초)
            
            axes[2, 0].plot(time_seconds, similarity_scores.cpu().numpy(), 'orange', linewidth=2, label='Cosine Similarity')
            
            # 동적 임계값 표시 (AST 신뢰도 기반)
            dynamic_threshold = global_confidence if global_confidence is not None else 0.6
            axes[2, 0].axhline(y=dynamic_threshold, color='red', linestyle='--', alpha=0.7, 
                              label=f'Dynamic Threshold ({dynamic_threshold:.3f})')
            
            # 선택된 앵커 영역 표시 (시간으로 변환)
            anchor_start_time = time_seconds[s]
            anchor_end_time = time_seconds[e]
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
            time_seconds = time_frames * (len(audio) / SR) / len(purity_scores)  # STFT 프레임 → 실제 시간(초)
            
            axes[2, 1].plot(time_seconds, purity_scores.cpu().numpy(), 'purple', linewidth=2, label='Purity Score')
            
            # 최고 순수도와 선택된 앵커 비교 (시간으로 변환)
            max_purity_idx = torch.argmax(purity_scores).item()
            max_purity_time = time_seconds[max_purity_idx]
            anchor_start_time = time_seconds[s]
            anchor_end_time = time_seconds[e]
            core_start_time = time_seconds[s + core_s_rel]
            core_end_time = time_seconds[s + core_e_rel]
            
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
            time_frames = np.arange(len(anchor_score))
            time_seconds = time_frames * (len(audio) / SR) / len(anchor_score)
            
            axes[2, 1].plot(time_seconds, anchor_score.cpu().numpy(), 'cyan', linewidth=2, label='Final Anchor Score')
            
            max_score_idx = torch.argmax(anchor_score).item()
            max_score_time = time_seconds[max_score_idx]
            anchor_start_time = time_seconds[s]
            anchor_end_time = time_seconds[e]
            
            axes[2, 1].axvline(max_score_time, color='cyan', linestyle='--', alpha=0.7, label=f'Max Score ({max_score_time:.2f}s)')
            axes[2, 1].axvspan(anchor_start_time, anchor_end_time, alpha=0.3, color='yellow', label=f'Selected Anchor ({anchor_start_time:.2f}-{anchor_end_time:.2f}s)')
            
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
        time_frames = np.arange(len(anchor_score))
        original_energy = np.sum(audio**2)
        src_energy = np.sum(src_amp**2)
        res_energy = np.sum(res**2)
        
        energy_data = [original_energy, src_energy, res_energy]
        energy_labels = ['Original', 'Source', 'Residual']
        colors = ['blue', 'green', 'red']
        
        # AST Attention Matrix (2D)
        if attention_matrix is not None:
            # Enhanced AST Attention Matrix Visualization
            attn_matrix_np = attention_matrix.cpu().numpy()
            
            # 어텐션 매트릭스 시각화 (개선된 버전)
            im = axes[2, 2].imshow(attn_matrix_np, aspect='auto', cmap='plasma', origin='lower')
            axes[2, 2].set_title(f'AST Attention Matrix\n({attn_matrix_np.shape[0]} freq × {attn_matrix_np.shape[1]} time patches)', 
                                fontweight='bold', fontsize=10)
            axes[2, 2].set_xlabel('Time Patches')
            axes[2, 2].set_ylabel('Frequency Patches')
            
            # Colorbar 추가
            cbar = plt.colorbar(im, ax=axes[2, 2], fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', rotation=270, labelpad=15)
            
            # 앵커 영역 하이라이트 (정확한 시간 매핑)
            if s is not None and e is not None:
                time_patches = attn_matrix_np.shape[1]
                total_time_frames = len(anchor_score)
                
                # STFT 프레임을 AST 패치로 변환
                patch_s = int((s / total_time_frames) * time_patches)
                patch_e = int((e / total_time_frames) * time_patches)
                
                # 시간 정보 계산
                anchor_start_time = (s / total_time_frames) * (len(audio) / SR)
                anchor_end_time = (e / total_time_frames) * (len(audio) / SR)
                
                # 앵커 영역 하이라이트
                axes[2, 2].axvspan(patch_s, patch_e, alpha=0.3, color='red', 
                                 label=f'Anchor ({anchor_start_time:.2f}-{anchor_end_time:.2f}s)')
                axes[2, 2].legend(fontsize=8)
                
                # X축을 시간으로 표시
                time_ticks = np.linspace(0, time_patches-1, 5)
                time_labels = [f'{(i/time_patches) * (len(audio)/SR):.2f}s' for i in time_ticks]
                axes[2, 2].set_xticks(time_ticks)
                axes[2, 2].set_xticklabels(time_labels, fontsize=8)
                
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
            
            # Amplification info
            amp_info = f' (Amp: {amplification_factor:.1f}x)' if amplification_factor > 1.0 else ''
            
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
        
        print(f"  📊 Debug plot saved: {debug_path}")
        
    except Exception as e:
        print(f"  ❌ Debug plot failed: {e}")

# =========================
# Backend Integration
# =========================
def send_to_backend(sound_type: str, sound_detail: str, decibel: float, angle: int = 0, 
                   backend_url: str = "http://13.238.200.232:8000/sound-events/", 
                   user_id: int = 6) -> bool:
    """
    백엔드로 분리된 소리 정보 전송
    
    Args:
        sound_type: 소리 타입 (danger/warning/help)
        sound_detail: 구체적인 소리 클래스명
        decibel: 데시벨 값
        angle: 방향각 (기본값: 0)
        backend_url: 백엔드 API URL
        user_id: 사용자 ID
    
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
            "occurred_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sound_icon": "string",
            "location_image_url": "string",
            "decibel": float(decibel),
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AST-Separator/1.0'
        }
        
        print(f"🔄 Sending to backend: {backend_url}")
        print(f"📤 Data: {data}")
        
        # SSL 경고 비활성화 (테스트용)
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        response = requests.post(
            backend_url, 
            json=data, 
            headers=headers,
            timeout=10.0,
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
    
    # Load audio
    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 AST-guided Source Separator (Final Integrated Version)\n{'='*64}")
    print(f"Input: {args.input} ({len(audio0)/SR:.3f}s)")
    print(f"Strategy: {args.strategy}")
    print(f"Features: Adaptive Masking, Energy Conservation, Classification, Energy Suppression")
    print(f"Debug visualization: {'OFF' if args.no_debug else 'ON'}")
    
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
    
    # Main processing loop
    for i in range(max(1, args.passes)):
        print(f"\n--- Pass {i + 1} ---")
        print(f"AST call #{i + 1} for attention extraction...")
        
        src_amp, res, er, used_mask, info = single_pass(
            cur, extractor, ast_model, mel_fb_m2f, used_mask_prev, prev_anchors, 
            i, args.output if not args.no_debug else None, prev_energy_ratio,
            separated_time_regions, previous_anchors
        )
        
        # Save separated source
        if info.get("separation_skipped", False):
            print(f"  ⚡ Separation skipped - using original audio")
            src_path = os.path.join(args.output, f"{i:02d}_{info['class_name']}.wav")
        else:
            src_path = os.path.join(args.output, f"{i:02d}_{info['class_name']}.wav")
        
        torchaudio.save(src_path, torch.from_numpy(src_amp).unsqueeze(0), SR)
        saved += 1
        
        print(f"  Separated: {info['class_name']} ({info['sound_type']})")
        print(f"  Confidence: {info['confidence']:.3f}")
        print(f"  Energy Ratio: {er:.3f}")
        print(f"  Elapsed: {info['elapsed']:.2f}s")
        
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
                    user_id=args.user_id
                )
                
                if backend_success:
                    print(f"  🌐 Backend transmission successful")
                else:
                    print(f"  ❌ Backend transmission failed")
                    
            except Exception as e:
                print(f"  ❌ Backend transmission error: {e}")
        else:
            print(f"  🚫 Backend transmission disabled")
        
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
        class_name, sound_type, class_id, confidence, top5_classes = classify_audio_segment(cur, extractor, ast_model)
        print(f"  🎯 Residual: {class_name} ({sound_type}) - Confidence: {confidence:.3f}")
        
        # Top 5 클래스 출력
        print(f"  📊 Top 5 residual predictions:")
        for i, (cls_name, prob, cls_id) in enumerate(top5_classes):
            cls_type = get_sound_type(cls_id)
            marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  " if i == 3 else "  "
            print(f"    {marker} {i+1}. {cls_name} ({cls_type}) - {prob:.3f}")
        
        if confidence >= RESIDUAL_CONFIDENCE_THRESHOLD:
            print(f"  ✅ High confidence residual detected, saving as sound...")
            residual_path = os.path.join(args.output, f"{saved:02d}_{class_name}.wav")
            torchaudio.save(residual_path, torch.from_numpy(cur).unsqueeze(0), SR)
            saved += 1
            
            # 백엔드로 잔여물 소리 정보 전송
            if not args.no_backend:
                try:
                    # 데시벨 계산
                    decibel = calculate_decibel_from_audio(cur)
                    
                    # 백엔드 전송
                    backend_success = send_to_backend(
                        sound_type=sound_type,
                        sound_detail=class_name,
                        decibel=decibel,
                        angle=args.angle,
                        backend_url=args.backend_url,
                        user_id=args.user_id
                    )
                    
                    if backend_success:
                        print(f"  🌐 Residual backend transmission successful")
                    else:
                        print(f"  ❌ Residual backend transmission failed")
                        
                except Exception as e:
                    print(f"  ❌ Residual backend transmission error: {e}")
            else:
                print(f"  🚫 Residual backend transmission disabled")
        else:
            print(f"  📝 Low confidence residual, saving as residual...")
            residual_path = os.path.join(args.output, f"{saved:02d}_residual.wav")
            torchaudio.save(residual_path, torch.from_numpy(cur).unsqueeze(0), SR)
    
    total_time = time.time() - total_t0
    print(f"\n{'='*64}")
    print(f"✅ Processing completed in {total_time:.2f}s")
    print(f"📁 Saved {saved} audio files to {args.output}")
    print(f"{'='*64}")

if __name__ == "__main__":
    main()
