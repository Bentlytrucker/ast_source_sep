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
MIN_ERATIO = 0.01

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
def ast_attention_freq_time(audio: np.ndarray, extractor, ast_model, T_out: int, F_out: int) -> Tuple[torch.Tensor, torch.Tensor, str, str, int, float]:
    """
    AST 어텐션에서 시간과 주파수 정보를 모두 추출하고 분류 결과도 함께 반환
    Returns: (time_attention, freq_attention, class_name, sound_type, class_id, confidence)
    """
    # 10초로 패딩
    target_len = int(10.0 * SR)
    if len(audio) < target_len:
        audio_padded = np.zeros(target_len, dtype=np.float32)
        audio_padded[:len(audio)] = audio
    else:
        audio_padded = audio[:target_len]
    
    feat = extractor(audio_padded, sampling_rate=SR, return_tensors="pt")
    out = ast_model(input_values=feat["input_values"], output_attentions=True, return_dict=True)
    attns = out.attentions
    
    # 분류 결과 추출
    logits = out.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = logits.argmax(dim=-1).item()
    confidence = probabilities[0, predicted_class_id].item()
    
    class_name = ast_model.config.id2label[predicted_class_id]
    sound_type = get_sound_type(predicted_class_id)
    
    if not attns or len(attns) == 0:
        return torch.ones(T_out) * 0.5, torch.ones(F_out) * 0.5, class_name, sound_type, predicted_class_id, confidence
    
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
    
    # 시간 어텐션 (주파수 차원으로 평균)
    time_attn = full_map.mean(dim=0)  # [101]
    time_attn_interp = F.interpolate(time_attn.view(1,1,-1), size=T_out, mode="linear", align_corners=False).view(-1)
    time_attn_smooth = smooth1d(time_attn_interp, SMOOTH_T)
    time_attn_norm = norm01(time_attn_smooth)
    
    # 주파수 어텐션 (시간 차원으로 평균)
    freq_attn = full_map.mean(dim=1)  # [12]
    freq_attn_interp = F.interpolate(freq_attn.view(1,1,-1), size=F_out, mode="linear", align_corners=False).view(-1)
    freq_attn_norm = norm01(freq_attn_interp)
    
    return time_attn_norm, freq_attn_norm, class_name, sound_type, predicted_class_id, confidence

@torch.no_grad()
def classify_audio_segment(audio: np.ndarray, extractor, ast_model) -> Tuple[str, str, int, float]:
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
    
    return class_name, sound_type, predicted_class_id, confidence

# =========================
# Core Separation Logic
# =========================
def calculate_purity(P: torch.Tensor) -> torch.Tensor:
    """Calculate spectral purity"""
    fbins, T = P.shape
    e = P.sum(dim=0)
    e_n = e / (e.max() + EPS)
    p = P / (P.sum(dim=0, keepdim=True) + EPS)
    H = -(p * (p + EPS).log()).sum(dim=0)
    Hn = H / np.log(max(2, fbins))
    pur = W_E * e_n + (1.0 - W_E) * (1.0 - Hn)
    return norm01(smooth1d(pur, SMOOTH_T))

def anchor_score(A_t: torch.Tensor, Pur: torch.Tensor) -> torch.Tensor:
    return norm01(smooth1d((A_t.clamp(0,1)**ALPHA_ATT) * (Pur.clamp(0,1)**BETA_PUR), SMOOTH_T))

def pick_anchor_region(score: torch.Tensor, La: int, core_pct: float, 
                      attention_map: torch.Tensor = None, purity_map: torch.Tensor = None,
                      previous_anchors: List[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
    """
    어텐션과 순수도를 기반으로 앵커 영역을 선택합니다.
    매 pass마다 AST 모델 추론을 통해 어텐션 맵을 업데이트합니다.
    """
    T = score.numel()

    # 어텐션과 순수도 기반 앵커 스코어 계산
    if attention_map is not None and purity_map is not None:
        # 어텐션과 순수도를 결합한 앵커 스코어
        if isinstance(purity_map, torch.Tensor):
            combined_score = anchor_score(attention_map, purity_map)
        else:
            # purity_map이 float인 경우, 어텐션만 사용
            combined_score = attention_map * purity_map  # float 값을 곱해서 가중치 적용
        print(f"    🎯 Using attention + purity based anchor selection")
    else:
        # 기본 스코어 사용
        combined_score = score
        print(f"    🎯 Using basic score based anchor selection")

    # Create mask to avoid previous anchor regions
    avoid_mask = torch.ones(T, dtype=torch.bool, device=score.device)
    if previous_anchors:
        for prev_s, prev_e in previous_anchors:
            # 이전 앵커 구간과 그 주변을 피하도록 마스크 생성
            buffer = int(La * 1.5)
            avoid_start = max(0, prev_s - buffer)
            avoid_end = min(T, prev_e + buffer)
            avoid_mask[avoid_start:avoid_end] = False
            print(f"    🚫 Avoiding previous anchor region: {prev_s}-{prev_e} (extended: {avoid_start}-{avoid_end})")
    
    # 1. Find the index of the highest score that avoids previous anchors
    if previous_anchors and avoid_mask.sum() > 0:
        candidate_score = combined_score.clone()
        candidate_score[~avoid_mask] = -float('inf')
        if candidate_score.max() > -float('inf'):
            peak_idx = int(torch.argmax(candidate_score).item())
            print(f"    ✅ Selected new anchor region avoiding {len(previous_anchors)} previous anchors")
        else:
            # 모든 영역이 피해야 할 영역이면, 피해야 할 영역을 무시하고 최고점 선택
            print(f"    ⚠️ All regions are avoided, selecting best available region")
            peak_idx = int(torch.argmax(combined_score).item())
    else:
        peak_idx = int(torch.argmax(combined_score).item())

    # 2. Calculate the anchor window centered on the peak.
    anchor_s = max(0, min(peak_idx - (La // 2), T - La))
    anchor_e = anchor_s + La

    # 3. Define the local score window within the anchor.
    local_score = score[anchor_s:anchor_e]
    
    # 4. Find the peak's index relative to the start of the anchor.
    peak_idx_rel = int(torch.argmax(local_score).item())

    # 5. Define the threshold for expanding the core.
    threshold = torch.quantile(local_score, core_pct)

    # 6. Expand left and right from the relative peak to define the core.
    core_s_rel = peak_idx_rel
    while core_s_rel > 0 and local_score[core_s_rel - 1] >= threshold:
        core_s_rel -= 1
        
    core_e_rel = peak_idx_rel
    while core_e_rel < La - 1 and local_score[core_e_rel + 1] >= threshold:
        core_e_rel += 1
    
    # 7. Validate that the selected anchor has meaningful energy
    anchor_energy = local_score.mean().item()
    if anchor_energy < 0.01:  # Very low energy threshold
        print(f"    ⚠️ Selected anchor has very low energy ({anchor_energy:.6f}), may not be effective")
    
    return anchor_s, anchor_e, core_s_rel, core_e_rel

def create_frequency_support(Ablk: torch.Tensor, ast_freq_attn: torch.Tensor) -> torch.Tensor:
    """Create frequency support mask from anchor block and AST attention"""
    fbins = Ablk.shape[0]
    
    # Calculate energy per frequency bin
    energy_per_freq = Ablk.sum(dim=1)
    energy_threshold = torch.quantile(energy_per_freq, OMEGA_Q_CONSERVATIVE)
    
    # Create binary mask
    omega = (energy_per_freq > energy_threshold).float()
    
    # Ensure minimum number of active bins
    if omega.sum() < OMEGA_MIN_BINS:
        _, top_indices = torch.topk(energy_per_freq, OMEGA_MIN_BINS)
        omega = torch.zeros_like(omega)
        omega[top_indices] = 1.0
    
    # Apply AST frequency attention weighting
    ast_threshold = torch.quantile(ast_freq_attn, AST_FREQ_QUANTILE_CONSERVATIVE)
    ast_mask = (ast_freq_attn >= ast_threshold).float()
    
    # Combine energy and attention masks
    omega = omega * ast_mask
    
    return omega

def create_template(Ablk: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Create spectral template from anchor block using frequency support"""
    # Weighted average across time using frequency support
    w_sm = (Ablk * omega.view(-1, 1)).sum(dim=1)
    w = (w_sm * omega)
    w = w / (w.sum() + EPS)
    
    return w

def presence_from_energy(Xmel: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Calculate presence from energy using frequency support"""
    om = omega.view(-1, 1)
    e_omega = (Xmel * om).sum(dim=0)
    e_omega = smooth1d(e_omega, 9)
    thr = torch.quantile(e_omega, 0.20)
    g_pres = (e_omega > torch.clamp(thr, min=1e-10)).float()
    return g_pres

def cos_similarity_over_omega(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, g_pres: torch.Tensor) -> torch.Tensor:
    """Calculate cosine similarity over frequency support"""
    Xo = Xmel * omega.view(-1, 1)
    wn = (w_bar * omega) / ((w_bar * omega).norm(p=2) + 1e-8)
    Xn = Xo / (Xo.norm(p=2, dim=0, keepdim=True) + 1e-8)
    cos_t_raw = (wn.view(-1,1) * Xn).sum(dim=0).clamp(0,1) * g_pres
    return cos_t_raw

def adaptive_masking_strategy(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, 
                           ast_freq_attn: torch.Tensor, P: torch.Tensor, mel_fb_m2f: torch.Tensor,
                           s: int, e: int, Ablk: torch.Tensor, confidence: float, strategy: str = "conservative") -> torch.Tensor:
    """
    적응적 마스킹 전략: 보수적/공격적 모드에 따른 동적 마스크 생성
    유사도 0.6 미만인 부분은 템플릿과 완벽히 일치하는 부분만 남기고 에너지 1%만 남김
    """
    fbins, T = P.shape
    
    # 1. 기본 코사인 유사도 계산
    g_pres = presence_from_energy(Xmel, omega)
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)
    
    # 2. 유사도 0.6 미만인 부분에 대한 특별 처리
    low_similarity_mask = (cos_t_raw < 0.6).float()
    high_similarity_mask = (cos_t_raw >= 0.6).float()
    
    # 3. 신뢰도 기반 분리 강도 조절
    # 원본 소리의 신뢰도만큼 에너지를 추출
    confidence_factor = confidence  # 신뢰도를 분리 강도로 사용
    
    # 전략에 따른 코사인 유사도 처리 (신뢰도 가중치 적용)
    if strategy == "conservative":
        # 보수적 방식: 제곱으로 약화 + 신뢰도 가중치
        cos_processed = (cos_t_raw ** 2) * confidence_factor
    else:  # aggressive
        # 공격적 방식: 시그모이드 적용 + 신뢰도 가중치
        cos_processed = torch.sigmoid(MASK_SIGMOID_SLOPE * (cos_t_raw - MASK_SIGMOID_CENTER)) * confidence_factor
    
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
    
    # 앵커 영역의 상위 20% 진폭 주파수 선택
    anchor_spec = P[:, s:e]
    anchor_max_amp = anchor_spec.max(dim=1).values
    amp_threshold = torch.quantile(anchor_max_amp, 0.8)
    high_amp_mask_lin = (anchor_max_amp >= amp_threshold).float()
    
    # AST 주파수 어텐션을 Linear 도메인으로 변환
    ast_freq_threshold = torch.quantile(ast_freq_attn, 0.4 if strategy == "conservative" else 0.2)
    ast_active_mask_mel = (ast_freq_attn >= ast_freq_threshold).float()
    
    # mel_fb_m2f 차원 확인 및 조정
    if mel_fb_m2f.shape[0] == 128 and mel_fb_m2f.shape[1] == 201:
        # mel_fb_m2f가 [128, 201] 형태인 경우
        ast_active_mask_lin = ((mel_fb_m2f.T @ ast_active_mask_mel).clamp_min(0.0) > 0.1).float()
    elif mel_fb_m2f.shape[0] == 201 and mel_fb_m2f.shape[1] == 128:
        # mel_fb_m2f가 [201, 128] 형태인 경우
        ast_active_mask_lin = ((mel_fb_m2f @ ast_active_mask_mel).clamp_min(0.0) > 0.1).float()
    else:
        # 차원이 맞지 않는 경우 기본값 사용
        print(f"    ⚠️ mel_fb_m2f shape mismatch: {mel_fb_m2f.shape}, using default mask")
        ast_active_mask_lin = torch.ones(omega_lin.shape[0], device=omega.device)
    
    # 주파수 가중치 결합
    freq_boost_mask = torch.maximum(high_amp_mask_lin, ast_active_mask_lin)
    
    # 5. 전략에 따른 가중치 적용 (템플릿 가중 주파수에 더 강한 분리)
    if strategy == "conservative":
        # 보수적 방식: 템플릿 가중 주파수에 3배 가중치
        freq_weight = 1.0 + 2.0 * freq_boost_mask  # [1.0, 3.0]
    else:  # aggressive
        # 공격적 방식: 템플릿 가중 주파수에 2배 가중치
        freq_weight = 1.0 + 1.0 * freq_boost_mask  # [1.0, 2.0]
    
    # 6. 기본 마스크 계산
    M_base = omega_lin.view(-1, 1) * cos_processed.view(1, -1)
    
    # 7. 주파수 가중치 적용
    M_weighted = M_base * freq_weight.view(-1, 1)
    
    # 8. 스펙트로그램 제한 (전략에 따라 다르게)
    spec_magnitude = P.sqrt()
    if strategy == "conservative":
        # 보수적 방식: 0.9 제한으로 더 강한 분리
        M_lin = torch.minimum(M_weighted, spec_magnitude * 0.9)
    else:  # aggressive
        # 공격적 방식: 1.0 제한으로 완전한 분리
        M_lin = torch.minimum(M_weighted, spec_magnitude)
    
    # 9. 유사도 0.6 미만인 부분에 대한 특별 처리
    # 앵커 부분 에너지의 1%에 해당하는 수치만 추출
    if low_similarity_mask.sum() > 0:  # 유사도 0.6 미만인 시간대가 있는 경우
        # 앵커 영역의 평균 에너지 계산
        anchor_energy = Ablk.mean().item()
        target_energy = anchor_energy * 0.01  # 앵커 에너지의 1%
        
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
            print(f"  📊 Low similarity regions: {scaled_count} time frames scaled to 1% anchor energy")
    
    # 마스크를 1.0으로 제한 (원본을 초과하지 않도록)
    M_lin = torch.clamp(M_lin, 0.0, 1.0)
    
    # 간단한 통계 출력
    print(f"  📊 Mask ({strategy}): mean={M_lin.mean().item():.3f}, conf={confidence_factor:.3f}, boost={freq_weight.max().item():.1f}x")
    
    return M_lin

# =========================
# Main Processing Pipeline
# =========================
def single_pass(audio: np.ndarray, extractor, ast_model,
                mel_fb_m2f: torch.Tensor,
                used_mask_prev: Optional[torch.Tensor],
                prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor]],
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
            
            # 에너지 억제 (1%만 남기기) - 더 강한 억제
            suppression_factor = 0.99  # 99% 억제하여 1%만 남김
            P_suppressed = P * (1.0 - time_mask * suppression_factor)
            P = P_suppressed
            
            print(f"    📉 Suppressed {class_name_prev} (conf: {confidence_prev:.3f}) to 1% (factor: {suppression_factor:.3f})")
        
        # 억제된 스펙트로그램을 오디오로 변환하여 AST 모델에 전달
        print(f"  🔄 Converting suppressed spectrogram back to audio for AST inference")
        mag_suppressed = torch.sqrt(P)  # Power에서 Magnitude로 변환
        stft_suppressed = mag_suppressed * torch.exp(1j * phase)  # 복소수 STFT 재구성
        audio_for_ast = torch.istft(stft_suppressed, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, 
                                   window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()

    # AST에서 시간과 주파수 어텐션 모두 추출 (분류 결과도 함께) - 억제된 오디오 사용
    A_t, ast_freq_attn, class_name, sound_type, class_id, confidence = ast_attention_freq_time(audio_for_ast, extractor, ast_model, T, N_MELS)
    Pur = calculate_purity(P)
    Sc = anchor_score(A_t, Pur)

    print(f"  🎯 Detected: {class_name} ({sound_type}) - Confidence: {confidence:.3f}")
    
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
    for prev_s, prev_e, prev_mask, prev_weight in prev_anchors:
        if prev_mask.shape[0] != T:
            prev_mask = align_len_1d(prev_mask, T, device=Sc.device, mode="linear")
        ca = (prev_s + prev_e) // 2
        sigma = (prev_e - prev_s) / 6.0
        idx = torch.arange(T, device=Sc.device) - ca
        Sc = Sc * (1 - 0.3 * torch.exp(-(idx**2)/(2*(sigma**2)+1e-8)))
        core_s = max(0, ca - La//2); core_e = min(T, ca + La//2)
        Sc[core_s:core_e] *= 0.2
    
    # Pick anchor and core regions using attention and purity (avoiding previous anchors)
    s, e, core_s_rel, core_e_rel = pick_anchor_region(Sc, La, TOP_PCT_CORE_IN_ANCHOR, 
                                                     A_t, global_purity, previous_anchors)
    
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
    
    # Create separation mask with adaptive strategy
    M_lin = adaptive_masking_strategy(Xmel, w_bar, omega, ast_freq_attn, P, mel_fb_m2f, s, e, Ablk, confidence, strategy)
    
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
    
    mag_masked = M_lin * mag
    mag_residual = mag * (1.0 - M_lin)  # 잔여물 = 원본 * (1 - 마스크)
    
    # 에너지 보존 검증
    original_energy = torch.sum(mag**2).item()
    masked_energy = torch.sum(mag_masked**2).item()
    residual_energy = torch.sum(mag_residual**2).item()
    total_energy = masked_energy + residual_energy
    energy_ratio = total_energy / (original_energy + 1e-8)
    
    print(f"  📊 Energy: Masked={masked_energy:.0f}, Residual={residual_energy:.0f}, Ratio={energy_ratio:.3f}")
    
    # 에너지 보존이 안 되면 마스크를 조정
    if energy_ratio > 1.01:  # 1% 허용 오차
        print(f"  ⚠️ Energy conservation issue detected, adjusting mask...")
        # 마스크를 원본 에너지의 95%로 제한
        energy_limit = 0.95
        if masked_energy > original_energy * energy_limit:
            scale_factor = (original_energy * energy_limit) / (masked_energy + 1e-8)
            M_lin = M_lin * scale_factor
            mag_masked = M_lin * mag
            mag_residual = torch.maximum(mag - mag_masked, torch.zeros_like(mag))
            print(f"  🔧 Mask scaled by factor: {scale_factor:.3f}")
    
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
                  similarity_scores=cos_t_raw, amplification_factor=amplification_factor)
    
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
               similarity_scores: torch.Tensor = None, amplification_factor: float = 1.0):
    """Create comprehensive debug visualization (9 plots)"""
    try:
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Pass {pass_idx + 1}: {class_name} (Conf: {confidence:.3f})', fontsize=16, fontweight='bold')
        
        # 1. Original Audio Waveform
        axes[0, 0].plot(audio, 'b-', alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title('Original Audio', fontweight='bold')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, len(audio))
        
        # 2. Separated Source Waveform
        axes[0, 1].plot(src_amp, 'g-', alpha=0.7, linewidth=0.5)
        axes[0, 1].set_title('Separated Source', fontweight='bold')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, len(src_amp))
        
        # 3. Residual Audio Waveform
        axes[0, 2].plot(res, 'r-', alpha=0.7, linewidth=0.5)
        axes[0, 2].set_title('Residual Audio', fontweight='bold')
        axes[0, 2].set_ylabel('Amplitude')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlim(0, len(res))
        
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
        
        # 6. AST Time Attention
        axes[1, 2].plot(A_t.cpu().numpy(), 'purple', linewidth=2)
        axes[1, 2].axvspan(s, e, alpha=0.3, color='yellow', label='Anchor Region')
        axes[1, 2].axvspan(s + core_s_rel, s + core_e_rel, alpha=0.5, color='red', label='Core Region')
        axes[1, 2].set_title('AST Time Attention', fontweight='bold')
        axes[1, 2].set_ylabel('Attention Weight')
        axes[1, 2].set_xlabel('Time Frame')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Time-based Cosine Similarity
        if similarity_scores is not None:
            axes[2, 0].plot(similarity_scores.cpu().numpy(), 'orange', linewidth=2, label='Cosine Similarity')
            axes[2, 0].axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Threshold (0.6)')
            axes[2, 0].axvspan(s, e, alpha=0.3, color='yellow', label='Anchor Region')
            axes[2, 0].set_title('Time-based Cosine Similarity', fontweight='bold')
            axes[2, 0].set_ylabel('Similarity Score')
            axes[2, 0].set_xlabel('Time Frame')
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
        
        # 8. Anchor Score
        axes[2, 1].plot(anchor_score.cpu().numpy(), 'cyan', linewidth=2)
        axes[2, 1].axvspan(s, e, alpha=0.3, color='yellow', label='Selected Anchor')
        axes[2, 1].set_title('Anchor Score (Time × Purity)', fontweight='bold')
        axes[2, 1].set_ylabel('Score')
        axes[2, 1].set_xlabel('Time Frame')
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
        
        bars = axes[2, 2].bar(energy_labels, energy_data, color=colors, alpha=0.7)
        axes[2, 2].set_title('Energy Distribution', fontweight='bold')
        axes[2, 2].set_ylabel('Energy')
        axes[2, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Add value labels on bars
        for bar, value in zip(bars, energy_data):
            height = bar.get_height()
            axes[2, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2e}', ha='center', va='bottom', fontsize=9)
        
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
        debug_path = os.path.join(out_dir, f'debug_pass_{pass_idx + 1}.png')
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Debug plot saved: {debug_path}")
        
    except Exception as e:
        print(f"  ❌ Debug plot failed: {e}")

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
    ast_model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        attn_implementation="eager"
    ).to(device).eval()
    
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
        
        # 분리된 시간대 정보 수집 (분리 건너뛰기가 아닌 경우만)
        if not info.get("separation_skipped", False):
            separated_time_regions.append({
                'time_mask': info['src_time_mask'],
                'class_name': info['class_name'],
                'confidence': info['confidence'],
                'pass_idx': i
            })
            
            # 앵커 구간 정보 수집
            if 'anchor_region' in info:
                previous_anchors.append(info['anchor_region'])
        
        # Update for next pass
        prev_energy_ratio = er
        cur = res
        used_mask_prev = used_mask
        
        # Early termination if energy ratio is too low
        if er < MIN_ERATIO:
            print(f"  ⚠️ Energy ratio {er:.3f} below threshold {MIN_ERATIO}, stopping...")
            break
    
    # Final residual classification
    if len(cur) > 0 and np.max(np.abs(cur)) > 1e-6:
        print(f"\n--- Final Residual Classification ---")
        class_name, sound_type, class_id, confidence = classify_audio_segment(cur, extractor, ast_model)
        print(f"  🎯 Residual: {class_name} ({sound_type}) - Confidence: {confidence:.3f}")
        
        if confidence >= RESIDUAL_CONFIDENCE_THRESHOLD:
            print(f"  ✅ High confidence residual detected, saving as sound...")
            residual_path = os.path.join(args.output, f"{saved:02d}_{class_name}.wav")
            torchaudio.save(residual_path, torch.from_numpy(cur).unsqueeze(0), SR)
            saved += 1
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
