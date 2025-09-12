#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST-guided Source Separator
A unified pipeline for audio source separation using AST (Audio Spectrogram Transformer) model:
- Enhanced Frequency Attention with AST model integration
- Adaptive masking strategy with conservative and aggressive modes
- Energy conservation with fallback mechanisms
- Comprehensive classification and analysis
- Multi-pass separation with intelligent anchor selection
"""

import os, time, warnings, argparse
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from transformers import ASTFeatureExtractor, ASTForAudioClassification

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# =========================
# Config


# =========================
SR                = 16000  # 샘플링 레이트 (Hz) - AST 모델과 호환되는 16kHz
WIN_SEC           = 4.096  # 분석 윈도우 길이 (초) - 4초 고정 윈도우로 처리
ANCHOR_SEC        = 0.512  # 앵커 구간 길이 (초) - 512ms 앵커로 소스 특성 추출
L_FIXED           = int(round(WIN_SEC * SR))  # 고정 오디오 길이 (샘플 수)

# === Final Output Processing ===
NORMALIZE_TARGET_PEAK = 0.95  # 최대 볼륨의 95% 크기로 표준화 (클리핑 방지)
RESIDUAL_CLIP_THR = 0.0005    # 최종 잔여물의 진폭이 이 값보다 작으면 0으로 만듦 (노이즈 제거)

# === Adaptive Masking Strategy ===
USE_ADAPTIVE_STRATEGY = True   # 적응적 마스킹 전략 사용 여부
FALLBACK_THRESHOLD = 0.1       # 에너지 보존 실패 시 fallback 전략 사용 임계값

# === Soft Masking Parameters ===
MASK_SIGMOID_CENTER = 0.6   # 마스크가 0.5가 되는 cosΩ 값 (중심점) - 낮을수록 더 강한 마스크
MASK_SIGMOID_SLOPE  = 20.0  # S-커브의 경사 - 높을수록 하드 마스크처럼 날카로워짐

# STFT Parameters (10ms hop)
N_FFT, HOP, WINLEN = 400, 160, 400  # FFT 크기, 홉 길이, 윈도우 길이 (10ms 홉으로 실시간 처리)
WINDOW = torch.hann_window(WINLEN)  # Hann 윈도우 함수
EPS = 1e-10  # 수치 안정성을 위한 작은 값

# Mel Scale Parameters
N_MELS = 128  # Mel 스펙트로그램 빈 수 - AST 모델과 호환

# Anchor Score Parameters
SMOOTH_T      = 19           # 시간축 스무딩 커널 크기 (홉 단위)
ALPHA_ATT     = 0.80         # 어텐션 가중치 지수 (낮을수록 어텐션 영향 감소)
BETA_PUR      = 1.20         # 순도(purity) 가중치 지수 (높을수록 순도 중시)
W_E           = 0.30         # 에너지와 엔트로피의 가중치 비율
TOP_PCT_CORE_IN_ANCHOR  = 0.50  # 앵커 내 코어 영역 비율 (50%)

# Ω & Template Parameters (Adaptive)
OMEGA_Q_CONSERVATIVE = 0.2   # 보수적 전략용
OMEGA_Q_AGGRESSIVE   = 0.7   # 공격적 전략용
OMEGA_DIL         = 2        # Ω 마스크 팽창 반복 횟수 (인접 주파수 포함)
OMEGA_MIN_BINS    = 5        # 최소 선택할 주파수 빈 수 (너무 적은 선택 방지)

# AST Frequency Attention Parameters (Adaptive)
AST_FREQ_QUANTILE_CONSERVATIVE = 0.4  # 보수적 전략용
AST_FREQ_QUANTILE_AGGRESSIVE   = 0.2  # 공격적 전략용

# Sound Type Classification
DANGER_IDS = {396, 397, 398, 399, 400, 426, 436}
HELP_IDS = {23, 14, 354, 355, 356, 359}
WARNING_IDS = {288, 364, 388, 389, 390, 439, 391, 392, 393, 395, 440, 441, 443, 456, 469, 470, 478, 479}

# Presence Gate Parameters
PRES_Q            = 0.20     # Presence gate quantile (상위 80% 에너지 구간에서 presence 판단)
PRES_SMOOTH_T     = 9        # Presence gate 시간축 스무딩 커널 크기

# Suppression Parameters (이전 패스 억제)
USED_THRESHOLD        = 0.65  # 사용된 프레임 판단 임계값 (65% 이상 마스크된 프레임)
USED_DILATE_MS        = 80    # 사용된 프레임 주변 확장 시간 (ms)
ANCHOR_SUPPRESS_MS    = 200   # 이전 앵커 중심 억제 반경 (ms)
ANCHOR_SUPPRESS_BASE  = 0.6   # 이전 앵커 억제 강도 (60% 억제)

# Loop Control Parameters
MAX_PASSES    = 3      # 최대 분리 패스 수 (3번까지 반복)
MIN_ERATIO    = 0.01   # 최소 에너지 비율 (1% 미만이면 중단)

# =========================
# Utils
# =========================
def norm01(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def ensure_odd(k: int) -> int:
    return k + 1 if (k % 2 == 0) else k

def smooth1d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1: return x
    ker = torch.ones(k, device=x.device) / k
    return F.conv1d(x.view(1,1,-1), ker.view(1,1,-1), padding=k//2).view(-1)

def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def align_len_1d(x: torch.Tensor, T: int, device=None, mode="linear"):
    if device is None: device = x.device
    xv = x.to(device).view(1,1,-1).float()
    if xv.size(-1) == T:
        out = xv.view(-1)
    else:
        out = F.interpolate(xv, size=T, mode=mode, align_corners=False).view(-1)
    return out.clamp(0,1)

def soft_sigmoid(x: torch.Tensor, center: float, slope: float, min_val: float = 0.0) -> torch.Tensor:
    sig = torch.sigmoid(slope * (x - center))
    return min_val + (1.0 - min_val) * sig

def get_sound_type(class_id: int) -> str:
    """클래스 ID를 기반으로 소리 타입 반환"""
    if class_id in DANGER_IDS:
        return "danger"
    elif class_id in HELP_IDS:
        return "help"
    elif class_id in WARNING_IDS:
        return "warning"
    else:
        return "other"

def calculate_decibel(audio: np.ndarray) -> Tuple[float, float, float]:
    """오디오의 데시벨 계산 (min, max, 평균)"""
    # RMS 계산
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return -np.inf, -np.inf, -np.inf
    
    # 데시벨 변환 (20 * log10(rms))
    db = 20 * np.log10(rms + 1e-10)
    
    # min, max, 평균 계산
    db_min = 20 * np.log10(np.min(np.abs(audio)) + 1e-10)
    db_max = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    db_mean = db
    
    return db_min, db_max, db_mean

def calculate_global_purity(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor) -> float:
    """전체 오디오에 대한 순수도 계산"""
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

@torch.no_grad()
def classify_audio_segment(audio: np.ndarray, extractor, ast_model) -> Tuple[str, str, int, float]:
    """오디오 세그먼트를 분류하여 클래스명, 타입, ID, 신뢰도 반환 (잔여물 분류용)"""
    # 10초로 패딩
    target_len = int(10.0 * SR)
    if len(audio) < target_len:
        audio_padded = np.zeros(target_len, dtype=np.float32)
        audio_padded[:len(audio)] = audio
    else:
        audio_padded = audio[:target_len]
    
    # AST 모델로 분류
    feat = extractor(audio_padded, sampling_rate=SR, return_tensors="pt")
    outputs = ast_model(input_values=feat["input_values"])
    
    # Top-1 클래스 추출 및 신뢰도 계산
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class_id = logits.argmax(dim=-1).item()
    confidence = probabilities[0, predicted_class_id].item()
    
    class_name = ast_model.config.id2label[predicted_class_id]
    sound_type = get_sound_type(predicted_class_id)
    
    return class_name, sound_type, predicted_class_id, confidence

# =========================
# Debug Visualization
# =========================
def debug_plot(pass_idx: int, Sc: torch.Tensor, a_raw: torch.Tensor, cos_t_raw: torch.Tensor, 
               C_t: torch.Tensor, P: torch.Tensor, M_lin: torch.Tensor, full_map: torch.Tensor,
               s: int, e: int, core_s_rel: int, core_e_rel: int, ast_freq_attn: torch.Tensor,
               src_amp: np.ndarray, res: np.ndarray, png: str, title: str = "",
               original_audio: np.ndarray = None, global_confidence: float = 0.0, 
               global_purity: float = 0.0):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    # 제목에 전체 신뢰도와 순수도 추가
    enhanced_title = f"{title}\nGlobal Confidence: {global_confidence:.3f} | Global Purity: {global_purity:.3f}"
    fig.suptitle(enhanced_title, fontsize=16, fontweight='bold')
    
    # === 첫 번째 행: 파형 (Waveforms) ===
    # 1. Original Audio Waveform
    ax = axes[0, 0]
    if original_audio is not None:
        time_axis = np.linspace(0, len(original_audio) / SR, len(original_audio))
        ax.plot(time_axis, original_audio, 'b-', alpha=0.7, linewidth=0.8)
        ax.set_title('Original Audio Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_ylim(-1, 1)  # Amplitude 범위 통일
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No original audio', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Original Audio Waveform')
        ax.set_ylim(-1, 1)
    
    # 2. Separated Source Waveform
    ax = axes[0, 1]
    time_axis = np.linspace(0, len(src_amp) / SR, len(src_amp))
    ax.plot(time_axis, src_amp, 'g-', alpha=0.7, linewidth=0.8)
    ax.set_title('Separated Source Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(-1, 1)  # Amplitude 범위 통일
    ax.grid(True, alpha=0.3)
    
    # 3. Residual Audio Waveform
    ax = axes[0, 2]
    time_axis = np.linspace(0, len(res) / SR, len(res))
    ax.plot(time_axis, res, 'r-', alpha=0.7, linewidth=0.8)
    ax.set_title('Residual Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_ylim(-1, 1)  # Amplitude 범위 통일
    ax.grid(True, alpha=0.3)
    
    # === 두 번째 행: Mel 스펙트로그램 및 분석 ===
    # 4. Anchor Score
    ax = axes[1, 0]
    T = Sc.numel()
    t_axis = np.arange(T) * HOP / SR
    ax.plot(t_axis, to_np(Sc), 'b-', linewidth=1.5, label='Anchor Score')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='red', label='Anchor Region')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='orange', label='Core Region')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score')
    ax.set_title('Anchor Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Amplitude & Cosine Similarity
    ax = axes[1, 1]
    t_axis = np.arange(a_raw.numel()) * HOP / SR
    ax.plot(t_axis, to_np(a_raw), 'g-', linewidth=1.5, label='Amplitude')
    ax2 = ax.twinx()
    ax2.plot(t_axis, to_np(cos_t_raw), 'r-', linewidth=1.5, label='Cosine Similarity')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude', color='g')
    ax2.set_ylabel('Cosine Similarity', color='r')
    ax.set_title('Amplitude & Cosine Similarity')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 6. AST Frequency Attention
    ax = axes[1, 2]
    ax.plot(to_np(ast_freq_attn), 'purple', linewidth=2)
    ax.set_title('AST Frequency Attention')
    ax.set_xlabel('Mel bins')
    ax.set_ylabel('Attention weight')
    ax.grid(True, alpha=0.3)
    
    # === 세 번째 행: Linear 스펙트로그램 ===
    # 7. Power Spectrogram
    ax = axes[2, 0]
    fbins, T = P.shape
    f_axis = np.arange(fbins) * SR / (2 * fbins)
    t_axis = np.arange(T) * HOP / SR
    im = ax.imshow(to_np(torch.log10(P + 1e-10)), aspect='auto', origin='lower', 
                   extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='viridis')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='red')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Power Spectrogram (log10)')
    plt.colorbar(im, ax=ax, label='log10(Power)')
    
    # 8. Generated Mask
    ax = axes[2, 1]
    im = ax.imshow(to_np(M_lin), aspect='auto', origin='lower',
                   extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='hot')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='cyan')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='yellow')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Generated Mask')
    plt.colorbar(im, ax=ax, label='Mask Value')
    
    # 9. Masked Spectrogram
    ax = axes[2, 2]
    masked_spec = P * M_lin
    im = ax.imshow(to_np(torch.log10(masked_spec + 1e-10)), aspect='auto', origin='lower',
                   extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='viridis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Masked Spectrogram (log10)')
    plt.colorbar(im, ax=ax, label='log10(Power)')
    
    plt.tight_layout()
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()

# =========================
# IO & Spectra
# =========================
def load_fixed_audio(path: str) -> np.ndarray:
    wav, sro = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sro != SR:
        wav = torchaudio.functional.resample(wav, sro, SR)
    wav = wav.squeeze().numpy().astype(np.float32)
    if len(wav) >= L_FIXED:
        return wav[:L_FIXED]
    out = np.zeros(L_FIXED, dtype=np.float32)
    out[:len(wav)] = wav
    return out

@torch.no_grad()
def stft_all(audio: np.ndarray, mel_fb_m2f: torch.Tensor):
    wav = torch.from_numpy(audio)
    st = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                    window=WINDOW, return_complex=True, center=True)
    mag = st.abs()
    P   = (mag * mag).clamp_min(EPS)
    phase = torch.angle(st)

    if mel_fb_m2f.shape[0] != N_MELS:
        mel_fb_m2f = mel_fb_m2f.T.contiguous()
    assert mel_fb_m2f.shape[0] == N_MELS and mel_fb_m2f.shape[1] == P.shape[0]
    mel_fb_m2f = mel_fb_m2f.to(P.dtype).to(P.device)
    mel_pow = (mel_fb_m2f @ P).clamp_min(EPS)
    return st, mag, P, phase, mel_pow

# =========================
# AST Attention (Time & Frequency)
# =========================
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

# =========================
# Attention & Purity
# =========================
def purity_from_P(P: torch.Tensor) -> torch.Tensor:
    fbins, T = P.shape
    e = P.sum(dim=0); e_n = e / (e.max() + EPS)
    p = P / (P.sum(dim=0, keepdim=True) + EPS)
    H = -(p * (p + EPS).log()).sum(dim=0)
    Hn = H / np.log(max(2, fbins))
    pur = W_E * e_n + (1.0 - W_E) * (1.0 - Hn)
    return norm01(smooth1d(pur, SMOOTH_T))

def anchor_score(A_t: torch.Tensor, Pur: torch.Tensor) -> torch.Tensor:
    return norm01(smooth1d((A_t.clamp(0,1)**ALPHA_ATT) * (Pur.clamp(0,1)**BETA_PUR), SMOOTH_T))

def pick_anchor_region(score: torch.Tensor, La: int, core_pct: float) -> Tuple[int, int, int, int]:
    """
    Finds the highest score peak, creates an anchor around it, and then finds
    the core region within that anchor, also centered on the peak.
    Returns: (anchor_start, anchor_end, core_start_relative, core_end_relative)
    """
    T = score.numel()

    # 1. Find the index of the absolute highest score.
    peak_idx = int(torch.argmax(score).item())

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
    
    # End index is exclusive, so add 1
    core_e_rel += 1

    return anchor_s, anchor_e, core_s_rel, core_e_rel

# =========================
# Ω & Template Generation (Adaptive Strategy)
# =========================
def omega_support_with_ast_freq(Ablk: torch.Tensor, ast_freq_attn: torch.Tensor, strategy: str = "conservative") -> torch.Tensor:
    """
    적응적 전략에 따른 AST 주파수 어텐션을 고려한 omega 지원 계산
    """
    # 전략에 따른 파라미터 선택
    if strategy == "conservative":
        omega_q = OMEGA_Q_CONSERVATIVE
        ast_freq_quantile = AST_FREQ_QUANTILE_CONSERVATIVE
    else:  # aggressive
        omega_q = OMEGA_Q_AGGRESSIVE
        ast_freq_quantile = AST_FREQ_QUANTILE_AGGRESSIVE
    
    # 기존 방식으로 계산된 마스크
    med = Ablk.median(dim=1).values
    th = torch.quantile(med, omega_q)
    mask_energy = (med >= th).float()
    
    # AST 주파수 어텐션에서 상위 주파수들 선택
    ast_freq_th = torch.quantile(ast_freq_attn, ast_freq_quantile)
    mask_ast_freq = (ast_freq_attn >= ast_freq_th).float()
    
    # 두 마스크를 결합 (OR 연산)
    mask = torch.maximum(mask_energy, mask_ast_freq)
    
    # 기존 팽창 연산
    for _ in range(OMEGA_DIL):
        mask = torch.maximum(mask, torch.roll(mask, 1))
        mask = torch.maximum(mask, torch.roll(mask, -1))
    
    # 최소 빈 수 보장
    if int(mask.sum().item()) < OMEGA_MIN_BINS:
        order = torch.argsort(med, descending=True)
        need = OMEGA_MIN_BINS - int(mask.sum().item())
        take = order[:need]
        mask[take] = 1.0
    
    return mask

def template_from_anchor_block(Ablk: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    om = omega.view(-1,1)
    w  = (Ablk * om).mean(dim=1) * omega
    w  = w / (w.sum() + EPS)
    w_sm = F.avg_pool1d(w.view(1,1,-1), kernel_size=3, stride=1, padding=1).view(-1)
    w   = (w_sm * omega); w = w / (w.sum() + EPS)
    return w

# =========================
# Presence gate & cosΩ
# =========================
def presence_from_energy(Xmel: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    om = omega.view(-1,1)
    e_omega = (Xmel * om).sum(dim=0)
    e_omega = smooth1d(e_omega, PRES_SMOOTH_T)
    thr = torch.quantile(e_omega, PRES_Q)
    thr = torch.clamp(thr, min=1e-10)
    return (e_omega > thr).float()

def amplitude_raw(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    om = omega.view(-1,1)
    Xo = Xmel * om
    denom = (w_bar*w_bar).sum() + EPS
    a_raw = (w_bar.view(1,-1) @ Xo).view(-1) / denom
    return a_raw.clamp_min(0.0)

def cos_similarity_over_omega(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, g_pres: torch.Tensor):
    om = omega.view(-1,1)
    Xo = Xmel * om
    wn = (w_bar * omega); wn = wn / (wn.norm(p=2) + 1e-8)
    Xn = Xo / (Xo.norm(p=2, dim=0, keepdim=True) + 1e-8)
    cos_raw = (wn.view(-1,1) * Xn).sum(dim=0).clamp(0,1)
    return cos_raw * g_pres

# =========================
# Adaptive Masking Strategy
# =========================
def adaptive_masking_strategy(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, 
                           ast_freq_attn: torch.Tensor, P: torch.Tensor, mel_fb_m2f: torch.Tensor,
                           s: int, e: int, strategy: str = "conservative") -> torch.Tensor:
    """
    적응적 마스킹 전략: 보수적/공격적 모드에 따른 동적 마스크 생성
    """
    fbins, T = P.shape
    
    # 1. 기본 코사인 유사도 계산
    g_pres = presence_from_energy(Xmel, omega)
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)
    
    # 2. 전략에 따른 코사인 유사도 처리
    if strategy == "conservative":
        # 보수적 방식: 제곱으로 약화
        cos_processed = cos_t_raw ** 2
    else:  # aggressive
        # 공격적 방식: 시그모이드 적용
        cos_processed = torch.sigmoid(MASK_SIGMOID_SLOPE * (cos_t_raw - MASK_SIGMOID_CENTER))
    
    # 3. 주파수 가중치 계산
    # Linear 도메인에서 직접 계산
    omega_lin = ((mel_fb_m2f.T @ omega).clamp_min(0.0) > 1e-12).float()
    
    # 앵커 영역의 상위 20% 진폭 주파수 선택
    anchor_spec = P[:, s:e]
    anchor_max_amp = anchor_spec.max(dim=1).values
    amp_threshold = torch.quantile(anchor_max_amp, 0.8)
    high_amp_mask_lin = (anchor_max_amp >= amp_threshold).float()
    
    # AST 주파수 어텐션을 Linear 도메인으로 변환
    ast_freq_threshold = torch.quantile(ast_freq_attn, 0.4 if strategy == "conservative" else 0.2)
    ast_active_mask_mel = (ast_freq_attn >= ast_freq_threshold).float()
    ast_active_mask_lin = ((mel_fb_m2f.T @ ast_active_mask_mel).clamp_min(0.0) > 0.1).float()
    
    # 주파수 가중치 결합
    freq_boost_mask = torch.maximum(high_amp_mask_lin, ast_active_mask_lin)
    
    # 4. 전략에 따른 가중치 적용
    if strategy == "conservative":
        # 보수적 방식: 2배 가중치
        freq_weight = 1.0 + freq_boost_mask  # [1.0, 2.0]
    else:  # aggressive
        # 공격적 방식: 30% 가중치
        freq_weight = 1.0 + 0.3 * freq_boost_mask  # [1.0, 1.3]
    
    # 5. 기본 마스크 계산
    M_base = omega_lin.view(-1, 1) * cos_processed.view(1, -1)
    
    # 6. 주파수 가중치 적용
    M_weighted = M_base * freq_weight.view(-1, 1)
    
    # 7. 스펙트로그램 제한
    spec_magnitude = P.sqrt()
    M_lin = torch.minimum(M_weighted, spec_magnitude)
    
    return M_lin

def adaptive_strategy_selection(prev_energy_ratio: float, pass_idx: int) -> str:
    """
    이전 결과를 기반으로 적응적 전략 선택
    """
    if not USE_ADAPTIVE_STRATEGY:
        return "conservative"  # 기본값
    
    # 첫 번째 패스는 보수적으로 시작
    if pass_idx == 0:
        return "conservative"
    
    # 이전에 에너지 보존 문제가 있었으면 보수적으로
    if prev_energy_ratio > 2.0:
        return "conservative"
    
    # 에너지가 너무 적게 추출되었으면 공격적으로
    if prev_energy_ratio < 1.2:
        return "aggressive"
    
    # 기본적으로 보수적 전략 유지
    return "conservative"

# =========================
# Single Pass Processing
# =========================
def single_pass(audio: np.ndarray, extractor, ast_model,
                mel_fb_m2f: torch.Tensor,
                used_mask_prev: Optional[torch.Tensor],
                prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor]],
                pass_idx: int, out_dir: Optional[str], prev_energy_ratio: float = 1.0,
                separated_time_regions: List[dict] = None):

    t0 = time.time()
    st, mag, P, phase, Xmel = stft_all(audio, mel_fb_m2f)
    fbins, T = P.shape
    La = int(round(ANCHOR_SEC * SR / HOP))

    # 적응적 전략 선택
    strategy = adaptive_strategy_selection(prev_energy_ratio, pass_idx)
    print(f"  🎯 Strategy: {strategy}")

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
            
            # 에너지 억제 (2%만 남기기)
            suppression_factor = 0.98  # 98% 억제하여 2%만 남김
            P_suppressed = P * (1.0 - time_mask * suppression_factor)
            P = P_suppressed
            
            print(f"    📉 Suppressed {class_name_prev} (conf: {confidence_prev:.3f}) to 2% (factor: {suppression_factor:.3f})")
        
        # 억제된 스펙트로그램을 오디오로 변환하여 AST 모델에 전달
        print(f"  🔄 Converting suppressed spectrogram back to audio for AST inference")
        mag_suppressed = torch.sqrt(P)  # Power에서 Magnitude로 변환
        stft_suppressed = mag_suppressed * torch.exp(1j * phase)  # 복소수 STFT 재구성
        audio_for_ast = torch.istft(stft_suppressed, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, 
                                   window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()

    # AST에서 시간과 주파수 어텐션 모두 추출 (분류 결과도 함께) - 억제된 오디오 사용
    A_t, ast_freq_attn, class_name, sound_type, class_id, confidence = ast_attention_freq_time(audio_for_ast, extractor, ast_model, T, N_MELS)
    Pur = purity_from_P(P)
    Sc = anchor_score(A_t, Pur)

    # Suppress used frames
    if used_mask_prev is not None:
        um = align_len_1d(used_mask_prev, T, device=Sc.device, mode="linear")
        k = int(round((USED_DILATE_MS/1000.0)*SR/HOP)); k = ensure_odd(max(1,k))
        ker = torch.ones(k, device=Sc.device)/k
        um = (F.conv1d(um.view(1,1,-1), ker.view(1,1,-1), padding=k//2).view(-1) > 0.2).float()
        Sc = Sc * (1 - 0.85 * um)

    # Enhanced suppression of previous anchors
    for (sa, ea, prev_w, prev_omega) in prev_anchors:
        ca = int(((sa+ea)/2) * SR / HOP)
        ca = max(0, min(T-1, ca))
        sigma = int(round((ANCHOR_SUPPRESS_MS/1000.0)*SR/HOP))
        idx = torch.arange(T, device=Sc.device) - ca
        Sc = Sc * (1 - ANCHOR_SUPPRESS_BASE * torch.exp(-(idx**2)/(2*(sigma**2)+1e-8)))
        core_s = max(0, ca - La//2); core_e = min(T, ca + La//2)
        Sc[core_s:core_e] *= 0.2
    
    # Pick anchor and core regions, centered on the peak score
    s, e, core_s_rel, core_e_rel = pick_anchor_region(Sc, La, TOP_PCT_CORE_IN_ANCHOR)
    
    # Create anchor block (Ablk) based on the core indices
    Ablk = Xmel[:, s:e].clone()
    if core_s_rel > 0:  Ablk[:, :core_s_rel] = 0
    if core_e_rel < La: Ablk[:, core_e_rel:] = 0

    # AST 주파수 어텐션을 고려한 Ω 계산
    omega = omega_support_with_ast_freq(Ablk, ast_freq_attn, strategy)
    w_bar = template_from_anchor_block(Ablk, omega)
    
    # 적응적 마스킹 전략 적용
    M_lin = adaptive_masking_strategy(Xmel, w_bar, omega, ast_freq_attn, P, mel_fb_m2f, s, e, strategy)
    
    # Subtraction in the complex STFT domain for precision
    stft_full = st
    
    # 마스크를 진폭에만 적용하고 위상은 그대로 유지
    mag_masked = M_lin * mag  # 진폭에 마스크 적용
    stft_src = mag_masked * torch.exp(1j * phase)  # 복소수 STFT 재구성
    
    # 잔여물 계산: 진폭 기반으로 올바르게 계산 (에너지 보존)
    mag_residual = torch.maximum(mag - mag_masked, torch.zeros_like(mag))
    stft_res = mag_residual * torch.exp(1j * phase)  # 잔여물 복소수 STFT 재구성
    
    # 디버깅: 뺄셈 결과 검증
    src_energy = torch.sum(torch.abs(stft_src)**2).item()
    res_energy = torch.sum(torch.abs(stft_res)**2).item()
    orig_energy = torch.sum(torch.abs(stft_full)**2).item()
    total_energy = src_energy + res_energy
    
    print(f"  🔍 Energy: Original={orig_energy:.6f}, Source={src_energy:.6f}, Residual={res_energy:.6f}")
    print(f"  🔍 Energy ratio: Src/Orig={src_energy/(orig_energy+1e-8):.3f}, Res/Orig={res_energy/(orig_energy+1e-8):.3f}")
    print(f"  🔍 Energy conservation: Total/Orig={total_energy/(orig_energy+1e-8):.3f}")
    
    # 에너지 보존 검증 및 정규화
    energy_ratio = total_energy / (orig_energy + 1e-8)
    if energy_ratio > 1.1:  # 총 에너지가 원본의 110%를 넘으면
        print(f"  ⚠️ WARNING: Energy not conserved! Total/Orig={energy_ratio:.3f}")
        # 에너지 정규화
        scale_factor = orig_energy / (total_energy + 1e-8)
        scale_tensor = torch.tensor(scale_factor, device=stft_src.device, dtype=stft_src.dtype)
        stft_src = stft_src * torch.sqrt(scale_tensor)
        stft_res = stft_res * torch.sqrt(scale_tensor)
        print(f"  🔧 Scaled energies by factor {scale_factor:.3f}")

    # Reconstruct both source and residual
    src_amp = torch.istft(stft_src, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, 
                         window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()
    res = torch.istft(stft_res, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, 
                      window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()

    # ER calculation
    e_src = float(np.sum(src_amp**2)); e_res = float(np.sum(res**2))
    er = e_src / (e_src + e_res + 1e-12)

    # Used-frame mask for next pass
    r_t = (M_lin * P).sum(dim=0) / (P.sum(dim=0) + 1e-8)
    used_mask = (r_t >= USED_THRESHOLD).float()

    elapsed = time.time() - t0
    
    # Global purity calculation
    g_pres = presence_from_energy(Xmel, omega)
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)
    global_purity = cos_t_raw.mean().item()
    
    # 앵커 중심 계산 (분리 건너뛰기 로직에서 사용)
    ca = (s + e) // 2
    
    # 분리 건너뛰기 조건 확인
    if should_skip_separation(confidence, Pur[ca], class_id):
        print(f"  ⚡ High confidence & purity detected! Skipping separation...")
        print(f"  📊 Confidence: {confidence:.3f} (≥0.8), Purity: {Pur[ca]:.3f} (≥0.7)")
        
        # 원본 오디오를 그대로 반환 (분리하지 않음)
        src_amp = audio.copy()
        res = np.zeros_like(audio)
        er = 1.0  # 전체가 소스로 간주
        
        # 정보 반환
        info = {
            "er": er,
            "elapsed": elapsed,
            "anchor": (s*HOP/SR, e*HOP/SR),
            "core": ((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR),
            "quality": 1.0,  # 분리하지 않았으므로 최고 품질
            "w_bar": w_bar,
            "omega": omega,
            "sound_type": sound_type,
            "sound_detail": class_name,
            "class_id": class_id,
            "confidence": confidence,
            "purity": Pur[ca],
            "separation_skipped": True,
            "strategy": strategy,
            "energy_ratio": 1.0,
            "db_min": db_min,
            "db_max": db_max,
            "db_mean": db_mean
        }
        
        return src_amp, res, er, None, info
    
    # Debug plot generation
    if out_dir is not None:
        # 필요한 변수들 계산
        a_raw = amplitude_raw(Xmel, w_bar, omega)
        cos_t_raw_debug = cos_similarity_over_omega(Xmel, w_bar, omega, presence_from_energy(Xmel, omega))
        C_t = cos_t_raw_debug  # 별칭
        
        # full_map은 AST 어텐션에서 추출 (간단한 버전)
        full_map = torch.zeros(12, 101)  # 기본 크기
        
        png = os.path.join(out_dir, f"debug_pass_{pass_idx+1}.png")
        debug_plot(pass_idx, Sc, a_raw, cos_t_raw_debug, C_t, P, M_lin, full_map,
                  s, e, core_s_rel, core_e_rel, ast_freq_attn, src_amp, res, png,
                  title=f"Pass {pass_idx+1} | Strategy: {strategy} | anchor {s*HOP/SR:.2f}-{e*HOP/SR:.2f}s | ER={er*100:.1f}% [Adaptive Mask]",
                  original_audio=audio, global_confidence=confidence, global_purity=global_purity)
    
    # 데시벨 계산
    db_min, db_max, db_mean = calculate_decibel(src_amp)
    
    # 분리된 소스의 시간대 정보 계산 (다음 패스에서 에너지 억제용)
    src_time_mask = (M_lin.sum(dim=0) > 1e-6).float()  # 분리된 시간 프레임들
    src_time_indices = torch.where(src_time_mask > 0)[0]  # 분리된 시간 인덱스들
    
    info = {
        "er": er,
        "elapsed": elapsed,
        "anchor": (s*HOP/SR, e*HOP/SR),
        "core": ((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR),
        "quality": float(M_lin.mean().item()),
        "w_bar": w_bar,
        "omega": omega,
        "stopped": False,
        "strategy": strategy,
        "energy_ratio": energy_ratio,
        "class_name": class_name,
        "sound_type": sound_type,
        "class_id": class_id,
        "confidence": confidence,
        "db_min": db_min,
        "db_max": db_max,
        "db_mean": db_mean,
        "src_time_mask": src_time_mask,  # 분리된 시간 마스크
        "src_time_indices": src_time_indices  # 분리된 시간 인덱스
    }
    return src_amp, res, er, used_mask, info

# =========================
# Main Processing Pipeline
# =========================
def main():
    ap = argparse.ArgumentParser(description="AST-guided Source Separator")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--passes", type=int, default=MAX_PASSES)
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--strategy", choices=["conservative", "aggressive", "adaptive"], default="adaptive")
    args = ap.parse_args()

    # 전역 전략 설정
    global USE_ADAPTIVE_STRATEGY
    if args.strategy == "adaptive":
        USE_ADAPTIVE_STRATEGY = True
    else:
        USE_ADAPTIVE_STRATEGY = False

    os.makedirs(args.output, exist_ok=True)

    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 AST-guided Source Separator\n{'='*64}")
    print(f"Input: fixed {WIN_SEC:.3f}s, Anchor: {ANCHOR_SEC:.3f}s, Passes: {args.passes}")
    print(f"Strategy: {args.strategy}")
    print(f"Features: Adaptive Masking, Energy Conservation, Classification, Decibel Analysis")

    # Mel FB: [F,M] -> [M,F]
    fbins = N_FFT//2 + 1
    mel_fb_f2m = torchaudio.functional.melscale_fbanks(
        n_freqs=fbins, f_min=0.0, f_max=SR/2, n_mels=N_MELS,
        sample_rate=SR, norm="slaney"
    )
    mel_fb_m2f = mel_fb_f2m.T.contiguous()

    # AST (CPU)
    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast_model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        attn_implementation="eager"
    ).eval()

    cur = audio0.copy()
    used_mask_prev = None
    prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor]] = []
    total_t0 = time.time()
    saved = 0
    prev_energy_ratio = 1.0
    separated_time_regions = []  # 이전에 분리된 시간대 정보 저장

    for i in range(max(1, args.passes)):
        print(f"\n▶ Pass {i+1}/{args.passes}")
        result = single_pass(
            cur, extractor, ast_model, mel_fb_m2f,
            used_mask_prev, prev_anchors,
            pass_idx=i, out_dir=None if args.no_debug else args.output,
            prev_energy_ratio=prev_energy_ratio,
            separated_time_regions=separated_time_regions
        )
        
        if result[0] is None:
            info = result[4]
            reason = info.get("reason", "stopped")
            print(f"  ⏹️ Stopped: {reason}")
            break
        
        src, res, er, used_mask_prev, info = result
        
        # 분류 정보 출력
        class_name = info['class_name']
        sound_type = info['sound_type']
        class_id = info['class_id']
        confidence = info['confidence']
        strategy = info['strategy']
        energy_ratio = info['energy_ratio']
        db_min, db_max, db_mean = info['db_min'], info['db_max'], info['db_mean']
        
        # 분리 건너뛰기 여부 확인
        separation_skipped = info.get('separation_skipped', False)
        
        if separation_skipped:
            print(f"⏱️ pass{i+1}: {info['elapsed']:.3f}s | anchor {info['anchor'][0]:.2f}-{info['anchor'][1]:.2f}s | SKIPPED")
            print(f"  🎯 Strategy: {strategy} | Separation: SKIPPED (High confidence & purity)")
            print(f"  🎵 Class: {class_name} (ID: {class_id}) | Type: {sound_type} | Confidence: {confidence:.3f}")
            print(f"  🔊 Decibel: min={db_min:.1f}dB, max={db_max:.1f}dB, mean={db_mean:.1f}dB")
        else:
            print(f"⏱️ pass{i+1}: {info['elapsed']:.3f}s | anchor {info['anchor'][0]:.2f}-{info['anchor'][1]:.2f}s"
                  f" | core {info['core'][0]:.2f}-{info['core'][1]:.2f}s | ER={er*100:.1f}%")
            print(f"  🎯 Strategy: {strategy} | Energy Ratio: {energy_ratio:.3f}")
            print(f"  🎵 Class: {class_name} (ID: {class_id}) | Type: {sound_type} | Confidence: {confidence:.3f}")
            print(f"  🔊 Decibel: min={db_min:.1f}dB, max={db_max:.1f}dB, mean={db_mean:.1f}dB")

        if er < MIN_ERATIO:
            print("  ⚠️ Too little energy; stopping.")
            break
        
        # 분리된 시간대 정보 수집 (다음 패스에서 에너지 억제용) - 분리 건너뛰기 시에는 수집하지 않음
        if not separation_skipped:
            src_time_mask = info.get('src_time_mask')
            if src_time_mask is not None:
                separated_time_regions.append({
                    'pass': i + 1,
                    'class_name': class_name,
                    'time_mask': src_time_mask,
                    'time_indices': info.get('src_time_indices', []),
                    'confidence': confidence
                })
                print(f"  📊 Collected {len(separated_time_regions)} separated time regions for energy suppression")

        # Peak Normalization for clear output
        peak = np.max(np.abs(src))
        if peak > 1e-8:
            gain = NORMALIZE_TARGET_PEAK / peak
            src = src * gain
            src = np.clip(src, -1.0, 1.0)

        # 클래스명을 포함한 파일명 생성
        safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_class_name = safe_class_name.replace(' ', '_')
        out_src = os.path.join(args.output, f"{i+1:02d}_{safe_class_name}_{sound_type}.wav")
        torchaudio.save(out_src, torch.from_numpy(src).unsqueeze(0), SR)
        print(f"    ✅ Saved (Normalized): {out_src}")

        cur = res
        prev_anchors.append((info["anchor"][0], info["anchor"][1], info["w_bar"], info["omega"]))
        prev_energy_ratio = energy_ratio
        saved += 1

    # Apply Hard Clipping to the final residual
    if RESIDUAL_CLIP_THR > 0:
        print(f"\nApplying residual clipping with threshold: {RESIDUAL_CLIP_THR}")
        cur[np.abs(cur) < RESIDUAL_CLIP_THR] = 0.0
    
    # Residual 분류 (신뢰도 0.7 기준)
    print(f"\n🔍 Classifying residual audio...")
    res_class_name, res_sound_type, res_class_id, res_confidence = classify_audio_segment(cur, extractor, ast_model)
    res_db_min, res_db_max, res_db_mean = calculate_decibel(cur)
    
    print(f"  🎵 Residual Class: {res_class_name} (ID: {res_class_id}) | Type: {res_sound_type} | Confidence: {res_confidence:.3f}")
    print(f"  🔊 Residual Decibel: min={res_db_min:.1f}dB, max={res_db_max:.1f}dB, mean={res_db_mean:.1f}dB")
    
    # 신뢰도 0.7 기준으로 파일명 결정
    if res_confidence >= 0.7:
        safe_res_class_name = "".join(c for c in res_class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_res_class_name = safe_res_class_name.replace(' ', '_')
        out_res = os.path.join(args.output, f"00_{safe_res_class_name}_{res_sound_type}.wav")
        print(f"  ✅ High confidence ({res_confidence:.3f} ≥ 0.7), using class name: {res_class_name}")
    else:
        out_res = os.path.join(args.output, "00_residual.wav")
        print(f"  ⚠️ Low confidence ({res_confidence:.3f} < 0.7), using generic name")
    
    torchaudio.save(out_res, torch.from_numpy(cur).unsqueeze(0), SR)

    total_elapsed = time.time() - total_t0
    print(f"\n💾 Residual: {out_res}")
    print(f"⏱️ Total: {total_elapsed:.3f}s")
    print(f"✅ Done. Separated: {saved}\n")

if __name__ == "__main__":
    main()
