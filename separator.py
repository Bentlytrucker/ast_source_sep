#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified AST-guided Source Separator (Raspberry Pi 5 — <5s target)

핵심 최적화
- Pass당 AST 1회: 어텐션 + 패치토큰 캐시 재사용
- FeatureExtractor의 멜 계산 제거: STFT에서 만든 Mel(Xmel) 재사용
- 템플릿 유사도 하드 임계(시간 마스크): cos >= 0.6 만 채택
- 위상 복원 최적화: phase_complex = STFT / (|STFT|+ε)
- AST 레이어 트리밍 옵션(--keep-ast-layers), Linear INT8 동적 양자화 옵션(--int8)
- 잔여물 분류 스킵 옵션(--skip-residual-cls)
- 디버그 플롯 임포트 지연(import matplotlib in function)

권장 실행 예:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python sep2_fast.py --input in.wav --output out --passes 1 --no-debug --int8 --skip-residual-cls --keep-ast-layers 6
"""

import os, time, warnings, argparse
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
# matplotlib는 디버그 시에만 임포트 (debug_plot 내부)

warnings.filterwarnings("ignore")

# ===== Threads (권장: 라즈베리파이 5) =====
torch.set_num_threads(2)          # 2~3 권장
torch.set_num_interop_threads(1)

# =========================
# Config
# =========================
SR                = 16000
WIN_SEC           = 4.096
ANCHOR_SEC        = 0.512
L_FIXED           = int(round(WIN_SEC * SR))

NORMALIZE_TARGET_PEAK = 0.95
RESIDUAL_CLIP_THR     = 0.0005

USE_ADAPTIVE_STRATEGY = True
FALLBACK_THRESHOLD    = 0.1

MASK_SIGMOID_CENTER = 0.6
MASK_SIGMOID_SLOPE  = 20.0

# STFT (10ms hop). center=False로 살짝 가속
N_FFT, HOP, WINLEN = 320, 160, 320
WINDOW = torch.hann_window(WINLEN, periodic=True) 
EPS = 1e-10
CENTER_STFT = True   # False 권장(속도)

# Mel
N_MELS = 128

# Anchor score
SMOOTH_T      = 19
ALPHA_ATT     = 0.80
BETA_PUR      = 1.20
W_E           = 0.30
TOP_PCT_CORE_IN_ANCHOR  = 0.50

# Ω support
OMEGA_Q_CONSERVATIVE = 0.2
OMEGA_Q_AGGRESSIVE   = 0.7
OMEGA_DIL         = 2
OMEGA_MIN_BINS    = 5

# AST freq attn
AST_FREQ_QUANTILE_CONSERVATIVE = 0.4
AST_FREQ_QUANTILE_AGGRESSIVE   = 0.2

# Presence gate
PRES_Q        = 0.20
PRES_SMOOTH_T = 9

# Suppression
USED_THRESHOLD        = 0.65
USED_DILATE_MS        = 80
ANCHOR_SUPPRESS_MS    = 200
ANCHOR_SUPPRESS_BASE  = 0.6

# Loop
MAX_PASSES    = 3
MIN_ERATIO    = 0.01

# Hard time-threshold (요청사항)
TIME_SIM_THRESHOLD = 0.6

# Sound Type Classification groups (AudioSet class IDs)
DANGER_IDS  = {396, 397, 398, 399, 400, 426, 436}
HELP_IDS    = {23, 14, 354, 355, 356, 359}
WARNING_IDS = {288, 364, 388, 389, 390, 439, 391, 392, 393, 395, 440, 441, 443, 456, 469, 470, 478, 479}

# =========================
# Utils
# =========================
def norm01(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def ensure_odd(k: int) -> int:
    return k + 1 if (k % 2 == 0) else k

def smooth1d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1: return x
    ker = torch.ones(k, device=x.device, dtype=x.dtype) / k
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

def get_sound_type(class_id: int) -> str:
    if class_id in DANGER_IDS: return "danger"
    if class_id in HELP_IDS: return "help"
    if class_id in WARNING_IDS: return "warning"
    return "other"

def calculate_decibel(audio: np.ndarray) -> Tuple[float, float, float]:
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0: return -np.inf, -np.inf, -np.inf
    db_mean = 20 * np.log10(rms + 1e-10)
    db_min = 20 * np.log10(np.min(np.abs(audio)) + 1e-10)
    db_max = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    return db_min, db_max, db_mean

# =========================
# Debug Visualization (지연 임포트)
# =========================
def debug_plot(pass_idx: int, Sc: torch.Tensor, a_raw: torch.Tensor, cos_t_raw: torch.Tensor,
               C_t: torch.Tensor, P: torch.Tensor, M_lin: torch.Tensor, full_map: torch.Tensor,
               s: int, e: int, core_s_rel: int, core_e_rel: int, ast_freq_attn: torch.Tensor,
               src_amp: np.ndarray, res: np.ndarray, png: str, title: str = ""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. Anchor score
    ax = axes[0, 0]
    T = Sc.numel()
    t_axis = np.arange(T) * HOP / SR
    ax.plot(t_axis, to_np(Sc), 'b-', linewidth=1.5, label='Anchor Score')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='red', label='Anchor')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='orange', label='Core')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Score'); ax.set_title('Anchor Selection')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Amp & Cos
    ax = axes[0, 1]
    t_axis = np.arange(a_raw.numel()) * HOP / SR
    ax.plot(t_axis, to_np(a_raw), 'g-', linewidth=1.5, label='Amplitude')
    ax2 = ax.twinx()
    ax2.plot(t_axis, to_np(cos_t_raw), 'r-', linewidth=1.5, label='Cos')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amp', color='g')
    ax2.set_ylabel('Cos', color='r')
    ax.set_title('Amplitude & Cosine Similarity')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    # 3. Spectrogram
    ax = axes[0, 2]
    fbins, T = P.shape
    f_axis = np.arange(fbins) * SR / (2 * fbins)
    t_axis = np.arange(T) * HOP / SR
    im = ax.imshow(to_np(torch.log10(P + 1e-10)), aspect='auto', origin='lower',
                   extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='viridis')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='red')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='orange')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Hz'); ax.set_title('Power Spectrogram (log10)')
    plt.colorbar(im, ax=ax, label='log10(Power)')

    # 4. Mask
    ax = axes[1, 0]
    im = ax.imshow(to_np(M_lin), aspect='auto', origin='lower',
                   extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='hot')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='cyan')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='yellow')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Hz'); ax.set_title('Mask')
    plt.colorbar(im, ax=ax, label='Mask')

    # 5. Source
    ax = axes[1, 1]
    t_audio = np.arange(len(src_amp)) / SR
    ax.plot(t_audio, src_amp, 'b-', linewidth=0.8)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amp'); ax.set_title('Separated Source'); ax.grid(True, alpha=0.3)

    # 6. Residual
    ax = axes[1, 2]
    t_audio = np.arange(len(res)) / SR
    ax.plot(t_audio, res, 'r-', linewidth=0.8)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Amp'); ax.set_title('Residual'); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(png, dpi=150, bbox_inches='tight'); plt.close()

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
    st = torch.stft(
        wav, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
        window=WINDOW, return_complex=True, center=CENTER_STFT
    )
    mag = st.abs()
    P   = (mag * mag).clamp_min(EPS)
    # 빠른 위상 복원: e^{jθ} = STFT / |STFT|
    phase_complex = st / (mag + 1e-8)

    # Mel pow (M x F) @ (F x T) = (M x T)
    if mel_fb_m2f.shape[0] != N_MELS:
        mel_fb_m2f = mel_fb_m2f.T.contiguous()
    mel_fb_m2f = mel_fb_m2f.to(P.dtype).to(P.device)
    mel_pow = (mel_fb_m2f @ P).clamp_min(EPS)
    return st, mag, P, phase_complex, mel_pow

# =========================
# AST: 1회 추론 (Xmel 재사용)
# =========================
@torch.no_grad()
def ast_forward_once_from_mel(Xmel: torch.Tensor, extractor, ast_model, T_out: int, F_out: int):
    """
    Xmel: [M, T] (power-mel). FeatureExtractor의 멜 계산을 생략하고,
    log-mel + 정규화(mean/std)만 적용해 AST에 직접 투입.
    반환:
      time_attn_T: [T_out] (interpolated & smoothed)
      freq_attn_M: [F_out] (interpolated)
      time_attn_patch: [Tp]
      freq_attn_patch: [Fp]
      patch_tokens: [Fp, Tp, D]
    """
    # 1) log-mel + 정규화
    log_mel = torch.log(Xmel + 1e-4)                 # [M, T]
    x = log_mel.T.contiguous()                       # [T, M]
    mean = torch.tensor(extractor.mean, device=x.device, dtype=x.dtype)
    std  = torch.tensor(extractor.std,  device=x.device, dtype=x.dtype)
    x = (x - mean) / (std + 1e-5)

    # 2) 시간축 길이 맞추기 (AST 기대 길이)
    T_AST = 1024
    if x.size(0) < T_AST:
        pad = torch.zeros(T_AST - x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    else:
        x = x[:T_AST, :]

    # 3) AST 1회 추론
    out = ast_model(input_values=x.unsqueeze(0), output_attentions=True,
                    output_hidden_states=True, return_dict=True)

    # 4) 어텐션 맵 (마지막 레이어, CLS->patches)
    A = out.attentions[-1]                              # [1, heads, seq, seq]
    cls_to_patches = A[0, :, 0, 2:].mean(dim=0)        # [num_patches]
    # 패치 2D reshape
    Fp = 12
    num_patches = cls_to_patches.numel()
    Tp = max(1, num_patches // Fp)
    cls_to_patches = cls_to_patches[:Fp*Tp]
    full_map = cls_to_patches.reshape(Fp, Tp)

    time_attn_patch = full_map.mean(dim=0)             # [Tp]
    freq_attn_patch = full_map.mean(dim=1)             # [Fp]

    # 출력 해상도로 보간
    time_attn_T = F.interpolate(time_attn_patch.view(1,1,-1), size=T_out, mode="linear", align_corners=False).view(-1)
    time_attn_T = norm01(smooth1d(time_attn_T, SMOOTH_T))
    freq_attn_M = F.interpolate(freq_attn_patch.view(1,1,-1), size=F_out, mode="linear", align_corners=False).view(-1)
    freq_attn_M = norm01(freq_attn_M)

    # 5) 마지막 레이어 히든 (특수토큰 2개 제외)
    last_hidden = out.hidden_states[-1][0]             # [seq, D]
    patch_tokens = last_hidden[2:2+Fp*Tp, :].contiguous().view(Fp, Tp, -1)  # [Fp,Tp,D]
    return time_attn_T, freq_attn_M, time_attn_patch, freq_attn_patch, patch_tokens

@torch.no_grad()
def classify_anchor_from_cache(ast_model,
                               time_attn_patch: torch.Tensor,
                               freq_attn_patch: torch.Tensor,
                               patch_tokens: torch.Tensor,
                               s: int, e: int, T_stft: int):
    """
    패치 어텐션(time×freq) 가중 평균으로 앵커 임베딩 생성 → classifier 통과.
    dropout 모듈 호출 없이 classifier만 사용.
    """
    Fp, Tp, D = patch_tokens.shape

    # STFT frame -> patch 시간축 매핑
    t0 = int(round(s * Tp / max(1, T_stft)))
    t1 = int(round(e * Tp / max(1, T_stft)))
    t0 = max(0, min(Tp - 1, t0))
    t1 = max(t0 + 1, min(Tp, t1))

    wt = time_attn_patch[t0:t1].clamp_min(0)
    wf = freq_attn_patch.clamp_min(0)
    if float(wt.sum()) == 0: wt = torch.ones_like(wt)
    if float(wf.sum()) == 0: wf = torch.ones_like(wf)
    wt = wt / (wt.sum() + 1e-8)
    wf = wf / (wf.sum() + 1e-8)

    W = (wf.view(Fp,1) * wt.view(1, t1 - t0)).unsqueeze(-1)  # [Fp, t_len, 1]
    anchor_embed = (patch_tokens[:, t0:t1, :] * W).sum(dim=(0,1))      # [D]

    classifier = ast_model.classifier
    anchor_embed = anchor_embed.to(next(classifier.parameters()).dtype).unsqueeze(0)  # [1,D]
    logits = classifier(anchor_embed)
    probs  = torch.softmax(logits, dim=-1)[0]
    class_id = int(probs.argmax().item())
    conf     = float(probs[class_id].item())
    class_name = ast_model.config.id2label[class_id]
    sound_type = get_sound_type(class_id)
    return class_name, sound_type, class_id, conf

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
    T = score.numel()
    peak_idx = int(torch.argmax(score).item())
    anchor_s = max(0, min(peak_idx - (La // 2), T - La))
    anchor_e = anchor_s + La
    local_score = score[anchor_s:anchor_e]
    peak_idx_rel = int(torch.argmax(local_score).item())
    threshold = torch.quantile(local_score, core_pct)
    core_s_rel = peak_idx_rel
    while core_s_rel > 0 and local_score[core_s_rel - 1] >= threshold: core_s_rel -= 1
    core_e_rel = peak_idx_rel
    while core_e_rel < La - 1 and local_score[core_e_rel + 1] >= threshold: core_e_rel += 1
    core_e_rel += 1
    return anchor_s, anchor_e, core_s_rel, core_e_rel

# =========================
# Ω & Template
# =========================
def omega_support_with_ast_freq(Ablk: torch.Tensor, ast_freq_attn: torch.Tensor, strategy: str = "conservative") -> torch.Tensor:
    if strategy == "conservative":
        omega_q = OMEGA_Q_CONSERVATIVE; ast_freq_quantile = AST_FREQ_QUANTILE_CONSERVATIVE
    else:
        omega_q = OMEGA_Q_AGGRESSIVE;   ast_freq_quantile = AST_FREQ_QUANTILE_AGGRESSIVE
    med = Ablk.median(dim=1).values
    th = torch.quantile(med, omega_q)
    mask_energy = (med >= th).float()
    ast_freq_th = torch.quantile(ast_freq_attn, ast_freq_quantile)
    mask_ast_freq = (ast_freq_attn >= ast_freq_th).float()
    mask = torch.maximum(mask_energy, mask_ast_freq)
    for _ in range(OMEGA_DIL):
        mask = torch.maximum(mask, torch.roll(mask, 1))
        mask = torch.maximum(mask, torch.roll(mask, -1))
    if int(mask.sum().item()) < OMEGA_MIN_BINS:
        order = torch.argsort(med, descending=True)
        need = OMEGA_MIN_BINS - int(mask.sum().item())
        mask[order[:need]] = 1.0
    return mask

def template_from_anchor_block(Ablk: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    om = omega.view(-1,1)
    w  = (Ablk * om).mean(dim=1) * omega
    w  = w / (w.sum() + EPS)
    w_sm = F.avg_pool1d(w.view(1,1,-1), kernel_size=3, stride=1, padding=1).view(-1)
    w   = (w_sm * omega); w = w / (w.sum() + EPS)
    return w

# =========================
# Presence & Similarity
# =========================
def presence_from_energy(Xmel: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    om = omega.view(-1,1)
    e_omega = (Xmel * om).sum(dim=0)
    e_omega = smooth1d(e_omega, PRES_SMOOTH_T)
    thr = torch.quantile(e_omega, PRES_Q).clamp(min=1e-10)
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
# Unified Masking (하드 시간 임계 포함)
# =========================
def unified_masking_strategy(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor,
                             ast_freq_attn: torch.Tensor, P: torch.Tensor, mel_fb_m2f: torch.Tensor,
                             s: int, e: int, strategy: str = "conservative") -> torch.Tensor:
    fbins, T = P.shape
    g_pres = presence_from_energy(Xmel, omega)
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)  # [T]

    # 하드 시간 컷
    time_mask_hard = (cos_t_raw >= TIME_SIM_THRESHOLD).float()

    # 전략별 cos 처리
    if strategy == "conservative":
        cos_processed = (cos_t_raw ** 2)
    else:
        cos_processed = torch.sigmoid(MASK_SIGMOID_SLOPE * (cos_t_raw - MASK_SIGMOID_CENTER))

    cos_processed = (cos_processed * time_mask_hard).clamp(0, 1)

    # Linear freq weighting
    omega_lin = ((mel_fb_m2f.T @ omega).clamp_min(0.0) > 1e-12).float()
    anchor_spec = P[:, s:e]
    anchor_max_amp = anchor_spec.max(dim=1).values
    amp_threshold = torch.quantile(anchor_max_amp, 0.8)
    high_amp_mask_lin = (anchor_max_amp >= amp_threshold).float()
    ast_freq_threshold = torch.quantile(ast_freq_attn, 0.4 if strategy == "conservative" else 0.2)
    ast_active_mask_mel = (ast_freq_attn >= ast_freq_threshold).float()
    ast_active_mask_lin = ((mel_fb_m2f.T @ ast_active_mask_mel).clamp_min(0.0) > 0.1).float()
    freq_boost_mask = torch.maximum(high_amp_mask_lin, ast_active_mask_lin)

    if strategy == "conservative":
        freq_weight = 1.0 + freq_boost_mask
    else:
        freq_weight = 1.0 + 0.3 * freq_boost_mask

    M_base     = omega_lin.view(-1, 1) * cos_processed.view(1, -1)
    M_weighted = M_base * freq_weight.view(-1, 1)

    # 이중 안전장치: 시간 하드마스크 적용
    M_weighted = M_weighted * time_mask_hard.view(1, -1)

    # 스펙트럼 한계
    spec_magnitude = P.sqrt()
    M_lin = torch.minimum(M_weighted, spec_magnitude)
    return M_lin

def adaptive_strategy_selection(prev_energy_ratio: float, pass_idx: int) -> str:
    if not USE_ADAPTIVE_STRATEGY: return "conservative"
    if pass_idx == 0: return "conservative"
    if prev_energy_ratio > 2.0: return "conservative"
    if prev_energy_ratio < 1.2: return "aggressive"
    return "conservative"

# =========================
# Single Pass
# =========================
def single_pass(audio: np.ndarray, extractor, ast_model,
                mel_fb_m2f: torch.Tensor,
                used_mask_prev: Optional[torch.Tensor],
                prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor]],
                pass_idx: int, out_dir: Optional[str], prev_energy_ratio: float = 1.0,
                profile: bool = False):

    t0 = time.time()
    # STFT & Mel
    st, mag, P, phase_complex, Xmel = stft_all(audio, mel_fb_m2f)
    fbins, T = P.shape
    La = int(round(ANCHOR_SEC * SR / HOP))

    # Strategy
    strategy = adaptive_strategy_selection(prev_energy_ratio, pass_idx)
    print(f"  🎯 Strategy: {strategy}")

    t_ast0 = time.time()
    # AST 1회 (Xmel 재사용)
    A_t, ast_freq_attn, time_attn_patch, freq_attn_patch, patch_tokens = \
        ast_forward_once_from_mel(Xmel, extractor, ast_model, T, N_MELS)
    t_ast1 = time.time()
    if profile:
        print(f"  [TIMING] AST fwd (from-mel): {t_ast1 - t_ast0:.3f}s")

    # Anchor score
    Pur = purity_from_P(P)
    Sc  = anchor_score(A_t, Pur)

    # Suppress used frames
    if used_mask_prev is not None:
        um = align_len_1d(used_mask_prev, T, device=Sc.device, mode="linear")
        k = int(round((USED_DILATE_MS/1000.0)*SR/HOP)); k = ensure_odd(max(1,k))
        ker = torch.ones(k, device=Sc.device)/k
        um = (F.conv1d(um.view(1,1,-1), ker.view(1,1,-1), padding=k//2).view(-1) > 0.2).float()
        Sc = Sc * (1 - 0.85 * um)

    # Suppress previous anchors
    for (sa, ea, prev_w, prev_omega) in prev_anchors:
        ca = int(((sa+ea)/2) * SR / HOP); ca = max(0, min(T-1, ca))
        sigma = int(round((ANCHOR_SUPPRESS_MS/1000.0)*SR/HOP))
        idx = torch.arange(T, device=Sc.device) - ca
        Sc = Sc * (1 - ANCHOR_SUPPRESS_BASE * torch.exp(-(idx**2)/(2*(sigma**2)+1e-8)))
        core_s = max(0, ca - La//2); core_e = min(T, ca + La//2)
        Sc[core_s:core_e] *= 0.2

    s, e, core_s_rel, core_e_rel = pick_anchor_region(Sc, La, TOP_PCT_CORE_IN_ANCHOR)

    # Anchor block
    Ablk = Xmel[:, s:e].clone()
    if core_s_rel > 0:  Ablk[:, :core_s_rel] = 0
    if core_e_rel < La: Ablk[:, core_e_rel:] = 0

    # Ω & template
    omega = omega_support_with_ast_freq(Ablk, ast_freq_attn, strategy)
    w_bar = template_from_anchor_block(Ablk, omega)

    # Mask
    M_lin = unified_masking_strategy(Xmel, w_bar, omega, ast_freq_attn, P, mel_fb_m2f, s, e, strategy)

    # ===== Separation (magnitude mask + phase preserve) =====
    mag_masked = M_lin * mag
    stft_src   = mag_masked * phase_complex
    mag_resid  = torch.maximum(mag - mag_masked, torch.zeros_like(mag))
    stft_res   = mag_resid * phase_complex
    # =======================================================

    # Energy check & scaling
    src_energy  = torch.sum(torch.abs(stft_src)**2).item()
    res_energy  = torch.sum(torch.abs(stft_res)**2).item()
    orig_energy = torch.sum(torch.abs(st)**2).item()
    total_energy = src_energy + res_energy
    energy_ratio = total_energy / (orig_energy + 1e-8)
    if energy_ratio > 1.02:
        scale = (orig_energy / (total_energy + 1e-8)) ** 0.5
        stft_src *= scale; stft_res *= scale
        # (선택) 재계산
        src_energy  = torch.sum(torch.abs(stft_src)**2).item()
        res_energy  = torch.sum(torch.abs(stft_res)**2).item()
        total_energy = src_energy + res_energy

    print(f"  🔍 Energy: Orig={orig_energy:.2f}, Src={src_energy:.2f}, Res={res_energy:.2f}, Total/Orig={total_energy/(orig_energy+1e-8):.3f}")

    # ISTFT
    t_i0 = time.time()
    src_amp = torch.istft(
        stft_src, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
        window=WINDOW, center=CENTER_STFT, length=L_FIXED
    ).detach().cpu().numpy()
    
    res = torch.istft(
        stft_res, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
        window=WINDOW, center=CENTER_STFT, length=L_FIXED
    ).detach().cpu().numpy()
    t_i1 = time.time()
    if profile:
        print(f"  [TIMING] ISTFT pair: {t_i1 - t_i0:.3f}s")

    # ER
    e_src = float(np.sum(src_amp**2)); e_res = float(np.sum(res**2))
    er = e_src / (e_src + e_res + 1e-12)

    # 앵커 전용 분류 (AST 재사용)
    class_name, sound_type, class_id, confidence = classify_anchor_from_cache(
        ast_model, time_attn_patch, freq_attn_patch, patch_tokens, s, e, T
    )
    db_min, db_max, db_mean = calculate_decibel(src_amp)

    # Debug
    if out_dir is not None:
        a_raw = amplitude_raw(Xmel, w_bar, omega)
        cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, presence_from_energy(Xmel, omega))
        C_t = cos_t_raw
        full_map = torch.zeros(12, max(1, time_attn_patch.numel()))
        png = os.path.join(out_dir, f"debug_pass_{pass_idx+1}.png")
        debug_plot(pass_idx, Sc, a_raw, cos_t_raw, C_t, P, M_lin, full_map,
                   s, e, core_s_rel, core_e_rel, ast_freq_attn, src_amp, res, png,
                   title=f"Pass {pass_idx+1} | {strategy} | anchor {s*HOP/SR:.2f}-{e*HOP/SR:.2f}s | ER={er*100:.1f}% [One-AST]")

    # Used frames
    r_t = (M_lin * P).sum(dim=0) / (P.sum(dim=0) + 1e-8)
    used_mask = (r_t >= USED_THRESHOLD).float()

    elapsed = time.time() - t0
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
        "db_mean": db_mean
    }
    return src_amp, res, er, used_mask, info

# =========================
# Model helpers
# =========================
def trim_ast_layers(ast_model, keep: int = 6):
    """
    AST 인코더 레이어를 앞에서 keep개만 남겨 연산량 절감.
    """
    try:
        enc = ast_model.audio_spectrogram_transformer.encoder
        enc.layer = torch.nn.ModuleList(list(enc.layer)[:keep])
        if hasattr(ast_model.config, "num_hidden_layers"):
            ast_model.config.num_hidden_layers = keep
        print(f"✅ Trimmed AST encoder to {keep} layers")
    except Exception as e:
        print(f"⚠️ AST layer trim skipped: {e}")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Unified AST-guided Source Separator (Fast)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--passes", type=int, default=1, help="Number of separation passes (default 1 for speed)")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--strategy", choices=["conservative", "aggressive", "adaptive"], default="adaptive")
    ap.add_argument("--int8", action="store_true", help="Dynamic-quantize AST Linear layers to int8")
    ap.add_argument("--keep-ast-layers", type=int, default=6, help="Keep only first N AST encoder layers (speed)")
    ap.add_argument("--skip-residual-cls", action="store_true", help="Skip residual AST classification at the end")
    ap.add_argument("--profile", action="store_true", help="Print timing for major steps")
    args = ap.parse_args()

    global USE_ADAPTIVE_STRATEGY
    USE_ADAPTIVE_STRATEGY = (args.strategy == "adaptive")

    os.makedirs(args.output, exist_ok=True)

    # Load audio
    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 Unified AST-guided Source Separator (Fast, One-AST/pass)\n{'='*64}")
    print(f"Input: fixed {WIN_SEC:.3f}s, Anchor: {ANCHOR_SEC:.3f}s, Passes: {args.passes}, Strategy: {args.strategy}")
    print(f"Options: int8={args.int8}, keep_layers={args.keep_ast_layers}, skip_residual_cls={args.skip_residual_cls}, no_debug={args.no_debug}")

    # Mel filter banks
    fbins = N_FFT//2 + 1
    mel_fb_f2m = torchaudio.functional.melscale_fbanks(
        n_freqs=fbins, f_min=0.0, f_max=SR/2, n_mels=N_MELS,
        sample_rate=SR, norm="slaney"
    )
    mel_fb_m2f = mel_fb_f2m.T.contiguous()

    # AST
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast_model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        attn_implementation="eager"
    ).eval()

    # Trim layers for speed
    if args.keep_ast_layers and args.keep_ast_layers > 0:
        trim_ast_layers(ast_model, keep=args.keep_ast_layers)

    # Dynamic int8 quantization (Linear only)
    if args.int8:
        import torch.nn as nn
        ast_model = torch.quantization.quantize_dynamic(
            ast_model, {nn.Linear}, dtype=torch.qint8
        ).eval()
        print("✅ AST dynamic quantized (int8)")

    cur = audio0.copy()
    used_mask_prev = None
    prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor]] = []
    total_t0 = time.time()
    saved = 0
    prev_energy_ratio = 1.0

    for i in range(max(1, args.passes)):
        pass_t0 = time.time()
        print(f"\n▶ Pass {i+1}/{args.passes}")
        src, res, er, used_mask_prev, info = single_pass(
            cur, extractor, ast_model, mel_fb_m2f,
            used_mask_prev, prev_anchors,
            pass_idx=i, out_dir=None if args.no_debug else args.output,
            prev_energy_ratio=prev_energy_ratio, profile=args.profile
        )

        class_name = info['class_name']; sound_type = info['sound_type']
        class_id = info['class_id']; confidence = info['confidence']
        strategy = info['strategy']; energy_ratio = info['energy_ratio']
        db_min, db_max, db_mean = info['db_min'], info['db_max'], info['db_mean']

        print(f"⏱️ pass{i+1}: {info['elapsed']:.3f}s | anchor {info['anchor'][0]:.2f}-{info['anchor'][1]:.2f}s"
              f" | core {info['core'][0]:.2f}-{info['core'][1]:.2f}s | ER={er*100:.1f}%")
        print(f"  🎵 Class(Anchor): {class_name} (ID {class_id}) [{sound_type}] conf={confidence:.3f}")
        print(f"  🔊 dB: min={db_min:.1f}, max={db_max:.1f}, mean={db_mean:.1f}")

        if er < MIN_ERATIO:
            print("  ⚠️ Too little energy; stopping.")
            break

        # Peak normalization
        peak = np.max(np.abs(src))
        if peak > 1e-8:
            src = np.clip(src * (NORMALIZE_TARGET_PEAK / peak), -1.0, 1.0)

        # Save source
        safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
        out_src = os.path.join(args.output, f"{i+1:02d}_{safe_class_name}_{sound_type}.wav")
        torchaudio.save(out_src, torch.from_numpy(src).unsqueeze(0), SR)
        print(f"    ✅ Saved: {out_src}")

        cur = res
        prev_anchors.append((info["anchor"][0], info["anchor"][1], info["w_bar"], info["omega"]))
        prev_energy_ratio = energy_ratio
        saved += 1
        if args.profile:
            print(f"  [TIMING] Pass wall time: {time.time() - pass_t0:.3f}s")

    # Residual post
    if RESIDUAL_CLIP_THR > 0:
        cur[np.abs(cur) < RESIDUAL_CLIP_THR] = 0.0

    # Residual classification (옵션)
    if not args.skip_residual_cls:
        print(f"\n🔍 Classifying residual (from-mel)...")
        # Mel 만들기
        _, _, _, _, Xmel_res = stft_all(cur, mel_fb_m2f)
        A_t_res, ast_freq_attn_res, tpatch_res, fpatch_res, ptokens_res = \
            ast_forward_once_from_mel(Xmel_res, extractor, ast_model, T_out=Xmel_res.shape[1], F_out=N_MELS)
        # 전 구간을 앵커처럼 분류
        r_class, r_type, r_id, r_conf = classify_anchor_from_cache(
            ast_model, tpatch_res, fpatch_res, ptokens_res, s=0, e=Xmel_res.shape[1], T_stft=Xmel_res.shape[1]
        )
        print(f"  🎵 Residual: {r_class} (ID {r_id}) [{r_type}] conf={r_conf:.3f}")
        # 파일명
        if r_conf >= 0.6:
            safe_res = "".join(c for c in r_class if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
            out_res = os.path.join(args.output, f"00_{safe_res}_{r_type}.wav")
        else:
            out_res = os.path.join(args.output, "00_residual.wav")
    else:
        print("\nℹ️ Skipping residual classification (--skip-residual-cls)")
        out_res = os.path.join(args.output, "00_residual.wav")

    torchaudio.save(out_res, torch.from_numpy(cur).unsqueeze(0), SR)

    total_elapsed = time.time() - total_t0
    print(f"\n💾 Residual: {out_res}")
    print(f"⏱️ Total: {total_elapsed:.3f}s")
    print(f"✅ Done. Separated: {saved}\n")

if __name__ == "__main__":
    main()
