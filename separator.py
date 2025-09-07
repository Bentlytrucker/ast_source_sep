#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST-guided Source Separator — Hard Time Threshold + Short Padding, Sigmoid Ω (Full)

What this version does (per your spec):
- Time selection is HARD by cosine threshold (tau=0.6), then we add a short
  temporal padding (2–3 frames) to relax edge choppiness:
      b(t) = 1[cosΩ(t) >= tau]
      time_weight(t) = dilation(b, K)   # K frames (10ms hop ⇒ ~K*10ms)
- Frequency selection uses a CONTINUOUS (sigmoid) Ω on mel bins instead of binary Ω.
  This reduces horizontal striping / mid-band holes when projected to linear freq:
      omega_soft(m) = σ(slope * (med_norm(m) - center_quantile))
- Separation is complex-domain masking; residual per-frame tiny energy pruning; peak-normalized sources.

Dependencies: torch, torchaudio, transformers, matplotlib, numpy
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
# Frequency-guided padding strength
FREQ_GUIDE_GAMMA = 1.2  # 1.0~1.5 권장; 클수록 cosΩ 높은 프레임만 강하게 살림

SR                = 16000
WIN_SEC           = 4.096
ANCHOR_SEC        = 0.512
L_FIXED           = int(round(WIN_SEC * SR))

# === Output Processing ===
NORMALIZE_TARGET_PEAK = 0.95
RESIDUAL_CLIP_THR     = 0.0005

# === Time mask: hard cosine threshold + short padding ===
TIME_TAU      = 0.7   # hard threshold on cosΩ(t)
TIME_PAD_K    = 3      # +/-K frames dilation (10ms hop ⇒ ~30ms). Use 2~3 as discussed.

# STFT (10ms hop)
N_FFT, HOP, WINLEN = 400, 160, 400
WINDOW = torch.hann_window(WINLEN)
EPS = 1e-10

# Mel
N_MELS = 128

# Anchor score
SMOOTH_T      = 19
ALPHA_ATT     = 0.80
BETA_PUR      = 1.20
W_E           = 0.30
TOP_PCT_CORE_IN_ANCHOR  = 0.50

# Ω & template (SOFT Ω settings)
OMEGA_Q           = 0.70     # center via quantile on median-normalized mel
OMEGA_SIG_SLOPE   = 10.0     # sigmoid slope (7~12 recommended)
OMEGA_MIN_BINS    = 10       # safety floor for coverage

# Suppression / usage
USED_THRESHOLD        = 0.65
USED_DILATE_MS        = 80
ANCHOR_SUPPRESS_MS    = 200
ANCHOR_SUPPRESS_BASE  = 0.6

# Residual pruning per pass (frame-energy ratio)
PRUNE_FRAME_RATIO_THR = 0.005  # 0.5%

# Loop
MAX_PASSES    = 3
MIN_ERATIO    = 0.005  # stop if ER < 1%

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
# Attention & Purity
# =========================
@torch.no_grad()
def ast_attention_time(audio: np.ndarray, extractor, ast_model, T_out: int) -> torch.Tensor:
    feat = extractor(audio, sampling_rate=SR, return_tensors="pt")
    out  = ast_model(input_values=feat["input_values"], output_attentions=True, return_dict=True)
    attns = out.attentions
    if not attns or len(attns)==0:
        return torch.ones(T_out)*0.5
    A = attns[-1]
    cls = A[:, :, 0, 1:].mean(dim=1).squeeze(0)
    Fp = 12
    Np = cls.numel()
    Tp = Np // Fp
    if Tp < 2: return torch.ones(T_out)*0.5
    grid = cls[:Fp*Tp].reshape(Fp, Tp).mean(dim=0)
    grid = norm01(grid)
    A_t  = F.interpolate(grid.view(1,1,-1), size=T_out, mode="linear", align_corners=False).view(-1)
    return norm01(smooth1d(A_t, SMOOTH_T))

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
    Peak-first anchor & core inside anchor (quantile-based expansion).
    Returns: (anchor_start, anchor_end, core_s_rel, core_e_rel)
    """
    T = score.numel()
    peak_idx = int(torch.argmax(score).item())
    anchor_s = max(0, min(peak_idx - (La // 2), T - La))
    anchor_e = anchor_s + La

    local_score = score[anchor_s:anchor_e]
    peak_idx_rel = int(torch.argmax(local_score).item())

    threshold = torch.quantile(local_score, core_pct)

    core_s_rel = peak_idx_rel
    while core_s_rel > 0 and local_score[core_s_rel - 1] >= threshold:
        core_s_rel -= 1

    core_e_rel = peak_idx_rel
    while core_e_rel < La - 1 and local_score[core_e_rel + 1] >= threshold:
        core_e_rel += 1

    core_e_rel += 1  # end-exclusive
    return anchor_s, anchor_e, core_s_rel, core_e_rel

def select_anchor_with_energy(score: torch.Tensor, La: int, core_pct: float,
                              P: torch.Tensor, max_tries: int = 5):
    """
    Try top peaks up to max_tries. Reject if anchor energy ratio <= 0.5%.
    Returns: (s,e,core_s_rel,core_e_rel, energy_ratio) or None if not found.
    """
    Sc = score.clone()
    T = score.numel()
    tried = 0
    for _ in range(max_tries):
        s, e, core_s_rel, core_e_rel = pick_anchor_region(Sc, La, core_pct)
        E_anchor = P[:, s:e].sum()
        E_total  = P.sum()
        ratio = float((E_anchor / (E_total + EPS)).item())
        if ratio > 0.005:
            return s, e, core_s_rel, core_e_rel, ratio
        # suppress this region and try next
        peak_idx = int(torch.argmax(Sc).item())
        sup_s = max(0, peak_idx - La)
        sup_e = min(T, peak_idx + La)
        Sc[sup_s:sup_e] = 0.0
        tried += 1
        if Sc.max() <= 0 or tried >= max_tries:
            break
    return None

# =========================
# Ω & Template
# =========================
def omega_support_soft(Ablk: torch.Tensor) -> torch.Tensor:
    """
    Continuous Ω(m) ∈ [0,1] using a sigmoid over normalized median energy per mel bin.
    - Normalize mel medians to [0,1]
    - Center at the OMEGA_Q quantile
    - Apply sigmoid with slope OMEGA_SIG_SLOPE
    - Light smoothing
    Ensures at least OMEGA_MIN_BINS have noticeable weight by gentle re-scaling.
    """
    med  = Ablk.median(dim=1).values           # [M]
    if float(med.max().item()) <= 1e-12:
        return torch.zeros_like(med)
    medn = (med - med.min()) / (med.max() - med.min() + 1e-8)
    c    = torch.quantile(medn, OMEGA_Q)
    omega_soft = torch.sigmoid(OMEGA_SIG_SLOPE * (medn - c))  # (0,1)
    # light smoothing (3-bin)
    omega_soft = F.avg_pool1d(omega_soft.view(1,1,-1), kernel_size=3, stride=1, padding=1).view(-1)
    # ensure some coverage
    if int((omega_soft > 0.01).sum().item()) < OMEGA_MIN_BINS:
        order = torch.argsort(medn, descending=True)
        take = order[:OMEGA_MIN_BINS]
        omega_soft[take] = torch.maximum(omega_soft[take], torch.tensor(0.5, device=omega_soft.device))
    return omega_soft.clamp(0,1)

def template_from_anchor_block(Ablk: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    om = omega.view(-1,1)
    w  = (Ablk * om).mean(dim=1)
    w  = (w * omega)
    s  = w.sum() + EPS
    w  = w / s
    w_sm = F.avg_pool1d(w.view(1,1,-1), kernel_size=3, stride=1, padding=1).view(-1)
    w   = (w_sm * omega)
    w   = w / (w.sum() + EPS)
    return w

# =========================
# Time similarity & freq mapping
# =========================
def cos_similarity_over_omega_no_gate(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    om = omega.view(-1,1)
    Xo = Xmel * om
    wn = (w_bar * omega); wn = wn / (wn.norm(p=2) + 1e-8)
    Xn = Xo / (Xo.norm(p=2, dim=0, keepdim=True) + 1e-8)
    cos_raw = (wn.view(-1,1) * Xn).sum(dim=0).clamp(0,1)
    return cos_raw

def omega_linear_continuous(mel_fb_m2f: torch.Tensor, omega: torch.Tensor, gamma_f: float = 0.9) -> torch.Tensor:
    """
    Continuous mapping from mel weights (omega in [0,1]) to linear freq weights q in [0,1],
    with light smoothing along frequency.
    mel_fb_m2f: [M,F], omega: [M]
    """
    M, Freq = mel_fb_m2f.shape
    if float(omega.sum().item()) <= 1e-12:
        return torch.zeros(Freq, device=mel_fb_m2f.device, dtype=mel_fb_m2f.dtype)
    om = omega / (omega.sum() + EPS)
    q = (mel_fb_m2f.T @ om).view(-1)  # [F]
    q = (q / (q.max() + EPS)).clamp(0,1)
    if gamma_f != 1.0:
        q = q.pow(gamma_f)
    # light smoothing (5-bin for slightly smoother profile)
    q = F.avg_pool1d(q.view(1,1,-1), kernel_size=5, stride=1, padding=2).view(-1).clamp(0,1)
    return q

# =========================
# Time mask: hard threshold + padding
# =========================
def freq_guided_edge_fade(cos_t: torch.Tensor, tau: float, K: int, gamma: float = 1.2) -> torch.Tensor:
    """
    Hard 시간 마스크(b) + 엣지 페이드(half-cosine)를 적용하되,
    페이드 구간의 크기를 cosΩ(t)^gamma 로 스케일하는 '주파수-가이드' 버전.
    시드(=b==1) 내부는 그대로 1 유지, 경계 패딩 프레임만 연속값으로 완충.
    """
    T = cos_t.numel()
    b = (cos_t >= tau).float()
    if K <= 0 or T == 0:
        return b

    # 기본은 시드 유지
    m = b.clone()

    # half-cosine 램프
    t = torch.arange(0, K+1, device=cos_t.device, dtype=cos_t.dtype)
    fade_in_full  = 0.5 - 0.5*torch.cos(torch.pi * (t / K))  # 0..1
    fade_out_full = 0.5 + 0.5*torch.cos(torch.pi * (t / K))  # 1..0

    # 엣지 검출
    d = torch.zeros_like(b)
    d[1:] = b[1:] - b[:-1]
    rising  = (d > 0).nonzero(as_tuple=False).view(-1)         # 0->1 발생 지점 t0 (첫 1의 인덱스)
    falling0 = (d < 0).nonzero(as_tuple=False).view(-1)        # 1->0 직후의 0 인덱스
    falling = (falling0 - 1).clamp(min=0)                      # 마지막 1의 인덱스 t1

    pow_cos = cos_t.clamp(0,1).pow(gamma)                      # 주파수 유사도 기반 가이드

    # Rising edge: ... 0 0 [패딩] 1 |
    for t0 in rising.tolist():
        start = max(0, t0 - K)
        L = t0 - start + 1
        fade = fade_in_full[-L:]                                # 0->1 램프 tail 정렬
        guide = pow_cos[start:t0+1]                             # 각 프레임의 cosΩ^γ
        cand = fade * guide                                     # 가이드된 페이드
        m[start:t0+1] = torch.maximum(m[start:t0+1], cand)

    # Falling edge: | 1 [패딩] 0 0 ...
    for t1 in falling.tolist():
        end = min(T-1, t1 + K)
        L = end - t1 + 1
        fade = fade_out_full[:L]                                # 1->0 램프 head 정렬
        guide = pow_cos[t1:end+1]
        cand = fade * guide
        m[t1:end+1] = torch.maximum(m[t1:end+1], cand)

    return m.clamp(0,1)


# =========================
# Debug
# =========================
def debug_plot(pass_idx:int, score:torch.Tensor, cos_t:torch.Tensor, time_weight:torch.Tensor,
               P:torch.Tensor, M_lin:torch.Tensor,
               s:int, e:int, core_s_rel:int, core_e_rel:int,
               out_png:str, title:str):
    fbins, T = P.shape
    t = np.arange(T) * HOP / SR
    fig, ax = plt.subplots(2,3, figsize=(15,8))

    # 1) Anchor score + windows
    ax[0,0].plot(t, to_np(score), lw=1.2)
    ax[0,0].axvspan(s*HOP/SR, e*HOP/SR, color='orange', alpha=0.20, label='anchor')
    cs = s + core_s_rel; ce = s + core_e_rel
    ax[0,0].axvspan(cs*HOP/SR, ce*HOP/SR, color='red', alpha=0.20, label='anchor-core')
    ax[0,0].legend(); ax[0,0].set_title("Anchor score"); ax[0,0].set_ylim([0,1.05])

    # 2) Scalars
    ax[0,1].plot(t, to_np(cos_t), label='cosΩ', lw=1.0)
    ax[0,1].plot(t, to_np(time_weight), label=f'time weight (hard τ={TIME_TAU}+pad K={TIME_PAD_K})', lw=1.0, alpha=0.85)
    ax[0,1].legend(); ax[0,1].set_ylim([0,1.05]); ax[0,1].set_title("Scalars")

    # 3) Mask stats
    m_mean = to_np(M_lin.mean(dim=0))
    ax[0,2].plot(t, m_mean); ax[0,2].set_ylim([0,1.05])
    ax[0,2].set_title("Mask frame-mean")

    # 4-6) Spectrograms (Linear Amplitude)
    spec_mag = P.sqrt()
    masked_mag = (M_lin * P).sqrt()

    vmin = 0.0
    vmax = to_np(spec_mag.max())

    im0 = ax[1,0].imshow(to_np(spec_mag), aspect='auto', origin='lower', cmap='viridis',
                         vmin=vmin, vmax=vmax,
                         extent=[0, t[-1] if len(t)>0 else 0, 0, fbins])
    ax[1,0].set_title("Spec (Linear Amp)"); plt.colorbar(im0, ax=ax[1,0])

    im1 = ax[1,1].imshow(to_np(M_lin), aspect='auto', origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                         extent=[0, t[-1] if len(t)>0 else 0, 0, fbins])
    ax[1,1].set_title("Mask"); plt.colorbar(im1, ax=ax[1,1])

    im2 = ax[1,2].imshow(to_np(masked_mag), aspect='auto', origin='lower', cmap='viridis',
                         vmin=vmin, vmax=vmax,
                         extent=[0, t[-1] if len(t)>0 else 0, 0, fbins])
    ax[1,2].set_title("Masked Spec (Linear Amp)"); plt.colorbar(im2, ax=ax[1,2])

    plt.suptitle(title); plt.tight_layout(); plt.savefig(out_png, dpi=120, bbox_inches='tight'); plt.close()
    print(f"  📊 Debug saved: {out_png}")

# =========================
# Single Pass (Hard mask + padding, Soft Ω)
# =========================
def single_pass(audio: np.ndarray, extractor, ast_model,
                mel_fb_m2f: torch.Tensor,
                used_mask_prev: Optional[torch.Tensor],
                prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor]],
                pass_idx:int, out_dir:Optional[str]):

    t0 = time.time()
    st, mag, P, phase, Xmel = stft_all(audio, mel_fb_m2f)
    fbins, T = P.shape
    La = int(round(ANCHOR_SEC * SR / HOP))

    # Anchor score
    A_t  = ast_attention_time(audio, extractor, ast_model, T)
    Pur  = purity_from_P(P)
    Sc   = anchor_score(A_t, Pur)

    # Suppress used frames (from previous pass hard usage)
    if used_mask_prev is not None:
        um = align_len_1d(used_mask_prev, T, device=Sc.device, mode="linear")
        k = int(round((USED_DILATE_MS/1000.0)*SR/HOP)); k = ensure_odd(max(1,k))
        ker = torch.ones(k, device=Sc.device)/k
        um = (F.conv1d(um.view(1,1,-1), ker.view(1,1,-1), padding=k//2).view(-1) > 0.2).float()
        Sc = Sc * (1 - 0.85 * um)

    # Enhance suppression of previous anchors
    for (sa, ea, prev_w, prev_omega) in prev_anchors:
        ca = int(((sa+ea)/2) * SR / HOP)
        ca = max(0, min(T-1, ca))
        sigma = int(round((ANCHOR_SUPPRESS_MS/1000.0)*SR/HOP))
        idx = torch.arange(T, device=Sc.device) - ca
        Sc = Sc * (1 - ANCHOR_SUPPRESS_BASE * torch.exp(-(idx**2)/(2*(sigma**2)+1e-8)))
        core_s = max(0, ca - La//2); core_e = min(T, ca + La//2)
        Sc[core_s:core_e] *= 0.2

    # Select anchor with energy validation (>= 0.5%)
    sel = select_anchor_with_energy(Sc, La, TOP_PCT_CORE_IN_ANCHOR, P, max_tries=5)
    if sel is None:
        info = {"stopped": True, "reason":"no-valid-anchor(<0.5% energy)", "elapsed": time.time()-t0}
        return None, None, 0.0, None, info
    s, e, core_s_rel, core_e_rel, anchor_ratio = sel

    # Construct anchor block (core only)
    Ablk = Xmel[:, s:e].clone()
    if core_s_rel > 0:  Ablk[:, :core_s_rel] = 0
    if core_e_rel < (e - s): Ablk[:, core_e_rel:] = 0

    # Ω (soft sigmoid) & template
    omega = omega_support_soft(Ablk)                 # [M] continuous in [0,1]
    w_bar = template_from_anchor_block(Ablk, omega)  # [M]

    # Cosine similarity over Ω
    cos_t_raw = cos_similarity_over_omega_no_gate(Xmel, w_bar, omega)   # [T] in [0,1]

    # HARD mask + temporal padding (dilation)
    time_weight = freq_guided_edge_fade(cos_t_raw, tau=TIME_TAU, K=TIME_PAD_K, gamma=FREQ_GUIDE_GAMMA)


    # Mel->linear continuous frequency weights
    q_lin = omega_linear_continuous(mel_fb_m2f, omega, gamma_f=0.9)     # [F]
    M_lin = q_lin.view(-1,1) * time_weight.view(1,-1)                   # [F,T]

    # Subtraction in complex STFT domain
    stft_full = st
    stft_src  = M_lin * stft_full
    stft_res  = stft_full - stft_src

    # Residual pruning per pass: zero frames with tiny energy ratio (<=0.5%)
    Pres = (stft_res.abs()**2).clamp_min(EPS)
    e_frame = Pres.sum(dim=0)                    # [T]
    e_tot   = e_frame.sum()
    r_frame = e_frame / (e_tot + EPS)
    low = (r_frame <= PRUNE_FRAME_RATIO_THR)
    if bool(low.any()):
        stft_res[:, low] = 0.0 + 0.0j

    # Reconstruct source and residual
    src_amp = torch.istft(stft_src, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()
    res     = torch.istft(stft_res, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()

    # ER calculation
    e_src = float(np.sum(src_amp**2)); e_res = float(np.sum(res**2))
    er = e_src / (e_src + e_res + 1e-12)

    # Used-frame mask for next pass (use HARD support b(t) to avoid over-suppression)
    b_hard = (cos_t_raw >= TIME_TAU).float()
    r_t_used = (q_lin.view(-1,1) * b_hard.view(1,-1) * P).sum(dim=0) / (P.sum(dim=0) + 1e-8)
    used_mask = (r_t_used >= USED_THRESHOLD).float()

    elapsed = time.time() - t0

    if out_dir:
        png = os.path.join(out_dir, f"debug_pass_{pass_idx+1}.png")
        debug_plot(pass_idx, Sc, cos_t_raw, time_weight, P, M_lin,
                   s, e, core_s_rel, core_e_rel, png,
                   title=(f"Pass {pass_idx+1} | anchor {s*HOP/SR:.2f}-{e*HOP/SR:.2f}s "
                          f"(r={anchor_ratio*100:.2f}%) | ER={er*100:.1f}% | τ={TIME_TAU}, padK={TIME_PAD_K}"))

    info = {
        "er": er,
        "elapsed": elapsed,
        "anchor": (s*HOP/SR, e*HOP/SR),
        "core":   ((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR),
        "quality": float(time_weight.mean().item()),
        "w_bar": w_bar,
        "omega": omega,
        "anchor_ratio": anchor_ratio,
        "stopped": False
    }
    return src_amp, res, er, used_mask, info

# =========================
# Main
# =========================
def main():
    # <-- FIX: declare globals BEFORE any use inside main()
    global TIME_TAU, TIME_PAD_K, OMEGA_SIG_SLOPE, OMEGA_Q

    ap=argparse.ArgumentParser(description="AST-guided Source Separator (Hard Threshold + Padding, Soft Ω)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--passes", type=int, default=MAX_PASSES)
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--tau", type=float, default=TIME_TAU, help="Cosine hard threshold (default 0.6)")
    ap.add_argument("--pad-k", type=int, default=TIME_PAD_K, help="Temporal padding (±K frames), default 3")
    ap.add_argument("--omega-slope", type=float, default=OMEGA_SIG_SLOPE, help="Sigmoid Ω slope (default 10.0)")
    ap.add_argument("--omega-q", type=float, default=OMEGA_Q, help="Sigmoid Ω center quantile (default 0.70)")
    args=ap.parse_args()

    # apply CLI overrides to globals
    TIME_TAU = float(args.tau)
    TIME_PAD_K = max(0, int(args.pad_k))
    OMEGA_SIG_SLOPE = float(args.omega_slope)
    OMEGA_Q = float(args.omega_q)

    os.makedirs(args.output, exist_ok=True)

    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 AST-guided Source Separator (Hard Threshold + Padding, Soft Ω)\n{'='*64}")
    print(f"Input: fixed {WIN_SEC:.3f}s, Anchor: {ANCHOR_SEC:.3f}s, Passes: {args.passes}")
    print(f"Time: hard τ={TIME_TAU}, padding ±K={TIME_PAD_K} frames (~{TIME_PAD_K*HOP/SR*1000:.0f} ms)")
    print(f"Ω: sigmoid with slope={OMEGA_SIG_SLOPE}, center quantile={OMEGA_Q}, min bins={OMEGA_MIN_BINS}")
    print(f"Anchor≥0.5% energy, continuous Ω→linear, residual frame pruning≤0.5%.")

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

    for i in range(max(1, args.passes)):
        print(f"\n▶ Pass {i+1}/{args.passes}")
        result = single_pass(
            cur, extractor, ast_model, mel_fb_m2f,
            used_mask_prev, prev_anchors,
            pass_idx=i, out_dir=None if args.no_debug else args.output
        )

        if result[0] is None:
            info = result[4]
            reason = info.get("reason", "stopped")
            print(f"  ⏹️ Stopped: {reason}")
            break

        src, res, er, used_mask_prev, info = result
        print(f"⏱️ pass{i+1}: {info['elapsed']:.3f}s | anchor {info['anchor'][0]:.2f}-{info['anchor'][1]:.2f}s"
              f" | core {info['core'][0]:.2f}-{info['core'][1]:.2f}s | ER={er*100:.1f}% | anchor_r={info['anchor_ratio']*100:.2f}%")

        if er < MIN_ERATIO:
            print("  ⚠️ Too little energy; stopping.")
            break

        # Peak Normalization
        peak = np.max(np.abs(src))
        if peak > 1e-8:
            gain = NORMALIZE_TARGET_PEAK / peak
            src = np.clip(src * gain, -1.0, 1.0)

        out_src = os.path.join(args.output, f"{i+1:02d}_source.wav")
        torchaudio.save(out_src, torch.from_numpy(src).unsqueeze(0), SR)
        print(f"    ✅ Saved (Normalized): {out_src}")

        cur = res
        prev_anchors.append((info["anchor"][0], info["anchor"][1], info["w_bar"], info["omega"]))
        saved += 1

    # Final residual amplitude clipping
    if RESIDUAL_CLIP_THR > 0:
        print(f"\nApplying residual clipping with threshold: {RESIDUAL_CLIP_THR}")
        cur[np.abs(cur) < RESIDUAL_CLIP_THR] = 0.0

    out_res = os.path.join(args.output, "00_residual.wav")
    torchaudio.save(out_res, torch.from_numpy(cur).unsqueeze(0), SR)

    total_elapsed = time.time() - total_t0
    print(f"\n💾 Residual: {out_res}")
    print(f"⏱️ Total: {total_elapsed:.3f}s")
    print(f"✅ Done. Separated: {saved}\n")

if __name__ == "__main__":
    main()
