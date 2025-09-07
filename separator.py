#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AST-guided Source Separator — Leaky+Saturating Cosine Gate (Full)

What this version does:
- Time selection/weighting is purely cosine-similarity-driven with a leaky + saturating gate:
  * cos <= tau - band_low         -> gate = 0   (hard off, no leak)
  * tau - band_low < cos < tau    -> gate rises 0 -> leak_at_tau (small leak)
  * tau <= cos < tau + band_high  -> gate rises leak_at_tau -> 1 (fast ramp)
  * cos >= tau + band_high        -> gate = 1   (saturated)
  -> time_weight = cos * gate  (so high-similarity regions follow cos exactly)
- No presence gate, no a_raw gate, no sigmoid smoothstep shaping elsewhere.
- Anchor selection with energy validation (>= 0.5% of total).
- Continuous mel->linear frequency weights (not binary).
- Complex-domain subtraction, per-pass residual pruning (<= 0.5% per-frame energy), peak-normalized sources.

Inputs/Outputs:
- Fixed 4.096 s mono input (resampled to 16 kHz if needed), multiple passes (default 3).
- Saves per-pass extracted sources and final residual.

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
SR                = 16000
WIN_SEC           = 4.096
ANCHOR_SEC        = 0.512
L_FIXED           = int(round(WIN_SEC * SR))

# === Final Output Processing ===
NORMALIZE_TARGET_PEAK = 0.95
RESIDUAL_CLIP_THR     = 0.0005

# === Cosine→Leaky + Saturating time gate ===
TIME_TAU          = 0.60   # center threshold
TIME_BAND_LOW     = 0.05   # (tau - band_low, tau): 0 -> leak_at_tau
TIME_BAND_HIGH    = 0.04   # (tau, tau + band_high): leak_at_tau -> 1; >= tau+band_high => 1
TIME_LEAK_AT_TAU  = 0.10   # gate value exactly at tau (small leak)
TIME_USE_COS_WEIGHTED = True  # True => time_weight = cos * gate (recommended)

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

# Ω & template
OMEGA_Q           = 0.70
OMEGA_DIL         = 2
OMEGA_MIN_BINS    = 10

# Suppression / usage
USED_THRESHOLD        = 0.65
USED_DILATE_MS        = 80
ANCHOR_SUPPRESS_MS    = 200
ANCHOR_SUPPRESS_BASE  = 0.6

# Residual pruning per pass (frame-energy ratio)
PRUNE_FRAME_RATIO_THR = 0.005  # 0.5%

# Loop
MAX_PASSES    = 3
MIN_ERATIO    = 0.01  # stop if ER < 1%

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
        # suppress this candidate region and try next
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
def omega_support(Ablk: torch.Tensor) -> torch.Tensor:
    med = Ablk.median(dim=1).values
    th  = torch.quantile(med, OMEGA_Q)
    mask = (med >= th).float()
    for _ in range(OMEGA_DIL):
        mask = torch.maximum(mask, torch.roll(mask, 1))
        mask = torch.maximum(mask, torch.roll(mask, -1))
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
    Continuous mapping from mel support (omega) to linear freq weights q in [0,1],
    with light smoothing along frequency.
    mel_fb_m2f: [M,F], omega: [M]
    """
    M, Freq = mel_fb_m2f.shape
    om = omega / (omega.sum() + EPS)
    q = (mel_fb_m2f.T @ om).view(-1)  # [F]
    q = (q / (q.max() + EPS)).clamp(0,1)
    if gamma_f != 1.0:
        q = q.pow(gamma_f)
    # light smoothing (3-bin)
    q = F.avg_pool1d(q.view(1,1,-1), kernel_size=3, stride=1, padding=1).view(-1).clamp(0,1)
    return q

# =========================
# Leaky + Saturating gate
# =========================
def leaky_saturating_gate(cos_t: torch.Tensor, tau: float,
                          band_low: float, band_high: float,
                          leak_at_tau: float) -> torch.Tensor:
    """
    Band-limited, leaky-ReLU-like gate with top saturation.

    cos <= tau - band_low          -> 0
    tau - band_low < cos < tau     -> 0 -> leak_at_tau  (linear)
    tau <= cos < tau + band_high   -> leak_at_tau -> 1  (linear)
    cos >= tau + band_high         -> 1  (saturate)

    Output in [0,1], continuous/connected at boundaries.
    """
    x   = cos_t.clamp(0, 1)
    lo  = tau - band_low
    mid = tau
    hi  = tau + band_high

    below    = (x <= lo).float()
    mid_low  = ((x > lo) & (x < mid)).float()
    mid_high = ((x >= mid) & (x < hi)).float()
    above    = (x >= hi).float()

    # lo..mid : 0 -> leak
    u_low  = ((x - lo) / (mid - lo + 1e-8)).clamp(0, 1)
    g_low  = u_low * leak_at_tau

    # mid..hi : leak -> 1
    u_high = ((x - mid) / (hi - mid + 1e-8)).clamp(0, 1)
    g_high = leak_at_tau + (1.0 - leak_at_tau) * u_high

    gate = below * 0.0 + mid_low * g_low + mid_high * g_high + above * 1.0
    return gate.clamp(0, 1)

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
    ax[0,1].plot(t, to_np(time_weight), label='time weight (leaky+saturating)', lw=1.0, alpha=0.85)
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
# Single Pass (Leaky+Saturating Cosine Gate)
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

    # Suppress used frames
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

    # Ω & template
    omega = omega_support(Ablk)
    w_bar = template_from_anchor_block(Ablk, omega)

    # Cosine similarity over Ω
    cos_t_raw = cos_similarity_over_omega_no_gate(Xmel, w_bar, omega)   # [T] in [0,1]

    # Leaky + Saturating time gate around tau
    gate = leaky_saturating_gate(
        cos_t_raw,
        tau=TIME_TAU,
        band_low=TIME_BAND_LOW,
        band_high=TIME_BAND_HIGH,
        leak_at_tau=TIME_LEAK_AT_TAU
    )  # [T] in [0,1]

    # Final time weight: high-similarity region (cos >= tau+band_high) => gate=1 => equals cos
    time_weight = cos_t_raw * gate if TIME_USE_COS_WEIGHTED else gate   # [T]

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

    # Used-frame mask for next pass
    P_lin = P
    r_t_used = (M_lin * P_lin).sum(dim=0) / (P_lin.sum(dim=0) + 1e-8)
    used_mask = (r_t_used >= USED_THRESHOLD).float()

    elapsed = time.time() - t0

    if out_dir:
        png = os.path.join(out_dir, f"debug_pass_{pass_idx+1}.png")
        debug_plot(pass_idx, Sc, cos_t_raw, time_weight, P, M_lin,
                   s, e, core_s_rel, core_e_rel, png,
                   title=(f"Pass {pass_idx+1} | anchor {s*HOP/SR:.2f}-{e*HOP/SR:.2f}s "
                          f"(r={anchor_ratio*100:.2f}%) | ER={er*100:.1f}%"))

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
    ap=argparse.ArgumentParser(description="AST-guided Source Separator (Leaky+Saturating Cosine Gate)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--passes", type=int, default=MAX_PASSES)
    ap.add_argument("--no-debug", action="store_true")
    args=ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 AST-guided Source Separator (Leaky+Saturating Cosine Gate)\n{'='*64}")
    print(f"Input: fixed {WIN_SEC:.3f}s, Anchor: {ANCHOR_SEC:.3f}s, Passes: {args.passes}")
    print(f"Gate: tau={TIME_TAU}, band_low={TIME_BAND_LOW}, band_high={TIME_BAND_HIGH}, leak@tau={TIME_LEAK_AT_TAU}, cos-weighted={TIME_USE_COS_WEIGHTED}")
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
