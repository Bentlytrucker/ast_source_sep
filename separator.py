#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cos-only Mask + Presence-Gated, Soft-Ω Demixer (AST-guided)
- C(t) = cosΩ, but forced to 0 if a_raw == 0 (hard rule)
- Only frames with C(t) >= 0.6 are considered (hard time gate)
- Mask uses ONLY cosΩ (no amplitude bias); extraction uses amplitude envelope
- Soft Ω (halo+floor) to avoid single-line artifacts
- Anchor: attn^α * purity^β ; inside-anchor top-50% core only (others zero-padded)
- Per-pass residual update; used-frames + prev-anchor suppression

Run:
  python separator.py --input mix.wav --output out --passes 3
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
SR           = 16000
WIN_SEC      = 4.096                 # fixed window (1.024*4)
ANCHOR_SEC   = 0.512                 # anchor window
L_FIXED      = int(round(WIN_SEC * SR))

# STFT (10ms hop)
N_FFT, HOP, WINLEN = 400, 160, 400
WINDOW = torch.hann_window(WINLEN)
EPS = 1e-10

# Mel
N_MELS = 128

# Anchor score
SMOOTH_T  = 19
ALPHA_ATT = 0.80
BETA_PUR  = 1.20
W_E       = 0.30
TOP_PCT_ANCHOR            = 0.70   # global
TOP_PCT_CORE_IN_ANCHOR    = 0.50   # inside anchor top-50% contig

# Ω & template
OMEGA_Q        = 0.70
OMEGA_DIL      = 2
OMEGA_MIN_BINS = 10

# Presence gate
PRES_Q         = 0.20
PRES_SMOOTH_T  = 9

# Cos-only mask shaping
HALO_BINS         = 3
OMEGA_CONTRAST    = 0.85
OMEGA_FLOOR       = 0.08
GAMMA_COS_IN_MASK = 1.35
FIRST_PASS_MASK_EXP = 0.65
WIENER_BETA       = 1.5
WIENER_ITERS      = 2

# Hard constraints (요청 반영)
ARAW_ZERO_THR  = 1e-8      # a_raw ≤ 이 값이면 0으로 간주
C_HARD_TAU     = 0.60      # C(t) ≥ 0.6만 취급

# Extraction amplitude (uses a_raw)
AMP_POWER      = 1.00
AMP_GAIN_MIN   = 0.85
AMP_GAIN_MAX   = 1.35
ENV_FADE_FR    = 4

# Oversubtraction (time-domain LS)
DO_OVERSUB_TIME  = True
CONF_TAU         = C_HARD_TAU   # 세그먼트 기준도 0.6로 통일
LS_ALPHA_CLAMP   = (0.98, 1.22)
LS_ALPHA_KAPPA   = 0.35
FADE_MS          = 50
MIN_SEG_MS       = 120
MERGE_GAP_MS     = 60

# Used-frames & prev-anchor suppression
USED_THRESHOLD       = 0.65
USED_DILATE_MS       = 80
ANCHOR_SUPPRESS_MS   = 200
ANCHOR_SUPPRESS_GAIN = 0.6

# Loop
MAX_PASSES    = 3
MIN_ERATIO    = 0.01

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
    P   = (mag * mag).clamp_min(EPS)               # [F,T]
    phase = torch.angle(st)

    # mel_fb_m2f: [M,F]
    if mel_fb_m2f.shape[0] != N_MELS:
        mel_fb_m2f = mel_fb_m2f.T.contiguous()
    assert mel_fb_m2f.shape[0] == N_MELS and mel_fb_m2f.shape[1] == P.shape[0]
    mel_fb_m2f = mel_fb_m2f.to(P.dtype).to(P.device)
    mel_pow = (mel_fb_m2f @ P).clamp_min(EPS)      # [M,T]
    return st, mag, P, phase, mel_pow

def bg_estimate(P: torch.Tensor, k: int = 101) -> torch.Tensor:
    return F.avg_pool1d(P.unsqueeze(0), kernel_size=k, stride=1, padding=k//2).squeeze(0).clamp_min(EPS)

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
    A = attns[-1]                               # [1, heads, seq, seq]
    cls = A[:, :, 0, 1:].mean(dim=1).squeeze(0) # [patches]
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

def pick_anchor_region(score: torch.Tensor, La: int) -> Tuple[int,int]:
    T = score.numel()
    thr = torch.quantile(score, TOP_PCT_ANCHOR)
    mask = (score >= thr).float().cpu().numpy().astype(np.int8)
    segs=[]
    i=0
    while i<T:
        if mask[i]==0: i+=1; continue
        j=i
        while j<T and mask[j]==1: j+=1
        segs.append((i,j)); i=j
    if not segs:
        ker = torch.ones(La, device=score.device)/La
        conv = F.conv1d(score.view(1,1,-1), ker.view(1,1,-1), padding=La//2).view(-1)
        c = int(torch.argmax(conv).item())
        s = max(0, min(T-La, c-La//2)); return s, s+La
    best_s, best_e, best_mean = 0, La, -1
    for a,b in segs:
        length=b-a
        if length>=La:
            local = score[a:b]
            ker = torch.ones(La, device=score.device)/La
            conv = F.conv1d(local.view(1,1,-1), ker.view(1,1,-1)).view(-1)  # valid
            k = int(torch.argmax(conv).item())
            s=a+k; e=s+La; m=float(conv[k].item())
        else:
            c=(a+b)//2; s=max(0, min(T-La, c-La//2)); e=s+La; m=float(score[s:e].mean().item())
        if m>best_mean: best_s, best_e, best_mean = s,e,m
    return best_s, best_e

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
    return mask  # [M]{0,1}

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
    e_omega = (Xmel * om).sum(dim=0)                 # [T]
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
    return cos_raw * g_pres  # presence==0 -> cos==0

# =========================
# Soft Ω (halo+floor) & mask
# =========================
def soften_omega_linear(omega_lin: torch.Tensor, halo_bins: int = HALO_BINS,
                        contrast: float = OMEGA_CONTRAST, floor_out: float = OMEGA_FLOOR):
    k = 2*halo_bins + 1
    x = omega_lin.view(1,1,-1).float()
    soft = F.avg_pool1d(x, kernel_size=k, stride=1, padding=halo_bins).view(-1)
    soft = floor_out + contrast * (soft - soft.min()) / (soft.max() - soft.min() + 1e-8) * (1.0 - floor_out)
    return soft.clamp(0,1)

def make_mask_cos_only(P: torch.Tensor, omega_lin: torch.Tensor, cos_t: torch.Tensor,
                       C_gate: torch.Tensor, pass_idx: int) -> torch.Tensor:
    # cos-only + hard C(t) gate
    cos_eff = cos_t.clamp(0,1).pow(GAMMA_COS_IN_MASK)
    cos_eff = cos_eff * C_gate  # <== only C>=tau frames survive (others 0)

    omega_soft = soften_omega_linear(omega_lin)
    M0 = (omega_soft.view(-1,1) * cos_eff.view(1,-1)).clamp(0,1)

    # small blur
    M0 = F.avg_pool2d(M0.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze()

    if pass_idx == 0 and FIRST_PASS_MASK_EXP != 1.0:
        M0 = M0.pow(FIRST_PASS_MASK_EXP).clamp(0,1)

    # Wiener
    m = M0.clone()
    for _ in range(WIENER_ITERS):
        sp  = (m ** WIENER_BETA) * P
        npw = ((1 - m) ** WIENER_BETA) * P
        m   = (sp / (sp + npw + EPS)).clamp(0.0, 1.0)
    # keep hard time gate after Wiener as well (idempotent)
    m = m * C_gate.view(1,-1)
    return m.clamp(0,1)

# =========================
# Reconstruction + amplitude gain + oversub
# =========================
def reconstruct_from_mask(mag: torch.Tensor, phase: torch.Tensor, M_lin: torch.Tensor):
    stft_full = torch.polar(mag, phase)
    stft_src  = (M_lin.clamp(0,1))**0.5 * stft_full
    stft_res  = stft_full - stft_src
    src = torch.istft(stft_src, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                      window=WINDOW, center=True, length=L_FIXED)
    res = torch.istft(stft_res, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                      window=WINDOW, center=True, length=L_FIXED)
    return src.detach().cpu().numpy(), res.detach().cpu().numpy()

def build_amp_envelope(a_raw: torch.Tensor, g_pres: torch.Tensor) -> np.ndarray:
    # normalize on presence frames
    a = a_raw.clone()
    pres_idx = (g_pres > 0.5)
    if pres_idx.any():
        med = a[pres_idx].median().item()
        if med > 1e-8:
            a = a / med
    a = a.pow(AMP_POWER)
    a = torch.clamp(a, AMP_GAIN_MIN, AMP_GAIN_MAX)
    # zero out where no presence
    a = a * (g_pres > 0.5).float()
    # mild smoothing
    a = smooth1d(a, 5)
    # upsample
    env = F.interpolate(a.view(1,1,-1), size=L_FIXED, mode="linear", align_corners=False).view(-1)
    # optional frame-edge fades
    if ENV_FADE_FR > 0:
        win = torch.ones_like(a)
        up  = 0.5 - 0.5*torch.cos(torch.linspace(0, np.pi, ENV_FADE_FR, device=a.device))
        win[:ENV_FADE_FR] = up; win[-ENV_FADE_FR:] = torch.flip(up, [0])
        win = F.interpolate(win.view(1,1,-1), size=L_FIXED, mode="linear", align_corners=False).view(-1)
        env = env * win
    return env.detach().cpu().numpy()

def segments_from_conf(C: torch.Tensor, tau: float, T: int) -> List[Tuple[int,int]]:
    mask = (C >= tau).float().cpu().numpy().astype(np.int32)
    segs=[]
    i=0
    while i<T:
        if mask[i]==0: i+=1; continue
        j=i
        while j<T and mask[j]==1: j+=1
        segs.append((i,j)); i=j
    min_len = int(round((MIN_SEG_MS/1000.0)*SR/HOP))
    merge_gap = int(round((MERGE_GAP_MS/1000.0)*SR/HOP))
    segs = [s for s in segs if s[1]-s[0] >= min_len]
    if not segs: return []
    merged=[segs[0]]
    for s,e in segs[1:]:
        ps,pe = merged[-1]
        if s - pe <= merge_gap: merged[-1]=(ps,e)
        else: merged.append((s,e))
    return merged

def oversub_time(mix: np.ndarray, src: np.ndarray, segs: List[Tuple[int,int]], C: torch.Tensor) -> np.ndarray:
    if not segs: return mix
    out = mix.copy()
    fade = int(round((FADE_MS/1000.0)*SR))
    for s,e in segs:
        s_s = max(0, s*HOP - fade); e_s = min(len(mix), e*HOP + fade)
        if e_s - s_s < 8: continue
        x = torch.from_numpy(mix[s_s:e_s]); y = torch.from_numpy(src[s_s:e_s])
        den = float((y*y).sum().item()) + 1e-8
        alpha = float((x*y).sum().item()/den)
        alpha = float(np.clip(alpha, LS_ALPHA_CLAMP[0], LS_ALPHA_CLAMP[1]))
        c_avg = float(C[s:e].mean().item())
        alpha *= (1.0 + LS_ALPHA_KAPPA * max(0.0, c_avg - CONF_TAU))
        if fade>0:
            w = np.ones(e_s - s_s, dtype=np.float32)
            up = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, fade, dtype=np.float32))
            w[:fade]=up; w[-fade:]=up[::-1]
        else:
            w = np.ones(e_s - s_s, dtype=np.float32)
        out[s_s:e_s] = out[s_s:e_s] - (alpha * y.numpy() * w)
    return out

# =========================
# Debug
# =========================
def debug_plot(pass_idx:int, score:torch.Tensor, a_raw:torch.Tensor,
               cos_t:torch.Tensor, C_t:torch.Tensor,
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

    # 2) Scalars: a_raw(norm) & cosΩ & C(t)
    ar = to_np(a_raw); ar_n = (ar - ar.min())/(ar.max()-ar.min()+1e-8)
    ax[0,1].plot(t, ar_n, label='a_raw (norm)', lw=1.0)
    ax[0,1].plot(t, to_np(cos_t), label='cosΩ', lw=1.0)
    ax[0,1].plot(t, to_np(C_t), label=f'C(t) (>= {C_HARD_TAU:.2f} only)', lw=1.0, alpha=0.85)
    ax[0,1].axhline(C_HARD_TAU, color='r', ls='--', alpha=0.7)
    ax[0,1].legend(); ax[0,1].set_ylim([0,1.05]); ax[0,1].set_title("Scalars")

    # 3) Mask stats (frame mean)
    m_mean = to_np(M_lin.mean(dim=0))
    ax[0,2].plot(t, m_mean, color='purple'); ax[0,2].set_ylim([0,1.05])
    ax[0,2].set_title("Mask frame-mean")

    # 4) Spec(dB)
    spec_db = 20 * torch.log10(P.sqrt() + 1e-10)
    im0 = ax[1,0].imshow(to_np(spec_db), aspect='auto', origin='lower', cmap='viridis',
                         extent=[0, t[-1] if len(t)>0 else 0, 0, fbins])
    ax[1,0].set_title("Spec(dB)"); plt.colorbar(im0, ax=ax[1,0])

    # 5) Mask(linear)
    im1 = ax[1,1].imshow(to_np(M_lin), aspect='auto', origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                         extent=[0, t[-1] if len(t)>0 else 0, 0, fbins])
    ax[1,1].set_title("Mask(linear)"); plt.colorbar(im1, ax=ax[1,1])

    # 6) Masked Spec(dB)
    masked_db = 20 * torch.log10((M_lin * P).sqrt() + 1e-10)
    im2 = ax[1,2].imshow(to_np(masked_db), aspect='auto', origin='lower', cmap='viridis',
                         extent=[0, t[-1] if len(t)>0 else 0, 0, fbins])
    ax[1,2].set_title("Masked Spec(dB)"); plt.colorbar(im2, ax=ax[1,2])

    plt.suptitle(title); plt.tight_layout(); plt.savefig(out_png, dpi=120, bbox_inches='tight'); plt.close()
    print(f"  📊 Debug saved: {out_png}")

# =========================
# Single pass
# =========================
def single_pass(audio: np.ndarray, extractor, ast_model,
                mel_fb_m2f: torch.Tensor,
                used_mask_prev: Optional[torch.Tensor],
                prev_anchors: List[Tuple[float,float]],
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

    # Suppress prev anchor centers
    for (sa, ea) in prev_anchors:
        ca = int(((sa+ea)/2) * SR / HOP)
        ca = max(0, min(T-1, ca))
        sigma = int(round((ANCHOR_SUPPRESS_MS/1000.0)*SR/HOP))
        idx = torch.arange(T, device=Sc.device) - ca
        Sc = Sc * (1 - ANCHOR_SUPPRESS_GAIN * torch.exp(-(idx**2)/(2*(sigma**2)+1e-8)))
        core_s = max(0, ca - La//2); core_e = min(T, ca + La//2)
        Sc[core_s:core_e] *= 0.2

    # Global anchor [s:e] (len=La)
    s, e = pick_anchor_region(Sc, La)

    # Inside-anchor: pick contiguous top-50% core; zero-pad others
    local = Sc[s:e]
    thr_in = torch.quantile(local, TOP_PCT_CORE_IN_ANCHOR)
    mk = (local >= thr_in).float().cpu().numpy().astype(np.int8)
    best = (0,0,0); i=0
    while i<mk.size:
        if mk[i]==0: i+=1; continue
        j=i
        while j<mk.size and mk[j]==1: j+=1
        if j-i>best[2]: best=(i,j,j-i)
        i=j
    core_s_rel, core_e_rel = best[0], best[1]
    Ablk = Xmel[:, s:e].clone()
    if core_s_rel>0:  Ablk[:, :core_s_rel] = 0
    if core_e_rel<La: Ablk[:, core_e_rel:] = 0

    # Ω & template
    omega = omega_support(Ablk)                       # [M]
    w_bar = template_from_anchor_block(Ablk, omega)   # [M]

    # Scalars
    a_raw = amplitude_raw(Xmel, w_bar, omega)         # [T]
    g_pres = presence_from_energy(Xmel, omega)        # [T]
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)  # [T], already 0 when presence==0

    # ===== Hard rule: if a_raw == 0 -> C(t) = 0  =====
    a_nz = (a_raw > ARAW_ZERO_THR).float()
    C_t = cos_t_raw * a_nz
    # ===== Hard gate: only frames with C(t) >= 0.6 considered =====
    C_gate = (C_t >= C_HARD_TAU).float()

    # Map Ω(mel)->Ω(linear) for freq weighting
    omega_lin = ((mel_fb_m2f.T @ omega).clamp_min(0.0) > 1e-12).float()  # [F]

    # Mask (cos-only with hard C-gate) → reconstruction
    M_lin = make_mask_cos_only(P, omega_lin, C_t, C_gate, pass_idx)
    src_masked, _res_dummy = reconstruct_from_mask(mag, phase, M_lin)

    # Frame-wise amplitude envelope (only where presence==1)
    env = build_amp_envelope(a_raw, g_pres)           # [L_FIXED]
    src_amp = src_masked * env                        # final source (time domain)

    # Residual
    res = audio - src_amp
    if DO_OVERSUB_TIME:
        segs = segments_from_conf(C_t, CONF_TAU, T)
        res = oversub_time(res, src_amp, segs, C_t)

    # ER (after amp & oversub)
    e_src = float(np.sum(src_amp**2)); e_res = float(np.sum(res**2))
    er = e_src / (e_src + e_res + 1e-12)

    # Used-frame mask for next pass
    r_t = (M_lin * P).sum(dim=0) / (P.sum(dim=0) + 1e-8)
    used_energy = (r_t >= USED_THRESHOLD).float()
    used_mask = torch.maximum(used_energy, (C_t >= max(0.9, C_HARD_TAU)).float())

    elapsed = time.time() - t0

    # Debug
    if out_dir:
        png = os.path.join(out_dir, f"debug_pass_{pass_idx+1}.png")
        debug_plot(pass_idx, Sc, a_raw, C_t, C_gate, P, M_lin,
                   s, e, core_s_rel, core_e_rel, png,
                   title=f"Pass {pass_idx+1} | anchor {s*HOP/SR:.2f}-{e*HOP/SR:.2f}s | ER={er*100:.1f}%")

    info = {
        "er": er,
        "elapsed": elapsed,
        "anchor": (s*HOP/SR, e*HOP/SR),
        "core":   ((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR),
    }
    return src_amp, res, er, used_mask, info

# =========================
# Main
# =========================
def main():
    ap=argparse.ArgumentParser(description="Cos-only Mask, Presence-Gated Soft-Ω Demixer (C(t) hard rules)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--passes", type=int, default=MAX_PASSES)
    ap.add_argument("--no-debug", action="store_true")
    args=ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 Cos-only Mask + Presence-Gated Soft-Ω Demixer\n{'='*64}")
    print(f"Input: fixed {WIN_SEC:.3f}s, Anchor: {ANCHOR_SEC:.3f}s, Passes: {args.passes}")

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
    prev_anchors: List[Tuple[float,float]] = []
    total_t0 = time.time()
    saved = 0

    for i in range(max(1, args.passes)):
        print(f"\n▶ Pass {i+1}/{args.passes}")
        src, res, er, used_mask_prev, info = single_pass(
            cur, extractor, ast_model, mel_fb_m2f,
            used_mask_prev, prev_anchors,
            pass_idx=i, out_dir=None if args.no_debug else args.output
        )
        print(f"⏱️ pass{i+1}: {info['elapsed']:.3f}s | anchor {info['anchor'][0]:.2f}-{info['anchor'][1]:.2f}s"
              f" | core {info['core'][0]:.2f}-{info['core'][1]:.2f}s | ER={er*100:.1f}%")

        if er < MIN_ERATIO:
            print("   ⚠️ Too little energy; stopping.")
            break

        out_src = os.path.join(args.output, f"{i+1:02d}_source.wav")
        torchaudio.save(out_src, torch.from_numpy(src).unsqueeze(0), SR)
        print(f"   ✅ Saved: {out_src}")

        cur = res
        prev_anchors.append(info["anchor"])
        saved += 1

    out_res = os.path.join(args.output, "00_residual.wav")
    torchaudio.save(out_res, torch.from_numpy(cur).unsqueeze(0), SR)

    total_elapsed = time.time() - total_t0
    print(f"\n💾 Residual: {out_res}")
    print(f"⏱️ Total: {total_elapsed:.3f}s")
    print(f"✅ Done. Separated: {saved}\n")

if __name__ == "__main__":
    main()
