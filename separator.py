#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anchor-Refined Ω-NNLS AST Demixer (CPU, <~5-8s typical)
- bg_estimate: avg_pool1d (fix conv1d channel mismatch)
- Anchor region: pick by attn^α * purity^β, then keep only longest
  contiguous core in top-50% (inside anchor) and zero-pad others
- Template: anchor mel-power block within Ω; slight in-Ω freq smooth (3-tap)
- Time confidence: C(t) = cos^p * a_gate^q  --> multiply into mask (reduces horizontal lines & over-sep)
- Ω-weighted NNLS back-projection (mel->linear)
- Ratio mask + mild blur + Wiener; stronger first-pass exponent
- Per-pass residual update + used-frame suppression + prev-anchor suppression
- Rich debug per pass

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
torch.set_num_threads(4)  # predictable CPU latency

# =========================
# Config
# =========================
SR = 16000
WIN_SEC   = 4.096                 # fixed input window (1.024*4)
L_FIXED   = int(round(WIN_SEC * SR))
ANCHOR_SEC = 0.512                # anchor length

# STFT (10ms hop)
N_FFT, HOP, WINLEN = 400, 160, 400
WINDOW = torch.hann_window(WINLEN)
EPS = 1e-10

# Mel
N_MELS = 128

# Anchor score
SMOOTH_T  = 19
ALPHA_ATT = 0.80                  # attn^alpha
BETA_PUR  = 1.20                  # purity^beta
W_E       = 0.30                  # purity = W_E*energy + (1-W_E)*(1-entropy)
TOP_PCT_ANCHOR = 0.70             # global anchor selection (pick window len=0.512)
TOP_PCT_CORE_IN_ANCHOR = 0.50     # inside anchor: keep longest contiguous top-50% frames

# Ω & template
OMEGA_Q        = 0.85
OMEGA_DIL      = 3
OMEGA_MIN_BINS = 12                # force multi-bin support to avoid single-line

# Time confidence (applies to mask)
CONF_TAU        = 0.9
CONF_EXP_AMP    = 0.85             # exponent for a_gate  (0.8~1.1)
CONF_EXP_COS    = 0.9             # exponent for cosΩ
GAMMA_C_IN_MASK = 0.85             # M *= C^gamma

# NNLS (Ω-weighted mel->linear)
NNLS_STEPS   = 8
NNLS_LAMBDA  = 6.0
NNLS_LR      = 0.10

# Mask / Wiener
MASK_GAMMA_OUT  = 0.9
WIENER_BETA     = 1.5
WIENER_ITERS    = 2

FIRST_PASS_EXP  = 0.8            # raise M^(FIRST_PASS_EXP) to extract more energy in pass1

# Used-frames & prev anchor suppression
USED_DILATE_MS       = 80         # odd kernel enforced
USED_THRESHOLD       = 0.65       # frame extraction ratio
ANCHOR_SUPPRESS_MS   = 200
ANCHOR_SUPPRESS_GAIN = 0.5

# Oversubtraction (time-domain LS with fades; optional – off by default)
DO_OVERSUB_TIME  = True
LS_ALPHA_CLAMP   = (0.98, 1.20)
LS_ALPHA_KAPPA   = 0.35           # extra gain above CONF_TAU
FADE_MS          = 50
MIN_SEG_MS       = 120
MERGE_GAP_MS     = 60

# Loop
MAX_PASSES = 3
MIN_ERATIO = 0.01

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
    # 🔧 conv1d → avg_pool1d 로 채널 불일치 문제 해결
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
    # 전역 앵커: 상위 TOP_PCT 구간 내에서 길이 La의 평균 스코어가 최대인 위치
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
        s = max(0, min(T-La, c-La//2))
        return s, s+La
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
def omega_support(anchor_mel: torch.Tensor) -> torch.Tensor:
    med = anchor_mel.median(dim=1).values        # [M]
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
    # Ablk: [M, La] (outside non-core frames already zero-padded)
    om = omega.view(-1,1)
    w  = (Ablk * om).mean(dim=1) * omega          # Ω 내부 평균
    w  = w / (w.sum() + EPS)
    # 🔒 단일빈 억제: Ω 내부에서만 아주 약한 3-tap 스무딩
    w_sm = F.avg_pool1d(w.view(1,1,-1), kernel_size=3, stride=1, padding=1).view(-1)
    w   = (w_sm * omega)
    w   = w / (w.sum() + EPS)
    return w  # [M]

# =========================
# Amplitude / Confidence
# =========================
def amplitude_raw_and_gate(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor):
    om = omega.view(-1,1)
    Xo = Xmel * om
    denom = (w_bar*w_bar).sum() + EPS
    a_raw = (w_bar.view(1,-1) @ Xo).view(-1) / denom    # absolute amplitude
    a_n = norm01(smooth1d(a_raw.clamp_min(0.0), SMOOTH_T))
    return a_raw.clamp_min(0.0), a_n

def cos_similarity_over_omega(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor):
    om = omega.view(-1,1)
    Xo = Xmel * om
    wn = (w_bar * omega); wn = wn / (wn.norm(p=2) + 1e-8)
    Xn = Xo / (Xo.norm(p=2, dim=0, keepdim=True) + 1e-8)
    cos = (wn.view(-1,1) * Xn).sum(dim=0).clamp(0,1)
    return cos  # [T]

def build_confidence(a_gate: torch.Tensor, cos_t: torch.Tensor) -> torch.Tensor:
    C = (a_gate.clamp(0,1)**CONF_EXP_AMP) * (cos_t.clamp(0,1)**CONF_EXP_COS)
    return norm01(C)

# =========================
# Ω-weighted NNLS back-proj
# =========================
def nnls_weighted_backproject(S_mel: torch.Tensor, mel_fb_m2f: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    M, Fbins = mel_fb_m2f.shape
    Mb, T = S_mel.shape
    assert Mb == M
    mel_pinv = torch.linalg.pinv(mel_fb_m2f)      # [F,M]
    S_lin = (mel_pinv @ S_mel).clamp_min(0.0)     # [F,T]

    omega = omega.view(-1,1)
    W = 1.0 + NNLS_LAMBDA * omega
    WT2 = (W*W)

    Mel  = mel_fb_m2f
    MelT = mel_fb_m2f.T

    for _ in range(NNLS_STEPS):
        R = (Mel @ S_lin) - S_mel
        G = MelT @ (WT2 * R)
        S_lin = (S_lin - NNLS_LR * G).clamp_min(0.0)
    return S_lin.clamp_min(0.0)

# =========================
# Mask & Reconstruction
# =========================
def make_mask(S_lin: torch.Tensor, P: torch.Tensor, omega_lin: torch.Tensor, C: torch.Tensor):
    # 기본 비율 마스크
    M0 = (S_lin / (P + EPS)).clamp(0.0, 1.0)
    # 부드러운 3x3
    M0 = F.avg_pool2d(M0.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze()
    # 시간 신뢰도 직접 반영
    M0 = M0 * (C.clamp(0,1).pow(GAMMA_C_IN_MASK).view(1,-1))
    # Wiener
    m = M0.clone()
    for _ in range(WIENER_ITERS):
        sp  = (m ** WIENER_BETA) * P
        npw = ((1 - m) ** WIENER_BETA) * P
        m   = (sp / (sp + npw + EPS)).clamp(0.0, 1.0)
    return m.clamp(0.0, 1.0)

def reconstruct(mag: torch.Tensor, phase: torch.Tensor, M_lin: torch.Tensor):
    stft_full = torch.polar(mag, phase)
    stft_src  = (M_lin.clamp(0,1))**0.5 * stft_full
    stft_res  = stft_full - stft_src
    src = torch.istft(stft_src, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                      window=WINDOW, center=True, length=L_FIXED)
    res = torch.istft(stft_res, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                      window=WINDOW, center=True, length=L_FIXED)
    e_src = float((src**2).sum()); e_tot = e_src + float((res**2).sum())
    er = (e_src / (e_tot + EPS)) if e_tot>0 else 0.0
    return src.detach().cpu().numpy(), res.detach().cpu().numpy(), er

# =========================
# Confidence segments & oversub
# =========================
def segments_from_conf(C: torch.Tensor, tau: float, T: int) -> List[Tuple[int,int]]:
    mask = (C >= tau).float().cpu().numpy().astype(np.int32)
    segs=[]
    i=0
    while i<T:
        if mask[i]==0: i+=1; continue
        j=i
        while j<T and mask[j]==1: j+=1
        segs.append((i,j)); i=j
    # 최소 길이/merge
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
        # clamp + confidence gain
        alpha = float(np.clip(alpha, LS_ALPHA_CLAMP[0], LS_ALPHA_CLAMP[1]))
        c_avg = float(C[s:e].mean().item())
        alpha *= (1.0 + LS_ALPHA_KAPPA * max(0.0, c_avg - CONF_TAU))
        # fades
        if fade>0:
            w = np.ones(e_s - s_s, dtype=np.float32)
            up = 0.5 - 0.5*np.cos(np.linspace(0, np.pi, fade, dtype=np.float32))
            w[:fade]=up; w[-fade:]=up[::-1]
        else:
            w = np.ones(e_s - s_s, dtype=np.float32)
        out[s_s:e_s] = out[s_s:e_s] - (alpha * y.numpy() * w)
    return out

# =========================
# Debug plot
# =========================
def debug_plot(pass_idx:int, score:torch.Tensor, a_raw:torch.Tensor, a_gate:torch.Tensor,
               cos_t:torch.Tensor, C:torch.Tensor, P:torch.Tensor, M_lin:torch.Tensor,
               s:int, e:int, core_s_rel:int, core_e_rel:int, out_png:str, title:str):
    fbins, T = P.shape
    t = np.arange(T) * HOP / SR
    fig, ax = plt.subplots(2,3, figsize=(15,8))

    # 1) Anchor score + anchor + anchor-core
    ax[0,0].plot(t, to_np(score), lw=1.2)
    ax[0,0].axvspan(s*HOP/SR, e*HOP/SR, color='orange', alpha=0.20, label='anchor')
    cs = s + core_s_rel; ce = s + core_e_rel
    ax[0,0].axvspan(cs*HOP/SR, ce*HOP/SR, color='red', alpha=0.20, label='anchor-core')
    ax[0,0].legend(); ax[0,0].set_title("Anchor score"); ax[0,0].set_ylim([0,1.05])

    # 2) a_raw (norm) / a_gate / cosΩ
    ar = to_np(a_raw); ag = to_np(a_gate); csim = to_np(cos_t)
    ar_n = (ar - ar.min()) / (ar.max()-ar.min()+1e-8)
    ax[0,1].plot(t, ar_n, label='a_raw (norm)', lw=1.0)
    ax[0,1].plot(t, ag,    label='a_gate', lw=1.0)
    ax[0,1].plot(t, csim,  label='cosΩ', lw=1.0, alpha=0.8)
    ax[0,1].legend(); ax[0,1].set_ylim([0,1.05]); ax[0,1].set_title("Scalars")

    # 3) Confidence C(t)
    ax[0,2].plot(t, to_np(C), color='purple'); ax[0,2].axhline(CONF_TAU, color='r', ls='--')
    ax[0,2].set_ylim([0,1.05]); ax[0,2].set_title("Confidence C(t)")

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

    # Suppress used frames (length-invariant)
    if used_mask_prev is not None:
        um = align_len_1d(used_mask_prev, T, device=Sc.device, mode="linear")
        k = int(round((USED_DILATE_MS/1000.0)*SR/HOP))
        k = ensure_odd(max(1, k))
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

    # Global anchor [s:e] (len=La)
    s, e = pick_anchor_region(Sc, La)

    # ---- NEW: refine anchor core inside [s:e] with top-50% contiguous window
    local = Sc[s:e]
    thr_in = torch.quantile(local, TOP_PCT_CORE_IN_ANCHOR)
    mk = (local >= thr_in).float().cpu().numpy().astype(np.int8)
    best = (0,0,0)
    i=0
    while i<mk.size:
        if mk[i]==0: i+=1; continue
        j=i
        while j<mk.size and mk[j]==1: j+=1
        if j-i>best[2]: best=(i,j,j-i)
        i=j
    core_s_rel, core_e_rel = best[0], best[1]
    # Build anchor mel block, zero-pad outside core
    Ablk = Xmel[:, s:e].clone()
    if core_s_rel>0:    Ablk[:, :core_s_rel] = 0
    if core_e_rel<La:   Ablk[:, core_e_rel:] = 0

    # Ω & template
    omega = omega_support(Ablk)                         # [M]
    w_bar = template_from_anchor_block(Ablk, omega)     # [M]

    # Scalars & confidence
    a_raw, a_gate = amplitude_raw_and_gate(Xmel, w_bar, omega)  # [T],[T]
    cos_t = cos_similarity_over_omega(Xmel, w_bar, omega)       # [T]
    C = build_confidence(a_gate, cos_t)                         # [T]

    # Source mel power (shape fixed, scale = a_raw)
    S_mel = w_bar.view(-1,1) * a_raw.view(1,-1)                 # [M,T]

    # Back-projection (Ω-weighted NNLS)
    S_lin = nnls_weighted_backproject(S_mel, mel_fb_m2f, omega) # [F,T]

    # Map Ω(mel) -> Ω(linear) (for possible local boosts; here only used to ensure multi-bin support)
    omega_lin = ((mel_fb_m2f.T @ omega).clamp_min(0.0) > 1e-12).float()  # [F]

    # Mask with time-confidence weighting
    M_lin = make_mask(S_lin, P, omega_lin, C)

    # First pass energy push
    if pass_idx == 0:
        M_lin = M_lin.pow(FIRST_PASS_EXP)

    # Reconstruct
    src, res, er = reconstruct(mag, phase, M_lin)

    # Optional oversubtraction on confident segments
    if DO_OVERSUB_TIME:
        segs = segments_from_conf(C, CONF_TAU, T)
        res = oversub_time(res, src, segs, C)

    # Used-frame mask for next pass
    r_t = (M_lin * P).sum(dim=0) / (P.sum(dim=0) + 1e-8)
    used_mask = (r_t >= USED_THRESHOLD).float()

    elapsed = time.time() - t0

    # Debug
    if out_dir:
        png = os.path.join(out_dir, f"debug_pass_{pass_idx+1}.png")
        debug_plot(pass_idx, Sc, a_raw, a_gate, cos_t, C, P, M_lin,
                   s, e, core_s_rel, core_e_rel, png,
                   title=f"Pass {pass_idx+1} | anchor {s*HOP/SR:.2f}-{e*HOP/SR:.2f}s | ER={er*100:.1f}%")

    info = {
        "er": er,
        "elapsed": elapsed,
        "anchor": (s*HOP/SR, e*HOP/SR),
        "core":   ((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR),
    }
    return src, res, er, used_mask, info

# =========================
# Main
# =========================
def main():
    ap=argparse.ArgumentParser(description="Anchor-Refined Ω-NNLS AST Demixer")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--passes", type=int, default=MAX_PASSES)
    ap.add_argument("--no-debug", action="store_true")
    args=ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 Anchor-Refined Ω-NNLS AST Demixer\n{'='*64}")
    print(f"Input: fixed {WIN_SEC:.3f}s, Anchor: {ANCHOR_SEC:.3f}s, Passes: {args.passes}")

    # Mel FB: [F,M] -> [M,F]
    fbins = N_FFT//2 + 1
    mel_fb_f2m = torchaudio.functional.melscale_fbanks(
        n_freqs=fbins, f_min=0.0, f_max=SR/2, n_mels=N_MELS,
        sample_rate=SR, norm="slaney"
    )
    mel_fb_m2f = mel_fb_f2m.T.contiguous()  # [M,F]

    # AST (CPU)
    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    ast_model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        attn_implementation="eager"  # avoid fallback warning
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
        print(f"⏱️ pass{i+1}: {info['elapsed']:.3f}s | anchor {info['anchor'][0]:.2f}-{info['anchor'][1]:.2f}s | core {info['core'][0]:.2f}-{info['core'][1]:.2f}s | ER={er*100:.1f}%")

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
