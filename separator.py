#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Version: AST-guided Source Separator
Features:
- Sigmoid Soft Masking based purely on Cosine Similarity.
- Peak-first anchor and core selection for stability.
- Frequency-domain subtraction for precision.
- Peak Normalization for consistent output volume.
- Residual Clipping for a clean residual file.
- Linear Amplitude visualization for intuitive analysis.
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
NORMALIZE_TARGET_PEAK = 0.95 # 최대 볼륨의 95% 크기로 표준화
RESIDUAL_CLIP_THR = 0.0005 # 최종 잔여물의 진폭이 이 값보다 작으면 0으로 만듦

# === Sigmoid Soft Masking Parameters ===
MASK_SIGMOID_CENTER = 0.6   # 마스크가 0.5가 되는 cosΩ 값 (중심점)
MASK_SIGMOID_SLOPE  = 15.0  # S-커브의 경사. 높을수록 하드 마스크처럼 날카로워짐

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
OMEGA_Q           = 0.40
OMEGA_DIL         = 2
OMEGA_MIN_BINS    = 10

# Presence gate
PRES_Q            = 0.20
PRES_SMOOTH_T     = 9

# Suppression
USED_THRESHOLD        = 0.65
USED_DILATE_MS        = 80
ANCHOR_SUPPRESS_MS    = 200
ANCHOR_SUPPRESS_BASE  = 0.6

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

def soft_sigmoid(x: torch.Tensor, center: float, slope: float, min_val: float = 0.0) -> torch.Tensor:
    sig = torch.sigmoid(slope * (x - center))
    return min_val + (1.0 - min_val) * sig

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
# Debug (Linear Amplitude Visualization)
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

    # 2) Scalars
    ar = to_np(a_raw); ar_n = (ar - ar.min())/(ar.max()-ar.min()+1e-8)
    ax[0,1].plot(t, ar_n, label='a_raw (norm)', lw=1.0)
    ax[0,1].plot(t, to_np(cos_t), label='cosΩ', lw=1.0)
    ax[0,1].plot(t, to_np(C_t), label='C(t) [Debug]', lw=1.0, alpha=0.85)
    ax[0,1].legend(); ax[0,1].set_ylim([0,1.05]); ax[0,1].set_title("Scalars")

    # 3) Mask stats
    m_mean = to_np(M_lin.mean(dim=0))
    ax[0,2].plot(t, m_mean, color='purple'); ax[0,2].set_ylim([0,1.05])
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
# Single Pass (Final Version: Sigmoid Mask on cosΩ)
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

    # Ω & template
    omega = omega_support(Ablk)
    w_bar = template_from_anchor_block(Ablk, omega)
    
    # Calculate cosΩ, the core of our mask
    g_pres = presence_from_energy(Xmel, omega)
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)

    # Map Ω(mel)->Ω(linear) for frequency weighting
    omega_lin = ((mel_fb_m2f.T @ omega).clamp_min(0.0) > 1e-12).float()

    # === Final Masking Logic: Sigmoid function on cosΩ ===
    soft_time_mask = torch.sigmoid(MASK_SIGMOID_SLOPE * (cos_t_raw - MASK_SIGMOID_CENTER))
    M_lin = omega_lin.view(-1, 1) * soft_time_mask.view(1, -1)
    
    # Subtraction in the complex STFT domain for precision
    stft_full = st
    stft_src  = M_lin * stft_full
    stft_res  = stft_full - stft_src

    # Reconstruct both source and residual from their respective spectrograms
    src_amp = torch.istft(stft_src, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()
    res = torch.istft(stft_res, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()

    # ER calculation
    e_src = float(np.sum(src_amp**2)); e_res = float(np.sum(res**2))
    er = e_src / (e_src + e_res + 1e-12)

    # Used-frame mask for next pass
    r_t = (M_lin * P).sum(dim=0) / (P.sum(dim=0) + 1e-8)
    used_mask = (r_t >= USED_THRESHOLD).float()

    elapsed = time.time() - t0
    
    if out_dir:
        # Calculate C(t) purely for visualization purposes
        a_raw = amplitude_raw(Xmel, w_bar, omega)
        a_nz_soft = soft_sigmoid(a_raw / (a_raw.median() + 1e-8), center=0.1, slope=10.0, min_val=0.0)
        C_t = cos_t_raw * a_nz_soft
        png = os.path.join(out_dir, f"debug_pass_{pass_idx+1}.png")
        debug_plot(pass_idx, Sc, a_raw, cos_t_raw, C_t, P, M_lin,
                   s, e, core_s_rel, core_e_rel, png,
                   title=f"Pass {pass_idx+1} | anchor {s*HOP/SR:.2f}-{e*HOP/SR:.2f}s | ER={er*100:.1f}% [Sigmoid Mask]")

    info = {
        "er": er,
        "elapsed": elapsed,
        "anchor": (s*HOP/SR, e*HOP/SR),
        "core":   ((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR),
        "quality": float(soft_time_mask.mean().item()),
        "w_bar": w_bar,
        "omega": omega,
        "stopped": False
    }
    return src_amp, res, er, used_mask, info

# =========================
# Main (Final Version)
# =========================
def main():
    ap=argparse.ArgumentParser(description="Final AST-guided Source Separator")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--passes", type=int, default=MAX_PASSES)
    ap.add_argument("--no-debug", action="store_true")
    args=ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    audio0 = load_fixed_audio(args.input)
    print(f"\n{'='*64}\n🎵 AST-guided Source Separator (Final Version)\n{'='*64}")
    print(f"Input: fixed {WIN_SEC:.3f}s, Anchor: {ANCHOR_SEC:.3f}s, Passes: {args.passes}")
    print(f"Features: Sigmoid Masking, Peak-first Selection, Normalization, Clipping")

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
              f" | core {info['core'][0]:.2f}-{info['core'][1]:.2f}s | ER={er*100:.1f}%")

        if er < MIN_ERATIO:
            print("  ⚠️ Too little energy; stopping.")
            break

        # Peak Normalization for clear output
        peak = np.max(np.abs(src))
        if peak > 1e-8:
            gain = NORMALIZE_TARGET_PEAK / peak
            src = src * gain
            src = np.clip(src, -1.0, 1.0)

        out_src = os.path.join(args.output, f"{i+1:02d}_source.wav")
        torchaudio.save(out_src, torch.from_numpy(src).unsqueeze(0), SR)
        print(f"    ✅ Saved (Normalized): {out_src}")

        cur = res
        prev_anchors.append((info["anchor"][0], info["anchor"][1], info["w_bar"], info["omega"]))
        saved += 1

    # Apply Hard Clipping to the final residual
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
