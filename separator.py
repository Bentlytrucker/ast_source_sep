#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cached AST-guided Source Separator
- separator.py의 전체 로직 유지
- AST 모델 호출을 3번으로 최적화 (attention map 캐싱)
- 소스 분리 시마다 백엔드 전송
"""

import os, time, warnings, argparse, json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import requests
from transformers import ASTFeatureExtractor, ASTForAudioClassification

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# =========================
# Config (separator.py와 동일)
# =========================
SR = 16000
WIN_SEC = 4.096
ANCHOR_SEC = 0.512
L_FIXED = int(round(WIN_SEC * SR))

NORMALIZE_TARGET_PEAK = 0.95
RESIDUAL_CLIP_THR = 0.0005

USE_ADAPTIVE_STRATEGY = True
FALLBACK_THRESHOLD = 0.1

MASK_SIGMOID_CENTER = 0.5  # 더 낮은 중심점으로 강한 분리
MASK_SIGMOID_SLOPE = 25.0  # 더 높은 경사로 날카로운 경계

N_FFT, HOP, WINLEN = 400, 160, 400
WINDOW = torch.hann_window(WINLEN)
EPS = 1e-10

N_MELS = 128

SMOOTH_T = 19
ALPHA_ATT = 0.80
BETA_PUR = 1.20
W_E = 0.30
TOP_PCT_CORE_IN_ANCHOR = 0.50

OMEGA_Q_CONSERVATIVE = 0.2
OMEGA_Q_AGGRESSIVE = 0.7
OMEGA_DIL = 2
OMEGA_MIN_BINS = 5

AST_FREQ_QUANTILE_CONSERVATIVE = 0.4
AST_FREQ_QUANTILE_AGGRESSIVE = 0.2

DANGER_IDS = {396, 397, 398, 399, 400, 426, 436}
HELP_IDS = {23, 14, 354, 355, 356, 359}
WARNING_IDS = {288, 364, 388, 389, 390, 439, 391, 392, 393, 395, 440, 441, 443, 456, 469, 470, 478, 479}

PRES_Q = 0.20
PRES_SMOOTH_T = 9

USED_THRESHOLD = 0.65
USED_DILATE_MS = 80
ANCHOR_SUPPRESS_MS = 200
ANCHOR_SUPPRESS_BASE = 0.6

MAX_PASSES = 3
MIN_ERATIO = 0.01

# Backend API
USER_ID = 6
BACKEND_URL = "http://13.238.200.232:8000/sound-events/"

# =========================
# Utils (separator.py와 동일)
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
    if class_id in DANGER_IDS:
        return "danger"
    elif class_id in HELP_IDS:
        return "help"
    elif class_id in WARNING_IDS:
        return "warning"
    else:
        # 서버가 "other"를 허용하지 않으므로 기본값을 "warning"으로 설정
        return "other"

def calculate_decibel(audio: np.ndarray) -> Tuple[float, float, float]:
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return -np.inf, -np.inf, -np.inf
    
    db = 20 * np.log10(rms + 1e-10)
    db_min = 20 * np.log10(np.min(np.abs(audio)) + 1e-10)
    db_max = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    db_mean = db
    
    return db_min, db_max, db_mean

def send_to_backend(sound_type: str, sound_detail: str, decibel: float) -> bool:
    try:
        data = {
            "user_id": USER_ID,
            "sound_type": sound_type,
            "sound_detail": sound_detail,
            "angle": 0,
            "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sound_icon": "string",
            "location_image_url": "string",
            "decibel": float(decibel),
        }
        
        # 더 긴 타임아웃과 재시도 로직 추가
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'CachedSeparator/1.0'
        }
        
        print(f"🔄 Sending to backend: {BACKEND_URL}")
        print(f"📤 Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            BACKEND_URL, 
            json=data, 
            headers=headers,
            timeout=3.0,  # 타임아웃을 3초로 설정
            verify=False  # SSL 인증서 검증 비활성화 (테스트용)
        )
        
        if response.status_code == 200:
            print(f"✅ Sent to backend: {sound_detail} ({sound_type})")
            return True
        else:
            print(f"❌ Backend error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectTimeout:
        print(f"❌ Backend connection timeout: {BACKEND_URL}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Backend connection error: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Backend request timeout: {BACKEND_URL}")
        return False
    except Exception as e:
        print(f"❌ Backend error: {e}")
        return False

# =========================
# Caching System
# =========================
class ASTCache:
    def __init__(self):
        self.attention_cache: Dict[str, torch.Tensor] = {}
        self.cls_head_cache: Dict[str, torch.Tensor] = {}
    
    def get_cache_key(self, audio: np.ndarray) -> str:
        return str(hash(audio.tobytes()))
    
    def cache_attention(self, audio: np.ndarray, attention_map: torch.Tensor, cls_features: torch.Tensor):
        key = self.get_cache_key(audio)
        self.attention_cache[key] = attention_map
        self.cls_head_cache[key] = cls_features
    
    def get_attention(self, audio: np.ndarray) -> Optional[torch.Tensor]:
        key = self.get_cache_key(audio)
        return self.attention_cache.get(key)
    
    def get_cls_features(self, audio: np.ndarray) -> Optional[torch.Tensor]:
        key = self.get_cache_key(audio)
        return self.cls_head_cache.get(key)

# 전역 캐시 인스턴스
ast_cache = ASTCache()

# =========================
# AST Processing (3번만 호출)
# =========================
@torch.no_grad()
def ast_attention_freq_time_cached(audio: np.ndarray, extractor, ast_model, T_out: int, F_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AST 어텐션에서 시간과 주파수 정보를 모두 추출 (캐싱 포함)
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
    # AST 모델이 CPU에서 실행되므로 입력도 CPU로 보장
    feat["input_values"] = feat["input_values"].cpu()
    outputs = ast_model(input_values=feat["input_values"], output_attentions=True, return_dict=True)
    
    # Attention map 추출
    attns = outputs.attentions
    if not attns or len(attns) == 0:
        time_attention = torch.ones(101) * 0.5
        freq_attention = torch.ones(12) * 0.5
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
    
    # CLS features 추출
    if hasattr(outputs, 'last_hidden_state'):
        cls_features = outputs.last_hidden_state[:, 0, :]  # CLS token features
    else:
        # SequenceClassifierOutput의 경우 logits를 사용
        cls_features = outputs.logits
    
    # 캐싱 (시간 어텐션만 캐시)
    ast_cache.cache_attention(audio, time_attention, cls_features)
    
    return time_attn_norm, freq_attn_norm

@torch.no_grad()
def extract_and_cache_attention(audio: np.ndarray, extractor, ast_model) -> Tuple[torch.Tensor, torch.Tensor]:
    """AST 모델 호출하여 attention map과 CLS features 추출 및 캐싱 (하위 호환성)"""
    # 캐시 확인
    cached_attention = ast_cache.get_attention(audio)
    cached_cls = ast_cache.get_cls_features(audio)
    
    if cached_attention is not None and cached_cls is not None:
        return cached_attention, cached_cls
    
    # 10초로 패딩
    target_len = int(10.0 * SR)
    if len(audio) < target_len:
        audio_padded = np.zeros(target_len, dtype=np.float32)
        audio_padded[:len(audio)] = audio
    else:
        audio_padded = audio[:target_len]
    
    feat = extractor(audio_padded, sampling_rate=SR, return_tensors="pt")
    # AST 모델이 CPU에서 실행되므로 입력도 CPU로 보장
    feat["input_values"] = feat["input_values"].cpu()
    outputs = ast_model(input_values=feat["input_values"], output_attentions=True, return_dict=True)
    
    # Attention map 추출
    attns = outputs.attentions
    if not attns or len(attns) == 0:
        attention_map = torch.ones(101) * 0.5
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
        attention_map = full_map.mean(dim=0)
    
    # CLS features 추출
    if hasattr(outputs, 'last_hidden_state'):
        cls_features = outputs.last_hidden_state[:, 0, :]  # CLS token features
    else:
        # SequenceClassifierOutput의 경우 logits를 사용
        cls_features = outputs.logits
    
    # 캐싱
    ast_cache.cache_attention(audio, attention_map, cls_features)
    
    return attention_map, cls_features

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

# =========================
# Audio Processing (separator.py와 동일)
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
    P = (mag * mag).clamp_min(EPS)
    phase = torch.angle(st)

    if mel_fb_m2f.shape[0] != N_MELS:
        mel_fb_m2f = mel_fb_m2f.T.contiguous()
    assert mel_fb_m2f.shape[0] == N_MELS and mel_fb_m2f.shape[1] == P.shape[0]
    mel_fb_m2f = mel_fb_m2f.to(P.dtype).to(P.device)
    mel_pow = (mel_fb_m2f @ P).clamp_min(EPS)
    return st, mag, P, phase, mel_pow

# =========================
# Core Separation Logic (separator.py와 동일)
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
    while core_s_rel > 0 and local_score[core_s_rel - 1] >= threshold:
        core_s_rel -= 1
        
    core_e_rel = peak_idx_rel
    while core_e_rel < La - 1 and local_score[core_e_rel + 1] >= threshold:
        core_e_rel += 1
    
    core_e_rel += 1
    return anchor_s, anchor_e, core_s_rel, core_e_rel

def omega_support_with_ast_freq(Ablk: torch.Tensor, ast_freq_attn: torch.Tensor, strategy: str = "conservative") -> torch.Tensor:
    if strategy == "conservative":
        omega_q = OMEGA_Q_CONSERVATIVE
        ast_freq_quantile = AST_FREQ_QUANTILE_CONSERVATIVE
    else:
        omega_q = OMEGA_Q_AGGRESSIVE
        ast_freq_quantile = AST_FREQ_QUANTILE_AGGRESSIVE
    
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
        take = order[:need]
        mask[take] = 1.0
    
    return mask

def template_from_anchor_block(Ablk: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    om = omega.view(-1,1)
    w = (Ablk * om).mean(dim=1) * omega
    w = w / (w.sum() + EPS)
    w_sm = F.avg_pool1d(w.view(1,1,-1), kernel_size=3, stride=1, padding=1).view(-1)
    w = (w_sm * omega); w = w / (w.sum() + EPS)
    return w

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

def unified_masking_strategy(Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, 
                           ast_freq_attn: torch.Tensor, P: torch.Tensor, mel_fb_m2f: torch.Tensor,
                           s: int, e: int, strategy: str = "conservative") -> torch.Tensor:
    """
    통합 마스킹 전략: sep.py와 save.py의 장점을 결합
    """
    fbins, T = P.shape
    
    # 1. 기본 코사인 유사도 계산
    g_pres = presence_from_energy(Xmel, omega)
    cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)
    
    # 2. 코사인 유사도 3단계 처리: 정교한 분리를 위한 구간별 차별화
    # 0.4 미만: 거의 없는 소리 (0.05로 매우 약하게)
    # 0.4~0.6: 원본 코사인 유사도 그대로 (자연스러운 전환)
    # 0.6 이상: 강한 분리 (0.7로 설정, 가중치 적용시 1.05까지)
    
    cos_processed = torch.where(
        cos_t_raw < 0.5, 
        torch.full_like(cos_t_raw, 0.05),  # 거의 없는 소리
        torch.where(
            cos_t_raw < 0.6,
            cos_t_raw,  # 원본 그대로
            torch.full_like(cos_t_raw, 0.7)  # 강한 분리 (가중치 고려)
        )
    )
    
    # 스무딩으로 부드러운 경계 생성
    cos_processed = smooth1d(cos_processed, 5)  # 5프레임 스무딩
    
    # 3. 주파수 가중치 계산 (두 전략 모두 적용)
    # Linear 도메인에서 직접 계산 (sep.py 방식)
    omega_lin = ((mel_fb_m2f.T @ omega).clamp_min(0.0) > 1e-12).float()
    
    # 앵커 영역의 상위 30% 진폭 주파수 선택
    anchor_spec = P[:, s:e]
    anchor_max_amp = anchor_spec.max(dim=1).values
    amp_threshold = torch.quantile(anchor_max_amp, 0.7)  # 상위 30% (0.7 quantile)
    high_amp_mask_lin = (anchor_max_amp >= amp_threshold).float()
    
    # AST 어텐션이 주목한 주파수 (Linear 도메인으로 변환)
    ast_freq_threshold = torch.quantile(ast_freq_attn, 0.4 if strategy == "conservative" else 0.2)
    ast_active_mask_mel = (ast_freq_attn >= ast_freq_threshold).float()
    ast_active_mask_lin = ((mel_fb_m2f.T @ ast_active_mask_mel).clamp_min(0.0) > 0.1).float()
    
    # 주파수 가중치: 앵커 상위 30% + 어텐션 주목 주파수
    freq_boost_mask = torch.maximum(high_amp_mask_lin, ast_active_mask_lin)
    
    # 가중치 적용: 두 조건을 모두 만족하는 주파수에 더 높은 가중치
    freq_weight = 1.0 + 0.5 * freq_boost_mask  # [1.0, 1.5] - 적당한 가중치
    
    # 5. 기본 마스크 계산
    M_base = omega_lin.view(-1, 1) * cos_processed.view(1, -1)
    
    # 6. 주파수 가중치 적용
    M_weighted = M_base * freq_weight.view(-1, 1)
    
    # 7. 스펙트로그램 제한
    spec_magnitude = P.sqrt()
    M_lin = torch.minimum(M_weighted, spec_magnitude)
    
    # 8. 마스크 후처리: 코사인 유사도에서 이미 스무딩 적용했으므로 추가 처리 불필요
    
    return M_lin

def adaptive_strategy_selection(prev_energy_ratio: float, pass_idx: int) -> str:
    if not USE_ADAPTIVE_STRATEGY:
        return "conservative"
    
    if pass_idx == 0:
        return "conservative"
    
    if prev_energy_ratio > 2.0:
        return "conservative"
    
    if prev_energy_ratio < 1.2:
        return "aggressive"
    
    return "conservative"

# =========================
# Debug Visualization (separator.py와 동일)
# =========================
def debug_plot(pass_idx: int, Sc: torch.Tensor, a_raw: torch.Tensor, cos_t_raw: torch.Tensor, 
               C_t: torch.Tensor, P: torch.Tensor, M_lin: torch.Tensor, full_map: torch.Tensor,
               s: int, e: int, core_s_rel: int, core_e_rel: int, ast_freq_attn: torch.Tensor,
               src_amp: np.ndarray, res: np.ndarray, png: str, title: str = "", P_res: torch.Tensor = None):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 공통 변수
    T = Sc.numel()
    fbins, T_spec = P.shape
    f_axis = np.arange(fbins) * SR / (2 * fbins)
    t_axis = np.arange(T_spec) * HOP / SR
    
    # 1. Anchor Score
    ax = axes[0, 0]
    t_axis_score = np.arange(T) * HOP / SR
    ax.plot(t_axis_score, to_np(Sc), 'b-', linewidth=1.5, label='Anchor Score')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='red', label='Anchor Region')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='orange', label='Core Region')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score')
    ax.set_title('Anchor Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Amplitude & Cosine Similarity
    ax = axes[0, 1]
    t_axis_amp = np.arange(a_raw.numel()) * HOP / SR
    ax.plot(t_axis_amp, to_np(a_raw), 'g-', linewidth=1.5, label='Amplitude')
    ax2 = ax.twinx()
    ax2.plot(t_axis_amp, to_np(cos_t_raw), 'r-', linewidth=1.5, label='Cosine Similarity')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude', color='g')
    ax2.set_ylabel('Cosine Similarity', color='r')
    ax.set_title('Amplitude & Cosine Similarity')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. Original Power Spectrogram
    ax = axes[0, 2]
    im = ax.imshow(to_np(torch.log10(P + 1e-10)), aspect='auto', origin='lower', 
                   extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='viridis')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='red')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Original Power Spectrogram (log10)')
    plt.colorbar(im, ax=ax, label='log10(Power)')
    
    # 4. Generated Mask
    ax = axes[1, 0]
    im = ax.imshow(to_np(M_lin), aspect='auto', origin='lower',
                   extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='hot')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='cyan')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='yellow')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Generated Mask')
    plt.colorbar(im, ax=ax, label='Mask Value')
    
    # 5. Separated Source Power Spectrogram
    ax = axes[1, 1]
    P_src = (M_lin * P.sqrt()) ** 2  # 분리된 소스의 power spectrogram
    im = ax.imshow(to_np(torch.log10(P_src + 1e-10)), aspect='auto', origin='lower',
                   extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='viridis')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='cyan')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='yellow')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Separated Source Power Spectrogram (log10)')
    plt.colorbar(im, ax=ax, label='log10(Power)')
    
    # 6. Residual Power Spectrogram
    ax = axes[1, 2]
    if P_res is not None:
        im = ax.imshow(to_np(torch.log10(P_res + 1e-10)), aspect='auto', origin='lower',
                       extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='viridis')
        ax.set_title('Residual Power Spectrogram (log10)')
    else:
        # 잔여물 power spectrogram 계산
        P_residual = P - P_src
        P_residual = torch.maximum(P_residual, torch.zeros_like(P_residual))
        im = ax.imshow(to_np(torch.log10(P_residual + 1e-10)), aspect='auto', origin='lower',
                       extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap='viridis')
        ax.set_title('Residual Power Spectrogram (log10)')
    ax.axvspan(s*HOP/SR, e*HOP/SR, alpha=0.3, color='red')
    ax.axvspan((s+core_s_rel)*HOP/SR, (s+core_e_rel)*HOP/SR, alpha=0.5, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=ax, label='log10(Power)')
    
    # 7. Separated Source Waveform
    ax = axes[2, 0]
    t_audio = np.arange(len(src_amp)) / SR
    ax.plot(t_audio, src_amp, 'b-', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Separated Source Waveform')
    ax.grid(True, alpha=0.3)
    
    # 8. Residual Waveform
    ax = axes[2, 1]
    t_audio = np.arange(len(res)) / SR
    ax.plot(t_audio, res, 'r-', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Residual Waveform')
    ax.grid(True, alpha=0.3)
    
    # 9. Energy Comparison
    ax = axes[2, 2]
    if P_res is not None:
        orig_energy = torch.sum(P).item()
        src_energy = torch.sum(P_src).item()
        res_energy = torch.sum(P_res).item()
    else:
        orig_energy = torch.sum(P).item()
        src_energy = torch.sum(P_src).item()
        P_residual = P - P_src
        P_residual = torch.maximum(P_residual, torch.zeros_like(P_residual))
        res_energy = torch.sum(P_residual).item()
    
    categories = ['Original', 'Source', 'Residual']
    energies = [orig_energy, src_energy, res_energy]
    colors = ['gray', 'blue', 'red']
    
    bars = ax.bar(categories, energies, color=colors, alpha=0.7)
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Distribution')
    ax.grid(True, alpha=0.3)
    
    # 에너지 비율 표시
    for i, (bar, energy) in enumerate(zip(bars, energies)):
        ratio = energy / orig_energy * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + orig_energy*0.01,
                f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()

# =========================
# Main Processing Pipeline
# =========================
def single_pass_cached(audio: np.ndarray, extractor, ast_model, mel_fb_m2f: torch.Tensor,
                      used_mask_prev: Optional[torch.Tensor],
                      prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor]],
                      pass_idx: int, out_dir: Optional[str], prev_energy_ratio: float = 1.0,
                      enable_debug: bool = True):
    
    t0 = time.time()
    st, mag, P, phase, Xmel = stft_all(audio, mel_fb_m2f)
    fbins, T = P.shape
    La = int(round(ANCHOR_SEC * SR / HOP))

    # 적응적 전략 선택
    strategy = adaptive_strategy_selection(prev_energy_ratio, pass_idx)
    print(f"  Strategy: {strategy}")

    # AST에서 시간과 주파수 어텐션 모두 추출 (캐싱 포함)
    A_t, ast_freq_attn = ast_attention_freq_time_cached(audio, extractor, ast_model, T, N_MELS)
    
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
    
    # Pick anchor and core regions
    s, e, core_s_rel, core_e_rel = pick_anchor_region(Sc, La, TOP_PCT_CORE_IN_ANCHOR)
    
    # Create anchor block
    Ablk = Xmel[:, s:e].clone()
    if core_s_rel > 0:  Ablk[:, :core_s_rel] = 0
    if core_e_rel < La: Ablk[:, core_e_rel:] = 0

    # Ω 계산
    omega = omega_support_with_ast_freq(Ablk, ast_freq_attn, strategy)
    w_bar = template_from_anchor_block(Ablk, omega)
    
    # 통합 마스킹 전략 적용
    M_lin = unified_masking_strategy(Xmel, w_bar, omega, ast_freq_attn, P, mel_fb_m2f, s, e, strategy)
    
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

    # Debug plot generation
    if enable_debug and out_dir is not None:
        # 필요한 변수들 계산
        a_raw = amplitude_raw(Xmel, w_bar, omega)
        cos_t_raw = cos_similarity_over_omega(Xmel, w_bar, omega, presence_from_energy(Xmel, omega))
        C_t = cos_t_raw  # 별칭
        
        # full_map은 AST 어텐션에서 추출 (간단한 버전)
        full_map = torch.zeros(12, 101)  # 기본 크기
        
        # 잔여 power spectrogram 계산
        P_src = (M_lin * P.sqrt()) ** 2  # 분리된 소스의 power spectrogram
        P_residual = P - P_src
        P_residual = torch.maximum(P_residual, torch.zeros_like(P_residual))
        
        png = os.path.join(out_dir, f"debug_pass_{pass_idx+1}.png")
        debug_plot(pass_idx, Sc, a_raw, cos_t_raw, C_t, P, M_lin, full_map,
                  s, e, core_s_rel, core_e_rel, ast_freq_attn, src_amp, res, png,
                  title=f"Pass {pass_idx+1} | Strategy: {strategy} | anchor {s*HOP/SR:.2f}-{e*HOP/SR:.2f}s | ER={er*100:.1f}% [Unified Mask]",
                  P_res=P_residual)

    # Used-frame mask for next pass
    r_t = (M_lin * P).sum(dim=0) / (P.sum(dim=0) + 1e-8)
    used_mask = (r_t >= USED_THRESHOLD).float()

    elapsed = time.time() - t0
    
    # 캐싱된 attention map을 사용하여 분류
    class_name, sound_type, class_id, confidence = classify_from_cached_attention(audio, ast_model, s, e)
    db_min, db_max, db_mean = calculate_decibel(src_amp)
    
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
# Main Function
# =========================
def main():
    global BACKEND_URL
    
    parser = argparse.ArgumentParser(description="Cached AST-guided Source Separator")
    parser.add_argument("--input", "-i", required=True, help="Input audio file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--max-passes", type=int, default=MAX_PASSES, help="Maximum separation passes")
    parser.add_argument("--min-eratio", type=float, default=MIN_ERATIO, help="Minimum energy ratio to continue")
    parser.add_argument("--backend-url", default=BACKEND_URL, help="Backend API URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug visualization")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug visualization")
    
    args = parser.parse_args()
    
    # Backend URL 설정
    BACKEND_URL = args.backend_url
    
    # Debug 옵션 설정
    enable_debug = args.debug or (not args.no_debug)  # 기본값은 True, --no-debug로 비활성화
    
    # Device 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Debug visualization: {'ON' if enable_debug else 'OFF'}")
    
    # Output directory 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 모델 로드 (CPU에서 실행)
    print("Loading AST model...")
    extractor = ASTFeatureExtractor.from_pretrained(args.model)
    ast_model = ASTForAudioClassification.from_pretrained(
        args.model,
        attn_implementation="eager"
    ).eval()  # CPU에서 실행
    
    # Mel filterbank 생성
    # Mel FB: [F,M] -> [M,F]
    fbins = N_FFT//2 + 1
    mel_fb_f2m = torchaudio.functional.melscale_fbanks(
        n_freqs=fbins, f_min=0.0, f_max=SR/2, n_mels=N_MELS,
        sample_rate=SR, norm="slaney"
    )
    mel_fb_m2f = mel_fb_f2m.T.contiguous()
    
    # 오디오 로드
    print(f"Loading audio: {args.input}")
    audio = load_fixed_audio(args.input)
    print(f"Audio length: {len(audio)/SR:.2f}s")
    
    # 메인 처리 루프
    print("\nStarting separation with cached AST...")
    start_time = time.time()
    
    current_audio = audio.copy()
    used_mask_prev = None
    prev_anchors = []
    sources = []
    total_ast_calls = 0
    
    for pass_idx in range(args.max_passes):
        print(f"\n--- Pass {pass_idx + 1} ---")
        
        # AST 모델 호출 (총 3번)
        if pass_idx < 3:
            total_ast_calls += 1
            print(f"AST call #{total_ast_calls} for attention extraction...")
        
        # 분리 실행
        src_amp, res, er, used_mask, info = single_pass_cached(
            current_audio, extractor, ast_model, mel_fb_m2f,
            used_mask_prev, prev_anchors, pass_idx, args.output,
            prev_energy_ratio=info.get("energy_ratio", 1.0) if pass_idx > 0 else 1.0,
            enable_debug=enable_debug
        )
        
        # 결과 저장
        src_path = os.path.join(args.output, f"{pass_idx:02d}_separated.wav")
        torchaudio.save(src_path, torch.from_numpy(src_amp).unsqueeze(0), SR)
        
        # 백엔드 전송 (other 타입은 제외)
        if info["sound_type"] != "other":
            try:
                success = send_to_backend(
                    info["sound_type"], 
                    info["class_name"], 
                    info["db_mean"]
                )
            except Exception as e:
                print(f"⚠️  Backend send failed: {e}")
                success = False
        else:
            print(f"⏭️  Skipping backend send for 'other' type: {info['class_name']}")
            success = True  # other 타입은 성공으로 처리
        
        # 정보 출력
        print(f"  Separated: {info['class_name']} ({info['sound_type']})")
        print(f"  Confidence: {info['confidence']:.3f}")
        print(f"  Decibel: {info['db_mean']:.1f} dB")
        print(f"  Energy Ratio: {er:.3f}")
        print(f"  Anchor: {info['anchor'][0]:.2f}-{info['anchor'][1]:.2f}s")
        print(f"  Elapsed: {info['elapsed']:.2f}s")
        print(f"  Backend: {'✅' if success else '❌'}")
        
        # 앵커 정보 저장
        prev_anchors.append((info['anchor'][0], info['anchor'][1], info['w_bar'], info['omega']))
        sources.append({
            "pass": pass_idx + 1,
            "class_name": info['class_name'],
            "sound_type": info['sound_type'],
            "confidence": info['confidence'],
            "decibel": info['db_mean'],
            "energy_ratio": er,
            "anchor": info['anchor'],
            "file": src_path
        })
        
        # 잔여물을 다음 패스의 입력으로 사용
        current_audio = res
        used_mask_prev = used_mask
        
        # 조기 종료 조건
        if er < args.min_eratio:
            print(f"  Early stop: Energy ratio {er:.3f} < {args.min_eratio}")
            break
    
    # 잔여물 저장
    res_path = os.path.join(args.output, f"{len(sources):02d}_residual.wav")
    torchaudio.save(res_path, torch.from_numpy(current_audio).unsqueeze(0), SR)
    
    # 최종 결과
    total_time = time.time() - start_time
    print(f"\n=== Separation Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total AST calls: {total_ast_calls}")
    print(f"Sources found: {len(sources)}")
    print(f"Residual saved: {res_path}")
    
    # 요약 출력
    print(f"\n=== Sources Summary ===")
    for i, src in enumerate(sources):
        print(f"{i+1}. {src['class_name']} ({src['sound_type']}) - {src['confidence']:.3f} - {src['decibel']:.1f}dB")
    
    # 성능 검증
    if total_time < 4.0:
        print(f"\n✅ SUCCESS: Completed in {total_time:.2f}s (< 4s target)")
    else:
        print(f"\n⚠️  WARNING: Took {total_time:.2f}s (>= 4s target)")
    
    if total_ast_calls <= 3:
        print(f"✅ SUCCESS: Used {total_ast_calls} AST calls (<= 3 target)")
    else:
        print(f"⚠️  WARNING: Used {total_ast_calls} AST calls (> 3 target)")

if __name__ == "__main__":
    main()
