#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Separator Module for Raspberry Pi
- separator.py의 최신 로직을 기반으로 구현
- Raspberry Pi 환경에 최적화
- 각도 정보를 백엔드에 전달
- 음원 분리 및 분류 기능
"""

import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import requests
from typing import List, Tuple, Optional, Dict, Any

# separator.py에서 사용하는 모듈들 import
warnings.filterwarnings("ignore")
torch.set_num_threads(2)  # Raspberry Pi에 최적화

try:
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
except ImportError:
    print("Warning: transformers not available. Sound separation will be disabled.")
    ASTFeatureExtractor = None
    ASTForAudioClassification = None

# =========================
# Global Constants (separator.py 최신 버전)
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
MIN_ERATIO = 0.001  # separator.py와 동일

# Audio amplification parameters
MIN_ANCHOR_ENERGY = 0.001
AMPLIFICATION_FACTOR = 10.0
MAX_AMPLIFICATION = 100.0

# Sound classification mappings (separator.py 최신 버전)
DANGER_IDS = {396, 397, 398, 399, 400, 426, 436}
HELP_IDS = {23, 14, 354, 355, 356, 359}
WARNING_IDS = {288, 364, 388, 389, 390, 439, 391, 392, 393, 395, 440, 441, 443, 456, 469, 470, 478, 479}

# Raspberry Pi 최적화 설정
# 메모리 사용량 최적화
TORCH_NUM_THREADS = 2  # Raspberry Pi에 적합한 스레드 수

# 분리 강도 조정 (Raspberry Pi 최적화)
MASK_THRESHOLD = 0.3        # Raspberry Pi에 적합한 임계값
MASK_SOFTNESS = 0.1         # 부드러운 전환을 위한 범위
USE_HARD_THRESHOLD = False  # 부드러운 threshold 사용

# Raspberry Pi용 threshold 설정
THRESHOLD_PRESETS = {
    "conservative": 0.4,    # 보수적 분리
    "balanced": 0.3,        # 균형잡힌 분리 (기본값)
    "aggressive": 0.2,      # 공격적 분리
    "very_aggressive": 0.1  # 매우 공격적 분리
}
CURRENT_THRESHOLD_PRESET = "balanced"  # Raspberry Pi 기본값

# 잔여물 증폭 설정 (Raspberry Pi 최적화)
RESIDUAL_AMPLIFY = True     # 잔여물 증폭 활성화
RESIDUAL_GAIN = 1.5         # Raspberry Pi에 적합한 증폭 배수
RESIDUAL_MAX_GAIN = 3.0     # 최대 증폭 배수 제한

# Backend API
USER_ID = 6
BACKEND_URL = "http://13.238.200.232:8000/sound-events/"

# =========================
# Utility Functions (separator.py에서 가져옴)
# =========================
def norm01(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def ensure_odd(k: int) -> int:
    return k + 1 if (k % 2 == 0) else k

def smooth1d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    ker = torch.ones(k, device=x.device) / k
    return F.conv1d(x.view(1,1,-1), ker.view(1,1,-1), padding=k//2).view(-1)

def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def align_len_1d(x: torch.Tensor, target_len: int, device: torch.device, mode: str = "linear") -> torch.Tensor:
    """1D 텐서의 길이를 목표 길이에 맞춤 (separator.py 최신 버전)"""
    if x.shape[0] == target_len:
        return x
    
    if mode == "linear":
        return F.interpolate(x.view(1, 1, -1), size=target_len, mode="linear", align_corners=False).view(-1)
    elif mode == "nearest":
        return F.interpolate(x.view(1, 1, -1), size=target_len, mode="nearest").view(-1)
    else:
        return F.interpolate(x.view(1, 1, -1), size=target_len, mode="linear", align_corners=False).view(-1)

def soft_sigmoid(x: torch.Tensor, center: float, slope: float, min_val: float = 0.0) -> torch.Tensor:
    sig = torch.sigmoid(slope * (x - center))
    return min_val + (1.0 - min_val) * sig

def amplify_residual(residual: np.ndarray, gain: float = 2.0, max_gain: float = 4.0) -> np.ndarray:
    """잔여물 증폭 (클리핑 방지)"""
    try:
        # 현재 RMS 계산
        current_rms = np.sqrt(np.mean(residual ** 2))
        if current_rms < 1e-8:  # 너무 작은 경우
            return residual
        
        # 증폭 적용
        amplified = residual * gain
        
        # 클리핑 방지 (최대 증폭 제한)
        max_amplified_rms = current_rms * max_gain
        current_amplified_rms = np.sqrt(np.mean(amplified ** 2))
        
        if current_amplified_rms > max_amplified_rms:
            # 최대 증폭으로 제한
            amplified = amplified * (max_amplified_rms / current_amplified_rms)
        
        # -1.0 ~ 1.0 범위로 클리핑
        amplified = np.clip(amplified, -1.0, 1.0)
        
        return amplified
        
    except Exception as e:
        print(f"[Separator] Residual amplification error: {e}")
        return residual


class SoundSeparator:
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", 
                 device: str = "auto", backend_url: str = BACKEND_URL, led_controller=None):
        """
        Sound Separator 초기화
        
        Args:
            model_name: AST 모델 이름
            device: 사용할 디바이스 (auto/cpu/cuda)
            backend_url: 백엔드 API URL
            led_controller: LED 컨트롤러 인스턴스
        """
        self.model_name = model_name
        self.backend_url = backend_url
        self.led_controller = led_controller
        
        # Device 설정 (Raspberry Pi 최적화)
        if device == "auto":
            # Raspberry Pi에서는 CPU 사용
            self.device = torch.device("cpu")
            print("[Separator] Raspberry Pi detected - using CPU")
        else:
            self.device = torch.device(device)
        
        self.extractor = None
        self.ast_model = None
        self.mel_fb_m2f = None
        self.is_available = False
        
        # 분리 관련 캐시 (Raspberry Pi 최적화 - 메모리 제한)
        self.attention_cache = {}
        self.freq_attention_cache = {}
        self.cls_head_cache = {}
        self.spectrogram_cache = {}
        
        # Raspberry Pi 최적화 설정
        torch.set_num_threads(TORCH_NUM_THREADS)
        
        print("[Separator] 🚀 Starting Sound Separator initialization...")
        self._initialize_model()
        
        if self.is_available:
            print("[Separator] ✅ Sound Separator ready for use")
        else:
            print("[Separator] ❌ Sound Separator initialization failed - separation will be disabled")
    
    def _initialize_model(self):
        """AST 모델 초기화 (Raspberry Pi 최적화)"""
        print("[Separator] 🔍 Starting model initialization...")
        
        try:
            # 1. Transformers 라이브러리 확인
            print("[Separator] 🔍 Checking transformers library...")
            if ASTFeatureExtractor is None or ASTForAudioClassification is None:
                print("[Separator] ❌ Transformers not available - 실전 모드에서는 필수입니다!")
                self.is_available = False
                return
            print("[Separator] ✅ Transformers library available")
            
            # 2. PyTorch 및 torchaudio 확인
            print("[Separator] 🔍 Checking PyTorch and torchaudio...")
            print(f"[Separator] PyTorch version: {torch.__version__}")
            print(f"[Separator] torchaudio version: {torchaudio.__version__}")
            print(f"[Separator] CUDA available: {torch.cuda.is_available()}")
            print("[Separator] ✅ PyTorch and torchaudio available")
            
            # 3. 디바이스 설정 확인
            print(f"[Separator] 🔍 Setting up device: {self.device}")
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Separator] Final device: {self.device}")
            
            # 4. 스레드 설정
            print(f"[Separator] 🔍 Setting threads: {TORCH_NUM_THREADS}")
            torch.set_num_threads(TORCH_NUM_THREADS)
            
            # 5. 메모리 정리
            print("[Separator] 🔍 Cleaning memory...")
            import gc
            gc.collect()
            
            # 6. AST Feature Extractor 로딩
            print(f"[Separator] 🔍 Loading AST Feature Extractor: {self.model_name}")
            self.extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
            print("[Separator] ✅ AST Feature Extractor loaded")
            
            # 7. AST Model 로딩
            print(f"[Separator] 🔍 Loading AST Model: {self.model_name}")
            self.ast_model = ASTForAudioClassification.from_pretrained(
                self.model_name,
                attn_implementation="eager"  # Raspberry Pi 호환성
            )
            print("[Separator] ✅ AST Model loaded from pretrained")
            
            # 8. 디바이스로 이동
            print(f"[Separator] 🔍 Moving model to device: {self.device}")
            self.ast_model = self.ast_model.to(self.device)
            self.ast_model.eval()
            print("[Separator] ✅ Model moved to device and set to eval mode")
            
            # 9. Mel filterbank 생성
            print("[Separator] 🔍 Creating Mel filterbank...")
            fbins = N_FFT//2 + 1
            mel_fb_f2m = torchaudio.functional.melscale_fbanks(
                n_freqs=fbins, f_min=0.0, f_max=SR/2, n_mels=N_MELS,
                sample_rate=SR, norm="slaney"
            )
            self.mel_fb_m2f = mel_fb_f2m.T.contiguous()
            print("[Separator] ✅ Mel filterbank created")
            
            # 10. 최종 메모리 정리
            print("[Separator] 🔍 Final memory cleanup...")
            gc.collect()
            
            # 11. 성공 확인
            self.is_available = True
            print("[Separator] ✅ AST model loaded successfully (Raspberry Pi optimized)")
            print(f"[Separator] Model device: {next(self.ast_model.parameters()).device}")
            print(f"[Separator] Model dtype: {next(self.ast_model.parameters()).dtype}")
            
        except Exception as e:
            print(f"[Separator] ❌ Model loading error: {e}")
            print("[Separator] 실전 모드에서는 모델 로딩이 필수입니다!")
            import traceback
            print("[Separator] Full error traceback:")
            traceback.print_exc()
            self.is_available = False
    
    def _get_sound_type(self, class_id: int) -> str:
        """클래스 ID를 소리 타입으로 변환"""
        if class_id in DANGER_IDS:
            return "danger"
        elif class_id in HELP_IDS:
            return "help"
        elif class_id in WARNING_IDS:
            return "warning"
        else:
            return "other"
    
    def _calculate_decibel(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """dB 레벨 계산"""
        try:
            # 오디오 데이터 검증
            if len(audio) == 0:
                return -np.inf, -np.inf, -np.inf
            
            # RMS 계산
            rms = np.sqrt(np.mean(audio**2))
            
            if rms <= 0:
                return -np.inf, -np.inf, -np.inf
            
            # dB 변환 (20 * log10(rms))
            db = 20 * np.log10(rms)
            
            # 유효한 dB 값인지 확인
            if np.isnan(db) or np.isinf(db):
                return -np.inf, -np.inf, -np.inf
            
            # min, max dB 계산 (간단한 방법)
            audio_abs = np.abs(audio)
            audio_abs = audio_abs[audio_abs > 1e-10]  # 매우 작은 값 제외
            
            if len(audio_abs) > 0:
                db_min = 20 * np.log10(np.min(audio_abs))
                db_max = 20 * np.log10(np.max(audio_abs))
            else:
                db_min = db_max = db
            
            return db_min, db_max, db
            
        except Exception as e:
            print(f"[Separator] dB calculation error: {e}")
            return -np.inf, -np.inf, -np.inf
    
    def _prepare_audio_for_classification(self, audio_raw: np.ndarray) -> np.ndarray:
        """
        분류용 정규화된 오디오 데이터 준비
        
        Args:
            audio_raw: 원본 int16 오디오 데이터
            
        Returns:
            정규화된 float32 오디오 데이터
        """
        try:
            # int16을 float32로 정규화 (-1.0 ~ 1.0 범위)
            audio_normalized = audio_raw.astype(np.float32) / 32767.0
            
            # 10초로 패딩
            target_len = int(10.0 * SR)
            if len(audio_normalized) < target_len:
                audio_padded = np.zeros(target_len, dtype=np.float32)
                audio_padded[:len(audio_normalized)] = audio_normalized
                return audio_padded
            else:
                return audio_normalized[:target_len]
                
        except Exception as e:
            print(f"[Separator] Error preparing audio for classification: {e}")
            return audio_raw.astype(np.float32) / 32767.0
    
    def _calculate_decibel_from_raw(self, audio_raw: np.ndarray) -> Tuple[float, float, float]:
        """
        Sound Trigger의 _calculate_db_level과 완전히 동일한 방법으로 데시벨 계산
        
        Args:
            audio_raw: 원본 int16 오디오 데이터 (모노)
            
        Returns:
            (db_min, db_max, db_mean)
        """
        try:
            if len(audio_raw) == 0:
                return -np.inf, -np.inf, -np.inf
            
            # Sound Trigger의 _calculate_db_level과 완전히 동일한 로직
            # 모노 데이터이므로 channels <= 1 조건에 해당
            audio_data = audio_raw.astype(np.float32)
            
            # RMS 계산 (Sound Trigger와 동일)
            rms = np.sqrt(np.mean(audio_data**2))
            
            if rms == 0:
                return -np.inf, -np.inf, -np.inf
            
            # dB 변환 (20 * log10(rms)) - Sound Trigger와 동일
            if rms > 0:
                db = 20 * np.log10(rms)
                
                # 유효한 dB 값인지 확인 (Sound Trigger와 동일)
                if np.isnan(db) or np.isinf(db):
                    return -np.inf, -np.inf, -np.inf
                
                # min, max dB 계산 (간단한 방법)
                audio_abs = np.abs(audio_data)
                audio_abs = audio_abs[audio_abs > 1e-10]  # 매우 작은 값 제외
                
                if len(audio_abs) > 0:
                    db_min = 20 * np.log10(np.min(audio_abs))
                    db_max = 20 * np.log10(np.max(audio_abs))
                    
                    # 유효성 검사
                    if np.isnan(db_min) or np.isinf(db_min):
                        db_min = db
                    if np.isnan(db_max) or np.isinf(db_max):
                        db_max = db
                else:
                    db_min = db_max = db
                
                return db_min, db_max, db
            else:
                return -np.inf, -np.inf, -np.inf
            
        except Exception as e:
            print(f"[Separator] Raw dB calculation error: {e}")
            import traceback
            traceback.print_exc()
            return -np.inf, -np.inf, -np.inf
    
    def _calculate_decibel_simple(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """라즈베리 파이 호환 dB 계산 방법"""
        try:
            if len(audio) == 0:
                return -np.inf, -np.inf, -np.inf
            
            # 안전한 자료형 변환 (라즈베리 파이 호환)
            if audio.dtype == np.int16:
                # int16을 float64로 변환하여 정밀도 향상
                audio_float = audio.astype(np.float64)
            elif audio.dtype == np.float32:
                audio_float = audio.astype(np.float64)
            elif audio.dtype == np.float64:
                audio_float = audio.copy()
            else:
                audio_float = audio.astype(np.float64)
            
            # 데이터 검증
            if np.all(audio_float == 0):
                return -np.inf, -np.inf, -np.inf
            
            # RMS 계산 (Sound Trigger와 동일한 방식)
            rms = np.sqrt(np.mean(audio_float**2))
            
            if rms <= 0:
                return -np.inf, -np.inf, -np.inf
            
            # dB 변환 (20 * log10(rms)) - Sound Trigger와 동일
            db = 20 * np.log10(rms)
            
            # 유효한 dB 값인지 확인
            if np.isnan(db) or np.isinf(db):
                return -np.inf, -np.inf, -np.inf
            
            # min, max dB 계산 (안전한 방법)
            audio_abs = np.abs(audio_float)
            # 0이 아닌 값들만 사용
            non_zero_mask = audio_abs > 1e-10
            audio_abs_nonzero = audio_abs[non_zero_mask]
            
            if len(audio_abs_nonzero) > 0:
                db_min = 20 * np.log10(np.min(audio_abs_nonzero))
                db_max = 20 * np.log10(np.max(audio_abs_nonzero))
                
                # 유효성 검사
                if np.isnan(db_min) or np.isinf(db_min):
                    db_min = db
                if np.isnan(db_max) or np.isinf(db_max):
                    db_max = db
            else:
                db_min = db_max = db
            
            # 최종 검증
            if np.isnan(db_min) or np.isinf(db_min) or np.isnan(db_max) or np.isinf(db_max) or np.isnan(db) or np.isinf(db):
                return -np.inf, -np.inf, -np.inf
            
            return db_min, db_max, db
            
        except Exception as e:
            print(f"[Separator] Simple dB calculation error: {e}")
            import traceback
            traceback.print_exc()
            return -np.inf, -np.inf, -np.inf
    
    def _send_to_backend(self, sound_type: str, sound_detail: str, decibel: float, angle: int) -> bool:
        """
        Send results to backend (including angle information)
        
        Args:
            sound_type: Sound type (danger/warning/help/other)
            sound_detail: Sound detail information
            decibel: dB level
            angle: Angle (0-359)
            
        Returns:
            Transmission success status
        """
        try:
            data = {
                "user_id": USER_ID,
                "sound_type": sound_type,
                "sound_detail": sound_detail,
                "angle": angle,  # Include angle information
                "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sound_icon": "string",
                "location_image_url": "string",
                "decibel": float(decibel),
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'SoundPipeline/1.0'
            }
            
            print(f"🔄 Sending to backend: {self.backend_url}")
            print(f"📤 Data: {data}")
            
            # Disable SSL warnings for testing
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            response = requests.post(
                self.backend_url, 
                json=data, 
                headers=headers,
                timeout=10.0,  # Increased timeout
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
            print(f"❌ Backend connection timeout: {self.backend_url}")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Backend connection error: {e}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Backend request error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error sending to backend: {e}")
            return False
    
    def _load_fixed_audio(self, path: str) -> np.ndarray:
        """오디오 파일 로드 - Sound Trigger와 동일한 방식"""
        try:
            import wave
            
            # Sound Trigger와 동일한 방식으로 WAV 파일 읽기
            with wave.open(path, 'rb') as wav_file:
                # WAV 파일 정보 확인
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Sound Trigger와 동일한 방식으로 데이터 읽기
                raw_audio = wav_file.readframes(n_frames)
                
                # Sound Trigger와 동일한 int16 변환
                if sample_width == 2:  # 16-bit
                    # Sound Trigger와 동일: np.frombuffer 사용
                    audio_data = np.frombuffer(raw_audio, dtype=np.int16)
                elif sample_width == 1:  # 8-bit
                    audio_data = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.int16) - 128
                else:
                    print(f"[Separator] Warning: Unsupported sample width: {sample_width}")
                    return np.zeros(L_FIXED, dtype=np.int16)
            
            # 데이터 검증
            if len(audio_data) == 0:
                print(f"[Separator] Warning: Empty audio data from {path}")
                return np.zeros(L_FIXED, dtype=np.int16)
            
            # 0 데이터 검증
            if np.all(audio_data == 0):
                print(f"[Separator] Warning: All audio data is zero from {path}")
                return np.zeros(L_FIXED, dtype=np.int16)
            
            # Sound Trigger와 동일한 모노 변환 방식
            if channels > 1:
                    # Sound Trigger의 _to_mono_int16과 동일한 로직
                    usable_len = (len(audio_data) // channels) * channels
                    if usable_len != len(audio_data):
                        audio_data = audio_data[:usable_len]
                    x = audio_data.reshape(-1, channels)
                    
                    # 채널 5가 있으면 그 채널만 사용 (Sound Trigger와 동일)
                    if channels >= 6:
                        mono = x[:, 5].astype(np.int16)
                    else:
                        # 일반 마이크 채널 평균 (가능하면 앞쪽 4채널만 평균)
                        mic_cols = min(channels, 4)
                        mono = np.mean(x[:, :mic_cols], axis=1).astype(np.int16)
                    
                    audio_data = mono
                
            # 샘플링 레이트 변환 (간단한 리샘플링)
            if framerate != SR:
                # 간단한 리샘플링 (선형 보간)
                ratio = SR / framerate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data.astype(np.float64)
                ).astype(np.int16)
                
            
            # 고정 길이로 조정
            if len(audio_data) >= L_FIXED:
                return audio_data[:L_FIXED]
            else:
                out = np.zeros(L_FIXED, dtype=np.int16)
                out[:len(audio_data)] = audio_data
                return out
                
        except Exception as e:
            print(f"[Separator] Error loading audio {path}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(L_FIXED, dtype=np.int16)
    
    def _classify_audio(self, audio_normalized: np.ndarray) -> Tuple[str, str, int, float]:
        """
        오디오 분류 (실전용) - 정규화된 데이터 사용
        
        Args:
            audio_normalized: 정규화된 float32 오디오 데이터 (-1.0 ~ 1.0)
            
        Returns:
            (class_name, sound_type, class_id, confidence)
        """
        if not self.is_available:
            print("[Separator] ❌ Model not available - 실전 모드에서는 모델이 필수입니다!")
            return "Unknown", "other", 0, 0.0
        
        try:
            # 이미 정규화된 데이터를 사용
            audio_float = audio_normalized.astype(np.float32)
            
            # 10초로 패딩 (이미 _prepare_audio_for_classification에서 처리됨)
            target_len = int(10.0 * SR)
            if len(audio_float) < target_len:
                audio_padded = np.zeros(target_len, dtype=np.float32)
                audio_padded[:len(audio_float)] = audio_float
            else:
                audio_padded = audio_float[:target_len]
            
            feat = self.extractor(audio_padded, sampling_rate=SR, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.ast_model(input_values=feat["input_values"].to(self.device))
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_id = logits.argmax(dim=-1).item()
                confidence = probabilities[0, predicted_class_id].item()
            
            class_name = self.ast_model.config.id2label[predicted_class_id]
            sound_type = self._get_sound_type(predicted_class_id)
            
            return class_name, sound_type, predicted_class_id, confidence
            
        except Exception as e:
            print(f"[Separator] ❌ Classification error: {e}")
            return "Unknown", "other", 0, 0.0
    
    def _save_separated_audio(self, audio: np.ndarray, class_name: str, sound_type: str, output_dir: str, suffix: str = "") -> str:
        """
        분리된 오디오를 파일로 저장
        
        Args:
            audio: 오디오 데이터 (int16 또는 float32)
            class_name: 분류된 클래스 이름
            sound_type: 소리 타입
            output_dir: 저장 디렉토리
            
        Returns:
            저장된 파일 경로
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일명 생성 (타임스탬프 + 클래스명 + 타입)
            import time
            timestamp = int(time.time())
            safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_class_name = safe_class_name.replace(' ', '_')
            
            filename = f"separated_{timestamp}_{safe_class_name}_{sound_type}{suffix}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # int16 데이터를 float32로 변환하여 저장
            if audio.dtype == np.int16:
                audio_float = audio.astype(np.float32) / 32767.0  # -1.0 ~ 1.0 범위로 정규화
            else:
                audio_float = audio.astype(np.float32)
            
            # 오디오 저장
            torchaudio.save(filepath, torch.from_numpy(audio_float).unsqueeze(0), SR)
            
            print(f"[Separator] Separated audio saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"[Separator] Error saving separated audio: {e}")
            return None
    
    def process_audio(self, audio_file: str, angle: int, output_dir: str = None) -> Dict[str, Any]:
        """
        오디오 파일 처리 및 분류
        
        Args:
            audio_file: 오디오 파일 경로
            angle: 각도 (0-359)
            output_dir: 분리된 소리 저장 디렉토리
            
        Returns:
            처리 결과 딕셔너리
        """
        print(f"[Separator] Processing audio: {audio_file}")
        print(f"[Separator] Angle: {angle}°")
        
        try:
            # 오디오 로드 (원본 raw 데이터)
            audio_raw = self._load_fixed_audio(audio_file)
            print(f"[Separator] Audio length: {len(audio_raw)/SR:.2f}s")
            
            # 분류용 정규화된 오디오 생성
            audio_normalized = self._prepare_audio_for_classification(audio_raw)
            
            # 분류 (정규화된 데이터 사용)
            class_name, sound_type, class_id, confidence = self._classify_audio(audio_normalized)
            
            # dB 계산 (원본 raw 데이터 사용 - Sound Trigger와 동일한 방법)
            db_min, db_max, db_mean = self._calculate_decibel_from_raw(audio_raw)
            
            print(f"[Separator] Classified: {class_name} ({sound_type})")
            print(f"[Separator] Confidence: {confidence:.3f}")
            print(f"[Separator] Decibel: {db_mean:.1f} dB")
            
            # 음원 분리 실행 (separator.py 최신 버전)
            separated_sources = []
            separated_file = None
            if output_dir:
                if self.is_available:
                    print(f"[Separator] Starting source separation...")
                    
                    # 콜백 함수 정의 (각 패스 완료 시마다 백엔드 전송 + LED 제어)
                    def on_pass_complete(source_info):
                        """각 패스 완료 시마다 백엔드 전송 + LED 제어"""
                        if source_info['sound_type'] != "other":
                            print(f"[Separator] Sending separated source to backend: {source_info['class_name']} ({source_info['sound_type']})")
                            backend_success = self._send_to_backend(
                                source_info['sound_type'], 
                                source_info['class_name'], 
                                source_info.get('db_mean', db_mean), 
                                angle
                            )
                            if backend_success:
                                print(f"[Separator] ✅ Backend transmission successful for {source_info['class_name']}")
                            else:
                                print(f"[Separator] ❌ Backend transmission failed for {source_info['class_name']}")
                            
                            # LED 제어 (각도 기반)
                            if self.led_controller:
                                print(f"[Separator] Activating directional LED for {source_info['class_name']} at {angle}°")
                                led_success = self.led_controller.activate_led(
                                    angle, 
                                    source_info['class_name'], 
                                    source_info['sound_type']
                                )
                                if led_success:
                                    print(f"[Separator] ✅ LED activated for {source_info['class_name']} at {angle}°")
                                else:
                                    print(f"[Separator] ❌ LED activation failed for {source_info['class_name']}")
                            else:
                                print(f"[Separator] No LED controller available for {source_info['class_name']}")
                    
                    separated_sources = self.separate_audio(audio_normalized, max_passes=MAX_PASSES, on_pass_complete=on_pass_complete)
                    
                    # 분리된 소리들을 파일로 저장
                    for i, source in enumerate(separated_sources):
                        if source['audio'] is not None:
                            source_file = self._save_separated_audio(
                                source['audio'], 
                                source['class_name'], 
                                source['sound_type'], 
                                output_dir,
                                suffix=f"_pass_{source['pass']}"
                            )
                            source['file'] = source_file
                            print(f"[Separator] Separated source {i+1}: {source['class_name']} ({source['sound_type']}) - {source['confidence']:.3f}")
                    
                    # 첫 번째 분리된 소리를 기본 separated_file로 설정
                    if separated_sources:
                        separated_file = separated_sources[0]['file']
                else:
                    # 분리 불가능한 경우 원본 데이터 저장
                    separated_file = self._save_separated_audio(audio_raw, class_name, sound_type, output_dir)
            
            # 백엔드 전송 (other 타입 제외) - 분리된 소리들은 이미 전송됨
            backend_success = False
            if sound_type != "other" and not separated_sources:
                # 분리된 소리가 없는 경우에만 원본 분류 결과 전송
                backend_success = self._send_to_backend(sound_type, class_name, db_mean, angle)
            elif sound_type == "other":
                print(f"[Separator] Skipping backend send for 'other' type: {class_name}")
                backend_success = True
            else:
                # 분리된 소리들이 이미 전송되었으므로 성공으로 처리
                backend_success = True
            
            # 결과 반환
            result = {
                "success": True,
                "class_name": class_name,
                "sound_type": sound_type,
                "class_id": class_id,
                "confidence": confidence,
                "angle": angle,
                "decibel": {
                    "min": db_min,
                    "max": db_max,
                    "mean": db_mean
                },
                "backend_success": backend_success,
                "audio_file": audio_file,
                "separated_file": separated_file,
                "separated_sources": separated_sources,  # 새로운 필드: 분리된 소리들
                "separation_enabled": self.is_available
            }
            
            return result
            
        except Exception as e:
            print(f"[Separator] Processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_file": audio_file,
                "angle": angle
            }
    
    # =========================
    # Source Separation Logic (separator.py에서 가져옴)
    # =========================
    
    def _get_cache_key(self, audio: np.ndarray) -> str:
        """캐시 키 생성"""
        return str(hash(audio.tobytes()))
    
    def _extract_and_cache_attention(self, audio: np.ndarray, T_out: int, F_out: int, anchor_region: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, int, float, List[Tuple[str, float, int]]]:
        """AST 모델 호출하여 시간/주파수 attention map과 CLS features 추출 및 캐싱 (separator.py 최신 버전)"""
        cache_key = self._get_cache_key(audio)
        
        # 캐시 확인
        if cache_key in self.attention_cache and cache_key in self.cls_head_cache:
            freq_attn = self.freq_attention_cache.get(cache_key)
            if freq_attn is not None:
                # 캐시된 분류 정보도 반환
                cached_class_info = self.cls_head_cache.get(cache_key + "_class_info")
                if cached_class_info:
                    return self.attention_cache[cache_key], freq_attn, self.cls_head_cache[cache_key], *cached_class_info
        
        # int16 데이터를 float32로 변환
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
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
        
        feat = self.extractor(audio_for_classification, sampling_rate=SR, return_tensors="pt")
        
        # Mel 스펙트로그램 추출 (캐싱용)
        mel_spec = feat["input_values"].squeeze(0)  # [N_MELS, T]
        
        with torch.no_grad():
            outputs = self.ast_model(input_values=feat["input_values"].to(self.device), output_attentions=True, return_dict=True)
        
        # 분류 결과 추출
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = logits.argmax(dim=-1).item()
        confidence = probabilities[0, predicted_class_id].item()
        
        class_name = self.ast_model.config.id2label[predicted_class_id]
        sound_type = self._get_sound_type(predicted_class_id)
        
        # Top 5 클래스 추출
        top5_probs, top5_indices = torch.topk(probabilities[0], 5)
        top5_classes = []
        for i in range(5):
            class_id = top5_indices[i].item()
            class_name_top5 = self.ast_model.config.id2label[class_id]
            prob = top5_probs[i].item()
            top5_classes.append((class_name_top5, prob, class_id))
        
        # Attention map 추출
        attns = outputs.attentions
        if not attns or len(attns) == 0:
            time_attention = torch.ones(T_out) * 0.5
            freq_attention = torch.ones(F_out) * 0.5
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
            time_attention = norm01(time_attn_smooth)
            
            # 주파수 어텐션 (시간 차원으로 평균) - 크롭된 맵 사용
            freq_attn_mel = full_map_cropped.mean(dim=1)  # [12] - Mel 스케일
            freq_attn_interp = F.interpolate(freq_attn_mel.view(1,1,-1), size=F_out, mode="linear", align_corners=False).view(-1)
            freq_attention = norm01(freq_attn_interp)
        
        # CLS features 추출
        if hasattr(outputs, 'last_hidden_state'):
            cls_features = outputs.last_hidden_state[:, 0, :]  # CLS token features
        else:
            cls_features = outputs.logits
        
        # 캐싱
        self.attention_cache[cache_key] = time_attention
        self.freq_attention_cache[cache_key] = freq_attention
        self.cls_head_cache[cache_key] = cls_features
        self.spectrogram_cache[cache_key] = mel_spec.clone()
        
        # 분류 정보도 캐싱
        class_info = (class_name, sound_type, predicted_class_id, confidence, top5_classes)
        self.cls_head_cache[cache_key + "_class_info"] = class_info
        
        return time_attention, freq_attention, cls_features, class_name, sound_type, predicted_class_id, confidence, top5_classes
    
    def _stft_all(self, audio: np.ndarray, mel_fb_m2f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """STFT 변환 및 Mel 스펙트로그램 생성 (separator.py 최신 버전)"""
        # int16 데이터를 float32로 변환 (STFT 요구사항)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        wav = torch.from_numpy(audio)
        st = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN,
                        window=WINDOW, return_complex=True, center=True)
        mag = st.abs()
        P = (mag * mag).clamp_min(EPS)
        phase = torch.angle(st)

        if mel_fb_m2f.shape[0] != N_MELS:
            mel_fb_m2f = mel_fb_m2f.T.contiguous()
        else:
            mel_fb_m2f = mel_fb_m2f
        assert mel_fb_m2f.shape[0] == N_MELS and mel_fb_m2f.shape[1] == P.shape[0]
        mel_fb_m2f = mel_fb_m2f.to(P.dtype).to(P.device)
        mel_pow = (mel_fb_m2f @ P).clamp_min(EPS)
        return st, mag, P, phase, mel_pow
    
    def _purity_from_P(self, P: torch.Tensor) -> torch.Tensor:
        """순수도 계산"""
        fbins, T = P.shape
        e = P.sum(dim=0); e_n = e / (e.max() + EPS)
        p = P / (P.sum(dim=0, keepdim=True) + EPS)
        H = -(p * (p + EPS).log()).sum(dim=0)
        Hn = H / np.log(max(2, fbins))
        pur = W_E * e_n + (1.0 - W_E) * (1.0 - Hn)
        return norm01(smooth1d(pur, SMOOTH_T))
    
    def _anchor_score(self, A_t: torch.Tensor, Pur: torch.Tensor) -> torch.Tensor:
        """앵커 스코어 계산"""
        return norm01(smooth1d((A_t.clamp(0,1)**ALPHA_ATT) * (Pur.clamp(0,1)**BETA_PUR), SMOOTH_T))
    
    def _pick_anchor_region(self, score: torch.Tensor, La: int, core_pct: float, P: torch.Tensor) -> Tuple[int, int, int, int]:
        """앵커 영역 선택"""
        T = score.numel()
        
        # 전체 스펙트로그램의 에너지 계산
        total_energy = P.sum(dim=0)  # [T]
        energy_threshold = torch.quantile(total_energy, 0.1)  # 하위 10% 에너지 임계값
        
        # 에너지가 너무 낮은 구간은 앵커 후보에서 제외
        valid_regions = total_energy > energy_threshold
        
        # 유효한 구간에서만 앵커 선택
        if valid_regions.sum() == 0:
            peak_idx = int(torch.argmax(score).item())
        else:
            valid_score = score.clone()
            valid_score[~valid_regions] = -float('inf')
            peak_idx = int(torch.argmax(valid_score).item())
        
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
    
    def _omega_support_with_ast_freq(self, Ablk: torch.Tensor, ast_freq_attn: torch.Tensor, strategy: str = "conservative") -> torch.Tensor:
        """주파수 지원 영역 계산 (separator.py 최신 버전)"""
        omega_q = OMEGA_Q_CONSERVATIVE
        ast_freq_quantile = AST_FREQ_QUANTILE_CONSERVATIVE
        
        med = Ablk.median(dim=1).values
        th = torch.quantile(med, omega_q)
        mask_energy = (med >= th).float()
        
        ast_freq_th = torch.quantile(ast_freq_attn, ast_freq_quantile)
        mask_ast_freq = (ast_freq_attn >= ast_freq_th).float()
        
        mask = torch.maximum(mask_energy, mask_ast_freq)
        
        # Dilation (2 iterations)
        for _ in range(2):
            mask = torch.maximum(mask, torch.roll(mask, 1))
            mask = torch.maximum(mask, torch.roll(mask, -1))
        
        if int(mask.sum().item()) < OMEGA_MIN_BINS:
            order = torch.argsort(med, descending=True)
            need = OMEGA_MIN_BINS - int(mask.sum().item())
            take = order[:need]
            mask[take] = 1.0
        
        return mask
    
    def _template_from_anchor_block(self, Ablk: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """앵커 블록에서 템플릿 생성 (separator.py 최신 버전)"""
        om = omega.view(-1,1)
        w = (Ablk * om).mean(dim=1) * omega
        w = w / (w.sum() + EPS)
        w_sm = F.avg_pool1d(w.view(1,1,-1), kernel_size=3, stride=1, padding=1).view(-1)
        w = (w_sm * omega); w = w / (w.sum() + EPS)
        return w
    
    def _presence_from_energy(self, Xmel: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """에너지 기반 존재감 계산 (separator.py 최신 버전)"""
        om = omega.view(-1,1)
        e_omega = (Xmel * om).sum(dim=0)
        e_omega = smooth1d(e_omega, 9)  # PRES_SMOOTH_T = 9
        thr = torch.quantile(e_omega, 0.20)  # PRES_Q = 0.20
        thr = torch.clamp(thr, min=1e-10)
        return (e_omega > thr).float()
    
    def _amplitude_raw(self, Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """원시 진폭 계산 (separator.py 최신 버전)"""
        om = omega.view(-1,1)
        Xo = Xmel * om
        denom = (w_bar*w_bar).sum() + EPS
        a_raw = (w_bar.view(1,-1) @ Xo).view(-1) / denom
        return a_raw.clamp_min(0.0)
    
    def _cos_similarity_over_omega(self, Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, g_pres: torch.Tensor):
        """오메가 영역에서 코사인 유사도 계산 (separator.py 최신 버전)"""
        om = omega.view(-1,1)
        Xo = Xmel * om
        wn = (w_bar * omega); wn = wn / (wn.norm(p=2) + 1e-8)
        Xn = Xo / (Xo.norm(p=2, dim=0, keepdim=True) + 1e-8)
        cos_raw = (wn.view(-1,1) * Xn).sum(dim=0).clamp(0,1)
        return cos_raw * g_pres
    
    def _purity_from_P(self, P: torch.Tensor) -> torch.Tensor:
        """순수도 계산 (separator.py 최신 버전)"""
        fbins, T = P.shape
        e = P.sum(dim=0); e_n = e / (e.max() + EPS)
        p = P / (P.sum(dim=0, keepdim=True) + EPS)
        H = -(p * (p + EPS).log()).sum(dim=0)
        Hn = H / np.log(max(2, fbins))
        pur = W_E * e_n + (1.0 - W_E) * (1.0 - Hn)
        return norm01(smooth1d(pur, SMOOTH_T))
    
    def _anchor_score(self, A_t: torch.Tensor, Pur: torch.Tensor) -> torch.Tensor:
        """앵커 스코어 계산 (separator.py 최신 버전)"""
        return norm01(smooth1d((A_t.clamp(0,1)**ALPHA_ATT) * (Pur.clamp(0,1)**BETA_PUR), SMOOTH_T))
    
    def _pick_anchor_region(self, score: torch.Tensor, La: int, core_pct: float, P: torch.Tensor) -> Tuple[int, int, int, int]:
        """앵커 영역 선택 (separator.py 최신 버전)"""
        T = score.numel()
        
        # 전체 스펙트로그램의 에너지 계산
        total_energy = P.sum(dim=0)  # [T]
        energy_threshold = torch.quantile(total_energy, 0.1)  # 하위 10% 에너지 임계값
        
        # 에너지가 너무 낮은 구간은 앵커 후보에서 제외
        valid_regions = total_energy > energy_threshold
        
        # 유효한 구간에서만 앵커 선택
        if valid_regions.sum() == 0:
            peak_idx = int(torch.argmax(score).item())
        else:
            valid_score = score.clone()
            valid_score[~valid_regions] = -float('inf')
            peak_idx = int(torch.argmax(valid_score).item())
        
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
    
    def _unified_masking_strategy(self, Xmel: torch.Tensor, w_bar: torch.Tensor, omega: torch.Tensor, 
                                 ast_freq_attn: torch.Tensor, P: torch.Tensor, s: int, e: int, strategy: str = "conservative") -> torch.Tensor:
        """통합 마스킹 전략 (separator.py 최신 버전)"""
        fbins, T = P.shape
        
        # Calculate cosΩ, the core of our mask
        g_pres = self._presence_from_energy(Xmel, omega)
        cos_t_raw = self._cos_similarity_over_omega(Xmel, w_bar, omega, g_pres)

        # Map Ω(mel)->Ω(linear) for frequency weighting
        if omega.shape[0] == self.mel_fb_m2f.shape[0]:
            omega_lin = ((self.mel_fb_m2f @ omega).clamp_min(0.0) > 1e-12).float()
        else:
            omega_lin = torch.ones(self.mel_fb_m2f.shape[1], device=omega.device)
        
        # Enhanced Masking Logic
        anchor_spec = P[:, s:e]
        anchor_energy = anchor_spec.sum().item()
        total_energy = P.sum().item()
        energy_ratio = anchor_energy / (total_energy + 1e-8)
        
        is_weak_sound = energy_ratio < 0.1
        
        # 기본 마스크: 코사인 유사도 제곱으로 약화
        cos_squared = cos_t_raw ** 2
        
        # Threshold 기반 마스킹 (sigmoid 대신)
        base_threshold = THRESHOLD_PRESETS.get(CURRENT_THRESHOLD_PRESET, MASK_THRESHOLD)
        
        if is_weak_sound:
            # 약한 소리의 경우 더 낮은 임계값 사용
            threshold = base_threshold * 0.8
            softness = MASK_SOFTNESS * 1.2
        else:
            # 일반 소리의 경우 기본 임계값 사용
            threshold = base_threshold
            softness = MASK_SOFTNESS
        
        # Threshold 기반 마스킹
        if USE_HARD_THRESHOLD:
            # 완전한 하드 threshold (0 또는 1)
            soft_time_mask = (cos_squared >= threshold).float()
            mask_type = "HARD"
        else:
            # 부드러운 threshold 마스킹
            # threshold ± softness 범위에서 선형 보간
            mask_input = (cos_squared - (threshold - softness)) / (2 * softness)
            soft_time_mask = torch.clamp(mask_input, 0.0, 1.0)
            mask_type = "SOFT"
        
        # 디버그 정보 출력 (Raspberry Pi 최적화 - 간소화)
        mask_mean = soft_time_mask.mean().item()
        cos_squared_mean = cos_squared.mean().item()
        print(f"[Separator] {mask_type} masking ({CURRENT_THRESHOLD_PRESET}) - Threshold: {threshold:.3f}, Cos² mean: {cos_squared_mean:.3f}, Mask mean: {mask_mean:.3f}")
        
        # 앵커 영역의 진폭 주파수 선택
        anchor_max_amp = anchor_spec.max(dim=1).values
        
        if is_weak_sound:
            amp_threshold = torch.quantile(anchor_max_amp, 0.6)
        else:
            amp_threshold = torch.quantile(anchor_max_amp, 0.7)
        
        high_amp_mask_lin = (anchor_max_amp >= amp_threshold).float()
        
        # 앵커 영역에서 활성화된 AST 주파수 선택
        anchor_ast_freq = ast_freq_attn.clone()
        
        if is_weak_sound:
            ast_freq_threshold = torch.quantile(anchor_ast_freq, 0.5)
        else:
            ast_freq_threshold = torch.quantile(anchor_ast_freq, 0.4)
        
        ast_active_mask_mel = (anchor_ast_freq >= ast_freq_threshold).float()
        
        # AST 주파수 마스크를 Mel에서 Linear 도메인으로 변환
        if ast_active_mask_mel.shape[0] == self.mel_fb_m2f.shape[0]:
            ast_active_mask_lin = ((self.mel_fb_m2f @ ast_active_mask_mel).clamp_min(0.0) > 0.2).float()
        else:
            ast_active_mask_lin = torch.ones(self.mel_fb_m2f.shape[1], device=ast_freq_attn.device)
        
        # 선택된 주파수 영역 결합
        if high_amp_mask_lin.shape[0] != ast_active_mask_lin.shape[0]:
            min_size = min(high_amp_mask_lin.shape[0], ast_active_mask_lin.shape[0])
            high_amp_mask_lin = high_amp_mask_lin[:min_size]
            ast_active_mask_lin = ast_active_mask_lin[:min_size]
        
        freq_boost_mask = torch.maximum(high_amp_mask_lin, ast_active_mask_lin)
        
        # 가중치 적용 (Raspberry Pi 최적화 - 단순화)
        if is_weak_sound:
            freq_weight = 1.0 + 0.6 * freq_boost_mask  # 감소된 가중치
        else:
            freq_weight = 1.0 + 0.4 * freq_boost_mask  # 감소된 가중치
        
        # 기본 마스크 계산
        M_base = omega_lin.view(-1, 1) * soft_time_mask.view(1, -1)
        
        # 주파수 가중치 적용
        M_weighted = M_base * freq_weight.view(-1, 1)
        
        # 마스크가 실제 스펙트로그램보다 크지 않도록 제한
        spec_magnitude = P.sqrt()
        
        if M_weighted.shape[0] != spec_magnitude.shape[0]:
            min_freq = min(M_weighted.shape[0], spec_magnitude.shape[0])
            M_weighted = M_weighted[:min_freq, :]
            spec_magnitude = spec_magnitude[:min_freq, :]
        
        M_lin = torch.minimum(M_weighted, spec_magnitude)
        
        # 마스크 강도 조정 (Raspberry Pi 최적화)
        if is_weak_sound:
            M_lin = torch.minimum(M_lin, spec_magnitude * 0.9)  # 더 보수적
        else:
            M_lin = torch.minimum(M_lin, spec_magnitude * 0.7)  # 더 보수적
        
        return M_lin
    
    def _single_pass_separation(self, audio: np.ndarray, mel_fb_m2f: torch.Tensor, used_mask_prev: Optional[torch.Tensor],
                               prev_anchors: List[Tuple[float,float,torch.Tensor,torch.Tensor,torch.Tensor]],
                               pass_idx: int, prev_energy_ratio: float = 1.0,
                               separated_time_regions: List[dict] = None,
                               previous_anchors: List[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, float, Optional[torch.Tensor], Dict[str, Any]]:
        """단일 패스 분리 (separator.py 최신 버전)"""
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
        st, mag, P, phase, Xmel = self._stft_all(audio, mel_fb_m2f)
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

        # 3. AST attention 추출 (앵커 영역 없이)
        time_attention, ast_freq_attn, _, class_name, sound_type, class_id, confidence, top5_classes = self._extract_and_cache_attention(audio_for_ast, T, N_MELS)
        
        A_t = time_attention
        Pur = self._purity_from_P(P)
        Sc = self._anchor_score(A_t, Pur)

        # 4. 이전에 사용된 프레임 억제
        if used_mask_prev is not None:
            um = align_len_1d(used_mask_prev, T, device=Sc.device, mode="linear")
            k = int(round((80/1000.0)*SR/HOP)); k = ensure_odd(max(1,k))  # 80ms dilation
            ker = torch.ones(k, device=Sc.device)/k
            um = (F.conv1d(um.view(1,1,-1), ker.view(1,1,-1), padding=k//2).view(-1) > 0.2).float()
            Sc = Sc * (1 - 0.85 * um)

        # 5. 이전 앵커들의 강화된 억제
        for (sa, ea, prev_w, prev_omega, prev_anchor_score) in prev_anchors:
            ca = int(((sa+ea)/2) * SR / HOP)
            ca = max(0, min(T-1, ca))
            sigma = int(round((200/1000.0)*SR/HOP))  # 200ms suppression
            idx = torch.arange(T, device=Sc.device) - ca
            Sc = Sc * (1 - 0.6 * torch.exp(-(idx**2)/(2*(sigma**2)+1e-8)))
            core_s = max(0, ca - La//2); core_e = min(T, ca + La//2)
            Sc[core_s:core_e] *= 0.2
        
        # 6. 앵커 영역 선택
        s, e, core_s_rel, core_e_rel = self._pick_anchor_region(Sc, La, TOP_PCT_CORE_IN_ANCHOR, P)
        
        # 7. 앵커 블록 생성
        Ablk = Xmel[:, s:e].clone()
        if core_s_rel > 0:  Ablk[:, :core_s_rel] = 0
        if core_e_rel < La: Ablk[:, core_e_rel:] = 0

        # 8. Ω 계산
        omega = self._omega_support_with_ast_freq(Ablk, ast_freq_attn, "conservative")
        w_bar = self._template_from_anchor_block(Ablk, omega)
        
        # 9. 앵커 영역 기반 분류 (더 정확한 분류)
        anchor_region = (s, e)
        _, _, _, class_name_anchor, sound_type_anchor, class_id_anchor, confidence_anchor, top5_classes_anchor = self._extract_and_cache_attention(audio_for_ast, T, N_MELS, anchor_region)
        
        # 앵커 기반 분류가 더 높은 신뢰도를 가지면 사용
        if confidence_anchor > confidence:
            class_name = class_name_anchor
            sound_type = sound_type_anchor
            class_id = class_id_anchor
            confidence = confidence_anchor
            top5_classes = top5_classes_anchor
            print(f"  🎯 Using anchor-based classification: {class_name} (Conf: {confidence:.3f})")
        
        # 10. 통합 마스킹 전략 적용
        M_lin = self._unified_masking_strategy(Xmel, w_bar, omega, ast_freq_attn, P, s, e, "conservative")
        
        # 11. STFT 도메인에서 분리
        stft_full = st
        
        # 마스크 적용
        if M_lin.shape[0] != mag.shape[0]:
            min_freq = min(M_lin.shape[0], mag.shape[0])
            M_lin = M_lin[:min_freq, :]
            mag = mag[:min_freq, :]
            phase = phase[:min_freq, :]
        
        mag_linear = mag
        mag_masked_linear = M_lin * mag_linear
        
        stft_src = mag_masked_linear * torch.exp(1j * phase)
        
        # 잔여물 계산
        mag_residual_linear = mag_linear - mag_masked_linear
        mag_residual_linear = torch.maximum(mag_residual_linear, torch.zeros_like(mag_residual_linear))
        stft_res = mag_residual_linear * torch.exp(1j * phase)
        
        # 12. 에너지 검증 및 정규화
        src_energy = torch.sum(torch.abs(stft_src)**2).item()
        res_energy = torch.sum(torch.abs(stft_res)**2).item()
        orig_energy = torch.sum(torch.abs(stft_full)**2).item()
        total_energy = src_energy + res_energy
        
        energy_ratio = total_energy / (orig_energy + 1e-8)
        if energy_ratio > 1.05:
            scale_factor = orig_energy / (total_energy + 1e-8)
            scale_tensor = torch.tensor(scale_factor, device=stft_src.device, dtype=stft_src.dtype)
            stft_src = stft_src * torch.sqrt(scale_tensor)
            stft_res = stft_res * torch.sqrt(scale_tensor)
        elif energy_ratio < 0.95:
            scale_factor = orig_energy / (total_energy + 1e-8)
            scale_tensor = torch.tensor(scale_factor, device=stft_src.device, dtype=stft_src.dtype)
            stft_src = stft_src * torch.sqrt(scale_tensor)
            stft_res = stft_res * torch.sqrt(scale_tensor)

        # 13. 시간 도메인으로 재구성
        if stft_src.shape[0] != N_FFT//2 + 1:
            target_freq = N_FFT//2 + 1
            if stft_src.shape[0] < target_freq:
                pad_size = target_freq - stft_src.shape[0]
                stft_src = F.pad(stft_src, (0, 0, 0, pad_size), mode='constant', value=0)
                stft_res = F.pad(stft_res, (0, 0, 0, pad_size), mode='constant', value=0)
            else:
                stft_src = stft_src[:target_freq, :]
                stft_res = stft_res[:target_freq, :]
        
        src_amp = torch.istft(stft_src, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, 
                             window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()
        res = torch.istft(stft_res, n_fft=N_FFT, hop_length=HOP, win_length=WINLEN, 
                          window=WINDOW, center=True, length=L_FIXED).detach().cpu().numpy()

        # 14. ER 계산
        e_src = float(np.sum(src_amp**2)); e_res = float(np.sum(res**2))
        er = e_src / (e_src + e_res + 1e-12)

        # 15. 시간 마스크 생성 (다음 패스용)
        if M_lin.shape[0] != P.shape[0]:
            min_freq = min(M_lin.shape[0], P.shape[0])
            M_lin = M_lin[:min_freq, :]
            P = P[:min_freq, :]
        
        r_t = (M_lin * P).sum(dim=0) / (P.sum(dim=0) + 1e-8)
        used_mask = (r_t >= 0.65).float()  # USED_THRESHOLD
        
        # 16. 시간 마스크 인덱스 추출
        src_time_indices = torch.where(used_mask > 0.5)[0]
        src_time_mask = used_mask

        # 17. dB 계산
        db_min, db_max, db_mean = self._calculate_decibel_simple(src_amp)

        elapsed = time.time() - t0
        
        info = {
            "src_amp": src_amp,
            "res": res,
            "er": er,
            "class_name": class_name,
            "sound_type": sound_type,
            "class_id": class_id,
            "confidence": confidence,
            "elapsed": elapsed,
            "separation_skipped": False,
            "src_time_mask": src_time_mask,
            "src_time_indices": src_time_indices,
            "anchor_region": (s, e),  # 앵커 구간 정보 추가
            "anchor_score": Sc,  # 현재 패스의 anchor score 추가
            "db_min": db_min,
            "db_max": db_max,
            "db_mean": db_mean,
            "amplification_factor": amplification_factor
        }
        
        return src_amp, res, er, used_mask, info
    
    def separate_audio(self, audio: np.ndarray, max_passes: int = MAX_PASSES, on_pass_complete=None) -> List[Dict[str, Any]]:
        """오디오 분리 실행 (separator.py 최신 버전)"""
        if not self.is_available:
            print("[Separator] ❌ Model not available for separation")
            return []
        
        print(f"[Separator] Starting audio separation with {max_passes} passes...")
        
        # Mel filterbank 생성
        fbins = N_FFT//2 + 1
        mel_fb_f2m = torchaudio.functional.melscale_fbanks(
            n_freqs=fbins, f_min=0.0, f_max=SR/2, n_mels=N_MELS,
            sample_rate=SR, norm="slaney"
        )
        mel_fb_m2f = mel_fb_f2m.T.contiguous()
        
        current_audio = audio.copy()
        used_mask_prev = None
        prev_anchors = []
        sources = []
        prev_energy_ratio = 1.0
        separated_time_regions = []  # 이전에 분리된 시간대 정보 저장
        previous_anchors = []  # 이전 패스에서 사용된 앵커 구간 정보
        
        for pass_idx in range(max_passes):
            print(f"[Separator] --- Pass {pass_idx + 1} ---")
            
            # 분리 실행
            src_amp, res, er, used_mask, info = self._single_pass_separation(
                current_audio, mel_fb_m2f, used_mask_prev, prev_anchors, pass_idx, prev_energy_ratio,
                separated_time_regions, previous_anchors
            )
            
            # 분리 결과 디버깅
            if src_amp is not None:
                src_rms = np.sqrt(np.mean(src_amp ** 2))
                res_rms = np.sqrt(np.mean(res ** 2))
                print(f"[Separator] Pass {pass_idx + 1} - Source RMS: {src_rms:.6f}, Residual RMS: {res_rms:.6f}, Energy ratio: {er:.6f}")
                print(f"[Separator] Pass {pass_idx + 1} - Class: {info['class_name']} ({info['sound_type']}), Confidence: {info['confidence']:.3f}")
            else:
                print(f"[Separator] Pass {pass_idx + 1} - Source is None!")
            
            # 결과 저장
            source_info = {
                "pass": pass_idx + 1,
                "class_name": info['class_name'],
                "sound_type": info['sound_type'],
                "confidence": info['confidence'],
                "energy_ratio": er,
                "anchor": info['anchor_region'],
                "audio": src_amp,
                "db_mean": info['db_mean'],
                "amplification_factor": info.get('amplification_factor', 1.0)
            }
            sources.append(source_info)
            
            # 콜백 함수 호출 (각 패스 완료 시마다)
            if on_pass_complete:
                on_pass_complete(source_info)
            
            # 분리된 시간대 정보 수집
            if not info.get("separation_skipped", False):
                separated_time_regions.append({
                    'time_mask': info['src_time_mask'],
                    'class_name': info['class_name'],
                    'confidence': info['confidence'],
                    'pass_idx': pass_idx
                })
                
                # 앵커 구간 정보 수집
                if 'anchor_region' in info and 'anchor_score' in info:
                    prev_anchors.append((
                        info['anchor_region'][0],  # prev_s
                        info['anchor_region'][1],  # prev_e
                        info['src_time_mask'],     # prev_mask
                        torch.ones_like(info['src_time_mask']),  # prev_weight (기본값)
                        info['anchor_score']       # prev_anchor_score
                    ))
            
            # 잔여물을 다음 패스의 입력으로 사용 (증폭 적용)
            if RESIDUAL_AMPLIFY and pass_idx < max_passes - 1:  # 마지막 패스가 아닌 경우만
                # 잔여물 증폭
                current_audio = amplify_residual(res, RESIDUAL_GAIN, RESIDUAL_MAX_GAIN)
                print(f"[Separator] Residual amplified by {RESIDUAL_GAIN}x for next pass")
            else:
                current_audio = res
            
            used_mask_prev = used_mask
            prev_energy_ratio = er
            
            # 조기 종료 조건
            residual_energy = float(np.sum(res**2))
            if er < MIN_ERATIO and residual_energy < 0.001:  # 잔여물 에너지도 낮을 때만 종료
                print(f"[Separator] Early stop: Energy ratio {er:.3f} < {MIN_ERATIO} and residual energy {residual_energy:.6f} too low")
                break
            elif er < MIN_ERATIO:
                print(f"[Separator] Energy ratio {er:.3f} < {MIN_ERATIO}, but residual energy {residual_energy:.6f} is sufficient, continuing...")
        
        # 최종 잔여물 분류
        if len(current_audio) > 0 and np.max(np.abs(current_audio)) > 1e-6:
            print(f"[Separator] --- Final Residual Classification ---")
            # 간단한 분류 (캐시된 정보 사용)
            cache_key = self._get_cache_key(current_audio)
            cached_class_info = self.cls_head_cache.get(cache_key + "_class_info")
            
            if cached_class_info:
                class_name, sound_type, class_id, confidence, top5_classes = cached_class_info
            else:
                # 새로운 분류 수행
                _, _, _, class_name, sound_type, class_id, confidence, top5_classes = self._extract_and_cache_attention(current_audio, 100, N_MELS)
            
            print(f"[Separator] Residual: {class_name} ({sound_type}) - Confidence: {confidence:.3f}")
            
            if confidence >= RESIDUAL_CONFIDENCE_THRESHOLD:
                print(f"[Separator] High confidence residual detected, saving as sound...")
                residual_info = {
                    "pass": len(sources) + 1,
                    "class_name": class_name,
                    "sound_type": sound_type,
                    "confidence": confidence,
                    "energy_ratio": 0.0,
                    "anchor": (0, 0),
                    "audio": current_audio,
                    "db_mean": self._calculate_decibel_simple(current_audio)[2],
                    "amplification_factor": 1.0
                }
                sources.append(residual_info)
                
                if on_pass_complete:
                    on_pass_complete(residual_info)
        
        print(f"[Separator] Separation completed. Found {len(sources)} sources.")
        return sources
    
    def is_model_available(self) -> bool:
        """모델 사용 가능 여부 확인"""
        return self.is_available
    
    def cleanup(self):
        """리소스 정리 (Raspberry Pi 최적화)"""
        # 캐시 정리
        self.attention_cache.clear()
        self.freq_attention_cache.clear()
        self.cls_head_cache.clear()
        self.spectrogram_cache.clear()
        
        # 메모리 정리
        import gc
        gc.collect()
        
        # GPU 메모리 정리 (만약 GPU를 사용했다면)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()




def create_sound_separator(model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", 
                          device: str = "auto", backend_url: str = BACKEND_URL, led_controller=None) -> SoundSeparator:
    """
    Sound Separator 인스턴스 생성 (실전용)
    
    Args:
        model_name: AST 모델 이름
        device: 사용할 디바이스
        backend_url: 백엔드 API URL
        led_controller: LED 컨트롤러 인스턴스
        
    Returns:
        SoundSeparator 인스턴스
    """
    return SoundSeparator(model_name, device, backend_url, led_controller)


def main():
    """실전용 메인 함수"""
    print("🎵 Sound Separator - 실전 모드")
    print("=" * 50)
    print("이 모듈은 sound_pipeline.py에서 사용됩니다.")
    print("직접 실행하지 마세요.")
    print("=" * 50)


if __name__ == "__main__":
    main()
