#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Separator Module
- separator.py 기반으로 구현
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
torch.set_num_threads(4)

try:
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
except ImportError:
    print("Warning: transformers not available. Sound separation will be disabled.")
    ASTFeatureExtractor = None
    ASTForAudioClassification = None

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

MASK_SIGMOID_CENTER = 0.6
MASK_SIGMOID_SLOPE = 20.0

N_FFT, HOP, WINLEN = 400, 160, 400
WINDOW = torch.hann_window(WINLEN)
EPS = 1e-10

N_MELS = 128

SMOOTH_T = 19
ALPHA_ATT = 0.60
BETA_PUR = 1.50
W_E = 0.40
TOP_PCT_CORE_IN_ANCHOR = 0.50

OMEGA_Q_CONSERVATIVE = 0.9
OMEGA_Q_AGGRESSIVE = 0.7
OMEGA_DIL = 2
OMEGA_MIN_BINS = 5

AST_FREQ_QUANTILE_CONSERVATIVE = 0.7
AST_FREQ_QUANTILE_AGGRESSIVE = 0.4

DANGER_IDS = {0,396, 397, 398, 399, 400, 426, 436}
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


class SoundSeparator:
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", 
                 device: str = "auto", backend_url: str = BACKEND_URL):
        """
        Sound Separator 초기화
        
        Args:
            model_name: AST 모델 이름
            device: 사용할 디바이스 (auto/cpu/cuda)
            backend_url: 백엔드 API URL
        """
        self.model_name = model_name
        self.backend_url = backend_url
        
        # Device 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.extractor = None
        self.ast_model = None
        self.mel_fb_m2f = None
        self.is_available = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """AST 모델 초기화 (실전용)"""
        try:
            if ASTFeatureExtractor is None or ASTForAudioClassification is None:
                print("[Separator] ❌ Transformers not available - 실전 모드에서는 필수입니다!")
                self.is_available = False
                return
            
            print(f"[Separator] Loading AST model: {self.model_name}")
            print(f"[Separator] Device: {self.device}")
            
            self.extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
            self.ast_model = ASTForAudioClassification.from_pretrained(self.model_name).to(self.device)
            self.ast_model.eval()
            
            # Mel filterbank 생성
            self.mel_fb_m2f = torchaudio.transforms.MelScale(n_mels=N_MELS, sample_rate=SR, n_stft=N_FFT//2+1).fb
            
            self.is_available = True
            print("[Separator] ✅ AST model loaded successfully")
            
        except Exception as e:
            print(f"[Separator] ❌ Model loading error: {e}")
            print("[Separator] 실전 모드에서는 모델 로딩이 필수입니다!")
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
                print(f"[Separator] Debug: Empty raw audio data")
                return -np.inf, -np.inf, -np.inf
            
            # 디버그: 원본 오디오 데이터 정보
            print(f"[Separator] Debug: Raw audio range: {audio_raw.min()} to {audio_raw.max()}")
            print(f"[Separator] Debug: Raw audio mean: {audio_raw.mean():.1f}, std: {audio_raw.std():.1f}")
            print(f"[Separator] Debug: Raw audio dtype: {audio_raw.dtype}")
            
            # Sound Trigger의 _calculate_db_level과 완전히 동일한 로직
            # 모노 데이터이므로 channels <= 1 조건에 해당
            audio_data = audio_raw.astype(np.float32)
            
            # RMS 계산 (Sound Trigger와 동일)
            rms = np.sqrt(np.mean(audio_data**2))
            print(f"[Separator] Debug: RMS: {rms:.6f}")
            
            if rms == 0:
                print(f"[Separator] Debug: RMS is zero")
                return -np.inf, -np.inf, -np.inf
            
            # dB 변환 (20 * log10(rms)) - Sound Trigger와 동일
            if rms > 0:
                db = 20 * np.log10(rms)
                print(f"[Separator] Debug: Calculated dB: {db:.3f}")
                
                # 유효한 dB 값인지 확인 (Sound Trigger와 동일)
                if np.isnan(db) or np.isinf(db):
                    print(f"[Separator] Debug: dB is NaN or inf: {db}")
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
                
                print(f"[Separator] Debug: dB range: {db_min:.1f} to {db_max:.1f} dB (mean: {db:.1f} dB)")
                
                return db_min, db_max, db
            else:
                print(f"[Separator] Debug: RMS is not positive: {rms}")
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
                print(f"[Separator] Debug: Empty audio data")
                return -np.inf, -np.inf, -np.inf
            
            # 디버그: 오디오 데이터 정보
            print(f"[Separator] Debug: Audio data range: {audio.min()} to {audio.max()}")
            print(f"[Separator] Debug: Audio data mean: {audio.mean():.1f}, std: {audio.std():.1f}")
            print(f"[Separator] Debug: Audio data type: {audio.dtype}")
            print(f"[Separator] Debug: Audio data shape: {audio.shape}")
            
            # 안전한 자료형 변환 (라즈베리 파이 호환)
            if audio.dtype == np.int16:
                # int16을 float64로 변환하여 정밀도 향상
                audio_float = audio.astype(np.float64)
                print(f"[Separator] Debug: Converted int16 to float64")
            elif audio.dtype == np.float32:
                audio_float = audio.astype(np.float64)
                print(f"[Separator] Debug: Converted float32 to float64")
            elif audio.dtype == np.float64:
                audio_float = audio.copy()
                print(f"[Separator] Debug: Using float64 directly")
            else:
                audio_float = audio.astype(np.float64)
                print(f"[Separator] Debug: Converted {audio.dtype} to float64")
            
            # 데이터 검증
            if np.all(audio_float == 0):
                print(f"[Separator] Debug: All audio data is zero")
                return -np.inf, -np.inf, -np.inf
            
            # RMS 계산 (Sound Trigger와 동일한 방식)
            rms = np.sqrt(np.mean(audio_float**2))
            print(f"[Separator] Debug: RMS: {rms:.6f}")
            
            if rms <= 0:
                print(f"[Separator] Debug: RMS is zero or negative: {rms}")
                return -np.inf, -np.inf, -np.inf
            
            # dB 변환 (20 * log10(rms)) - Sound Trigger와 동일
            db = 20 * np.log10(rms)
            print(f"[Separator] Debug: Calculated dB: {db:.3f}")
            
            # 유효한 dB 값인지 확인
            if np.isnan(db) or np.isinf(db):
                print(f"[Separator] Debug: dB is NaN or inf: {db}")
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
            
            print(f"[Separator] Debug: dB range: {db_min:.1f} to {db_max:.1f} dB (mean: {db:.1f} dB)")
            
            # 최종 검증
            if np.isnan(db_min) or np.isinf(db_min) or np.isnan(db_max) or np.isinf(db_max) or np.isnan(db) or np.isinf(db):
                print(f"[Separator] Debug: Final validation failed - returning -inf")
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
            
            print(f"[Separator] Debug: Loading audio file: {path}")
            
            # Sound Trigger와 동일한 방식으로 WAV 파일 읽기
            with wave.open(path, 'rb') as wav_file:
                # WAV 파일 정보 확인
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                print(f"[Separator] Debug: WAV file info - channels: {channels}, sample_width: {sample_width}, framerate: {framerate}, n_frames: {n_frames}")
                
                # Sound Trigger와 동일한 방식으로 데이터 읽기
                raw_audio = wav_file.readframes(n_frames)
                print(f"[Separator] Debug: Raw audio length: {len(raw_audio)} bytes")
                
                # Sound Trigger와 동일한 int16 변환
                if sample_width == 2:  # 16-bit
                    # Sound Trigger와 동일: np.frombuffer 사용
                    audio_data = np.frombuffer(raw_audio, dtype=np.int16)
                    print(f"[Separator] Debug: Converted to int16 array, length: {len(audio_data)}")
                    print(f"[Separator] Debug: Raw audio data range: {audio_data.min()} to {audio_data.max()}")
                    print(f"[Separator] Debug: Raw audio data mean: {audio_data.mean():.1f}, std: {audio_data.std():.1f}")
                    print(f"[Separator] Debug: Non-zero samples: {np.count_nonzero(audio_data)} / {len(audio_data)}")
                elif sample_width == 1:  # 8-bit
                    audio_data = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.int16) - 128
                    print(f"[Separator] Debug: Converted 8-bit to int16, length: {len(audio_data)}")
                    print(f"[Separator] Debug: Raw audio data range: {audio_data.min()} to {audio_data.max()}")
                    print(f"[Separator] Debug: Raw audio data mean: {audio_data.mean():.1f}, std: {audio_data.std():.1f}")
                    print(f"[Separator] Debug: Non-zero samples: {np.count_nonzero(audio_data)} / {len(audio_data)}")
                else:
                    print(f"[Separator] Warning: Unsupported sample width: {sample_width}")
                    return np.zeros(L_FIXED, dtype=np.int16)
                
                # 데이터 검증
                if len(audio_data) == 0:
                    print(f"[Separator] Warning: Empty audio data from {path}")
                    return np.zeros(L_FIXED, dtype=np.int16)
                
                # 0 데이터 검증
                if np.all(audio_data == 0):
                    print(f"[Separator] ❌ CRITICAL: All audio data is zero!")
                    print(f"[Separator] Debug: This indicates a problem with the WAV file or loading process")
                    print(f"[Separator] Debug: File: {path}")
                    print(f"[Separator] Debug: Channels: {channels}, Sample width: {sample_width}, Framerate: {framerate}")
                    print(f"[Separator] Debug: Raw audio bytes: {len(raw_audio)}")
                    print(f"[Separator] Debug: First 20 bytes: {raw_audio[:20]}")
                    print(f"[Separator] Debug: Last 20 bytes: {raw_audio[-20:]}")
                    
                    # 파일 크기 확인
                    import os
                    file_size = os.path.getsize(path)
                    print(f"[Separator] Debug: File size: {file_size} bytes")
                    
                    # 0이 아닌 데이터가 있는지 확인
                    non_zero_count = np.count_nonzero(audio_data)
                    print(f"[Separator] Debug: Non-zero samples: {non_zero_count} / {len(audio_data)}")
                    
                    return np.zeros(L_FIXED, dtype=np.int16)
                
                # Sound Trigger와 동일한 모노 변환 방식
                if channels > 1:
                    print(f"[Separator] Debug: Converting {channels} channels to mono")
                    # Sound Trigger의 _to_mono_int16과 동일한 로직
                    usable_len = (len(audio_data) // channels) * channels
                    if usable_len != len(audio_data):
                        print(f"[Separator] Debug: Truncating audio data from {len(audio_data)} to {usable_len}")
                        audio_data = audio_data[:usable_len]
                    x = audio_data.reshape(-1, channels)
                    print(f"[Separator] Debug: Reshaped to {x.shape}")
                    
                    # 채널 5가 있으면 그 채널만 사용 (Sound Trigger와 동일)
                    if channels >= 6:
                        print(f"[Separator] Debug: Using channel 5 (post-processed)")
                        mono = x[:, 5].astype(np.int16)
                    else:
                        # 일반 마이크 채널 평균 (가능하면 앞쪽 4채널만 평균)
                        mic_cols = min(channels, 4)
                        print(f"[Separator] Debug: Averaging first {mic_cols} channels")
                        mono = np.mean(x[:, :mic_cols], axis=1).astype(np.int16)
                    
                    audio_data = mono
                    print(f"[Separator] Debug: Converted to mono using Sound Trigger method, new length: {len(audio_data)}")
                    print(f"[Separator] Debug: Mono audio range: {audio_data.min()} to {audio_data.max()}")
                    print(f"[Separator] Debug: Mono audio mean: {audio_data.mean():.1f}, std: {audio_data.std():.1f}")
                    print(f"[Separator] Debug: Mono non-zero samples: {np.count_nonzero(audio_data)} / {len(audio_data)}")
                
                # 샘플링 레이트 변환 (간단한 리샘플링)
                if framerate != SR:
                    print(f"[Separator] Debug: Resampling from {framerate}Hz to {SR}Hz")
                    print(f"[Separator] Debug: Before resampling - range: {audio_data.min()} to {audio_data.max()}")
                    print(f"[Separator] Debug: Before resampling - non-zero: {np.count_nonzero(audio_data)} / {len(audio_data)}")
                    
                    # 간단한 리샘플링 (선형 보간)
                    ratio = SR / framerate
                    new_length = int(len(audio_data) * ratio)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), new_length),
                        np.arange(len(audio_data)),
                        audio_data.astype(np.float64)
                    ).astype(np.int16)
                    
                    print(f"[Separator] Debug: Resampled to length: {len(audio_data)}")
                    print(f"[Separator] Debug: After resampling - range: {audio_data.min()} to {audio_data.max()}")
                    print(f"[Separator] Debug: After resampling - non-zero: {np.count_nonzero(audio_data)} / {len(audio_data)}")
                
                # 디버그: 오디오 데이터 범위 확인
                print(f"[Separator] Debug: Final audio range: {audio_data.min()} to {audio_data.max()}")
                print(f"[Separator] Debug: Final audio mean: {audio_data.mean():.1f}, std: {audio_data.std():.1f}")
                print(f"[Separator] Debug: Final audio length: {len(audio_data)} samples ({len(audio_data)/SR:.2f}s)")
                print(f"[Separator] Debug: Final audio dtype: {audio_data.dtype}")
                
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
    
    def _save_separated_audio(self, audio: np.ndarray, class_name: str, sound_type: str, output_dir: str) -> str:
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
            
            filename = f"separated_{timestamp}_{safe_class_name}_{sound_type}.wav"
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
            
            # 분리된 소리 저장 (원본 데이터 사용)
            separated_file = None
            if output_dir:
                separated_file = self._save_separated_audio(audio_raw, class_name, sound_type, output_dir)
            
            # 백엔드 전송 (other 타입 제외)
            backend_success = False
            if sound_type != "other":
                backend_success = self._send_to_backend(sound_type, class_name, db_mean, angle)
            else:
                print(f"[Separator] Skipping backend send for 'other' type: {class_name}")
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
                "separated_file": separated_file
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
    
    def is_model_available(self) -> bool:
        """모델 사용 가능 여부 확인"""
        return self.is_available
    
    def cleanup(self):
        """리소스 정리"""
        # PyTorch 모델은 자동으로 정리됨
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()




def create_sound_separator(model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", 
                          device: str = "auto", backend_url: str = BACKEND_URL) -> SoundSeparator:
    """
    Sound Separator 인스턴스 생성 (실전용)
    
    Args:
        model_name: AST 모델 이름
        device: 사용할 디바이스
        backend_url: 백엔드 API URL
        
    Returns:
        SoundSeparator 인스턴스
    """
    return SoundSeparator(model_name, device, backend_url)


def main():
    """실전용 메인 함수"""
    print("🎵 Sound Separator - 실전 모드")
    print("=" * 50)
    print("이 모듈은 sound_pipeline.py에서 사용됩니다.")
    print("직접 실행하지 마세요.")
    print("=" * 50)


if __name__ == "__main__":
    main()
