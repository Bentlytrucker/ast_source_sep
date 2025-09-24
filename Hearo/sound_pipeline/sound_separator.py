#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Separator Module for Raspberry Pi
- legacv_sep.py 기반으로 구현
- separator.py의 검증된 분리 로직 활용
- 차원 불일치 문제 해결
- Raspberry Pi 환경에 최적화
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

# separator.py의 핵심 함수들 import
try:
    # 상대 경로로 separator.py import
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from separator import (
        single_pass, ast_attention_freq_time_cached, classify_audio_segment,
        load_fixed_audio, norm01, presence_from_energy, cos_similarity_over_omega,
        adaptive_masking_strategy, adaptive_strategy_selection, stft_all,
        calculate_sound_occurrence_time
    )
    SEPARATOR_AVAILABLE = True
    print("[Separator] ✅ separator.py functions imported successfully")
except ImportError as e:
    print(f"[Separator] ⚠️ Could not import separator.py functions: {e}")
    SEPARATOR_AVAILABLE = False

# =========================
# Global Constants (legacv_sep.py와 동일)
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

OMEGA_Q_CONSERVATIVE = 0.9
OMEGA_Q_AGGRESSIVE = 0.7
OMEGA_DIL = 2
OMEGA_MIN_BINS = 5

AST_FREQ_QUANTILE_CONSERVATIVE = 0.7
AST_FREQ_QUANTILE_AGGRESSIVE = 0.4

DANGER_IDS = {396, 397, 398, 399, 400, 426, 436}
HELP_IDS = {23, 14, 354, 355, 356, 359}
WARNING_IDS = {0,288, 364, 388, 389, 390, 439, 391, 392, 393, 395, 440, 441, 443, 456, 469, 470, 478, 479}

PRES_Q = 0.20
PRES_SMOOTH_T = 9

USED_THRESHOLD = 0.65
USED_DILATE_MS = 80
ANCHOR_SUPPRESS_MS = 200
ANCHOR_SUPPRESS_BASE = 0.6

MAX_PASSES = 3
MIN_ERATIO = 0.005

# Backend API
USER_ID = 6
BACKEND_URL = "http://13.238.200.232:8000/sound-events/"

# =========================
# Utility Functions (legacv_sep.py에서 가져옴)
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

def amplify_residual(residual: np.ndarray, gain: float = 2.0, max_gain: float = 4.0) -> np.ndarray:
    """잔여물 증폭 (클리핑 방지)"""
    try:
        current_rms = np.sqrt(np.mean(residual ** 2))
        if current_rms < 1e-8:
            return residual
        
        amplified = residual * gain
        max_amplified_rms = current_rms * max_gain
        current_amplified_rms = np.sqrt(np.mean(amplified ** 2))
        
        if current_amplified_rms > max_amplified_rms:
            amplified = amplified * (max_amplified_rms / current_amplified_rms)
        
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
            led_controller: LED 컨트롤러 (선택사항)
        """
        self.model_name = model_name
        self.backend_url = backend_url
        self.led_controller = led_controller
        
        # Device 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.extractor = None
        self.ast_model = None
        self.mel_fb_m2f = None
        self.is_available = False
        
        # 분리 관련 캐시
        self.attention_cache = {}
        self.freq_attention_cache = {}
        self.cls_head_cache = {}
        self.spectrogram_cache = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """AST 모델 초기화 (Raspberry Pi 최적화)"""
        print("[Separator] 🔍 Starting Sound Separator initialization...")
        
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
            print(f"[Separator] 🔍 Setting threads: 2")
            torch.set_num_threads(2)

            # 5. 메모리 정리
            print("[Separator] 🔍 Cleaning memory...")
            import gc
            gc.collect()
            
            # 6. AST Feature Extractor 로딩
            print(f"[Separator] 🔍 Loading AST Feature Extractor: {self.model_name}")
            self.extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
            print("[Separator] ✅ AST Feature Extractor loaded")
            
            # 7. AST Model 로딩 (구버전 transformers 호환성)
            print(f"[Separator] 🔍 Loading AST Model: {self.model_name}")
            try:
                # 최신 transformers 버전용
                self.ast_model = ASTForAudioClassification.from_pretrained(
                    self.model_name,
                    attn_implementation="eager"
                )
                print("[Separator] ✅ AST Model loaded with eager attention")
            except TypeError as e:
                if "attn_implementation" in str(e):
                    print("[Separator] 🔍 Trying without attn_implementation parameter...")
                    # 구버전 transformers용
                    self.ast_model = ASTForAudioClassification.from_pretrained(
                        self.model_name
                    )
                    print("[Separator] ✅ AST Model loaded without attn_implementation")
                else:
                    raise e
            
            # 8. 모델 양자화 (라즈베리파이 최적화)
            print("[Separator] 🔍 Applying model quantization...")
            try:
                # Dynamic quantization 적용
                self.ast_model = torch.quantization.quantize_dynamic(
                    self.ast_model, 
                    {torch.nn.Linear, torch.nn.Conv1d}, 
                    dtype=torch.qint8
                )
                print("[Separator] ✅ Model quantization completed")
            except Exception as e:
                print(f"[Separator] ⚠️ Quantization failed, using original model: {e}")
            
            # 9. 디바이스로 이동
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
            print(f"[Separator] ✅ Mel filterbank created: {self.mel_fb_m2f.shape}")
            print(f"[Separator] Expected: [N_MELS={N_MELS}, fbins={fbins}]")
            
            # 10. 최종 메모리 정리
            print("[Separator] 🔍 Final memory cleanup...")
            gc.collect()
            
            # 11. 성공 확인
            self.is_available = True
            print("[Separator] ✅ AST model loaded successfully (Raspberry Pi optimized)")
            print(f"[Separator] Model device: {next(self.ast_model.parameters()).device}")
            print(f"[Separator] Model dtype: {next(self.ast_model.parameters()).dtype}")
            print("[Separator] ✅ Sound Separator ready for use")
            
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
    
    def _calculate_decibel_from_raw(self, audio_raw: np.ndarray) -> Tuple[float, float, float]:
        """Sound Trigger와 동일한 방법으로 데시벨 계산"""
        try:
            if len(audio_raw) == 0:
                return -np.inf, -np.inf, -np.inf
            
            # Sound Trigger와 동일한 로직
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
    
    def _prepare_audio_for_classification(self, audio_raw: np.ndarray) -> np.ndarray:
        """분류용 정규화된 오디오 데이터 준비"""
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
    
    def _send_to_backend(self, sound_type: str, sound_detail: str, decibel: float, angle: int, occurred_at: str = None) -> bool:
        """백엔드로 결과 전송"""
        try:
            # 소리 발생시간이 제공되지 않으면 현재 시간 사용
            if occurred_at is None:
                occurred_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            
            data = {
                "user_id": USER_ID,
                "sound_type": sound_type,
                "sound_detail": sound_detail,
                "angle": angle,
                "occurred_at": occurred_at,
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
                timeout=10.0,
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
                
                # Channel 0만 사용 (ReSpeaker USB Mic Array의 후처리된 오디오)
                mono = x[:, 0].astype(np.int16)
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
        """오디오 분류 (실전용) - 정규화된 데이터 사용"""
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
        """분리된 오디오를 파일로 저장"""
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
    
    def separate_audio(self, audio: np.ndarray, angle: int, max_passes: int = 3, on_pass_complete=None) -> List[Dict[str, Any]]:
        """separator.py의 최신 멀티패스 음원 분리 로직을 직접 사용"""
        # separator.py의 상수/클래스 import (이미 상단에서 import했다고 가정)
        from separator import multi_pass_separation, ASTProcessor, SR, L_INPUT, L_MODEL

        # 오디오 길이 맞추기 (4.096초/10.24초)
        if len(audio) < L_INPUT:
            audio_4sec = np.pad(audio, (0, L_INPUT - len(audio)))
        else:
            audio_4sec = audio[:L_INPUT]
        if len(audio) < L_MODEL:
            audio_10sec = np.pad(audio, (0, L_MODEL - len(audio)))
        else:
            audio_10sec = audio[:L_MODEL]

        # ASTProcessor 인스턴스 생성 (CPU 고정)
        ast_processor = ASTProcessor()

        # separator.py의 멀티패스 분리 호출
        results = multi_pass_separation(audio_4sec, audio_10sec, ast_processor, max_passes=max_passes)

        # 결과 변환 및 후처리
        separated_sources = []
        for idx, result in enumerate(results):
            # dB 계산 (원하면 기존 방식 활용)
            db_min = db_max = db_mean = None
            try:
                db_min, db_max, db_mean = self._calculate_decibel_from_raw(result.separated_audio)
            except Exception:
                pass
            source_info = {
                'audio': result.separated_audio,
                'class_name': result.classification['predicted_class'],
                'sound_type': 'other',  # 필요시 분류
                'confidence': result.classification['confidence'],
                'class_id': -1,
                'pass': idx + 1,
                'db_min': db_min,
                'db_max': db_max,
                'db_mean': db_mean,
                'occurred_at': None,
                'separation_mask': result.mask
            }
            separated_sources.append(source_info)
            if on_pass_complete:
                on_pass_complete(source_info)
        return separated_sources
    
    def _separate_with_separator_py(self, audio: np.ndarray, max_passes: int = 3, on_pass_complete=None) -> List[Dict[str, Any]]:
        """
        separator.py의 single_pass 함수를 사용한 음원 분리
        - 매 패스마다 AST 모델 호출 (최대 3번)
        - 첫 번째 패스에서 silence 감지 시 즉시 종료
        - 같은 클래스 소리 생략 기능
        - 마지막 잔여소리 추론 제거
        """
        print(f"[Separator] Using separator.py logic for {max_passes} passes...")
        print(f"[Separator] 🧠 AST model will be called for each pass (max {max_passes} times)")
        
        separated_sources = []
        used_mask_prev = None
        prev_anchors = []
        prev_energy_ratio = 1.0
        separated_time_regions = []
        previous_anchors = []
        
        # 중복 클래스 추적
        detected_classes = set()
        silence_threshold = 0.0001  # silence 감지 임계값 (Speech가 Silence로 잘못 감지되지 않도록 낮춤)
        
        # mel filterbank 생성 (separator.py와 동일)
        fbins = N_FFT//2 + 1
        mel_fb_f2m = torchaudio.functional.melscale_fbanks(
            n_freqs=fbins, f_min=0.0, f_max=SR/2, n_mels=N_MELS,
            sample_rate=SR, norm="slaney"
        )
        mel_fb_m2f = mel_fb_f2m.T.contiguous()
        
        for pass_idx in range(max_passes):
            print(f"[Separator] --- Pass {pass_idx + 1} ---")
            
            try:
                # separator.py의 single_pass 함수 호출
                # mel_fb_m2f를 전달하되, separator.py 내부에서 차원 불일치 처리
                result = single_pass(
                    audio=audio,
                    extractor=self.extractor,
                    ast_model=self.ast_model,
                    mel_fb_m2f=mel_fb_m2f,  # mel filterbank 전달
                    used_mask_prev=used_mask_prev,
                    prev_anchors=prev_anchors,
                    pass_idx=pass_idx,
                    out_dir=None,  # 파일 저장은 별도로 처리
                    prev_energy_ratio=prev_energy_ratio,
                    separated_time_regions=separated_time_regions,
                    previous_anchors=previous_anchors
                )
                
                if result is None:
                    print(f"[Separator] Pass {pass_idx + 1} completed - no more sources to separate")
                    break
                
                # separator.py의 single_pass는 (src_amp, res, er, None, info) 튜플을 반환
                src_amp, res, er, _, info = result
                
                # 잔여 오디오 에너지 검사 - 의미없는 수준이면 패스 중단 (separator.py와 동일)
                residual_energy = np.sum(res ** 2)
                original_energy = np.sum(audio ** 2)
                residual_ratio = residual_energy / (original_energy + 1e-10)
                
                # 에너지가 너무 낮거나 신뢰도가 너무 낮으면 중단 (separator.py와 동일)
                if residual_ratio < 0.02 or info['confidence'] < 0.10:  # 잔여 에너지 2% 미만 또는 신뢰도 10% 미만
                    print(f"[Separator] Pass {pass_idx + 1}: Stopping separation - Residual energy: {residual_ratio:.3f}, Confidence: {info['confidence']:.3f}")
                    break
                
                # 분리된 오디오와 분류 정보 추출
                separated_audio = src_amp  # 분리된 오디오
                class_name = info.get('class_name', 'unknown')
                sound_type = info.get('sound_type', 'other')
                confidence = info.get('confidence', 0.0)
                class_id = info.get('class_id', -1)
                
                if separated_audio is not None:
                    # 무효한 앵커 검사 (silence 구간으로 인한 무효 앵커)
                    if info.get('invalid_anchor', False):
                        print(f"[Separator] Pass {pass_idx + 1}: Invalid anchor detected (likely silence region), stopping separation")
                        return []
                    
                    # Silence 감지 (RMS 기반) - 비활성화
                    # rms = np.sqrt(np.mean(separated_audio**2))
                    # if rms < silence_threshold:
                    #     if pass_idx == 0:
                    #         # 첫 번째 패스에서 silence 감지 시 즉시 종료
                    #         print(f"[Separator] Pass {pass_idx + 1}: Silence detected in first pass (RMS: {rms:.6f}), stopping separation immediately")
                    #         return []
                    #     else:
                    #         # 이후 패스에서 silence 감지 시 해당 패스만 건너뛰기
                    #         print(f"[Separator] Pass {pass_idx + 1}: Silence detected (RMS: {rms:.6f}), skipping this pass")
                    #         current_audio = res
                    #         used_mask_prev = info.get('src_time_mask')
                    #         continue
                    
                    # 중복 클래스 확인
                    if class_name in detected_classes:
                        print(f"[Separator] Pass {pass_idx + 1}: Duplicate class '{class_name}' detected, skipping")
                        # 잔여물을 다음 패스의 입력으로 사용
                        current_audio = res
                        used_mask_prev = info.get('src_time_mask')
                        continue
                    
                    # dB 계산
                    db_min, db_max, db_mean = self._calculate_decibel_from_raw(separated_audio)
                    
                    # 소리 발생시간 계산 (separator.py와 동일한 로직)
                    occurred_at = None
                    if 'separation_mask' in info and info['separation_mask'] is not None and SEPARATOR_AVAILABLE:
                        try:
                            from datetime import datetime, timedelta
                            inference_start_time = datetime.utcnow()  # 실제로는 녹음 시작 시간 사용
                            occurred_at = calculate_sound_occurrence_time(
                                info['separation_mask'], 
                                inference_start_time, 
                                audio_duration=len(audio)/SR
                            )
                            print(f"  🕐 Sound occurrence time: {occurred_at}")
                        except NameError:
                            # calculate_sound_occurrence_time 함수가 import되지 않은 경우
                            from datetime import datetime
                            occurred_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                            print(f"  🕐 Sound occurrence time (fallback): {occurred_at}")
                    else:
                        from datetime import datetime
                        occurred_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    source_info = {
                        'audio': separated_audio,
                        'class_name': class_name,
                        'sound_type': sound_type,
                        'confidence': confidence,
                        'class_id': class_id,
                        'pass': pass_idx + 1,
                        'db_min': db_min,
                        'db_max': db_max,
                        'db_mean': db_mean,
                        'occurred_at': occurred_at,
                        'separation_mask': info.get('separation_mask')
                    }
                    
                    separated_sources.append(source_info)
                    detected_classes.add(class_name)  # 감지된 클래스 추가
                    
                    # 콜백 호출
                    if on_pass_complete:
                        on_pass_complete(source_info)
                    
                    print(f"[Separator] Pass {pass_idx + 1}: {class_name} ({sound_type}) - Confidence: {confidence:.3f}")
                else:
                    print(f"[Separator] Pass {pass_idx + 1}: No audio separated")
                
                # 다음 패스를 위한 상태 업데이트
                # info에서 필요한 정보 추출
                used_mask_prev = info.get('src_time_mask')  # 사용된 시간 마스크
                # prev_anchors는 info에서 직접 가져올 수 없으므로 빈 리스트로 유지
                # prev_energy_ratio는 info에서 직접 가져올 수 없으므로 1.0으로 유지
                
                # 분리된 시간 영역 정보 업데이트
                if separated_audio is not None:
                    time_region = {
                        'time_mask': info.get('src_time_mask'),
                        'class_name': class_name,
                        'confidence': confidence,
                        'pass': pass_idx + 1
                    }
                    separated_time_regions.append(time_region)
                
                # 잔여물을 다음 패스의 입력으로 사용 (마지막 패스가 아닌 경우만)
                if pass_idx < max_passes - 1:
                    current_audio = res
                else:
                    # 마지막 패스: 잔여물 추론 제거
                    print(f"[Separator] Pass {pass_idx + 1}: Last pass completed, skipping residual inference")
                    break
                
            except Exception as e:
                print(f"[Separator] Error in pass {pass_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"[Separator] Separation completed: {len(separated_sources)} sources found")
        print(f"[Separator] Detected classes: {list(detected_classes)}")
        return separated_sources
    
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
                            # 소리 발생시간 계산 (separator.py와 동일한 로직)
                            occurred_at = None
                            if 'separation_mask' in source_info and source_info['separation_mask'] is not None and SEPARATOR_AVAILABLE:
                                try:
                                    from datetime import datetime, timedelta
                                    inference_start_time = datetime.utcnow()  # 실제로는 녹음 시작 시간 사용
                                    occurred_at = calculate_sound_occurrence_time(
                                        source_info['separation_mask'], 
                                        inference_start_time, 
                                        audio_duration=len(audio_normalized)/SR
                                    )
                                    print(f"  🕐 Sound occurrence time: {occurred_at}")
                                except NameError:
                                    # calculate_sound_occurrence_time 함수가 import되지 않은 경우
                                    from datetime import datetime
                                    occurred_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                                    print(f"  🕐 Sound occurrence time (fallback): {occurred_at}")
                            else:
                                from datetime import datetime
                                occurred_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                            
                            print(f"[Separator] Sending separated source to backend: {source_info['class_name']} ({source_info['sound_type']})")
                            backend_success = self._send_to_backend(
                                source_info['sound_type'], 
                                source_info['class_name'], 
                                source_info.get('db_mean', db_mean), 
                                angle,
                                occurred_at=occurred_at
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
                    
                    # separator.py의 single_pass 함수 사용
                    if SEPARATOR_AVAILABLE:
                        separated_sources = self._separate_with_separator_py(audio_normalized, max_passes=MAX_PASSES, on_pass_complete=on_pass_complete)
                    else:
                        print("[Separator] ⚠️ separator.py not available, using fallback separation")
                        separated_sources = []
                    
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
    
    def is_model_available(self) -> bool:
        """모델 사용 가능 여부 확인"""
        return self.is_available
    
    def cleanup(self):
        """리소스 정리"""
        # 캐시 정리
        self.attention_cache.clear()
        self.freq_attention_cache.clear()
        self.cls_head_cache.clear()
        self.spectrogram_cache.clear()
    
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
        led_controller: LED 컨트롤러 (선택사항)
        
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
