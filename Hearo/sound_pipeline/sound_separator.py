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
        """AST 모델 초기화"""
        try:
            if ASTFeatureExtractor is None or ASTForAudioClassification is None:
                print("[Separator] Transformers not available, using mock mode")
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
            print("[Separator] AST model loaded successfully")
            
        except Exception as e:
            print(f"[Separator] Model loading error: {e}")
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
    
    def _calculate_decibel_simple(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """Sound Trigger와 동일한 dB 계산 방법"""
        try:
            if len(audio) == 0:
                print(f"[Separator] Debug: Empty audio data")
                return -np.inf, -np.inf, -np.inf
            
            # 디버그: 오디오 데이터 정보
           # print(f"[Separator] Debug: Audio data range: {audio.min():.3f} to {audio.max():.3f}")
            #print(f"[Separator] Debug: Audio data mean: {audio.mean():.3f}, std: {audio.std():.3f}")
            
            # RMS 계산
            rms = np.sqrt(np.mean(audio**2))
            #print(f"[Separator] Debug: RMS: {rms:.6f}")
            
            if rms <= 0:
                print(f"[Separator] Debug: RMS is zero or negative: {rms}")
                return -np.inf, -np.inf, -np.inf
            
            # dB 변환 (20 * log10(rms))
            db = 20 * np.log10(rms)
            #print(f"[Separator] Debug: Calculated dB: {db:.3f}")
            
            # 유효한 dB 값인지 확인
            if np.isnan(db) or np.isinf(db):
                print(f"[Separator] Debug: dB is NaN or inf: {db}")
                return -np.inf, -np.inf, -np.inf
            
            # min, max는 mean과 동일하게 설정 (간단하게)
            return db, db, db
            
        except Exception as e:
            print(f"[Separator] Simple dB calculation error: {e}")
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
        """오디오 파일 로드 및 고정 길이로 조정 (원시 int16 값 유지)"""
        try:
            # torchaudio로 로드 (정규화된 float 값)
            wav, sro = torchaudio.load(path)
            
            # 스테레오를 모노로 변환
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # 샘플링 레이트 변환
            if sro != SR:
                wav = torchaudio.functional.resample(wav, sro, SR)
            
            # 정규화된 float 값을 원시 int16 값으로 변환
            # torchaudio는 -1.0~1.0 범위로 정규화하므로, int16 범위로 복원
            wav_int16 = (wav * 32767).squeeze().numpy().astype(np.int16)
            
            # 데이터 검증
            if len(wav_int16) == 0:
                print(f"[Separator] Warning: Empty audio data from {path}")
                return np.zeros(L_FIXED, dtype=np.int16)
            
            # 디버그: 오디오 데이터 범위 확인
            print(f"[Separator] Debug: Loaded audio range: {wav_int16.min()} to {wav_int16.max()}")
            print(f"[Separator] Debug: Loaded audio mean: {wav_int16.mean():.1f}, std: {wav_int16.std():.1f}")
            
            # 고정 길이로 조정
            if len(wav_int16) >= L_FIXED:
                return wav_int16[:L_FIXED]
            else:
                out = np.zeros(L_FIXED, dtype=np.int16)
                out[:len(wav_int16)] = wav_int16
                return out
                
        except Exception as e:
            print(f"[Separator] Error loading audio {path}: {e}")
            return np.zeros(L_FIXED, dtype=np.int16)
    
    def _classify_audio(self, audio: np.ndarray) -> Tuple[str, str, int, float]:
        """
        오디오 분류 (간단한 버전)
        
        Args:
            audio: 오디오 데이터 (int16 또는 float32)
            
        Returns:
            (class_name, sound_type, class_id, confidence)
        """
        if not self.is_available:
            # Mock 분류 결과
            return "Unknown", "other", 0, 0.5
        
        try:
            # int16 데이터를 float32로 정규화 (AST 모델용)
            if audio.dtype == np.int16:
                audio_float = audio.astype(np.float32) / 32767.0  # -1.0 ~ 1.0 범위로 정규화
            else:
                audio_float = audio.astype(np.float32)
            
            # 10초로 패딩
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
            print(f"[Separator] Classification error: {e}")
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
            # 오디오 로드
            audio = self._load_fixed_audio(audio_file)
            print(f"[Separator] Audio length: {len(audio)/SR:.2f}s")
            
            # 분류
            class_name, sound_type, class_id, confidence = self._classify_audio(audio)
            
            # dB 계산 (Sound Trigger와 동일한 방법 사용)
            db_min, db_max, db_mean = self._calculate_decibel_simple(audio)
            
            print(f"[Separator] Classified: {class_name} ({sound_type})")
            print(f"[Separator] Confidence: {confidence:.3f}")
            print(f"[Separator] Decibel: {db_mean:.1f} dB")
            
            # 분리된 소리 저장
            separated_file = None
            if output_dir:
                separated_file = self._save_separated_audio(audio, class_name, sound_type, output_dir)
            
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


class MockSoundSeparator:
    """
    모델이 없을 때 사용하는 Mock 클래스
    테스트용으로 랜덤 분류 결과 반환
    """
    
    def __init__(self, backend_url: str = BACKEND_URL):
        self.backend_url = backend_url
        self.is_available = False
        print("[Separator] Mock Sound Separator initialized (no real model)")
    
    def process_audio(self, audio_file: str, angle: int, output_dir: str = None) -> Dict[str, Any]:
        """Mock 처리 결과 반환"""
        import random
        
        # Mock 분류 결과
        mock_classes = [
            ("Gunshot", "danger", 396, 0.85),
            ("Scream", "danger", 397, 0.90),
            ("Car horn", "warning", 288, 0.75),
            ("Dog bark", "warning", 364, 0.80),
            ("Help", "help", 23, 0.70),
            ("Unknown", "other", 0, 0.50)
        ]
        
        class_name, sound_type, class_id, confidence = random.choice(mock_classes)
        
        # Mock dB 계산
        db_mean = random.uniform(60, 120)
        
        print(f"[Separator] Mock classified: {class_name} ({sound_type})")
        print(f"[Separator] Mock confidence: {confidence:.3f}")
        print(f"[Separator] Mock decibel: {db_mean:.1f} dB")
        
        # Mock 분리된 소리 저장
        separated_file = None
        if output_dir:
            separated_file = self._save_separated_audio_mock(audio_file, class_name, sound_type, output_dir)
        
        # Mock 백엔드 전송 (실제 전송 시도)
        backend_success = True
        if sound_type != "other":
            print(f"[Separator] Mock backend send: {sound_type} at {angle}°")
            # 실제 백엔드 전송 시도
            backend_success = self._send_to_backend_mock(sound_type, class_name, db_mean, angle)
        
        return {
            "success": True,
            "class_name": class_name,
            "sound_type": sound_type,
            "class_id": class_id,
            "confidence": confidence,
            "angle": angle,
            "decibel": {
                "min": db_mean - 10,
                "max": db_mean + 10,
                "mean": db_mean
            },
            "backend_success": backend_success,
            "audio_file": audio_file,
            "separated_file": separated_file
        }
    
    def _save_separated_audio_mock(self, audio_file: str, class_name: str, sound_type: str, output_dir: str) -> str:
        """Mock 분리된 오디오 저장"""
        try:
            import shutil
            import time
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일명 생성
            timestamp = int(time.time())
            safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_class_name = safe_class_name.replace(' ', '_')
            
            filename = f"separated_{timestamp}_{safe_class_name}_{sound_type}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # 원본 파일을 복사 (Mock)
            shutil.copy2(audio_file, filepath)
            
            print(f"[Separator] Mock separated audio saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"[Separator] Mock error saving separated audio: {e}")
            return None
    
    def _send_to_backend_mock(self, sound_type: str, sound_detail: str, decibel: float, angle: int) -> bool:
        """Mock backend transmission (actually tries to send)"""
        try:
            data = {
                "user_id": USER_ID,
                "sound_type": sound_type,
                "sound_detail": sound_detail,
                "angle": angle,
                "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sound_icon": "string",
                "location_image_url": "string",
                "decibel": float(decibel),
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'SoundPipeline/1.0'
            }
            
            print(f"🔄 Mock sending to backend: {self.backend_url}")
            print(f"📤 Mock data: {data}")
            
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
                print(f"✅ Mock sent to backend: {sound_detail} ({sound_type}) at {angle}°")
                return True
            else:
                print(f"❌ Mock backend error: {response.status_code}")
                print(f"❌ Mock response: {response.text}")
                return False
                
        except requests.exceptions.ConnectTimeout:
            print(f"❌ Mock backend connection timeout: {self.backend_url}")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Mock backend connection error: {e}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Mock backend request error: {e}")
            return False
        except Exception as e:
            print(f"❌ Mock unexpected error sending to backend: {e}")
            return False
    
    def is_model_available(self) -> bool:
        """Mock은 항상 사용 가능"""
        return True
    
    def cleanup(self):
        """Mock 정리"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_sound_separator(model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", 
                          device: str = "auto", backend_url: str = BACKEND_URL) -> SoundSeparator:
    """
    Sound Separator 인스턴스 생성
    모델이 없으면 Mock 버전 반환
    
    Args:
        model_name: AST 모델 이름
        device: 사용할 디바이스
        backend_url: 백엔드 API URL
        
    Returns:
        SoundSeparator 또는 MockSoundSeparator 인스턴스
    """
    separator = SoundSeparator(model_name, device, backend_url)
    
    if not separator.is_model_available():
        print("[Separator] Real model not available, using mock separator")
        return MockSoundSeparator(backend_url)
    
    return separator


def main():
    """테스트용 메인 함수"""
    import tempfile
    import numpy as np
    
    # 테스트용 오디오 파일 생성
    test_audio = np.random.randn(16000).astype(np.float32)  # 1초 오디오
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        torchaudio.save(f.name, torch.from_numpy(test_audio).unsqueeze(0), SR)
        test_file = f.name
    
    try:
        with create_sound_separator() as separator:
            print("Sound Separator 테스트 시작...")
            
            result = separator.process_audio(test_file, 180)
            
            print("처리 결과:")
            print(f"  Success: {result['success']}")
            if result['success']:
                print(f"  Class: {result['class_name']}")
                print(f"  Type: {result['sound_type']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Angle: {result['angle']}°")
                print(f"  Decibel: {result['decibel']['mean']:.1f} dB")
                print(f"  Backend: {'✅' if result['backend_success'] else '❌'}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
            print("Sound Separator 테스트 완료")
    
    finally:
        # 테스트 파일 정리
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    main()
