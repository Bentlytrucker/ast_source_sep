#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Trigger Module
- 100dB 이상 소리 감지 시 1.024x4초 녹음
- triggered_record.py 기반으로 구현
"""

import pyaudio
import wave
import numpy as np
import time
import sys
import os
import threading
from typing import Optional, Tuple

# ===== 설정 =====
FORMAT = pyaudio.paInt16
RATE = 16000
CHUNK = 1024
THRESHOLD_DB = 30  # 30dB 임계값 (테스트용)
RECORD_SECONDS = 1.024 * 4  # 1.024 x 4초 = 4.096초
TARGET_DEVICE_KEYWORDS = ("ReSpeaker", "seeed", "SEEED")  # 장치명에 포함될 키워드

# Multi-channel configuration for microphone array
# channel 0: processed audio for ASR
# channel 1-4: 4 microphones' raw data
# channel 5: playback (factory firmware)
RESPEAKER_CHANNELS = 6  # Use 6 channels for full microphone array support
RESPEAKER_WIDTH = 2

class SoundTrigger:
    def __init__(self, output_dir: str = "recordings", led_controller=None, continuous_mode: bool = True):
        """
        Sound Trigger 초기화
        
        Args:
            output_dir: 녹음 파일 저장 디렉토리
            led_controller: LED 컨트롤러 인스턴스 (wake up용)
            continuous_mode: True for continuous recording, False for triggered recording
        """
        self.output_dir = output_dir
        self.led_controller = led_controller
        self.continuous_mode = continuous_mode
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.device_index = None
        self.max_in_ch = 0
        self.desired_channels = RESPEAKER_CHANNELS  # Use 6 channels by default
        
        # 동시 녹음 제한 (최대 2개)
        self.active_recordings = 0
        self.max_concurrent_recordings = 2
        self.recording_lock = threading.Lock()
        
        # Continuous recording state
        self.is_recording = False
        self.stop_recording = False
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 장치 초기화
        self._initialize_device()
        
    def _initialize_device(self):
        """ReSpeaker 장치 초기화"""
        self.device_index, self.max_in_ch = self._find_respeaker_input_device()
        
        # Try to use 6 channels for full microphone array support
        # channel 0: processed audio for ASR, channel 1-4: 4 microphones' raw data, channel 5: playback
        if self.max_in_ch >= 6:
            self.desired_channels = 6
        elif self.max_in_ch > 0:
            self.desired_channels = self.max_in_ch
        else:
            self.desired_channels = 1
        
        info = self.p.get_device_info_by_index(self.device_index)
        print(f"[Device] index={self.device_index}, name='{info.get('name')}', maxInputChannels={self.max_in_ch}")
        print(f"[Open] channels={self.desired_channels}, rate={RATE}")
        print(f"[Mode] {'Continuous' if self.continuous_mode else 'Triggered'} recording mode")
        
        if self.desired_channels == 6:
            print("[Channel Layout] 0:ASR, 1-4:Microphones, 5:Playback")
        
        self.stream = self.p.open(
            format=FORMAT,
            channels=self.desired_channels,
            rate=RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK
        )
        
    def _find_respeaker_input_device(self) -> Tuple[int, int]:
        """ReSpeaker 입력 장치 찾기"""
        device_index = None
        max_in_ch = 0
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            name = info.get('name', '')
            is_input = int(info.get('maxInputChannels', 0)) > 0
            if is_input and any(k.lower() in name.lower() for k in TARGET_DEVICE_KEYWORDS):
                device_index = i
                max_in_ch = int(info.get('maxInputChannels', 0))
                break
        
        # 못 찾았으면 기본 입력 장치 사용
        if device_index is None:
            default_idx = self.p.get_default_input_device_info().get('index', None)
            if default_idx is None:
                print("입력 장치를 찾을 수 없습니다.", file=sys.stderr)
                sys.exit(1)
            info = self.p.get_device_info_by_index(default_idx)
            device_index = default_idx
            max_in_ch = int(info.get('maxInputChannels', 0))
        
        return device_index, max_in_ch
    
    def _to_mono_int16(self, interleaved: np.ndarray, num_channels: int) -> np.ndarray:
        """멀티채널 int16 interleaved -> 모노 int16 (Channel 0만 사용)"""
        if num_channels <= 1:
            return interleaved.astype(np.int16)

        # 길이가 채널 수로 딱 나눠떨어지도록 잘라서 reshape
        usable_len = (len(interleaved) // num_channels) * num_channels
        if usable_len != len(interleaved):
            interleaved = interleaved[:usable_len]
        x = interleaved.reshape(-1, num_channels)

        # Channel 0만 사용 (ReSpeaker USB Mic Array의 후처리된 오디오)
        mono = x[:, 0].astype(np.int16)
        
        return mono
    
    def _save_multichannel_audio(self, frames_data: list, filename: str, num_channels: int) -> str:
        """Save multi-channel audio data to WAV file"""
        # Concatenate all frame data
        all_frames_data = b''.join(frames_data)
        
        # Create output filename
        output_path = os.path.join(self.output_dir, filename)
        
        # Save WAV file with all channels
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(num_channels)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(all_frames_data)
        wf.close()
        
        return output_path
    
    def _extract_channel_data(self, interleaved: np.ndarray, num_channels: int, channel: int) -> np.ndarray:
        """Extract specific channel data from interleaved multi-channel audio"""
        if num_channels <= 1:
            return interleaved.astype(np.int16)
        
        usable_len = (len(interleaved) // num_channels) * num_channels
        if usable_len != len(interleaved):
            interleaved = interleaved[:usable_len]
        x = interleaved.reshape(-1, num_channels)
        
        # Extract the specified channel
        if channel < num_channels:
            return x[:, channel].astype(np.int16)
        else:
            return x[:, 0].astype(np.int16)  # Fallback to channel 0

    def _calculate_db_level(self, interleaved: np.ndarray, num_channels: int) -> float:
        """dB 레벨 계산 (Channel 0만 사용)"""
        if num_channels <= 1:
            audio_data = interleaved.astype(np.float32)
        else:
            usable_len = (len(interleaved) // num_channels) * num_channels
            x = interleaved[:usable_len].reshape(-1, num_channels)

            # Channel 0만 사용 (ReSpeaker USB Mic Array의 후처리된 오디오)
            audio_data = x[:, 0].astype(np.float32)

        # RMS 계산
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Debug output removed
        
        if rms == 0:
            return -np.inf
        
        # dB 변환 (20 * log10(rms))
        # RMS가 0이 아닌 경우에만 dB 계산
        if rms > 0:
            db = 20 * np.log10(rms)
            
            # 유효한 dB 값인지 확인
            if np.isnan(db) or np.isinf(db):
                return -np.inf
                
            return db
        else:
            return -np.inf

    def _level_for_trigger(self, interleaved: np.ndarray, num_channels: int) -> float:
        """트리거 판정 레벨(RMS 또는 abs max) - Channel 0만 사용"""
        if num_channels <= 1:
            return float(np.max(np.abs(interleaved)))
        usable_len = (len(interleaved) // num_channels) * num_channels
        x = interleaved[:usable_len].reshape(-1, num_channels)

        # Channel 0만 사용 (ReSpeaker USB Mic Array의 후처리된 오디오)
        ch = x[:, 0].astype(np.int16)

        return float(np.max(np.abs(ch)))

    def start_continuous_recording(self, duration_seconds: float = None) -> Optional[str]:
        """
        Start continuous recording for voice recognition
        
        Args:
            duration_seconds: Recording duration (None for indefinite)
            
        Returns:
            Path to recorded multi-channel audio file
        """
        print(f"Starting continuous recording with {self.desired_channels} channels...")
        print("Channel layout: 0=ASR, 1-4=Microphones, 5=Playback")
        
        frames_data = []
        self.is_recording = True
        self.stop_recording = False
        
        # Calculate target samples if duration is specified
        target_samples = int(duration_seconds * RATE) if duration_seconds else None
        samples_collected = 0
        
        try:
            while self.is_recording and not self.stop_recording:
                # Read multi-channel audio data
                raw = self.stream.read(CHUNK, exception_on_overflow=False)
                frames_data.append(raw)
                
                if target_samples:
                    samples_collected += CHUNK
                    if samples_collected >= target_samples:
                        break
                        
            # Save multi-channel audio file
            timestamp = int(time.time())
            filename = f"continuous_recording_{timestamp}_{self.desired_channels}ch.wav"
            output_path = self._save_multichannel_audio(frames_data, filename, self.desired_channels)
            
            print(f"Continuous recording saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error during continuous recording: {e}")
            return None
        finally:
            self.is_recording = False
    
    def stop_continuous_recording(self):
        """Stop continuous recording"""
        self.stop_recording = True
        self.is_recording = False

    def start_monitoring(self) -> Optional[str]:
        """
        Start sound monitoring
        
        Returns:
            Recorded file path (when triggered), None (no trigger)
        """
        if self.continuous_mode:
            return self.start_continuous_recording(RECORD_SECONDS)
        
        print("Waiting for sounds above 100dB... (Press Ctrl+C to stop)")
        
        recording = False
        frames_bytes = []
        samples_collected = 0
        target_samples = int(RECORD_SECONDS * RATE)
        
        try:
            # Monitor with a reasonable timeout
            import time
            start_time = time.time()
            timeout = 300  # 5 minutes timeout (longer for testing)
            
            while time.time() - start_time < timeout:
                raw = self.stream.read(CHUNK, exception_on_overflow=False)
                data_i16 = np.frombuffer(raw, dtype=np.int16)

                # Calculate dB level
                db_level = self._calculate_db_level(data_i16, self.desired_channels)
                
                # Check trigger (above 100dB)
                if not recording and db_level >= THRESHOLD_DB:
                    # 동시 녹음 제한 확인
                    with self.recording_lock:
                        if self.active_recordings >= self.max_concurrent_recordings:
                            print(f"Sound detected! ({db_level:.1f}dB) but max concurrent recordings reached ({self.max_concurrent_recordings})")
                            continue
                        
                        self.active_recordings += 1
                        print(f"Sound detected! ({db_level:.1f}dB) micarray wake up...")
                        print(f"Active recordings: {self.active_recordings}/{self.max_concurrent_recordings}")
                    
                    # Execute wake up if LED controller is available
                    if self.led_controller:
                        self.led_controller.wakeup_from_sleep()
                    
                    print("Recording started...")
                    recording = True
                    frames_bytes = []
                    samples_collected = 0

                if recording:
                    # Store raw multi-channel data for triggered mode
                    frames_bytes.append(raw)
                    samples_collected += CHUNK

                    if samples_collected >= target_samples:
                        recording = False
                        print("녹음 종료. 파일 저장 중...")

                        # 파일명에 타임스탬프 포함
                        timestamp = int(time.time())
                        
                        # Save multi-channel file if available, otherwise mono
                        if self.desired_channels > 1:
                            output_filename = f"triggered_recording_{timestamp}_{self.desired_channels}ch.wav"
                            output_path = self._save_multichannel_audio(frames_bytes, output_filename, self.desired_channels)
                        else:
                            # Fallback to mono for single channel
                            mono_frames = []
                            for frame in frames_bytes:
                                data_i16 = np.frombuffer(frame, dtype=np.int16)
                                mono = self._to_mono_int16(data_i16, self.desired_channels)
                                mono_frames.append(mono.tobytes())
                            
                            output_filename = os.path.join(self.output_dir, f"triggered_recording_{timestamp}.wav")
                            all_frames_data = b''.join(mono_frames)
                            
                            wf = wave.open(output_filename, 'wb')
                            wf.setnchannels(1)
                            wf.setsampwidth(self.p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(all_frames_data)
                            wf.close()
                            output_path = output_filename

                        # 동시 녹음 카운터 감소
                        with self.recording_lock:
                            self.active_recordings -= 1
                            print(f"Recording completed. Active recordings: {self.active_recordings}/{self.max_concurrent_recordings}")

                        print(f"Saved: {output_path}")
                        print("Waiting for sounds above 100dB...")
                        
                        return output_path
            
            # Timeout reached
            print("Monitoring timeout (5 minutes) - no sound detected")
            return None

        except KeyboardInterrupt:
            print("\nMonitoring stopped...")
            return None
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        finally:
            if self.p:
                self.p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """테스트용 메인 함수"""
    with SoundTrigger() as trigger:
        recorded_file = trigger.start_monitoring()
        if recorded_file:
            print(f"녹음 완료: {recorded_file}")


if __name__ == "__main__":
    main()
