import pyaudio
import wave
import numpy as np
import time
import sys

# ===== 설정 =====
FORMAT = pyaudio.paInt16
RATE = 16000
CHUNK = 1024
THRESHOLD = 500          # 환경에 맞게 조정
RECORD_SECONDS = 5
TARGET_DEVICE_KEYWORDS = ("ReSpeaker", "seeed", "SEEED")  # 장치명에 포함될 키워드

# ===== 초기화 및 장치 탐색 =====
p = pyaudio.PyAudio()

def find_respeaker_input_device():
    device_index = None
    max_in_ch = 0
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get('name', '')
        is_input = int(info.get('maxInputChannels', 0)) > 0
        if is_input and any(k.lower() in name.lower() for k in TARGET_DEVICE_KEYWORDS):
            device_index = i
            max_in_ch = int(info.get('maxInputChannels', 0))
            break
    # 못 찾았으면 기본 입력 장치 사용
    if device_index is None:
        default_idx = p.get_default_input_device_info().get('index', None)
        if default_idx is None:
            print("입력 장치를 찾을 수 없습니다.", file=sys.stderr)
            sys.exit(1)
        info = p.get_device_info_by_index(default_idx)
        device_index = default_idx
        max_in_ch = int(info.get('maxInputChannels', 0))
    return device_index, max_in_ch

device_index, max_in_ch = find_respeaker_input_device()

# ReSpeaker v2.0이면 6채널(0~3: 마이크, 4: reference, 5: post-processed)을 시도
# 안 되면 사용 가능한 최대 입력 채널로 열고, 최종적으로 1채널 변환 저장
DESIRED_CHANNELS = 6 if max_in_ch >= 6 else max_in_ch if max_in_ch > 0 else 1

info = p.get_device_info_by_index(device_index)
print(f"[Device] index={device_index}, name='{info.get('name')}', maxInputChannels={max_in_ch}")
print(f"[Open] channels={DESIRED_CHANNELS}, rate={RATE}")

stream = p.open(format=FORMAT,
                channels=DESIRED_CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

print("임계값 이상 소리를 대기 중입니다... (Ctrl+C로 종료)")

recording = False
frames_bytes = []
samples_collected = 0
target_samples = RECORD_SECONDS * RATE

def to_mono_int16(interleaved: np.ndarray, num_channels: int) -> np.ndarray:
    """멀티채널 int16 interleaved -> 모노 int16 (채널5 우선, 없으면 마이크 평균)"""
    if num_channels <= 1:
        return interleaved.astype(np.int16)

    # 길이가 채널 수로 딱 나눠떨어지도록 잘라서 reshape
    usable_len = (len(interleaved) // num_channels) * num_channels
    if usable_len != len(interleaved):
        interleaved = interleaved[:usable_len]
    x = interleaved.reshape(-1, num_channels)

    # 채널 5가 있으면 (ReSpeaker post-processed/beamformed) 그 채널만 사용
    if num_channels >= 6:
        mono = x[:, 5].astype(np.int16)
    else:
        # 일반 마이크 채널 평균 (가능하면 앞쪽 4채널만 평균)
        mic_cols = min(num_channels, 4)
        mono = np.mean(x[:, :mic_cols], axis=1).astype(np.int16)
    return mono

def level_for_trigger(interleaved: np.ndarray, num_channels: int) -> float:
    """트리거 판정 레벨(RMS 또는 abs max). 채널5가 있으면 그 채널 기준."""
    if num_channels <= 1:
        return float(np.max(np.abs(interleaved)))
    usable_len = (len(interleaved) // num_channels) * num_channels
    x = interleaved[:usable_len].reshape(-1, num_channels)

    if num_channels >= 6:
        ch = x[:, 5].astype(np.int16)
    else:
        ch = np.mean(x[:, :min(num_channels, 4)], axis=1).astype(np.int16)

    return float(np.max(np.abs(ch)))

try:
    while True:
        raw = stream.read(CHUNK, exception_on_overflow=False)
        data_i16 = np.frombuffer(raw, dtype=np.int16)

        # 트리거 체크
        if not recording and level_for_trigger(data_i16, DESIRED_CHANNELS) > THRESHOLD:
            print("소리 감지! 녹음 시작...")
            recording = True
            frames_bytes = []
            samples_collected = 0

        if recording:
            mono = to_mono_int16(data_i16, DESIRED_CHANNELS)
            frames_bytes.append(mono.tobytes())
            samples_collected += len(mono)

            if samples_collected >= target_samples:
                recording = False
                print("녹음 종료. 파일 저장 중...")

                output_filename = f"triggered_recording_{int(time.time())}.wav"
                wf = wave.open(output_filename, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames_bytes))
                wf.close()

                print(f"저장 완료: {output_filename}")
                print("임계값 이상 소리를 다시 대기 중...")

except KeyboardInterrupt:
    print("\n종료합니다...")
finally:
    try:
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()
