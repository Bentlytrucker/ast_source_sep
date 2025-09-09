# Sound Pipeline

실시간 소리 감지, 분석 및 LED 제어를 위한 통합 파이프라인입니다.

## 기능

- **100dB 이상 소리 감지**: 실시간으로 100dB 이상의 소리를 감지하여 자동 녹음
- **각도 계산**: ReSpeaker 마이크 배열을 사용한 소리 방향 감지
- **음원 분리 및 분류**: AST 모델을 사용한 소리 분류 (danger/warning/help/other)
- **LED 제어**: 분류 결과에 따른 LED 색상 출력
- **백엔드 전송**: 분석 결과를 백엔드 서버로 전송

## LED 색상 매핑

- **Danger** (위험): 빨간색 (0xFF0000)
- **Warning** (경고): 노란색 (0xFFFF00)  
- **Help** (도움): 초록색 (0x00FF00)
- **Other** (기타): 파란색 (0x0000FF)

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. ReSpeaker 장치 설정

ReSpeaker USB 마이크 배열이 연결되어 있어야 합니다:
- Vendor ID: 0x2886
- Product ID: 0x0018

### 3. 권한 설정 (Linux)

USB 장치 접근을 위한 권한 설정:

```bash
sudo usermod -a -G dialout $USER
```

## 사용법

### 기본 실행

```bash
python sound_pipeline.py
```

### 옵션 설정

```bash
python sound_pipeline.py --output ./output --model MIT/ast-finetuned-audioset-10-10-0.4593 --device auto
```

### 명령행 옵션

- `--output, -o`: 출력 디렉토리 (기본값: pipeline_output)
- `--model, -m`: AST 모델 이름 (기본값: MIT/ast-finetuned-audioset-10-10-0.4593)
- `--device, -d`: 사용할 디바이스 (auto/cpu/cuda, 기본값: auto)
- `--backend-url`: 백엔드 API URL

## 파이프라인 구조

```
소리 감지 (100dB+) → 녹음 (1.024×4초) → 각도 계산 → 음원 분리/분류 → LED 출력 → 백엔드 전송
```

### 컴포넌트

1. **SoundTrigger**: 100dB 이상 소리 감지 및 녹음
2. **DOACalculator**: 소리 방향 각도 계산
3. **SoundSeparator**: 음원 분리 및 분류
4. **LEDController**: LED 색상 제어

## 파일 구조

```
sound_pipeline/
├── sound_pipeline.py      # 메인 파이프라인
├── sound_trigger.py       # 소리 감지 및 녹음
├── doa_calculator.py      # 각도 계산
├── sound_separator.py     # 음원 분리 및 분류
├── led_controller.py      # LED 제어
├── requirements.txt       # 의존성 목록
└── README.md             # 이 파일
```

## 백엔드 API

분석 결과는 다음 형식으로 백엔드에 전송됩니다:

```json
{
    "user_id": 6,
    "sound_type": "danger",
    "sound_detail": "Gunshot",
    "angle": 180,
    "occurred_at": "2024-01-01T12:00:00Z",
    "sound_icon": "string",
    "location_image_url": "string",
    "decibel": 85.5
}
```

## 문제 해결

### 1. USB 장치를 찾을 수 없음

- ReSpeaker 장치가 올바르게 연결되어 있는지 확인
- USB 케이블 및 전원 공급 확인
- `lsusb` (Linux) 또는 장치 관리자 (Windows)에서 장치 확인

### 2. 모델 로딩 실패

- 인터넷 연결 확인 (모델 다운로드용)
- 충분한 디스크 공간 확인
- PyTorch 및 transformers 라이브러리 버전 확인

### 3. 오디오 장치 오류

- 마이크 권한 확인
- 다른 애플리케이션에서 마이크 사용 중인지 확인
- 오디오 드라이버 업데이트

## 개발자 정보

이 파이프라인은 다음 기존 모듈들을 기반으로 구축되었습니다:

- `triggered_record.py`: 소리 감지 및 녹음
- `DOA.py`: 각도 계산
- `separator.py`: 음원 분리 및 분류
- `led_control.py`: LED 제어

## 라이선스

이 프로젝트는 원본 모듈들의 라이선스를 따릅니다.
