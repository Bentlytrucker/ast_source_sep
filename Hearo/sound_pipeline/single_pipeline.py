

import os
import sys
import time
import threading
import queue
import argparse
from typing import Optional, Dict, Any, Set
from datetime import datetime, timedelta


from sound_trigger import SoundTrigger
from doa_calculator import create_doa_calculator
from sound_separator import create_sound_separator
from led_controller import create_led_controller


class SingleSoundPipeline:
    def __init__(self, output_dir: str = "pipeline_output", 
                 model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 device: str = "auto", backend_url: str = "http://13.238.200.232:8000/sound-events/"):
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        self.backend_url = backend_url
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 상태 관리
        self.is_running = False
        
        # 각 오디오 파일마다 독립적인 중복 클래스 전송 방지를 위한 세트
        self.current_sent_classes: Set[str] = set()
        
        # 통계
        self.stats = {
            "total_detected": 0,
            "successful_separations": 0,
            "backend_sends": 0,
            "led_activations": 0,
            "duplicate_skips": 0
        }
    
    def _initialize_components(self):
        """컴포넌트들 초기화"""
        print("=== Single Thread Pipeline Initialization ===")
        
        # Initialize Sound Trigger
        print("1. Initializing Sound Trigger...")
        self.sound_trigger = SoundTrigger(os.path.join(self.output_dir, "recordings"), None)
        
        # Initialize DOA Calculator
        print("2. Initializing DOA Calculator...")
        self.doa_calculator = create_doa_calculator()
        
        # Initialize LED Controller first (needed for Sound Separator)
        print("3. Initializing LED Controller...")
        self.led_controller = create_led_controller()
        if self.led_controller is None:
            print("⚠️ LED Controller not available - LED control disabled")
        
        # Initialize Sound Separator with LED Controller
        print("4. Initializing Sound Separator...")
        self.sound_separator = create_sound_separator(
            self.model_name, 
            self.device, 
            self.backend_url, 
            self.led_controller  # LED 컨트롤러 주입
        )
        
        # 모델 초기화 상태 확인
        if hasattr(self.sound_separator, 'is_available') and self.sound_separator.is_available:
            print("✅ Sound Separator initialized successfully")
        else:
            print("❌ Sound Separator initialization failed!")
            print("🔍 Checking model availability...")
            if hasattr(self.sound_separator, 'is_model_available'):
                print(f"Model available: {self.sound_separator.is_model_available()}")
            else:
                print("is_model_available method not found")
        
        print("=== Single Thread Pipeline Ready ===")
    
    def _main_loop(self):
        """메인 루프 - 소리 감지부터 분리까지 순차 처리"""
        while self.is_running:
            try:
                # 1. 소리 감지 및 녹음
                recorded_file = self.sound_trigger.start_monitoring()
                
                if recorded_file and self.is_running:
                    self.stats["total_detected"] += 1
                    print(f"\n🎵 Processing: {os.path.basename(recorded_file)}")
                    
                    # 녹음 완료 시점 기록
                    recording_end_time = datetime.utcnow()
                    
                    # 2. 음원 분리 및 백엔드 전송
                    separation_result = self._process_separation(recorded_file, recording_end_time)
                    
                    if separation_result["success"]:
                        separated_sources = separation_result.get("separated_sources", [])
                        print(f"✅ Separation completed: {len(separated_sources)} sources")
                    else:
                        print(f"❌ Separation failed: {separation_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                continue
    
    
    def _process_separation(self, audio_file: str, recording_end_time: datetime) -> Dict[str, Any]:
        """음원 분리 및 각 패스마다 백엔드 전송 (중복 클래스 전송 방지)"""
        try:
            # 1. Calculate DOA
            angle = self.doa_calculator.get_direction_with_retry(max_retries=2)
            if angle is None:
                angle = 0
            
            print(f"📍 Direction: {angle}°")
            
            # 2. 오디오 로드
            audio_data = self.sound_separator._load_fixed_audio(audio_file)
            if audio_data is None:
                return {"success": False, "error": "Failed to load audio"}
            
            # 3. 음원 분리 수행 (각 패스마다 즉시 처리)
            print("🔍 Starting source separation...")
            
            # 현재 오디오 파일의 중복 클래스 세트 초기화
            self.current_sent_classes.clear()
            
            # 각 패스 완료 시마다 즉시 처리하는 콜백 함수
            def on_pass_complete(source_info):
                class_name = source_info['class_name']
                sound_type = source_info['sound_type']
                pass_num = source_info.get('pass', 0)
                
                # 소리 발생시간 계산 (separator.py와 동일한 로직)
                occurred_at = None
                if 'separation_mask' in source_info and source_info['separation_mask'] is not None:
                    try:
                        # 녹음 끝난 시점 사용
                        inference_start_time = recording_end_time
                        audio_duration = len(audio_data) / 16000  # 16kHz 가정
                        
                        # separator.py의 calculate_sound_occurrence_time 함수 사용
                        from sound_pipeline.separator import calculate_sound_occurrence_time
                        occurred_at = calculate_sound_occurrence_time(
                            source_info['separation_mask'], 
                            inference_start_time, 
                            audio_duration=audio_duration
                        )
                        print(f"  🕐 Sound occurrence time: {occurred_at}")
                    except Exception as e:
                        print(f"  ⚠️ Sound occurrence time calculation failed: {e}")
                        occurred_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    occurred_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # 현재 오디오 파일 내에서만 중복 클래스 체크
                if class_name in self.current_sent_classes:
                    print(f"⏭️ SKIP: {class_name} ({sound_type}) - Duplicate")
                    self.stats["duplicate_skips"] += 1
                    return
                
                # 백엔드 전송 (other 타입 및 silence 클래스 제외)
                backend_success = False
                if self.backend_url and sound_type != "other" and class_name.lower() != "silence":
                    # source_info에 occurred_at 추가
                    source_info_with_time = source_info.copy()
                    source_info_with_time['occurred_at'] = occurred_at
                    backend_success = self._send_to_backend(source_info_with_time, angle)
                    if backend_success:
                        self.stats["backend_sends"] += 1
                elif sound_type == "other":
                    print(f"⏭️ SKIP: {class_name} ({sound_type}) - Backend send skipped for 'other' type")
                elif class_name.lower() == "silence":
                    print(f"⏭️ SKIP: {class_name} ({sound_type}) - Backend send skipped for 'silence' class")
                
                # LED 활성화
                if self.led_controller:
                    self.led_controller.activate_led(angle, class_name, sound_type)
                    self.stats["led_activations"] += 1
                
                # 현재 오디오 파일의 전송된 클래스 기록
                self.current_sent_classes.add(class_name)
                
                # 간소화된 출력
                backend_status = "✅" if backend_success else "❌"
                print(f"🎵 {class_name} ({sound_type}) - Backend: {backend_status}")
            
            # 분리 실행 (각 패스마다 즉시 처리)
            separated_sources = self.sound_separator.separate_audio(audio_data, angle, max_passes=2, on_pass_complete=on_pass_complete)
            
            if separated_sources:
                print(f"✅ Separation completed: {len(separated_sources)} sources")
                self.stats["successful_separations"] += 1
                return {"success": True, "separated_sources": separated_sources}
            else:
                print("❌ No sources separated (Silence detected or no valid sounds)")
                return {"success": False, "error": "No sources separated"}
                
        except Exception as e:
            print(f"❌ Separation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _send_to_backend(self, source: Dict[str, Any], angle: int) -> bool:
        """백엔드로 분리된 소리 전송"""
        try:
            import requests
            from datetime import datetime
            
            # 소리 발생시간이 source에 포함되어 있으면 사용, 없으면 현재 시간 사용
            occurred_at = source.get('occurred_at', datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
            
            # 백엔드 전송 데이터 구성
            data = {
                "user_id": 6,
                "sound_type": source['sound_type'],
                "sound_detail": source['class_name'],
                "angle": angle,
                "occurred_at": occurred_at,
                "sound_icon": "string",
                "location_image_url": "string",
                "decibel": source.get('db_mean', 60.0)  # 실제 계산된 값 사용
            }
            
            # 백엔드로 전송
            response = requests.post(self.backend_url, json=data, timeout=5)
            
            if response.status_code == 200:
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def start(self):
        """파이프라인 시작 - 하나의 스레드에서 순차 실행"""
        if self.is_running:
            print("⚠️ Pipeline is already running")
            return
        
        print("🚀 Starting Single Thread Sound Pipeline...")
        print("=" * 60)
        print("Mode: Sound Detection → Source Separation → Backend/LED")
        print("=" * 60)
        
        # 컴포넌트들 초기화
        self._initialize_components()
        
        # 메인 루프 시작
        self.is_running = True
        
        print("\n✅ Single Thread Sound Pipeline started successfully!")
        print("📡 Monitoring for sounds above 100dB...")
        print("🔍 Will process audio separation and send to backend")
        print("💡 Will activate LED for each separated source")
        print("⏭️ Will skip duplicate classes")
        print("\nPress Ctrl+C to stop")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n🛑 Stopping pipeline...")
            self.stop()
    
    def stop(self):
        """파이프라인 중지"""
        if not self.is_running:
            print("⚠️ Pipeline is not running")
            return
        
        print("🛑 Stopping Single Thread Sound Pipeline...")
        
        # LED 끄기
        if self.led_controller:
            self.led_controller.turn_off()
        
        self.is_running = False
        print("✅ Single Thread Sound Pipeline stopped")
        self._print_statistics()
    
    def _print_statistics(self):
        """통계 출력"""
        print("\n=== Single Thread Pipeline Statistics ===")
        print(f"Total detected: {self.stats['total_detected']}")
        print(f"Successful separations: {self.stats['successful_separations']}")
        print(f"Backend sends: {self.stats['backend_sends']}")
        print(f"LED activations: {self.stats['led_activations']}")
        print(f"Duplicate skips: {self.stats['duplicate_skips']}")
        print("==========================================\n")
    
    def cleanup(self):
        """리소스 정리"""
        if self.is_running:
            self.stop()
        
        # 컴포넌트 정리
        if hasattr(self, 'sound_trigger') and self.sound_trigger:
            self.sound_trigger.cleanup()
        if hasattr(self, 'doa_calculator') and self.doa_calculator:
            self.doa_calculator.cleanup()
        if hasattr(self, 'sound_separator') and self.sound_separator:
            self.sound_separator.cleanup()
        if hasattr(self, 'led_controller') and self.led_controller:
            self.led_controller.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Single Thread Sound Pipeline - Fast Classification + Source Separation")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--backend-url", default="http://13.238.200.232:8000/sound-events/", help="Backend API URL")
    
    args = parser.parse_args()
    
    print("🎵 Single Thread Sound Pipeline v2.0")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Backend URL: {args.backend_url}")
    print("=" * 60)
    
    # 파이프라인 실행
    with SingleSoundPipeline(
        output_dir=args.output,
        model_name=args.model,
        device=args.device,
        backend_url=args.backend_url
    ) as pipeline:
        pipeline.start()


if __name__ == "__main__":
    main()
