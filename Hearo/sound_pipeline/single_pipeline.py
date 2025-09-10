#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Thread Sound Pipeline - Raspberry Pi용
- Fast Classification → Source Separation (Sequential)
- 중복 클래스 전송 방지
- 각 패스마다 백엔드 전송 + LED 활성화
"""

import os
import sys
import time
import threading
import queue
import argparse
from typing import Optional, Dict, Any, Set

# 파이프라인 모듈들 import
from sound_trigger import SoundTrigger
from doa_calculator import create_doa_calculator
from sound_separator import create_sound_separator
from led_controller import create_led_controller


class SingleSoundPipeline:
    """Single Thread Sound Pipeline - Raspberry Pi용"""
    
    def __init__(self, output_dir: str = "pipeline_output", 
                 model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 device: str = "auto", backend_url: str = "http://13.238.200.232:8000/sound-events/"):
        """
        Sound Pipeline 초기화
        
        Args:
            output_dir: 출력 디렉토리
            model_name: AST 모델 이름
            device: 사용할 디바이스
            backend_url: 백엔드 API URL
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        self.backend_url = backend_url
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 상태 관리
        self.is_running = False
        
        # 중복 클래스 전송 방지를 위한 세트
        self.sent_classes: Set[str] = set()
        
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
        
        # Initialize Sound Separator
        print("3. Initializing Sound Separator...")
        self.sound_separator = create_sound_separator(self.model_name, self.device, self.backend_url)
        
        # Initialize LED Controller
        print("4. Initializing LED Controller...")
        self.led_controller = create_led_controller()
        if self.led_controller is None:
            print("⚠️ LED Controller not available - LED control disabled")
        
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
                    
                    # 2. 음원 분리 및 백엔드 전송
                    separation_result = self._process_separation(recorded_file)
                    
                    if separation_result["success"]:
                        separated_sources = separation_result.get("separated_sources", [])
                        print(f"✅ Separation completed: {len(separated_sources)} sources")
                    else:
                        print(f"❌ Separation failed: {separation_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Main loop error: {e}")
                continue
    
    
    def _process_separation(self, audio_file: str) -> Dict[str, Any]:
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
            
            # 각 패스 완료 시마다 즉시 처리하는 콜백 함수
            def on_pass_complete(source_info):
                class_name = source_info['class_name']
                sound_type = source_info['sound_type']
                confidence = source_info.get('confidence', 0.0)
                pass_num = source_info.get('pass', 0)
                
                print(f"🎵 PASS {pass_num}: {class_name} ({sound_type}) - Confidence: {confidence:.3f}")
                
                # 중복 클래스 체크
                if class_name in self.sent_classes:
                    print(f"⏭️ SKIP: Duplicate class '{class_name}' already processed in previous pass")
                    print(f"    📋 Already sent classes: {list(self.sent_classes)}")
                    self.stats["duplicate_skips"] += 1
                    print("-" * 30)
                    return
                
                # 백엔드 전송
                if self.backend_url:
                    print(f"📡 [SINGLE_PIPELINE] Sending to backend: {class_name} at {angle}°")
                    self._send_to_backend(source_info, angle)
                    self.stats["backend_sends"] += 1
                
                # LED 활성화 (형식에 맞춰서)
                if self.led_controller:
                    print(f"💡 Activating LED: {class_name} ({sound_type})")
                    self.led_controller.activate_led(angle, class_name, sound_type)
                    self.stats["led_activations"] += 1
                
                # 전송된 클래스 기록
                self.sent_classes.add(class_name)
                
                print(f"✅ PASS {pass_num} COMPLETED: {class_name} ({sound_type})")
                print("-" * 30)
            
            # 분리 실행 (각 패스마다 즉시 처리)
            separated_sources = self.sound_separator.separate_audio(audio_data, max_passes=3, on_pass_complete=on_pass_complete)
            
            if separated_sources:
                print(f"✅ Separation completed: {len(separated_sources)} sources")
                self.stats["successful_separations"] += 1
                return {"success": True, "separated_sources": separated_sources}
            else:
                print("❌ No sources separated")
                return {"success": False, "error": "No sources separated"}
                
        except Exception as e:
            print(f"❌ Separation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _send_to_backend(self, source: Dict[str, Any], angle: int):
        """백엔드로 분리된 소리 전송"""
        try:
            import requests
            from datetime import datetime
            
            # 백엔드 전송 데이터 구성
            data = {
                "user_id": 6,
                "sound_type": source['sound_type'],
                "sound_detail": source['class_name'],
                "angle": angle,
                "occurred_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sound_icon": "string",
                "location_image_url": "string",
                "decibel": 60.0  # 기본값, 실제로는 계산된 값 사용
            }
            
            print(f"📤 [SINGLE_PIPELINE] Data: {data}")
            
            # 백엔드로 전송
            response = requests.post(self.backend_url, json=data, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ [SINGLE_PIPELINE] Sent to backend: {source['class_name']} ({source['sound_type']}) at {angle}°")
            else:
                print(f"❌ [SINGLE_PIPELINE] Backend send failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Backend send error: {e}")
    
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
        print(f"Unique classes sent: {len(self.sent_classes)}")
        if self.sent_classes:
            print(f"Sent classes: {list(self.sent_classes)}")
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
