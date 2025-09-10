#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Thread Sound Pipeline - Raspberry Pi용
- Fast Classification Thread: 빠른 분류 + Danger 시 빨간 LED
- Source Separation Thread: 음원 분리 + 각 패스마다 백엔드 전송 + LED 활성화
"""

import os
import sys
import time
import threading
import queue
import argparse
from typing import Optional, Dict, Any

# 파이프라인 모듈들 import
from sound_trigger import SoundTrigger
from doa_calculator import create_doa_calculator
from sound_separator import create_sound_separator
from led_controller import create_led_controller


class FastClassificationThread:
    """빠른 분류 스레드 - Danger 감지 시 즉시 빨간 LED"""
    
    def __init__(self, output_dir: str, model_name: str, device: str):
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        self.is_running = False
        self.thread = None
        
        # 컴포넌트들
        self.sound_trigger = None
        self.doa_calculator = None
        self.sound_separator = None
        self.led_controller = None
        
        # 통계
        self.stats = {
            "total_detected": 0,
            "danger_detected": 0,
            "fast_classifications": 0
        }
    
    def _initialize_components(self):
        """빠른 분류용 컴포넌트 초기화"""
        print("=== Fast Classification Thread Initialization ===")
        
        # Initialize Sound Trigger
        print("1. Initializing Sound Trigger...")
        self.sound_trigger = SoundTrigger(os.path.join(self.output_dir, "recordings"), None)
        
        # Initialize DOA Calculator
        print("2. Initializing DOA Calculator...")
        self.doa_calculator = create_doa_calculator()
        
        # Initialize Sound Separator (for fast classification only)
        print("3. Initializing Sound Separator...")
        self.sound_separator = create_sound_separator(self.model_name, self.device, None)  # No backend
        
        # Initialize LED Controller
        print("4. Initializing LED Controller...")
        self.led_controller = create_led_controller()
        if self.led_controller is None:
            print("⚠️ LED Controller not available - LED control disabled")
        
        print("=== Fast Classification Thread Ready ===")
    
    def _fast_classify(self, audio_file: str) -> Dict[str, Any]:
        """빠른 분류 수행"""
        try:
            # 1. Calculate DOA
            angle = self.doa_calculator.get_direction_with_retry(max_retries=2)
            if angle is None:
                angle = 0
            
            # 2. Fast classification (no separation)
            result = self.sound_separator.process_audio(audio_file, angle, None)  # No output dir
            
            if result["success"]:
                sound_type = result["sound_type"]
                class_name = result["class_name"]
                confidence = result["confidence"]
                
                # 3. Danger 감지 시 즉시 빨간 LED (형식에 맞춰서)
                if sound_type == "danger":
                    print(f"🚨 DANGER DETECTED! {class_name} (confidence: {confidence:.3f})")
                    if self.led_controller:
                        self.led_controller.activate_led(angle, class_name, sound_type)
                    self.stats["danger_detected"] += 1
                else:
                    print(f"✅ {sound_type.upper()}: {class_name} (confidence: {confidence:.3f})")
                
                self.stats["fast_classifications"] += 1
                
                return {
                    "success": True,
                    "sound_type": sound_type,
                    "class_name": class_name,
                    "confidence": confidence,
                    "angle": angle,
                    "is_danger": sound_type == "danger"
                }
            else:
                print(f"❌ Fast classification failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"❌ Fast classification error: {e}")
            return {"success": False, "error": str(e)}
    
    def _run(self):
        """스레드 실행 함수"""
        print("🚀 Fast Classification Thread started!")
        print("📡 Monitoring for sounds above 100dB...")
        print("🔴 Will immediately light RED LED for DANGER sounds")
        print("Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                # Wait for sound detection
                recorded_file = self.sound_trigger.start_monitoring()
                
                if recorded_file and self.is_running:
                    self.stats["total_detected"] += 1
                    print(f"\n🎵 Fast classifying: {os.path.basename(recorded_file)}")
                    
                    # Fast classification
                    result = self._fast_classify(recorded_file)
                    
                    if result["success"]:
                        print(f"📍 Direction: {result['angle']}°")
                        print(f"⚡ Fast classification completed in {time.time():.2f}s")
                    
                    # 파일을 Source Separation Thread로 전달
                    self._send_to_separation_thread(recorded_file)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping Fast Classification Thread...")
            self.stop()
    
    def _send_to_separation_thread(self, audio_file: str):
        """Source Separation Thread로 파일 전달"""
        # IPC 통신 (파일 시스템 기반)
        separation_queue_dir = os.path.join(self.output_dir, "separation_queue")
        os.makedirs(separation_queue_dir, exist_ok=True)
        
        # 큐 파일 생성
        queue_file = os.path.join(separation_queue_dir, f"queue_{int(time.time() * 1000)}.txt")
        with open(queue_file, 'w') as f:
            f.write(audio_file)
        
        print(f"📤 Sent to separation thread: {os.path.basename(audio_file)}")
    
    def start(self):
        """스레드 시작"""
        if self.is_running:
            print("⚠️ Fast Classification Thread is already running")
            return
        
        self._initialize_components()
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("✅ Fast Classification Thread started successfully!")
    
    def stop(self):
        """스레드 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        # LED 끄기
        if self.led_controller:
            self.led_controller.turn_off()
        
        print("✅ Fast Classification Thread stopped")
        self._print_statistics()
    
    def _print_statistics(self):
        """통계 출력"""
        print("\n=== Fast Classification Statistics ===")
        print(f"Total detected: {self.stats['total_detected']}")
        print(f"Danger detected: {self.stats['danger_detected']}")
        print(f"Fast classifications: {self.stats['fast_classifications']}")
        print("=====================================\n")


class SourceSeparationThread:
    """음원 분리 스레드 - 각 패스마다 백엔드 전송 + LED 활성화"""
    
    def __init__(self, output_dir: str, model_name: str, device: str, backend_url: str):
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        self.backend_url = backend_url
        self.is_running = False
        self.thread = None
        
        # 컴포넌트들
        self.doa_calculator = None
        self.sound_separator = None
        self.led_controller = None
        
        # 통계
        self.stats = {
            "total_processed": 0,
            "successful_separations": 0,
            "backend_sends": 0,
            "led_activations": 0
        }
    
    def _initialize_components(self):
        """음원 분리용 컴포넌트 초기화"""
        print("=== Source Separation Thread Initialization ===")
        
        # Initialize DOA Calculator
        print("1. Initializing DOA Calculator...")
        self.doa_calculator = create_doa_calculator()
        
        # Initialize Sound Separator (with backend)
        print("2. Initializing Sound Separator...")
        self.sound_separator = create_sound_separator(self.model_name, self.device, self.backend_url)
        
        # Initialize LED Controller
        print("3. Initializing LED Controller...")
        self.led_controller = create_led_controller()
        if self.led_controller is None:
            print("⚠️ LED Controller not available - LED control disabled")
        
        print("=== Source Separation Thread Ready ===")
    
    def _process_separation(self, audio_file: str) -> Dict[str, Any]:
        """음원 분리 및 각 패스마다 백엔드 전송"""
        try:
            # 1. Calculate DOA
            angle = self.doa_calculator.get_direction_with_retry(max_retries=2)
            if angle is None:
                angle = 0
            
            # 2. Sound separation with backend sending
            separated_output_dir = os.path.join(self.output_dir, "separated")
            result = self.sound_separator.process_audio(audio_file, angle, separated_output_dir)
            
            if result["success"]:
                separated_sources = result.get("separated_sources", [])
                
                # 3. 각 분리된 소리마다 백엔드 전송 및 LED 활성화
                for i, source in enumerate(separated_sources):
                    if source.get('audio') is not None:
                        print(f"🎵 Processing separated source {i+1}: {source['class_name']}")
                        
                        # 백엔드 전송
                        if self.backend_url:
                            self._send_to_backend(source, angle)
                            self.stats["backend_sends"] += 1
                        
                        # LED 활성화 (형식에 맞춰서)
                        if self.led_controller:
                            self.led_controller.activate_led(angle, source['class_name'], source['sound_type'])
                            self.stats["led_activations"] += 1
                        
                        print(f"✅ Source {i+1} processed: {source['class_name']} ({source['sound_type']})")
                
                self.stats["successful_separations"] += 1
                return result
            else:
                print(f"❌ Separation failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"❌ Separation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _send_to_backend(self, source: Dict[str, Any], angle: int):
        """백엔드로 분리된 소리 전송"""
        try:
            # TODO: 백엔드 전송 로직 구현
            print(f"📡 Sending to backend: {source['class_name']} at {angle}°")
            # 실제 백엔드 전송 코드는 sound_separator.py에 구현됨
        except Exception as e:
            print(f"❌ Backend send error: {e}")
    
    def _run(self):
        """스레드 실행 함수"""
        print("🚀 Source Separation Thread started!")
        print("🔍 Processing queued audio files for separation...")
        print("📡 Will send each separated source to backend")
        print("💡 Will activate LED for each separated source")
        print("Press Ctrl+C to stop")
        
        separation_queue_dir = os.path.join(self.output_dir, "separation_queue")
        os.makedirs(separation_queue_dir, exist_ok=True)
        
        try:
            while self.is_running:
                # 큐에서 파일 확인
                queue_files = [f for f in os.listdir(separation_queue_dir) if f.endswith('.txt')]
                
                if queue_files:
                    # 가장 오래된 파일 처리
                    queue_files.sort()
                    queue_file = os.path.join(separation_queue_dir, queue_files[0])
                    
                    try:
                        with open(queue_file, 'r') as f:
                            audio_file = f.read().strip()
                        
                        if os.path.exists(audio_file):
                            self.stats["total_processed"] += 1
                            print(f"\n🎵 Separating: {os.path.basename(audio_file)}")
                            
                            # 음원 분리 처리
                            result = self._process_separation(audio_file)
                            
                            if result["success"]:
                                print(f"✅ Separation completed: {len(result.get('separated_sources', []))} sources")
                            else:
                                print(f"❌ Separation failed: {result.get('error', 'Unknown error')}")
                        
                        # 큐 파일 삭제
                        os.remove(queue_file)
                        
                    except Exception as e:
                        print(f"❌ Queue processing error: {e}")
                        if os.path.exists(queue_file):
                            os.remove(queue_file)
                else:
                    # 큐가 비어있으면 잠시 대기
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping Source Separation Thread...")
            self.stop()
    
    def start(self):
        """스레드 시작"""
        if self.is_running:
            print("⚠️ Source Separation Thread is already running")
            return
        
        self._initialize_components()
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("✅ Source Separation Thread started successfully!")
    
    def stop(self):
        """스레드 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        
        # LED 끄기
        if self.led_controller:
            self.led_controller.turn_off()
        
        print("✅ Source Separation Thread stopped")
        self._print_statistics()
    
    def _print_statistics(self):
        """통계 출력"""
        print("\n=== Source Separation Statistics ===")
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"Successful separations: {self.stats['successful_separations']}")
        print(f"Backend sends: {self.stats['backend_sends']}")
        print(f"LED activations: {self.stats['led_activations']}")
        print("=====================================\n")


class DualSoundPipeline:
    """Dual Thread Sound Pipeline - Raspberry Pi용"""
    
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
        
        # 스레드들
        self.fast_classification_thread = None
        self.source_separation_thread = None
        
        # 상태 관리
        self.is_running = False
    
    def start(self):
        """파이프라인 시작"""
        if self.is_running:
            print("⚠️ Pipeline is already running")
            return
        
        print("🚀 Starting Dual Thread Sound Pipeline...")
        print("=" * 60)
        print("Thread 1: Fast Classification (Danger LED)")
        print("Thread 2: Source Separation (Backend + LED)")
        print("=" * 60)
        
        # 스레드들 초기화
        self.fast_classification_thread = FastClassificationThread(
            self.output_dir, self.model_name, self.device
        )
        self.source_separation_thread = SourceSeparationThread(
            self.output_dir, self.model_name, self.device, self.backend_url
        )
        
        # 스레드들 시작
        self.is_running = True
        
        print("\n🔄 Starting Fast Classification Thread...")
        self.fast_classification_thread.start()
        
        print("\n🔄 Starting Source Separation Thread...")
        self.source_separation_thread.start()
        
        print("\n✅ Dual Thread Sound Pipeline started successfully!")
        print("📡 Both threads are now running independently")
        print("Press Ctrl+C to stop")
        
        try:
            # 메인 스레드에서 대기
            while self.is_running:
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping pipeline...")
            self.stop()
    
    def stop(self):
        """파이프라인 중지"""
        if not self.is_running:
            print("⚠️ Pipeline is not running")
            return
        
        print("🛑 Stopping Dual Thread Sound Pipeline...")
        
        # 스레드들 중지
        if self.fast_classification_thread:
            self.fast_classification_thread.stop()
        
        if self.source_separation_thread:
            self.source_separation_thread.stop()
        
        self.is_running = False
        print("✅ Dual Thread Sound Pipeline stopped")
    
    def cleanup(self):
        """리소스 정리"""
        if self.is_running:
            self.stop()
        
        # 컴포넌트 정리
        if self.fast_classification_thread:
            if self.fast_classification_thread.sound_trigger:
                self.fast_classification_thread.sound_trigger.cleanup()
            if self.fast_classification_thread.doa_calculator:
                self.fast_classification_thread.doa_calculator.cleanup()
            if self.fast_classification_thread.sound_separator:
                self.fast_classification_thread.sound_separator.cleanup()
            if self.fast_classification_thread.led_controller:
                self.fast_classification_thread.led_controller.cleanup()
        
        if self.source_separation_thread:
            if self.source_separation_thread.doa_calculator:
                self.source_separation_thread.doa_calculator.cleanup()
            if self.source_separation_thread.sound_separator:
                self.source_separation_thread.sound_separator.cleanup()
            if self.source_separation_thread.led_controller:
                self.source_separation_thread.led_controller.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dual Thread Sound Pipeline - Fast Classification + Source Separation")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--backend-url", default="http://13.238.200.232:8000/sound-events/", help="Backend API URL")
    
    args = parser.parse_args()
    
    print("🎵 Dual Thread Sound Pipeline v2.0")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Backend URL: {args.backend_url}")
    print("=" * 60)
    
    # 파이프라인 실행
    with DualSoundPipeline(
        output_dir=args.output,
        model_name=args.model,
        device=args.device,
        backend_url=args.backend_url
    ) as pipeline:
        pipeline.start()


if __name__ == "__main__":
    main()
