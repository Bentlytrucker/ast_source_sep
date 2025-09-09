#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Pipeline Main Module
- 전체 파이프라인 통합 및 실행
- 100dB 이상 소리 감지 → 녹음 → 각도 계산 → 음원 분리 → LED 출력
"""

import os
import sys
import time
import threading
import queue
from typing import Optional, Dict, Any
import argparse

# 파이프라인 모듈들 import
from sound_trigger import SoundTrigger
from doa_calculator import create_doa_calculator
from sound_separator import create_sound_separator
from led_controller import create_led_controller


class SoundPipeline:
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
        
        # 컴포넌트들
        self.sound_trigger = None
        self.doa_calculator = None
        self.sound_separator = None
        self.led_controller = None
        
        # 상태 관리
        self.is_running = False
        self.processing_queue = queue.Queue()
        self.worker_thread = None
        
        # 통계
        self.stats = {
            "total_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "danger_detected": 0,
            "warning_detected": 0,
            "help_detected": 0,
            "other_detected": 0
        }
    
    def _initialize_components(self):
        """Initialize all components"""
        print("=== Sound Pipeline Initialization ===")
        
        # Initialize Sound Trigger (connected to LED controller)
        print("1. Initializing Sound Trigger...")
        self.sound_trigger = SoundTrigger(os.path.join(self.output_dir, "recordings"), self.led_controller)
        
        # Initialize DOA Calculator
        print("2. Initializing DOA Calculator...")
        self.doa_calculator = create_doa_calculator()
        
        # Initialize Sound Separator
        print("3. Initializing Sound Separator...")
        self.sound_separator = create_sound_separator(self.model_name, self.device, self.backend_url)
        
        # Initialize LED Controller
        print("4. Initializing LED Controller...")
        self.led_controller = create_led_controller()
        
        print("=== Initialization Complete ===")
        self._print_status()
    
    def _print_status(self):
        """컴포넌트 상태 출력"""
        print("\n=== 컴포넌트 상태 ===")
        print(f"Sound Trigger: {'✅' if self.sound_trigger else '❌'}")
        print(f"DOA Calculator: {'✅' if self.doa_calculator and self.doa_calculator.is_device_available() else '❌'}")
        print(f"Sound Separator: {'✅' if self.sound_separator and self.sound_separator.is_model_available() else '❌'}")
        print(f"LED Controller: {'✅' if self.led_controller and self.led_controller.is_device_available() else '❌'}")
        print("==================\n")
    
    def _process_audio_file(self, audio_file: str) -> Dict[str, Any]:
        """
        오디오 파일 처리
        
        Args:
            audio_file: 처리할 오디오 파일 경로
            
        Returns:
            처리 결과
        """
        print(f"\n🎵 Processing: {os.path.basename(audio_file)}")
        
        try:
            # 1. Calculate DOA
            print("📍 Calculating direction...")
            angle = self.doa_calculator.get_direction_with_retry(max_retries=3)
            if angle is None:
                angle = 0  # Default value
            print(f"📍 Direction: {angle}°")
            
            # 2. Sound separation and classification
            print("🔍 Analyzing sound...")
            separated_output_dir = os.path.join(self.output_dir, "separated")
            result = self.sound_separator.process_audio(audio_file, angle, separated_output_dir)
            
            if not result["success"]:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                return result
            
            # 3. LED control (5 seconds duration, 10 seconds blinking for danger)
            print("💡 Setting LED...")
            sound_type = result["sound_type"]
            led_success = self.led_controller.set_sound_type_color(sound_type)
            
            # 4. Update statistics
            self.stats["total_processed"] += 1
            if result["success"]:
                self.stats["successful_processing"] += 1
                self.stats[f"{sound_type}_detected"] += 1
            else:
                self.stats["failed_processing"] += 1
            
            # 5. Print results
            print(f"✅ Processing completed:")
            print(f"   Class: {result['class_name']}")
            print(f"   Type: {result['sound_type']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Angle: {result['angle']}°")
            print(f"   Decibel: {result['decibel']['mean']:.1f} dB")
            print(f"   Backend: {'✅' if result['backend_success'] else '❌'}")
            print(f"   LED: {'✅' if led_success else '❌'}")
            if result.get('separated_file'):
                print(f"   Separated: {os.path.basename(result['separated_file'])}")
            
            return result
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            self.stats["total_processed"] += 1
            self.stats["failed_processing"] += 1
            return {"success": False, "error": str(e)}
    
    def _worker_thread_func(self):
        """Worker thread function"""
        while self.is_running:
            try:
                # Get work from queue (1 second timeout)
                audio_file = self.processing_queue.get(timeout=1.0)
                
                if audio_file is None:  # Exit signal
                    break
                
                # Process audio file
                self._process_audio_file(audio_file)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                # Timeout - continue running
                continue
            except Exception as e:
                print(f"❌ Worker thread error: {e}")
                continue
    
    def _start_worker_thread(self):
        """Start worker thread"""
        self.worker_thread = threading.Thread(target=self._worker_thread_func, daemon=True)
        self.worker_thread.start()
        print("🔄 Worker thread started")
    
    def _stop_worker_thread(self):
        """Stop worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            # Send exit signal
            self.processing_queue.put(None)
            self.worker_thread.join(timeout=5.0)
            print("🔄 Worker thread stopped")
    
    def start(self):
        """Start pipeline"""
        if self.is_running:
            print("⚠️ Pipeline is already running")
            return
        
        print("🚀 Starting Sound Pipeline...")
        
        # Initialize components
        self._initialize_components()
        
        # Start worker thread
        self.is_running = True
        self._start_worker_thread()
        
        # LED initialization - start in sleep mode
        print("💤 Starting micarray in sleep mode...")
        self.led_controller.turn_off()  # sleep mode
        
        print("✅ Sound Pipeline started successfully!")
        print("📡 Monitoring for sounds above 100dB...")
        print("Press Ctrl+C to stop")
        
        try:
            # Main loop - sound detection and processing
            while self.is_running:
                # Wait for sound detection
                recorded_file = self.sound_trigger.start_monitoring()
                
                if recorded_file and self.is_running:
                    # Add to processing queue
                    self.processing_queue.put(recorded_file)
                    print(f"📝 Added to processing queue: {os.path.basename(recorded_file)}")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping pipeline...")
            self.stop()
    
    def stop(self):
        """Stop pipeline"""
        if not self.is_running:
            print("⚠️ Pipeline is not running")
            return
        
        print("🛑 Stopping Sound Pipeline...")
        
        # Stop running
        self.is_running = False
        
        # Stop worker thread
        self._stop_worker_thread()
        
        # Turn off LED
        if self.led_controller:
            self.led_controller.turn_off()
        
        # Print statistics
        self._print_statistics()
        
        print("✅ Sound Pipeline stopped")
    
    def _print_statistics(self):
        """Print statistics"""
        print("\n=== Processing Statistics ===")
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"Successful: {self.stats['successful_processing']}")
        print(f"Failed: {self.stats['failed_processing']}")
        print(f"Danger detected: {self.stats['danger_detected']}")
        print(f"Warning detected: {self.stats['warning_detected']}")
        print(f"Help detected: {self.stats['help_detected']}")
        print(f"Other detected: {self.stats['other_detected']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful_processing'] / self.stats['total_processed']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print("================\n")
    
    def cleanup(self):
        """Clean up resources"""
        if self.is_running:
            self.stop()
        
        # Clean up components
        if self.sound_trigger:
            self.sound_trigger.cleanup()
        if self.doa_calculator:
            self.doa_calculator.cleanup()
        if self.sound_separator:
            self.sound_separator.cleanup()
        if self.led_controller:
            self.led_controller.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Sound Pipeline - Real-time Sound Detection and Analysis")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--backend-url", default="http://13.238.200.232:8000/sound-events/", help="Backend API URL")
    
    args = parser.parse_args()
    
    print("🎵 Sound Pipeline v1.0")
    print("=" * 50)
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Backend URL: {args.backend_url}")
    print("=" * 50)
    
    # 파이프라인 실행
    with SoundPipeline(
        output_dir=args.output,
        model_name=args.model,
        device=args.device,
        backend_url=args.backend_url
    ) as pipeline:
        pipeline.start()


if __name__ == "__main__":
    main()
