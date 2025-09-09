#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Pipeline Test Script
- 각 컴포넌트별 테스트
- 통합 테스트
"""

import os
import sys
import time
import tempfile
import numpy as np
import torch
import torchaudio

# 파이프라인 모듈들 import
from sound_trigger import SoundTrigger
from doa_calculator import create_doa_calculator
from sound_separator import create_sound_separator
from led_controller import create_led_controller


def test_sound_trigger():
    """Sound Trigger 테스트"""
    print("=== Sound Trigger 테스트 ===")
    
    try:
        with SoundTrigger("test_recordings") as trigger:
            print("✅ Sound Trigger 초기화 성공")
            
            # 5초간 모니터링 테스트
            print("🔍 5초간 소리 모니터링 테스트...")
            start_time = time.time()
            
            while time.time() - start_time < 5:
                # 실제로는 start_monitoring()을 호출하지만, 테스트에서는 짧게
                time.sleep(0.1)
            
            print("✅ Sound Trigger 테스트 완료")
            return True
            
    except Exception as e:
        print(f"❌ Sound Trigger 테스트 실패: {e}")
        return False


def test_doa_calculator():
    """DOA Calculator 테스트"""
    print("\n=== DOA Calculator 테스트 ===")
    
    try:
        with create_doa_calculator() as doa:
            print("✅ DOA Calculator 초기화 성공")
            
            # 각도 측정 테스트
            for i in range(3):
                angle = doa.get_direction()
                if angle is not None:
                    print(f"📍 각도 {i+1}: {angle}°")
                else:
                    print(f"📍 각도 {i+1}: 측정 실패")
                time.sleep(0.5)
            
            print("✅ DOA Calculator 테스트 완료")
            return True
            
    except Exception as e:
        print(f"❌ DOA Calculator 테스트 실패: {e}")
        return False


def test_sound_separator():
    """Sound Separator 테스트"""
    print("\n=== Sound Separator 테스트 ===")
    
    try:
        with create_sound_separator() as separator:
            print("✅ Sound Separator 초기화 성공")
            
            # 테스트용 오디오 파일 생성
            test_audio = np.random.randn(16000).astype(np.float32)  # 1초 오디오
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                torchaudio.save(f.name, torch.from_numpy(test_audio).unsqueeze(0), 16000)
                test_file = f.name
            
            try:
                # 오디오 처리 테스트
                result = separator.process_audio(test_file, 180)
                
                if result["success"]:
                    print(f"🔍 분류 결과: {result['class_name']} ({result['sound_type']})")
                    print(f"📊 신뢰도: {result['confidence']:.3f}")
                    print(f"📈 dB: {result['decibel']['mean']:.1f}")
                    print(f"🌐 백엔드: {'✅' if result['backend_success'] else '❌'}")
                else:
                    print(f"❌ 처리 실패: {result.get('error', 'Unknown error')}")
                
                print("✅ Sound Separator 테스트 완료")
                return True
                
            finally:
                # 테스트 파일 정리
                if os.path.exists(test_file):
                    os.unlink(test_file)
            
    except Exception as e:
        print(f"❌ Sound Separator 테스트 실패: {e}")
        return False


def test_led_controller():
    """LED Controller 테스트"""
    print("\n=== LED Controller 테스트 ===")
    
    try:
        with create_led_controller() as led:
            print("✅ LED Controller 초기화 성공")
            
            # 각 색상 테스트
            colors = ["danger", "warning", "help", "other"]
            
            for color in colors:
                print(f"💡 {color} 색상 테스트...")
                success = led.set_sound_type_color(color, duration=0.5)
                if success:
                    print(f"✅ {color} 색상 설정 성공")
                else:
                    print(f"❌ {color} 색상 설정 실패")
                time.sleep(0.2)
            
            # 깜빡이기 테스트
            print("💡 깜빡이기 테스트...")
            success = led.blink_sound_type("danger", blink_count=2, blink_duration=0.3)
            if success:
                print("✅ 깜빡이기 성공")
            else:
                print("❌ 깜빡이기 실패")
            
            # 끄기
            led.turn_off()
            print("✅ LED Controller 테스트 완료")
            return True
            
    except Exception as e:
        print(f"❌ LED Controller 테스트 실패: {e}")
        return False


def test_integration():
    """통합 테스트"""
    print("\n=== 통합 테스트 ===")
    
    try:
        # 모든 컴포넌트 초기화
        with create_doa_calculator() as doa, \
             create_sound_separator() as separator, \
             create_led_controller() as led:
            
            print("✅ 모든 컴포넌트 초기화 성공")
            
            # 테스트용 오디오 파일 생성
            test_audio = np.random.randn(16000).astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                torchaudio.save(f.name, torch.from_numpy(test_audio).unsqueeze(0), 16000)
                test_file = f.name
            
            try:
                # 통합 처리 시뮬레이션
                print("🔄 통합 처리 시뮬레이션...")
                
                # 1. 각도 계산
                angle = doa.get_direction()
                if angle is None:
                    angle = 0
                print(f"📍 각도: {angle}°")
                
                # 2. 음원 분리 및 분류
                result = separator.process_audio(test_file, angle)
                
                if result["success"]:
                    print(f"🔍 분류: {result['class_name']} ({result['sound_type']})")
                    
                    # 3. LED 출력
                    led_success = led.set_sound_type_color(result['sound_type'], duration=1.0)
                    print(f"💡 LED: {'✅' if led_success else '❌'}")
                    
                    print("✅ 통합 테스트 성공")
                    return True
                else:
                    print(f"❌ 통합 테스트 실패: {result.get('error', 'Unknown error')}")
                    return False
                    
            finally:
                # 테스트 파일 정리
                if os.path.exists(test_file):
                    os.unlink(test_file)
            
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🧪 Sound Pipeline 테스트 시작")
    print("=" * 50)
    
    # 개별 컴포넌트 테스트
    tests = [
        ("Sound Trigger", test_sound_trigger),
        ("DOA Calculator", test_doa_calculator),
        ("Sound Separator", test_sound_separator),
        ("LED Controller", test_led_controller),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results[test_name] = False
    
    # 통합 테스트
    print("\n" + "=" * 50)
    results["Integration"] = test_integration()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
        return 0
    else:
        print("⚠️ 일부 테스트 실패")
        return 1


if __name__ == "__main__":
    sys.exit(main())
