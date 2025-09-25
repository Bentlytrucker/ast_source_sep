#!/usr/bin/env python3
"""
LED Controller 테스트 스크립트
"""

import sys
import os
import time

# 현재 디렉토리를 path에 추가
sys.path.append(os.path.dirname(__file__))

from led_controller import create_led_controller

def test_led_controller():
    """LED 컨트롤러 테스트"""
    print("🧪 LED Controller 테스트 시작...")
    print("=" * 50)
    
    # LED 컨트롤러 생성
    led_controller = create_led_controller()
    
    if led_controller is None:
        print("❌ LED 컨트롤러 생성 실패")
        return False
    
    print(f"✅ LED 컨트롤러 생성 성공")
    print(f"🔍 하드웨어 사용 가능: {led_controller.is_device_available()}")
    
    # 기본 기능 테스트
    print("\n📋 기본 기능 테스트:")
    
    try:
        # 1. 색상 설정 테스트
        print("1. 빨간색 LED 테스트...")
        result = led_controller.set_color(0xFF0000, duration=2.0)
        print(f"   결과: {'성공' if result else '실패'}")
        
        time.sleep(0.5)
        
        # 2. 소리 타입별 색상 테스트
        sound_types = ["danger", "warning", "help"]
        for sound_type in sound_types:
            print(f"2. {sound_type} 타입 LED 테스트...")
            result = led_controller.set_sound_type_color(sound_type, duration=1.0)
            print(f"   결과: {'성공' if result else '실패'}")
            time.sleep(0.5)
        
        # 3. 방향성 LED 테스트
        print("3. 방향성 LED 테스트 (각도: 90°)...")
        result = led_controller.activate_led(90, "test_sound", "danger")
        print(f"   결과: {'성공' if result else '실패'}")
        
        time.sleep(1)
        
        # 4. LED 끄기
        print("4. LED 끄기 테스트...")
        result = led_controller.turn_off()
        print(f"   결과: {'성공' if result else '실패'}")
        
        print("\n✅ 모든 테스트 완료")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 정리
        try:
            led_controller.cleanup()
        except:
            pass

if __name__ == "__main__":
    success = test_led_controller()
    sys.exit(0 if success else 1)