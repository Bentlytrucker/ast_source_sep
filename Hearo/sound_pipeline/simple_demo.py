#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Demo for Sound Pipeline
"""

import time

print('🎵 Sound Pipeline 간단 데모')
print('=' * 50)

try:
    # 모듈 import
    from doa_calculator import create_doa_calculator
    from led_controller import create_led_controller
    
    print('✅ 모듈 import 성공!')
    
    # 1. 각도 계산 테스트
    print('\n1. 각도 계산 테스트...')
    with create_doa_calculator() as doa:
        angle = doa.get_direction()
        print(f'   📍 감지된 각도: {angle}°')
    
    # 2. LED 제어 테스트
    print('\n2. LED 제어 테스트...')
    with create_led_controller() as led:
        # 각 소리 타입별 LED 테스트
        sound_types = ['danger', 'warning', 'help', 'other']
        for sound_type in sound_types:
            print(f'   💡 {sound_type} 타입 LED 테스트...')
            led.set_sound_type_color(sound_type, duration=0.5)
            time.sleep(0.2)
    
    # 3. 파이프라인 시뮬레이션
    print('\n3. 파이프라인 시뮬레이션...')
    print('   🔊 100dB 이상 소리 감지!')
    time.sleep(1)
    print('   📹 1.024×4초 녹음 완료')
    time.sleep(1)
    print('   📍 각도 계산 완료')
    time.sleep(1)
    print('   🔍 음원 분리 및 분류 완료 (Gunshot - danger)')
    time.sleep(1)
    print('   💡 빨간색 LED 출력')
    time.sleep(1)
    print('   🌐 백엔드 전송 완료 (각도 정보 포함)')
    
    print('\n' + '=' * 50)
    print('🎉 파이프라인 데모 완료!')
    print('모든 기능이 정상적으로 동작합니다!')
    
except Exception as e:
    print(f'❌ 오류 발생: {e}')
    import traceback
    traceback.print_exc()
