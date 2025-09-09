#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test Script for Sound Pipeline
"""

print('🎵 Sound Pipeline 빠른 테스트')
print('=' * 50)

try:
    # 모듈 import 테스트
    from sound_trigger import SoundTrigger
    from doa_calculator import create_doa_calculator
    from sound_separator import create_sound_separator
    from led_controller import create_led_controller
    
    print('✅ 모든 모듈 import 성공!')
    
    # 각 컴포넌트 초기화 테스트
    print('\n1. Sound Trigger 초기화...')
    trigger = SoundTrigger('test_output')
    print('   ✅ Sound Trigger 초기화 성공')
    
    print('\n2. DOA Calculator 초기화...')
    with create_doa_calculator() as doa:
        print('   ✅ DOA Calculator 초기화 성공')
        angle = doa.get_direction()
        print(f'   📍 각도: {angle}°')
    
    print('\n3. Sound Separator 초기화...')
    with create_sound_separator() as separator:
        print('   ✅ Sound Separator 초기화 성공')
        print(f'   🤖 모델 사용 가능: {separator.is_model_available()}')
    
    print('\n4. LED Controller 초기화...')
    with create_led_controller() as led:
        print('   ✅ LED Controller 초기화 성공')
        led.set_sound_type_color('danger', duration=1.0)
        print('   💡 LED 테스트 완료')
    
    print('\n' + '=' * 50)
    print('🎉 모든 컴포넌트 초기화 성공!')
    print('파이프라인이 정상적으로 동작할 준비가 되었습니다!')
    
except Exception as e:
    print(f'❌ 오류 발생: {e}')
    import traceback
    traceback.print_exc()
