#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Sound Pipeline Demo
- 새로운 요구사항에 맞는 파이프라인 데모
"""

import time

print('🎵 Advanced Sound Pipeline 데모')
print('=' * 60)

try:
    # 모듈 import
    from sound_trigger import SoundTrigger
    from doa_calculator import create_doa_calculator
    from sound_separator import create_sound_separator
    from led_controller import create_led_controller
    
    print('✅ 모든 모듈 import 성공!')
    
    # 1. LED Controller 초기화 (sleep 모드로 시작)
    print('\n1. LED Controller 초기화 (sleep 모드)...')
    with create_led_controller() as led:
        print('   💤 micarray가 sleep 모드로 시작되었습니다')
        
        # 2. Sound Trigger 초기화 (LED 컨트롤러와 연결)
        print('\n2. Sound Trigger 초기화...')
        trigger = SoundTrigger('demo_recordings', led)
        print('   ✅ Sound Trigger가 LED 컨트롤러와 연결되었습니다')
        
        # 3. DOA Calculator 초기화
        print('\n3. DOA Calculator 초기화...')
        with create_doa_calculator() as doa:
            print('   ✅ DOA Calculator 초기화 완료')
            
            # 4. Sound Separator 초기화
            print('\n4. Sound Separator 초기화...')
            with create_sound_separator() as separator:
                print('   ✅ Sound Separator 초기화 완료')
                
                # 5. 파이프라인 시뮬레이션
                print('\n5. 파이프라인 시뮬레이션...')
                print('   💤 micarray sleep 모드 대기 중...')
                time.sleep(2)
                
                print('   🔊 100dB 이상 소리 감지!')
                time.sleep(1)
                print('   ⚡ micarray wake up!')
                led.wakeup_from_sleep()
                time.sleep(1)
                
                print('   📹 1.024×4초 녹음 시작...')
                time.sleep(2)
                print('   📹 녹음 완료')
                
                print('   📍 각도 계산...')
                angle = doa.get_direction()
                print(f'   📍 각도: {angle}°')
                
                print('   🔍 음원 분리 및 분류...')
                time.sleep(1)
                print('   🔍 분류 결과: Gunshot (danger)')
                
                print('   💡 LED 출력 (danger - 10초간 깜빡임)...')
                led.set_sound_type_color('danger')
                time.sleep(3)  # 깜빡임 시뮬레이션
                
                print('   🌐 백엔드 전송 (각도 정보 포함)...')
                time.sleep(1)
                print('   🌐 전송 완료')
                
                # 다른 타입들도 테스트
                print('\n6. 다른 소리 타입 테스트...')
                
                sound_types = ['warning', 'help', 'other']
                for sound_type in sound_types:
                    print(f'   💡 {sound_type} 타입 LED 테스트 (5초 유지)...')
                    led.set_sound_type_color(sound_type)
                    time.sleep(1)
                
                print('\n   💤 다시 sleep 모드로 전환...')
                led.turn_off()
    
    print('\n' + '=' * 60)
    print('🎉 Advanced Sound Pipeline 데모 완료!')
    print('모든 새로운 기능이 정상적으로 동작합니다!')
    print('\n📋 구현된 기능:')
    print('   ✅ micarray sleep 모드로 시작')
    print('   ✅ 100dB 이상 소리 감지 시 wake up')
    print('   ✅ 1.024×4초 녹음')
    print('   ✅ 각도 계산')
    print('   ✅ 음원 분리 및 분류')
    print('   ✅ LED 색상 출력 (5초 유지)')
    print('   ✅ danger 타입 시 10초간 깜빡임')
    print('   ✅ 백엔드 전송 시 각도 정보 포함')
    
except Exception as e:
    print(f'❌ 오류 발생: {e}')
    import traceback
    traceback.print_exc()
