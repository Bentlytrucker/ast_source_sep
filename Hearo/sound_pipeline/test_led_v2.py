#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB PixelRing V2 직접 테스트
"""

import sys
import os

# usb_pixel_ring_v2.py 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_separation', 'pixel_ring', 'pixel_ring'))

print('🔍 USB PixelRing V2 테스트 시작...')

try:
    from usb_pixel_ring_v2 import PixelRing, find
    print('✅ USB PixelRing V2 모듈 import 성공!')
    
    # ReSpeaker 장치 찾기
    pixel_ring = find()
    if pixel_ring:
        print('✅ ReSpeaker 장치 발견 및 PixelRing 초기화 성공!')
        
        # LED 테스트
        pixel_ring.set_brightness(0x001)
        
        print('🔴 빨간색 LED 테스트...')
        pixel_ring.mono(0xFF0000)  # 빨간색
        import time
        time.sleep(3)
        
        print('🟢 초록색 LED 테스트...')
        pixel_ring.mono(0x00FF00)  # 초록색
        time.sleep(3)
        
        print('🟡 노란색 LED 테스트...')
        pixel_ring.mono(0xFFFF00)  # 노란색
        time.sleep(3)
        
        print('🔵 파란색 LED 테스트...')
        pixel_ring.mono(0x0000FF)  # 파란색
        time.sleep(3)
        
        print('⚡ Wake up 애니메이션 테스트...')
        pixel_ring.wakeup(180)
        time.sleep(3)
        
        print('👂 Listen 애니메이션 테스트...')
        pixel_ring.listen()
        time.sleep(3)
        
        print('💭 Think 애니메이션 테스트...')
        pixel_ring.think()
        time.sleep(3)
        
        pixel_ring.off()
        print('⚫ LED 꺼짐!')
        print('✅ LED 테스트 완료!')
        
    else:
        print('❌ ReSpeaker 장치를 찾을 수 없습니다.')
        
except ImportError as e:
    print(f'❌ USB PixelRing V2 모듈 import 실패: {e}')
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f'❌ LED 테스트 실패: {e}')
    import traceback
    traceback.print_exc()
