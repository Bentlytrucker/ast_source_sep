#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LED 테스트 스크립트
"""

import sys
import os

# pixel_ring 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_separation', 'pixel_ring'))

print('🔍 LED 테스트 시작...')

try:
    from pixel_ring.usb_pixel_ring_v2 import PixelRing
    print('✅ pixel_ring 모듈 import 성공!')
    
    # USB 장치 찾기
    import usb.core
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if dev:
        print('✅ ReSpeaker 장치 발견!')
        
        # PixelRing 초기화 테스트
        try:
            pixel_ring = PixelRing(dev)
            print('✅ PixelRing 초기화 성공!')
            
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
            
            pixel_ring.off()
            print('⚫ LED 꺼짐!')
            print('✅ LED 테스트 완료!')
            
        except Exception as e:
            print(f'❌ PixelRing 초기화 실패: {e}')
            import traceback
            traceback.print_exc()
    else:
        print('❌ ReSpeaker 장치를 찾을 수 없습니다.')
        
except ImportError as e:
    print(f'❌ pixel_ring 모듈 import 실패: {e}')
    print('   경로를 확인해주세요.')
    import traceback
    traceback.print_exc()
