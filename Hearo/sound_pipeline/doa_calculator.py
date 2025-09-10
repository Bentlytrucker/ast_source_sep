#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOA (Direction of Arrival) Calculator Module
- DOA.py 기반으로 구현
- 실시간 각도 계산 기능
"""

import usb.core
import usb.util
import time
import sys
from typing import Optional, Tuple
import os

# DOA.py에서 사용하는 tuning 모듈 import를 위한 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source_separation', 'usb_4_mic_array'))

try:
    from tuning import Tuning
except ImportError:
    print("Warning: tuning module not found. DOA calculation will be disabled.")
    Tuning = None

class DOACalculator:
    def __init__(self):
        """
        DOA Calculator 초기화
        """
        self.dev = None
        self.mic_tuning = None
        self.is_available = False
        
        self._initialize_device()
    
    def _initialize_device(self):
        """ReSpeaker USB 장치 초기화"""
        try:
            # ReSpeaker USB 장치 찾기 (Vendor ID: 0x2886, Product ID: 0x0018)
            self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            
            if self.dev and Tuning:
                self.mic_tuning = Tuning(self.dev)
                self.is_available = True
                print(f"[DOA] ReSpeaker device found and initialized")
                print(f"[DOA] Initial direction: {self.mic_tuning.direction}")
            else:
                print("[DOA] ReSpeaker device not found or tuning module unavailable")
                self.is_available = False
                
        except Exception as e:
            print(f"[DOA] Initialization error: {e}")
            self.is_available = False
    
    def get_direction(self) -> Optional[int]:
        """
        현재 소리 방향 각도 가져오기
        
        Returns:
            각도 (0-359도), None (장치 사용 불가)
        """
        if not self.is_available or not self.mic_tuning:
            return None
        
        try:
            direction = self.mic_tuning.direction
            return direction
        except Exception as e:
            print(f"[DOA] Error getting direction: {e}")
            return None
    
    def get_direction_with_retry(self, max_retries: int = 3) -> Optional[int]:
        """
        재시도 로직이 포함된 각도 가져오기
        
        Args:
            max_retries: 최대 재시도 횟수
            
        Returns:
            각도 (0-359도), None (장치 사용 불가)
        """
        for attempt in range(max_retries):
            direction = self.get_direction()
            if direction is not None:
                return direction
            
            if attempt < max_retries - 1:
                print(f"[DOA] Retry {attempt + 1}/{max_retries}")
                time.sleep(0.1)
        
        return None
    
    def is_device_available(self) -> bool:
        """
        DOA 장치 사용 가능 여부 확인
        
        Returns:
            True (사용 가능), False (사용 불가)
        """
        return self.is_available
    
    def cleanup(self):
        """리소스 정리"""
        # USB 장치는 특별한 정리가 필요하지 않음
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class MockDOACalculator:
    """
    DOA 장치가 없을 때 사용하는 Mock 클래스
    테스트용으로 랜덤 각도 반환
    """
    
    def __init__(self):
        self.is_available = False
        print("[DOA] Mock DOA Calculator initialized (no real device)")
    
    def get_direction(self) -> Optional[int]:
        """Mock 각도 반환 (0-359도 중 랜덤)"""
        import random
        return random.randint(0, 359)
    
    def get_direction_with_retry(self, max_retries: int = 3) -> Optional[int]:
        """Mock 각도 반환"""
        return self.get_direction()
    
    def is_device_available(self) -> bool:
        """Mock 장치는 항상 사용 가능"""
        return True
    
    def cleanup(self):
        """Mock 정리"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def create_doa_calculator() -> DOACalculator:
    """
    DOA Calculator 인스턴스 생성
    장치가 없으면 Mock 버전 반환
    
    Returns:
        DOACalculator 또는 MockDOACalculator 인스턴스
    """
    calculator = DOACalculator()
    
    if not calculator.is_device_available():
        print("[DOA] Real device not available, using mock calculator")
        return MockDOACalculator()
    
    return calculator


def main():
    """테스트용 메인 함수"""
    with create_doa_calculator() as doa:
        print("DOA Calculator 테스트 시작...")
        
        for i in range(10):
            direction = doa.get_direction()
            if direction is not None:
                print(f"Direction {i+1}: {direction}°")
            else:
                print(f"Direction {i+1}: Failed to get direction")
            
            time.sleep(1)
        
        print("DOA Calculator 테스트 완료")


if __name__ == "__main__":
    main()
