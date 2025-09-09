#!/usr/bin/env python3


import argparse
import sys
import os
import logging
from webcam_detector import WebcamDetector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description='Yolo-X TFLite 모델을 사용한 실시간 웹캠 객체 탐지',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
키보드 단축키:
  ESC 또는 'q': 프로그램 종료
  'c': 신뢰도 임계값 감소 (0.1씩)
  'v': 신뢰도 임계값 증가 (0.1씩)

예시:
  python main.py
  python main.py --model Yolo-X_w8a8.tflite --camera 0 --confidence 0.6
  python main.py --help
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='Yolo-X_w8a8.tflite',
        help='TFLite 모델 파일 경로 (기본값: Yolo-X_w8a8.tflite)'
    )
    
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
        help='웹캠 인덱스 (기본값: 0)'
    )
    
    parser.add_argument(
        '--confidence', 
        type=float, 
        default=0.6,
        help='신뢰도 임계값 (기본값: 0.5)'
    )
    
    parser.add_argument(
        '--nms', 
        type=float, 
        default=0.4,
        help='NMS 임계값 (기본값: 0.4)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """인수 유효성 검사"""
    # 모델 파일 존재 확인
    if not os.path.exists(args.model):
        logger.error(f"모델 파일을 찾을 수 없습니다: {args.model}")
        return False
    
    # 신뢰도 임계값 범위 확인
    if not 0.0 <= args.confidence <= 1.0:
        logger.error(f"신뢰도 임계값은 0.0과 1.0 사이여야 합니다: {args.confidence}")
        return False
    
    # NMS 임계값 범위 확인
    if not 0.0 <= args.nms <= 1.0:
        logger.error(f"NMS 임계값은 0.0과 1.0 사이여야 합니다: {args.nms}")
        return False
    
    # 웹캠 인덱스 확인
    if args.camera < 0:
        logger.error(f"웹캠 인덱스는 0 이상이어야 합니다: {args.camera}")
        return False
    
    return True


def print_system_info():
    """시스템 정보 출력"""
    import cv2
    import tensorflow as tf
    import numpy as np
    
    logger.info("=== 시스템 정보 ===")
    logger.info(f"OpenCV 버전: {cv2.__version__}")
    logger.info(f"TensorFlow 버전: {tf.__version__}")
    logger.info(f"NumPy 버전: {np.__version__}")
    logger.info(f"Python 버전: {sys.version}")
    
    # GPU 정보 확인
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            logger.info(f"GPU 디바이스: {len(gpu_devices)}개 발견")
            for i, device in enumerate(gpu_devices):
                logger.info(f"  GPU {i}: {device.name}")
        else:
            logger.info("GPU 디바이스: 없음 (CPU 모드)")
    except Exception as e:
        logger.warning(f"GPU 정보 확인 실패: {e}")


def main():
    """메인 함수"""
    try:
        # 인수 파싱
        args = parse_arguments()
        
        # 인수 유효성 검사
        if not validate_arguments(args):
            sys.exit(1)
        
        # 시스템 정보 출력
        print_system_info()
        
        # 설정 정보 출력
        logger.info("=== 실행 설정 ===")
        logger.info(f"모델 파일: {args.model}")
        logger.info(f"웹캠 인덱스: {args.camera}")
        logger.info(f"신뢰도 임계값: {args.confidence}")
        logger.info(f"NMS 임계값: {args.nms}")
        
        # 웹캠 탐지기 생성 및 실행
        detector = WebcamDetector(
            model_path=args.model,
            camera_index=args.camera,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms
        )
        
        logger.info("실시간 객체 탐지를 시작합니다...")
        detector.run()
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        logger.error(f"예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)
    
    logger.info("프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
