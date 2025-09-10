#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source Separation Thread 실행 스크립트
- 음원 분리 + 각 패스마다 백엔드 전송 + LED 활성화
"""

import os
import sys
import time
import argparse
from sound_pipeline_dual import SourceSeparationThread


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Source Separation Thread - Audio separation + Backend + LED")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--backend-url", default="http://13.238.200.232:8000/sound-events/", help="Backend API URL")
    
    args = parser.parse_args()
    
    print("🚀 Source Separation Thread v2.0")
    print("=" * 50)
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Backend URL: {args.backend_url}")
    print("=" * 50)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # Source Separation Thread 시작
    thread = SourceSeparationThread(args.output, args.model, args.device, args.backend_url)
    
    try:
        thread.start()
        print("\n✅ Source Separation Thread started successfully!")
        print("🔍 Processing queued audio files for separation...")
        print("📡 Will send each separated source to backend")
        print("💡 Will activate LED for each separated source")
        print("Press Enter to stop...")
        
        # 사용자 입력 대기
        input()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Source Separation Thread...")
    finally:
        thread.stop()
        print("✅ Source Separation Thread stopped")


if __name__ == "__main__":
    main()
