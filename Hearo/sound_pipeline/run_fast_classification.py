#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Classification Thread 실행 스크립트
- 빠른 분류 + Danger 시 빨간 LED
"""

import os
import sys
import time
import argparse
from sound_pipeline_dual import FastClassificationThread


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fast Classification Thread - Quick classification + Danger LED")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    print("🚀 Fast Classification Thread v2.0")
    print("=" * 50)
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # Fast Classification Thread 시작
    thread = FastClassificationThread(args.output, args.model, args.device)
    
    try:
        thread.start()
        print("\n✅ Fast Classification Thread started successfully!")
        print("📡 Monitoring for sounds above 100dB...")
        print("🔴 Will immediately light RED LED for DANGER sounds")
        print("Press Enter to stop...")
        
        # 사용자 입력 대기
        input()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping Fast Classification Thread...")
    finally:
        thread.stop()
        print("✅ Fast Classification Thread stopped")


if __name__ == "__main__":
    main()
