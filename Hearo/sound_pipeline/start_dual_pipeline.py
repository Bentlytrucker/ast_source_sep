#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Thread Sound Pipeline 시작 스크립트
- 한 번 실행하면 자동으로 두 개 터미널에서 각각 실행
"""

import os
import sys
import time
import subprocess
import argparse


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Start Dual Thread Sound Pipeline")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--model", "-m", default="MIT/ast-finetuned-audioset-10-10-0.4593", help="AST model name")
    parser.add_argument("--device", "-d", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--backend-url", default="http://13.238.200.232:8000/sound-events/", help="Backend API URL")
    
    args = parser.parse_args()
    
    print("🎵 Dual Thread Sound Pipeline Launcher v2.0")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Backend URL: {args.backend_url}")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    print("\n🔄 Starting both threads in separate terminals...")
    
    # Windows에서 새 터미널 창들 열기
    if os.name == 'nt':  # Windows
        # Fast Classification Thread
        cmd1 = f'start "Fast Classification Thread" cmd /k "cd /d {os.getcwd()} && python run_fast_classification.py --output {args.output} --model {args.model} --device {args.device}"'
        
        # Source Separation Thread  
        cmd2 = f'start "Source Separation Thread" cmd /k "cd /d {os.getcwd()} && python run_source_separation.py --output {args.output} --model {args.model} --device {args.device} --backend-url {args.backend_url}"'
        
        try:
            subprocess.Popen(cmd1, shell=True)
            print("✅ Fast Classification Thread started in separate terminal")
            time.sleep(2)  # 잠시 대기
            
            subprocess.Popen(cmd2, shell=True)
            print("✅ Source Separation Thread started in separate terminal")
            
        except Exception as e:
            print(f"❌ Failed to start threads: {e}")
            return
    
    else:  # Linux/Mac
        # Fast Classification Thread
        cmd1 = f'gnome-terminal --title="Fast Classification Thread" -- bash -c "cd {os.getcwd()} && python run_fast_classification.py --output {args.output} --model {args.model} --device {args.device}; exec bash"'
        
        # Source Separation Thread
        cmd2 = f'gnome-terminal --title="Source Separation Thread" -- bash -c "cd {os.getcwd()} && python run_source_separation.py --output {args.output} --model {args.model} --device {args.device} --backend-url {args.backend_url}; exec bash"'
        
        try:
            subprocess.Popen(cmd1, shell=True)
            print("✅ Fast Classification Thread started in separate terminal")
            time.sleep(2)  # 잠시 대기
            
            subprocess.Popen(cmd2, shell=True)
            print("✅ Source Separation Thread started in separate terminal")
            
        except Exception as e:
            print(f"❌ Failed to start threads: {e}")
            return
    
    print("\n✅ Both threads started successfully!")
    print("📡 Each thread is running in its own terminal window")
    print("🔴 Fast Classification Thread: Monitors for sounds and lights RED LED for DANGER")
    print("🔍 Source Separation Thread: Processes queued files and sends to backend")
    print("\n💡 To stop the threads:")
    print("   - Close the terminal windows manually")
    print("   - Or press Enter in each terminal window")
    print("\nPress Ctrl+C to exit this launcher")


if __name__ == "__main__":
    try:
        main()
        # 메인 스레드에서 대기
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n🛑 Launcher stopped")
        print("💡 Note: The separate terminal windows are still running")
        print("   Close them manually to stop the threads")
