#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous Recording Demo for Voice Recognition
Demonstrates the updated SoundTrigger with continuous recording mode
"""

import os
import sys
import time
import argparse
from sound_trigger import SoundTrigger

def demo_continuous_recording(duration: float = 10.0, output_dir: str = "demo_recordings"):
    """
    Demonstrate continuous recording with microphone array
    
    Args:
        duration: Recording duration in seconds
        output_dir: Output directory for recordings
    """
    print("Continuous Recording Demo for Voice Recognition")
    print("=" * 50)
    print(f"Recording duration: {duration} seconds")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SoundTrigger in continuous mode
    try:
        with SoundTrigger(output_dir=output_dir, continuous_mode=True) as trigger:
            print("SoundTrigger initialized successfully!")
            print("Starting continuous recording...")
            print("This will record ALL sounds, not just triggered events")
            print()
            
            # Start continuous recording
            recorded_file = trigger.start_continuous_recording(duration_seconds=duration)
            
            if recorded_file:
                print(f"\n✅ Recording completed successfully!")
                print(f"📁 File saved: {recorded_file}")
                
                # Show file info
                file_size = os.path.getsize(recorded_file)
                print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                
                # Estimate data rate
                data_rate = file_size / duration
                print(f"📈 Data rate: {data_rate:.0f} bytes/sec ({data_rate/1024:.1f} KB/sec)")
                
                return recorded_file
            else:
                print("❌ Recording failed!")
                return None
                
    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted by user")
        return None
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        return None

def demo_triggered_vs_continuous():
    """Compare triggered vs continuous recording modes"""
    print("\n" + "=" * 50)
    print("Comparing Triggered vs Continuous Recording Modes")
    print("=" * 50)
    
    # Test triggered mode
    print("\n1. Testing Triggered Mode (traditional)")
    try:
        with SoundTrigger(output_dir="demo_recordings", continuous_mode=False) as trigger:
            print("   Waiting for sound trigger (30dB threshold)...")
            print("   Make some noise to trigger recording!")
            
            start_time = time.time()
            recorded_file = trigger.start_monitoring()
            
            if recorded_file:
                elapsed = time.time() - start_time
                print(f"   ✅ Triggered recording: {os.path.basename(recorded_file)} (after {elapsed:.1f}s)")
            else:
                print("   ⏰ No trigger detected within timeout")
                
    except Exception as e:
        print(f"   ❌ Triggered mode error: {e}")
    
    # Test continuous mode
    print("\n2. Testing Continuous Mode (new)")
    try:
        with SoundTrigger(output_dir="demo_recordings", continuous_mode=True) as trigger:
            print("   Starting 5-second continuous recording...")
            
            recorded_file = trigger.start_continuous_recording(duration_seconds=5.0)
            
            if recorded_file:
                print(f"   ✅ Continuous recording: {os.path.basename(recorded_file)}")
            else:
                print("   ❌ Continuous recording failed")
                
    except Exception as e:
        print(f"   ❌ Continuous mode error: {e}")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Continuous Recording Demo")
    parser.add_argument("--duration", "-d", type=float, default=10.0, 
                       help="Recording duration in seconds (default: 10.0)")
    parser.add_argument("--output", "-o", default="demo_recordings",
                       help="Output directory (default: demo_recordings)")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Compare triggered vs continuous modes")
    
    args = parser.parse_args()
    
    if args.compare:
        demo_triggered_vs_continuous()
    else:
        recorded_file = demo_continuous_recording(args.duration, args.output)
        
        if recorded_file:
            print(f"\n🎉 Demo completed! Check the recording: {recorded_file}")
            print("\nThis demonstrates the new continuous recording capability")
            print("perfect for voice recognition applications where you want to")
            print("capture ALL audio continuously rather than waiting for triggers.")

if __name__ == "__main__":
    main()