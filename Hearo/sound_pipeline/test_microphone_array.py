#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for ReSpeaker microphone array with 6-channel recording
Based on the provided pyaudio example but enhanced for voice recognition
"""

import pyaudio
import wave
import numpy as np
import os
import time
from typing import Optional

# Configuration based on the provided example
RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6  # 6 channels: 0=ASR, 1-4=Microphones, 5=Playback
RESPEAKER_WIDTH = 2
CHUNK = 1024
RECORD_SECONDS = 5
TARGET_DEVICE_KEYWORDS = ("ReSpeaker", "seeed", "SEEED")

def find_respeaker_device():
    """Find ReSpeaker device index"""
    p = pyaudio.PyAudio()
    device_index = None
    max_in_ch = 0
    
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get('name', '')
        max_inputs = int(info.get('maxInputChannels', 0))
        max_outputs = int(info.get('maxOutputChannels', 0))
        
        print(f"  {i}: {name} (In: {max_inputs}, Out: {max_outputs})")
        
        # Check if this is a ReSpeaker device
        is_input = max_inputs > 0
        if is_input and any(k.lower() in name.lower() for k in TARGET_DEVICE_KEYWORDS):
            device_index = i
            max_in_ch = max_inputs
            print(f"  -> Found ReSpeaker device: {name}")
    
    # Fallback to default input device if ReSpeaker not found
    if device_index is None:
        default_info = p.get_default_input_device_info()
        device_index = default_info.get('index')
        max_in_ch = int(default_info.get('maxInputChannels', 0))
        print(f"  -> Using default input device: {default_info.get('name')}")
    
    p.terminate()
    return device_index, max_in_ch

def record_multichannel_audio(device_index: int, max_channels: int, duration: float = RECORD_SECONDS) -> Optional[str]:
    """Record multi-channel audio from ReSpeaker"""
    
    # Determine number of channels to use
    channels_to_use = min(RESPEAKER_CHANNELS, max_channels) if max_channels >= RESPEAKER_CHANNELS else max_channels
    
    print(f"Starting recording...")
    print(f"Device index: {device_index}")
    print(f"Channels: {channels_to_use}")
    print(f"Sample rate: {RESPEAKER_RATE} Hz")
    print(f"Duration: {duration} seconds")
    
    if channels_to_use == 6:
        print("Channel layout:")
        print("  Channel 0: Processed audio for ASR")
        print("  Channel 1-4: 4 microphones' raw data")
        print("  Channel 5: Playback (factory firmware)")
    
    p = pyaudio.PyAudio()
    
    try:
        # Open audio stream
        stream = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=channels_to_use,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )
        
        print("* Recording started...")
        
        frames = []
        num_chunks = int(RESPEAKER_RATE / CHUNK * duration)
        
        for i in range(num_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Show progress
            if i % (num_chunks // 10) == 0:
                progress = (i / num_chunks) * 100
                print(f"  Recording progress: {progress:.0f}%")
        
        print("* Recording completed")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"microphone_array_test_{timestamp}_{channels_to_use}ch.wav"
        
        # Save multi-channel WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels_to_use)
        wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
        wf.setframerate(RESPEAKER_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"* Audio saved as: {filename}")
        
        # Create individual channel files for analysis
        if channels_to_use > 1:
            extract_individual_channels(filename, channels_to_use)
        
        return filename
        
    except Exception as e:
        print(f"Error during recording: {e}")
        return None
    finally:
        p.terminate()

def extract_individual_channels(multichannel_file: str, num_channels: int):
    """Extract individual channels from multi-channel recording"""
    print(f"Extracting individual channels from {multichannel_file}...")
    
    # Read the multi-channel file
    wf = wave.open(multichannel_file, 'rb')
    frames = wf.readframes(wf.getnframes())
    wf.close()
    
    # Convert to numpy array
    audio_data = np.frombuffer(frames, dtype=np.int16)
    
    # Reshape to separate channels
    if len(audio_data) % num_channels == 0:
        multichannel_data = audio_data.reshape(-1, num_channels)
        
        # Extract each channel
        base_name = os.path.splitext(multichannel_file)[0]
        
        for ch in range(num_channels):
            channel_data = multichannel_data[:, ch]
            channel_filename = f"{base_name}_channel_{ch}.wav"
            
            # Save individual channel
            wf = wave.open(channel_filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(RESPEAKER_RATE)
            wf.writeframes(channel_data.astype(np.int16).tobytes())
            wf.close()
            
            # Analyze channel
            rms = np.sqrt(np.mean(channel_data.astype(np.float32)**2))
            max_val = np.max(np.abs(channel_data))
            
            channel_type = "ASR" if ch == 0 else f"Mic {ch}" if 1 <= ch <= 4 else "Playback" if ch == 5 else "Unknown"
            print(f"  Channel {ch} ({channel_type}): RMS={rms:.2f}, Max={max_val}, File={channel_filename}")
    else:
        print("Error: Audio data length not divisible by number of channels")

def main():
    """Main test function"""
    print("ReSpeaker Microphone Array Test")
    print("=" * 40)
    
    # Find ReSpeaker device
    device_index, max_channels = find_respeaker_device()
    
    if device_index is None:
        print("No suitable audio input device found!")
        return
    
    print(f"\nUsing device {device_index} with {max_channels} input channels")
    
    # Record audio
    recorded_file = record_multichannel_audio(device_index, max_channels)
    
    if recorded_file:
        print(f"\nTest completed successfully!")
        print(f"Multi-channel recording saved as: {recorded_file}")
        print("Individual channel files created for analysis.")
        
        # Display file info
        file_size = os.path.getsize(recorded_file)
        print(f"File size: {file_size} bytes ({file_size/1024:.1f} KB)")
    else:
        print("Test failed!")

if __name__ == "__main__":
    main()