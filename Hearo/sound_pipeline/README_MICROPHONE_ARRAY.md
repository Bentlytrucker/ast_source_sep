# ReSpeaker Microphone Array Integration for Voice Recognition

## Overview

This document describes the modifications made to the sound pipeline to support the ReSpeaker microphone array with 6-channel recording for continuous voice recognition.

## Changes Made

### 1. Multi-Channel Support (`sound_trigger.py`)

The `SoundTrigger` class has been enhanced to support:

- **6-Channel Recording**: Full microphone array support
  - Channel 0: Processed audio for ASR (Automatic Speech Recognition)
  - Channels 1-4: 4 microphones' raw data
  - Channel 5: Playback (factory firmware)

- **Continuous Recording Mode**: New mode for voice recognition
  - Records continuously instead of waiting for sound triggers
  - Captures ALL audio for voice processing
  - Better suited for voice recognition applications

### 2. Pipeline Integration (`single_pipeline.py`)

The main pipeline has been updated to:
- Use continuous recording mode by default
- Handle multi-channel audio files
- Process voice data continuously for recognition

## Configuration

### Hardware Setup
```python
# Multi-channel configuration for microphone array
RESPEAKER_CHANNELS = 6  # Use 6 channels for full microphone array support
RESPEAKER_RATE = 16000  # Sample rate
CHUNK = 1024           # Buffer size
```

### Channel Layout
- **Channel 0**: Processed audio for ASR (recommended for voice recognition)
- **Channels 1-4**: Individual microphone raw data (for beamforming/DOA)
- **Channel 5**: Playback reference (factory firmware)

## Usage Examples

### 1. Basic Test of Microphone Array

```bash
cd /workspace/Hearo/sound_pipeline
python test_microphone_array.py
```

This will:
- Detect your ReSpeaker device
- Record 5 seconds of 6-channel audio
- Extract individual channels for analysis
- Show channel statistics (RMS, max values)

### 2. Continuous Recording Demo

```bash
# Record for 10 seconds continuously
python continuous_recording_demo.py --duration 10

# Compare triggered vs continuous modes
python continuous_recording_demo.py --compare
```

### 3. Running the Full Pipeline

```bash
# Run with continuous recording mode (default)
python single_pipeline.py --output pipeline_output
```

## API Changes

### SoundTrigger Class

#### New Constructor Parameters
```python
SoundTrigger(
    output_dir="recordings", 
    led_controller=None, 
    continuous_mode=True  # NEW: Enable continuous recording
)
```

#### New Methods
```python
# Start continuous recording
start_continuous_recording(duration_seconds=None) -> Optional[str]

# Stop continuous recording
stop_continuous_recording()

# Extract specific channel data
_extract_channel_data(interleaved, num_channels, channel) -> np.ndarray

# Save multi-channel audio
_save_multichannel_audio(frames_data, filename, num_channels) -> str
```

## File Output Format

### Multi-Channel Files
- **Filename**: `continuous_recording_{timestamp}_6ch.wav` or `triggered_recording_{timestamp}_6ch.wav`
- **Format**: 6-channel WAV file, 16-bit, 16kHz
- **Size**: ~6x larger than mono files due to 6 channels

### Individual Channel Files
When using the test script, individual channel files are created:
- `recording_channel_0.wav` - ASR-processed audio
- `recording_channel_1.wav` - Microphone 1
- `recording_channel_2.wav` - Microphone 2  
- `recording_channel_3.wav` - Microphone 3
- `recording_channel_4.wav` - Microphone 4
- `recording_channel_5.wav` - Playback reference

## Voice Recognition Integration

### Recommended Channel Usage

For voice recognition applications:
1. **Primary**: Use Channel 0 (processed audio) for ASR
2. **Alternative**: Use individual microphone channels (1-4) for:
   - Beamforming
   - Direction of arrival (DOA)
   - Noise cancellation
   - Multi-microphone processing

### Benefits for Voice Recognition

1. **Continuous Capture**: No missed speech due to trigger delays
2. **Multi-Channel Data**: Better noise suppression and directional processing
3. **Processed Audio**: Channel 0 provides factory-optimized audio for ASR
4. **Raw Microphone Data**: Channels 1-4 for custom processing algorithms

## Performance Considerations

### Data Rate
- 6-channel recording: ~192 KB/sec (vs 32 KB/sec for mono)
- Storage: ~11.5 MB per minute of recording
- Processing: 6x more data to process

### Optimization Tips
1. Use Channel 0 for primary ASR processing
2. Process other channels only when needed
3. Implement buffering for real-time applications
4. Consider compression for long-term storage

## Troubleshooting

### Device Detection Issues
```bash
# List all audio devices
python -c "
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f'{i}: {info[\"name\"]} (In: {info[\"maxInputChannels\"]})')
p.terminate()
"
```

### Channel Count Issues
- If ReSpeaker shows < 6 channels, check firmware
- Fallback to available channel count automatically
- Verify USB connection and drivers

### Recording Issues
- Check microphone permissions
- Verify device is not in use by other applications
- Test with lower channel count first

## Migration from Previous Version

### Code Changes Required
```python
# OLD: Triggered mode only
trigger = SoundTrigger(output_dir)

# NEW: Continuous mode (recommended for voice recognition)
trigger = SoundTrigger(output_dir, continuous_mode=True)

# OLD: Mono output only
# NEW: Multi-channel output automatically based on device capabilities
```

### File Format Changes
- Output files now include channel count in filename
- Multi-channel WAV format instead of mono
- Individual channel extraction available

## Testing Your Setup

1. **Hardware Test**: Run `test_microphone_array.py`
2. **Recording Test**: Run `continuous_recording_demo.py`
3. **Pipeline Test**: Run `single_pipeline.py` with `--output test_output`

All tests should complete without errors and produce audio files with the expected channel count.