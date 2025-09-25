# Duplicate Classification Detection in Single Separator

## Overview

This document describes the implementation of duplicate classification detection in the `single_separator.py` file. The feature restricts anchor regions and selects new anchors when pass2's classification result is the same as pass1's result.

## Changes Made

### 1. Multi-pass Separation Function Enhanced

**File**: `single_separator.py`
**Function**: `multi_pass_separation()`

#### New Features:
- **Classification Tracking**: Added `previous_classifications` list to track classification results from previous passes
- **Duplicate Detection**: Added logic to detect when current pass classification matches previous pass classifications
- **Anchor History**: Track previous anchor regions for restriction when duplicates are detected

#### Key Changes:
```python
# Track previous classifications
previous_classifications = []

# Duplicate detection logic (pass 2 and beyond)
if i >= 1 and len(previous_classifications) > 0:
    temp_attention_matrix, temp_classification, _ = ast_processor.process(current_10sec)
    current_class = temp_classification['predicted_class']
    
    if current_class in previous_classifications:
        is_duplicate_detected = True
        # Collect previous anchor information for restriction
        for prev_result in results:
            previous_anchors.append(prev_result.anchor_frames)
```

### 2. Process Single Pass Function Enhanced

**Function**: `process_single_pass()`

#### New Parameters:
- `previous_classifications`: List of previous pass classifications
- `is_duplicate_detected`: Boolean flag indicating duplicate detection
- `previous_anchors`: List of previous anchor regions for restriction

### 3. Anchor Selection Function Enhanced

**Function**: `find_attention_anchor()`

#### New Features:
- **Previous Anchor Restriction**: When duplicates are detected, previous anchor regions are heavily suppressed (0.01x factor)
- **Enhanced Scoring System**: When duplicates are detected, the anchor selection uses enhanced scoring based on:
  - **Continuity Weight**: Prefers longer continuous segments
  - **Attention Weight**: Prefers higher attention scores
  - **Position Diversity Weight**: Prefers positions distant from previous anchors

#### Implementation Details:
```python
# Duplicate detection handling
if is_duplicate_detected and previous_anchors:
    print(f"🔄 Duplicate classification detected - restricting previous anchor regions")
    for anchor_start, anchor_end in previous_anchors:
        # Convert anchors to patch coordinates
        start_patch = int(anchor_start * num_time_patches / max_frames)
        end_patch = int(anchor_end * num_time_patches / max_frames)
        
        # Heavily suppress previous anchor regions (0.01x factor)
        att_masked[:, max(0, start_patch):min(num_time_patches, end_patch)] *= 0.01

# Enhanced scoring for duplicate detection
if is_duplicate_detected:
    continuity_weight = 1.0 + (len(segment) / 10.0)
    attention_weight = 1.0 + segment_attention
    position_weight = 1.0 + min(2.0, min_distance / 10.0)  # Distance from previous anchors
    
    score = len(segment) * segment_attention * freq_weight * continuity_weight * attention_weight * position_weight
```

## How It Works

### Step-by-Step Process:

1. **Pass 1**: Processes audio normally, stores classification result and anchor region
2. **Pass 2+**: Before processing, performs temporary AST classification to check for duplicates
3. **Duplicate Detection**: If current classification matches any previous classification:
   - Sets `is_duplicate_detected = True`
   - Collects all previous anchor regions
4. **Anchor Restriction**: Previous anchor regions are heavily suppressed in attention matrix
5. **Enhanced Anchor Selection**: Uses improved scoring to select diverse, high-attention, continuous regions
6. **Logging**: Provides detailed feedback about duplicate detection and anchor selection

### Benefits:

1. **Avoids Repetitive Separation**: Prevents the system from repeatedly separating the same sound source
2. **Explores New Regions**: Forces the system to look at different parts of the audio
3. **Improved Diversity**: Results in more diverse source separation across passes
4. **Better Coverage**: Ensures different audio regions are explored in subsequent passes

## Example Output

When duplicate detection is triggered, you'll see output like:
```
=== Pass 2/2 ===
🔄 Duplicate classification detected: 'Speech' (same as previous pass)
  🔄 Duplicate classification detected - restricting previous anchor regions
    Suppressed anchor region: frames 45-78 (patches 12-21)
    Segment at freq 3, pos 85-92: attention=0.847, continuity=1.80, position=2.15, score=8.42
  Processing pass 2...
```

## Configuration

The duplicate detection behavior can be controlled through the existing configuration parameters:

- `SeparationConfig.ATTENTION_PERCENTILE`: Controls attention threshold (default: 80)
- `SeparationConfig.CONTINUITY_GAP`: Controls segment continuity (default: 1)
- Anchor suppression factor: 0.01 (heavily suppresses previous regions)
- Position diversity weight: Up to 2.0x boost for distant positions

## Technical Notes

- The duplicate detection only activates from pass 2 onwards
- Previous anchor suppression uses a 0.01 multiplier (99% reduction)
- The enhanced scoring system provides up to 10x improvement for optimal segments
- All changes are backward compatible - existing functionality is preserved when no duplicates are detected

## Testing

To test the duplicate detection:
1. Use audio with repeated sound sources
2. Run multi-pass separation with `max_passes=2` or higher
3. Look for "🔄 Duplicate classification detected" messages in output
4. Verify that different anchor regions are selected in subsequent passes