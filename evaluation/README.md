# AudioSet Evaluation Suite

Comprehensive evaluation tools for sound separation and classification using Google AudioSet data.

## Overview

This evaluation suite provides tools to:
- Load and preprocess AudioSet data
- Run sound separation and classification evaluation
- Generate comprehensive reports and visualizations
- Analyze performance metrics

## Structure

```
evaluation_suite/
├── __init__.py              # Package initialization
├── config.py                # Configuration settings
├── data_loader.py           # AudioSet data loading utilities
├── evaluator.py             # Main evaluation engine
├── visualizer.py            # Visualization tools
├── run_evaluation.py        # Main evaluation script
└── README.md               # This file
```

## Quick Start

### 1. Run Full Evaluation
```bash
cd evaluation_suite
python run_evaluation.py --max-files 100
```

### 2. Quick Evaluation
```bash
python run_evaluation.py --quick --max-files 50
```

### 3. Evaluate Specific Classes
```bash
python run_evaluation.py --classes "Siren" "Alarm" "Explosion" --max-files 30
```

### 4. Generate Visualizations
```bash
python visualizer.py --results-dir ../evaluation_suite/results
```

## Configuration

Edit `config.py` to customize:
- Target classes for evaluation
- Maximum files per class
- Output directories
- Performance metrics

## Target Classes

The evaluation focuses on these sound categories:

### Siren & Alarm Classes
- Siren
- Civil defense siren
- Buzzer
- Smoke detector, smoke alarm
- Fire alarm
- Alarm
- Alarm clock

### Explosion & Breaking Sounds
- Explosion
- Boom
- Splinter
- Crack
- Glass
- Chink, clink
- Shatter
- Smash, crash
- Breaking
- Crushing
- Crumpling, crinkling

### Human Sounds
- Baby cry, infant cry
- Screaming

### Door & Household Sounds
- Door
- Doorbell
- Ding-dong
- Knock
- Water
- Dishes, pots, and pans
- Boiling

### Telephone Sounds
- Telephone
- Telephone bell ringing
- Ringtone
- Telephone dialing, DTMF
- Dial tone

## Output

The evaluation generates:
- JSON report with detailed results
- Separated audio files (optional)
- Debug plots (optional)
- HTML report
- Visualization plots

## Performance Metrics

- **Classification Accuracy**: Correct class predictions
- **Confidence Scores**: Model confidence in predictions
- **Energy Ratio**: Quality of sound separation
- **Processing Time**: Computational efficiency
- **Success Rate**: Percentage of successful evaluations

## Dependencies

- librosa
- matplotlib
- seaborn
- pandas
- numpy
- subprocess
- json
- os
- time

## Notes

- Ensure AudioSet data is downloaded before running evaluation
- The evaluation uses the main `test.py` script for sound separation
- Results are saved with timestamps for easy tracking
- Failed evaluations are logged with error messages
