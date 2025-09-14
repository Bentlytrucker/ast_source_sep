"""
Configuration settings for AudioSet evaluation
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIOSET_DATA_DIR = os.path.join(BASE_DIR, "audioset_data", "target_classes")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_suite", "results")
TEST_SCRIPT = os.path.join(BASE_DIR, "test.py")

# Target classes for evaluation
TARGET_CLASSES = [
    # Siren & Alarm classes
    "Siren",
    "Civil defense siren", 
    "Buzzer",
    "Smoke detector, smoke alarm",
    "Fire alarm",
    "Alarm",
    "Alarm clock",
    
    # Explosion & Breaking sounds
    "Explosion",
    "Boom",
    "Splinter",
    "Crack",
    "Glass",
    "Chink, clink",
    "Shatter",
    "Smash, crash",
    "Breaking",
    "Crushing",
    "Crumpling, crinkling",
    
    # Human sounds
    "Baby cry, infant cry",
    "Screaming",
    
    # Door & Household sounds
    "Door",
    "Doorbell",
    "Ding-dong",
    "Knock",
    "Water",
    "Dishes, pots, and pans",
    "Boiling",
    
    # Telephone sounds
    "Telephone",
    "Telephone bell ringing",
    "Ringtone",
    "Telephone dialing, DTMF",
    "Dial tone"
]

# Evaluation settings
MAX_FILES_PER_CLASS = 50  # Limit files per class for faster evaluation
MAX_TOTAL_FILES = 1600    # Total files to evaluate (32 classes * 50 files)
BATCH_SIZE = 10           # Files to process in parallel
TIMEOUT_SECONDS = 30      # Timeout per file

# Output settings
SAVE_SEPARATED_AUDIO = True
SAVE_DEBUG_PLOTS = True
GENERATE_REPORTS = True

# Performance metrics
METRICS = [
    "classification_accuracy",
    "confidence_scores", 
    "energy_ratio",
    "processing_time",
    "separation_quality"
]
