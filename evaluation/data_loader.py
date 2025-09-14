"""
AudioSet data loading and preprocessing utilities
"""
import os
import glob
import random
from typing import List, Dict, Tuple
import librosa
import numpy as np
from config import AUDIOSET_DATA_DIR, TARGET_CLASSES, MAX_FILES_PER_CLASS, MAX_TOTAL_FILES

class AudioSetLoader:
    """Load and manage AudioSet data for evaluation"""
    
    def __init__(self, data_dir: str = AUDIOSET_DATA_DIR):
        self.data_dir = data_dir
        self.audio_files = []
        self.class_mapping = {}
        
    def scan_audio_files(self) -> Dict[str, List[str]]:
        """Scan for available audio files and group by class"""
        print(f"Scanning audio files in: {self.data_dir}")
        
        # Find all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac']
        all_files = []
        
        for ext in audio_extensions:
            pattern = os.path.join(self.data_dir, "train_audio", ext)
            all_files.extend(glob.glob(pattern))
            
        print(f"Found {len(all_files)} audio files")
        
        # Group files by class (based on filename patterns or metadata)
        class_files = {}
        for class_name in TARGET_CLASSES:
            class_files[class_name] = []
            
        # For now, randomly distribute files (in real implementation, 
        # you'd use actual class labels from AudioSet metadata)
        random.shuffle(all_files)
        files_per_class = len(all_files) // len(TARGET_CLASSES)
        
        for i, class_name in enumerate(TARGET_CLASSES):
            start_idx = i * files_per_class
            end_idx = start_idx + files_per_class
            if i == len(TARGET_CLASSES) - 1:  # Last class gets remaining files
                end_idx = len(all_files)
            class_files[class_name] = all_files[start_idx:end_idx]
            
        # Limit files per class
        for class_name in class_files:
            if len(class_files[class_name]) > MAX_FILES_PER_CLASS:
                class_files[class_name] = random.sample(
                    class_files[class_name], MAX_FILES_PER_CLASS
                )
                
        return class_files
    
    def get_evaluation_files(self) -> List[Tuple[str, str]]:
        """Get list of (file_path, class_name) tuples for evaluation"""
        class_files = self.scan_audio_files()
        
        evaluation_files = []
        total_files = 0
        
        for class_name, files in class_files.items():
            for file_path in files:
                if total_files >= MAX_TOTAL_FILES:
                    break
                evaluation_files.append((file_path, class_name))
                total_files += 1
                
        print(f"Selected {len(evaluation_files)} files for evaluation")
        return evaluation_files
    
    def load_audio(self, file_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=target_sr)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def validate_audio(self, file_path: str) -> bool:
        """Validate if audio file is suitable for evaluation"""
        try:
            audio = self.load_audio(file_path)
            if audio is None:
                return False
                
            # Check duration (should be reasonable for evaluation)
            duration = len(audio) / 16000  # Assuming 16kHz
            if duration < 1.0 or duration > 30.0:
                return False
                
            # Check for silence
            if np.max(np.abs(audio)) < 0.01:
                return False
                
            return True
        except:
            return False

def get_evaluation_dataset() -> List[Tuple[str, str]]:
    """Convenience function to get evaluation dataset"""
    loader = AudioSetLoader()
    return loader.get_evaluation_files()
