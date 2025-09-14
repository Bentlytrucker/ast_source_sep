"""
Main evaluation engine for sound separation and classification
"""
import os
import json
import time
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import TEST_SCRIPT, OUTPUT_DIR, TIMEOUT_SECONDS, SAVE_SEPARATED_AUDIO

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    file_path: str
    class_name: str
    detected_class: str
    confidence: float
    energy_ratio: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    output_files: List[str] = None

class SoundSeparationEvaluator:
    """Main evaluator for sound separation and classification"""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
    def evaluate_single_file(self, file_path: str, class_name: str) -> EvaluationResult:
        """Evaluate a single audio file"""
        print(f"Evaluating: {os.path.basename(file_path)}")
        
        start_time = time.time()
        
        try:
            # Create output directory for this file
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            file_output_dir = os.path.join(self.output_dir, f"eval_{file_id}")
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Run test.py
            cmd = [
                "python", TEST_SCRIPT,
                "--input", file_path,
                "--output", file_output_dir
            ]
            
            if not SAVE_SEPARATED_AUDIO:
                cmd.append("--no-save")
                
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS
            )
            
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse output for results
                detected_class, confidence, energy_ratio = self._parse_output(result.stdout)
                
                # Find output files
                output_files = []
                if os.path.exists(file_output_dir):
                    output_files = [f for f in os.listdir(file_output_dir) if f.endswith('.wav')]
                
                return EvaluationResult(
                    file_path=file_path,
                    class_name=class_name,
                    detected_class=detected_class,
                    confidence=confidence,
                    energy_ratio=energy_ratio,
                    processing_time=processing_time,
                    success=True,
                    output_files=output_files
                )
            else:
                return EvaluationResult(
                    file_path=file_path,
                    class_name=class_name,
                    detected_class="",
                    confidence=0.0,
                    energy_ratio=0.0,
                    processing_time=processing_time,
                    success=False,
                    error_message=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return EvaluationResult(
                file_path=file_path,
                class_name=class_name,
                detected_class="",
                confidence=0.0,
                energy_ratio=0.0,
                processing_time=TIMEOUT_SECONDS,
                success=False,
                error_message="Timeout"
            )
        except Exception as e:
            return EvaluationResult(
                file_path=file_path,
                class_name=class_name,
                detected_class="",
                confidence=0.0,
                energy_ratio=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _parse_output(self, stdout: str) -> Tuple[str, float, float]:
        """Parse test.py output to extract results"""
        detected_class = "Unknown"
        confidence = 0.0
        energy_ratio = 0.0
        
        lines = stdout.strip().split('\n')
        for line in lines:
            if "Class:" in line:
                # Extract class name and confidence
                parts = line.split("Class:")[1].strip()
                if "(" in parts:
                    detected_class = parts.split("(")[0].strip()
                    if "Conf:" in parts:
                        conf_str = parts.split("Conf:")[1].split(")")[0].strip()
                        try:
                            confidence = float(conf_str)
                        except:
                            confidence = 0.0
            elif "Energy Ratio:" in line:
                try:
                    energy_ratio = float(line.split("Energy Ratio:")[1].strip())
                except:
                    energy_ratio = 0.0
                    
        return detected_class, confidence, energy_ratio
    
    def evaluate_batch(self, evaluation_files: List[Tuple[str, str]]) -> List[EvaluationResult]:
        """Evaluate a batch of files"""
        print(f"Starting batch evaluation of {len(evaluation_files)} files...")
        
        results = []
        for i, (file_path, class_name) in enumerate(evaluation_files):
            print(f"\n--- Evaluation {i+1}/{len(evaluation_files)} ---")
            result = self.evaluate_single_file(file_path, class_name)
            results.append(result)
            
            # Print result summary
            if result.success:
                print(f"✅ Success - {result.processing_time:.2f}s")
                print(f"   Class: {result.detected_class} (Conf: {result.confidence:.3f})")
                print(f"   Energy Ratio: {result.energy_ratio:.3f}")
                print(f"   Files Saved: {len(result.output_files) if result.output_files else 0}")
            else:
                print(f"❌ Failed - {result.processing_time:.2f}s")
                print(f"   Error: {result.error_message}")
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        if not self.results:
            return {}
            
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        # Calculate statistics
        total_files = len(self.results)
        success_rate = len(successful_results) / total_files if total_files > 0 else 0
        
        avg_confidence = np.mean([r.confidence for r in successful_results]) if successful_results else 0
        avg_energy_ratio = np.mean([r.energy_ratio for r in successful_results]) if successful_results else 0
        avg_processing_time = np.mean([r.processing_time for r in self.results])
        
        # Class distribution
        class_counts = {}
        for result in successful_results:
            class_counts[result.detected_class] = class_counts.get(result.detected_class, 0) + 1
        
        # Sound type distribution
        sound_types = {"help": 0, "warning": 0, "danger": 0, "other": 0}
        for result in successful_results:
            if "(help)" in result.detected_class:
                sound_types["help"] += 1
            elif "(warning)" in result.detected_class:
                sound_types["warning"] += 1
            elif "(danger)" in result.detected_class:
                sound_types["danger"] += 1
            else:
                sound_types["other"] += 1
        
        report = {
            "summary": {
                "total_files": total_files,
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": success_rate,
                "total_time": sum(r.processing_time for r in self.results),
                "average_time_per_file": avg_processing_time
            },
            "performance": {
                "average_confidence": avg_confidence,
                "average_energy_ratio": avg_energy_ratio
            },
            "class_distribution": class_counts,
            "sound_types": sound_types,
            "detailed_results": [
                {
                    "file": os.path.basename(r.file_path),
                    "class": r.class_name,
                    "detected": r.detected_class,
                    "confidence": r.confidence,
                    "energy_ratio": r.energy_ratio,
                    "success": r.success,
                    "error": r.error_message
                }
                for r in self.results
            ]
        }
        
        return report
    
    def save_results(self, report: Dict, filename: str = "evaluation_report.json"):
        """Save evaluation results to file"""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
        
    def print_summary(self, report: Dict):
        """Print evaluation summary"""
        summary = report["summary"]
        performance = report["performance"]
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total files evaluated: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total time: {summary['total_time']:.2f}s")
        print(f"Average time per file: {summary['average_time_per_file']:.2f}s")
        
        print(f"\n--- PERFORMANCE METRICS ---")
        print(f"Average Confidence: {performance['average_confidence']:.3f}")
        print(f"Average Energy Ratio: {performance['average_energy_ratio']:.3f}")
        
        print(f"\n--- CLASS DISTRIBUTION ---")
        for class_name, count in report["class_distribution"].items():
            print(f"  {class_name}: {count}")
            
        print(f"\n--- SOUND TYPES ---")
        for sound_type, count in report["sound_types"].items():
            print(f"  {sound_type}: {count}")
