"""
Advanced audio separation metrics for comprehensive evaluation
"""
import os
import sys
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional

# Add parent directory to import evaluation_metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation_metrics import compute_all_metrics, evaluate_separation_quality

class AdvancedAudioMetrics:
    """Advanced metrics calculator for audio separation evaluation"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """Load audio file safely"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def calculate_separation_metrics(self, original_file: str, 
                                   separated_files: List[str]) -> Dict:
        """Calculate comprehensive separation metrics"""
        metrics = {
            'separation_quality': {},
            'source_metrics': [],
            'overall_score': 0.0
        }
        
        # Load original audio
        original_audio = self.load_audio_file(original_file)
        if original_audio is None:
            return metrics
            
        # Load separated sources
        separated_sources = []
        for file_path in separated_files:
            audio = self.load_audio_file(file_path)
            if audio is not None:
                separated_sources.append(audio)
        
        if not separated_sources:
            return metrics
            
        # Calculate comprehensive metrics
        try:
            separation_results = evaluate_separation_quality(
                original_audio, separated_sources, fs=self.sample_rate
            )
            
            metrics['separation_quality'] = separation_results['overall_metrics']
            metrics['source_metrics'] = separation_results['metrics_per_source']
            
            # Calculate overall score (weighted combination)
            overall_score = self._calculate_overall_score(separation_results)
            metrics['overall_score'] = overall_score
            
        except Exception as e:
            print(f"Error calculating separation metrics: {e}")
            
        return metrics
    
    def _calculate_overall_score(self, separation_results: Dict) -> float:
        """Calculate weighted overall score from multiple metrics"""
        weights = {
            'SDR': 0.3,    # Signal quality
            'SIR': 0.25,   # Separation quality  
            'SAR': 0.2,    # Artifact level
            'STOI': 0.15,  # Intelligibility
            'PESQ': 0.1    # Perceptual quality
        }
        
        overall_metrics = separation_results.get('overall_metrics', {})
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in overall_metrics:
                value = overall_metrics[metric]
                
                # Normalize values to 0-1 scale
                if metric in ['SDR', 'SIR', 'SAR']:
                    # dB values: assume 0-30 dB range maps to 0-1
                    normalized_value = min(1.0, max(0.0, value / 30.0))
                elif metric == 'STOI':
                    # Already 0-1
                    normalized_value = value
                elif metric == 'PESQ':
                    # 1-5 range maps to 0-1
                    normalized_value = (value - 1.0) / 4.0
                else:
                    normalized_value = 0.0
                
                score += weight * normalized_value
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def calculate_classification_metrics(self, predicted_class: str, 
                                       confidence: float,
                                       expected_class: str = None) -> Dict:
        """Calculate classification-specific metrics"""
        metrics = {
            'confidence': confidence,
            'is_correct': False,
            'confidence_penalty': 0.0
        }
        
        if expected_class:
            # Check if prediction is correct
            is_correct = predicted_class.lower() in expected_class.lower() or \
                        expected_class.lower() in predicted_class.lower()
            metrics['is_correct'] = is_correct
            
            # Confidence penalty for wrong predictions
            if not is_correct:
                metrics['confidence_penalty'] = confidence
        
        return metrics
    
    def calculate_efficiency_metrics(self, processing_time: float,
                                   file_size_mb: float = 0.0) -> Dict:
        """Calculate computational efficiency metrics"""
        metrics = {
            'processing_time': processing_time,
            'throughput': 0.0,  # files per minute
            'efficiency_score': 0.0
        }
        
        if processing_time > 0:
            metrics['throughput'] = 60.0 / processing_time  # files per minute
            
            # Efficiency score (higher is better, normalized)
            # Assume 10 seconds is baseline, score decreases with longer times
            baseline_time = 10.0
            if processing_time <= baseline_time:
                metrics['efficiency_score'] = 1.0
            else:
                metrics['efficiency_score'] = baseline_time / processing_time
        
        return metrics
    
    def generate_comprehensive_report(self, evaluation_results: List[Dict]) -> Dict:
        """Generate comprehensive evaluation report with all metrics"""
        report = {
            'summary': {
                'total_files': len(evaluation_results),
                'successful_evaluations': 0,
                'average_processing_time': 0.0,
                'average_confidence': 0.0,
                'average_overall_score': 0.0
            },
            'classification_performance': {
                'accuracy': 0.0,
                'average_confidence': 0.0,
                'confidence_distribution': []
            },
            'separation_performance': {
                'average_sdr': 0.0,
                'average_sir': 0.0,
                'average_sar': 0.0,
                'average_stoi': 0.0,
                'average_pesq': 0.0
            },
            'efficiency_metrics': {
                'average_processing_time': 0.0,
                'average_throughput': 0.0,
                'efficiency_score': 0.0
            },
            'detailed_results': []
        }
        
        successful_results = []
        classification_scores = []
        separation_scores = []
        processing_times = []
        
        for result in evaluation_results:
            if result.get('success', False):
                successful_results.append(result)
                
                # Classification metrics
                confidence = result.get('confidence', 0.0)
                classification_scores.append(confidence)
                
                # Processing time
                proc_time = result.get('processing_time', 0.0)
                processing_times.append(proc_time)
                
                # Separation metrics (if available)
                if 'separation_metrics' in result:
                    sep_metrics = result['separation_metrics']
                    if 'separation_quality' in sep_metrics:
                        separation_scores.append(sep_metrics['separation_quality'])
        
        # Calculate summary statistics
        if successful_results:
            report['summary']['successful_evaluations'] = len(successful_results)
            report['summary']['average_processing_time'] = np.mean(processing_times)
            report['summary']['average_confidence'] = np.mean(classification_scores)
            
            if separation_scores:
                # Calculate average separation metrics
                avg_sdr = np.mean([s.get('SDR', 0) for s in separation_scores])
                avg_sir = np.mean([s.get('SIR', 0) for s in separation_scores])
                avg_sar = np.mean([s.get('SAR', 0) for s in separation_scores])
                avg_stoi = np.mean([s.get('STOI', 0) for s in separation_scores])
                avg_pesq = np.mean([s.get('PESQ', 0) for s in separation_scores])
                
                report['separation_performance'] = {
                    'average_sdr': avg_sdr,
                    'average_sir': avg_sir,
                    'average_sar': avg_sar,
                    'average_stoi': avg_stoi,
                    'average_pesq': avg_pesq
                }
                
                # Overall score
                overall_scores = [s.get('overall_score', 0) for s in separation_scores]
                report['summary']['average_overall_score'] = np.mean(overall_scores)
        
        # Classification performance
        if classification_scores:
            report['classification_performance'] = {
                'accuracy': len(successful_results) / len(evaluation_results),
                'average_confidence': np.mean(classification_scores),
                'confidence_distribution': {
                    'min': np.min(classification_scores),
                    'max': np.max(classification_scores),
                    'std': np.std(classification_scores)
                }
            }
        
        # Efficiency metrics
        if processing_times:
            report['efficiency_metrics'] = {
                'average_processing_time': np.mean(processing_times),
                'average_throughput': 60.0 / np.mean(processing_times),
                'efficiency_score': np.mean([10.0 / max(t, 0.1) for t in processing_times])
            }
        
        return report

def print_advanced_metrics_summary(report: Dict):
    """Print comprehensive metrics summary"""
    print("\n" + "="*80)
    print("🎯 ADVANCED AUDIO SEPARATION EVALUATION RESULTS")
    print("="*80)
    
    # Summary
    summary = report['summary']
    print(f"📊 Total Files: {summary['total_files']}")
    print(f"✅ Successful: {summary['successful_evaluations']}")
    print(f"⏱️  Avg Processing Time: {summary['average_processing_time']:.2f}s")
    print(f"🎯 Avg Confidence: {summary['average_confidence']:.3f}")
    print(f"🏆 Overall Score: {summary['average_overall_score']:.3f}")
    
    # Classification Performance
    print(f"\n📈 CLASSIFICATION PERFORMANCE:")
    class_perf = report['classification_performance']
    print(f"  Accuracy: {class_perf['accuracy']:.1%}")
    print(f"  Avg Confidence: {class_perf['average_confidence']:.3f}")
    
    # Separation Performance
    print(f"\n🎵 SEPARATION PERFORMANCE:")
    sep_perf = report['separation_performance']
    print(f"  SDR: {sep_perf['average_sdr']:.2f} dB")
    print(f"  SIR: {sep_perf['average_sir']:.2f} dB")
    print(f"  SAR: {sep_perf['average_sar']:.2f} dB")
    print(f"  STOI: {sep_perf['average_stoi']:.3f}")
    print(f"  PESQ: {sep_perf['average_pesq']:.2f}")
    
    # Efficiency
    print(f"\n⚡ EFFICIENCY METRICS:")
    eff_metrics = report['efficiency_metrics']
    print(f"  Avg Processing Time: {eff_metrics['average_processing_time']:.2f}s")
    print(f"  Throughput: {eff_metrics['average_throughput']:.1f} files/min")
    print(f"  Efficiency Score: {eff_metrics['efficiency_score']:.3f}")
    
    print("="*80)
