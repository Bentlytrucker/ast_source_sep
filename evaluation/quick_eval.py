"""
Quick evaluation script for testing with limited files
"""
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_suite.config import OUTPUT_DIR, TARGET_CLASSES
from evaluation_suite.data_loader import get_evaluation_dataset
from evaluation_suite.evaluator import SoundSeparationEvaluator

def quick_evaluation(max_files=20):
    """Run quick evaluation with limited files"""
    print("="*60)
    print("QUICK AUDIOSET EVALUATION")
    print("="*60)
    print(f"Max files: {max_files}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, f"quick_eval_{timestamp}")
    
    # Initialize evaluator
    evaluator = SoundSeparationEvaluator(output_dir)
    
    # Get evaluation dataset
    print("\nLoading evaluation dataset...")
    evaluation_files = get_evaluation_dataset()
    
    if not evaluation_files:
        print("❌ No audio files found for evaluation!")
        return
    
    # Limit files
    if len(evaluation_files) > max_files:
        evaluation_files = evaluation_files[:max_files]
        print(f"Limited to {max_files} files for quick evaluation")
    
    print(f"Found {len(evaluation_files)} files for evaluation")
    
    # Run evaluation
    print("\nStarting quick evaluation...")
    start_time = time.time()
    results = evaluator.evaluate_batch(evaluation_files)
    total_time = time.time() - start_time
    
    # Generate and save report
    print("\nGenerating report...")
    report = evaluator.generate_report()
    evaluator.save_results(report, "quick_evaluation_report.json")
    
    # Print summary
    evaluator.print_summary(report)
    
    print(f"\n✅ Quick evaluation completed in {total_time:.2f}s!")
    print(f"Results saved to: {output_dir}")
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick AudioSet evaluation")
    parser.add_argument("--max-files", type=int, default=20,
                       help="Maximum number of files to evaluate")
    
    args = parser.parse_args()
    quick_evaluation(args.max_files)
