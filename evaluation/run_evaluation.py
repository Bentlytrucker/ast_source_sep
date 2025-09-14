"""
Main script to run comprehensive AudioSet evaluation
"""
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_suite.config import OUTPUT_DIR, TARGET_CLASSES
from evaluation_suite.data_loader import get_evaluation_dataset
from evaluation_suite.evaluator import SoundSeparationEvaluator

def main():
    parser = argparse.ArgumentParser(description="Run AudioSet evaluation")
    parser.add_argument("--max-files", type=int, default=100, 
                       help="Maximum number of files to evaluate")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                       help="Output directory for results")
    parser.add_argument("--classes", nargs="+", default=TARGET_CLASSES,
                       help="Specific classes to evaluate")
    parser.add_argument("--quick", action="store_true",
                       help="Quick evaluation with fewer files")
    
    args = parser.parse_args()
    
    print("="*60)
    print("AUDIOSET COMPREHENSIVE EVALUATION")
    print("="*60)
    print(f"Target classes: {len(args.classes)}")
    print(f"Max files: {args.max_files}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    
    # Initialize evaluator
    evaluator = SoundSeparationEvaluator(output_dir)
    
    # Get evaluation dataset
    print("\nLoading evaluation dataset...")
    evaluation_files = get_evaluation_dataset()
    
    if not evaluation_files:
        print("❌ No audio files found for evaluation!")
        return
    
    # Limit files if specified
    if args.max_files and len(evaluation_files) > args.max_files:
        evaluation_files = evaluation_files[:args.max_files]
        print(f"Limited to {args.max_files} files for evaluation")
    
    print(f"Found {len(evaluation_files)} files for evaluation")
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate_batch(evaluation_files)
    
    # Generate and save report
    print("\nGenerating report...")
    report = evaluator.generate_report()
    evaluator.save_results(report)
    
    # Print summary
    evaluator.print_summary(report)
    
    print(f"\n✅ Evaluation completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
