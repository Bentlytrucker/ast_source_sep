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
import pandas as pd

def print_detailed_performance_table(report):
    """Print detailed performance table"""
    print("\n" + "="*100)
    print("📊 DETAILED PERFORMANCE TABLE")
    print("="*100)
    
    # Create DataFrame from detailed results
    detailed_results = report.get("detailed_results", [])
    if not detailed_results:
        print("No detailed results available")
        return
    
    # Prepare data for table
    table_data = []
    for result in detailed_results:
        row = {
            "File": result.get("file", "")[:30] + "..." if len(result.get("file", "")) > 30 else result.get("file", ""),
            "Class": result.get("class", ""),
            "Detected": result.get("detected", ""),
            "Confidence": f"{result.get('confidence', 0):.3f}",
            "Energy Ratio": f"{result.get('energy_ratio', 0):.3f}",
            "Success": "✅" if result.get("success", False) else "❌",
            "Error": result.get("error", "")[:20] + "..." if len(result.get("error", "")) > 20 else result.get("error", "")
        }
        table_data.append(row)
    
    # Create and display DataFrame
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    
    # Summary statistics table
    print("\n" + "="*80)
    print("📈 SUMMARY STATISTICS")
    print("="*80)
    
    summary_data = []
    
    # Overall metrics
    summary = report.get("summary", {})
    summary_data.append({
        "Metric": "Total Files",
        "Value": summary.get("total_files", 0),
        "Unit": "files"
    })
    summary_data.append({
        "Metric": "Success Rate",
        "Value": f"{summary.get('success_rate', 0):.1%}",
        "Unit": "%"
    })
    summary_data.append({
        "Metric": "Avg Processing Time",
        "Value": f"{summary.get('average_time_per_file', 0):.2f}",
        "Unit": "seconds"
    })
    
    # Performance metrics
    performance = report.get("performance", {})
    summary_data.append({
        "Metric": "Avg Confidence",
        "Value": f"{performance.get('average_confidence', 0):.3f}",
        "Unit": "score"
    })
    summary_data.append({
        "Metric": "Avg Energy Ratio",
        "Value": f"{performance.get('average_energy_ratio', 0):.3f}",
        "Unit": "ratio"
    })
    
    # Class distribution
    class_dist = report.get("class_distribution", {})
    if class_dist:
        summary_data.append({
            "Metric": "Unique Classes",
            "Value": len(class_dist),
            "Unit": "classes"
        })
        most_common = max(class_dist.items(), key=lambda x: x[1]) if class_dist else ("", 0)
        summary_data.append({
            "Metric": "Most Common Class",
            "Value": f"{most_common[0]} ({most_common[1]})",
            "Unit": "count"
        })
    
    # Sound types
    sound_types = report.get("sound_types", {})
    if sound_types:
        for sound_type, count in sound_types.items():
            summary_data.append({
                "Metric": f"{sound_type.title()} Sounds",
                "Value": count,
                "Unit": "files"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Class performance breakdown
    if class_dist:
        print("\n" + "="*60)
        print("🎯 CLASS PERFORMANCE BREAKDOWN")
        print("="*60)
        
        class_data = []
        for class_name, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            class_data.append({
                "Class": class_name,
                "Count": count,
                "Percentage": f"{(count / summary.get('total_files', 1)) * 100:.1f}%"
            })
        
        class_df = pd.DataFrame(class_data)
        print(class_df.to_string(index=False))
    
    print("="*100)

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
    
    # Print detailed performance table
    print_detailed_performance_table(report)
    
    print(f"\n✅ Evaluation completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
