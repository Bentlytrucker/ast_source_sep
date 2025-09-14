"""
Visualization tools for evaluation results
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from config import OUTPUT_DIR

class EvaluationVisualizer:
    """Create visualizations for evaluation results"""
    
    def __init__(self, results_dir: str = OUTPUT_DIR):
        self.results_dir = results_dir
        
    def load_results(self, report_file: str = "evaluation_report.json") -> Dict:
        """Load evaluation results from JSON file"""
        report_path = os.path.join(self.results_dir, report_file)
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_class_distribution(self, report: Dict, save_path: str = None):
        """Plot class distribution"""
        class_counts = report["class_distribution"]
        
        if not class_counts:
            print("No class data to plot")
            return
            
        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = plt.bar(range(len(classes)), counts)
        plt.xlabel("Detected Classes")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_distribution(self, report: Dict, save_path: str = None):
        """Plot confidence score distribution"""
        detailed_results = report["detailed_results"]
        confidences = [r["confidence"] for r in detailed_results if r["success"]]
        
        if not confidences:
            print("No confidence data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.title("Confidence Score Distribution")
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_energy_ratio_distribution(self, report: Dict, save_path: str = None):
        """Plot energy ratio distribution"""
        detailed_results = report["detailed_results"]
        energy_ratios = [r["energy_ratio"] for r in detailed_results if r["success"] and r["energy_ratio"] > 0]
        
        if not energy_ratios:
            print("No energy ratio data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(energy_ratios, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel("Energy Ratio")
        plt.ylabel("Frequency")
        plt.title("Energy Ratio Distribution")
        plt.axvline(np.mean(energy_ratios), color='red', linestyle='--',
                   label=f'Mean: {np.mean(energy_ratios):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_processing_time_distribution(self, report: Dict, save_path: str = None):
        """Plot processing time distribution"""
        detailed_results = report["detailed_results"]
        processing_times = [r.get("processing_time", 0) for r in detailed_results]
        
        if not processing_times:
            print("No processing time data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(processing_times, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel("Processing Time (seconds)")
        plt.ylabel("Frequency")
        plt.title("Processing Time Distribution")
        plt.axvline(np.mean(processing_times), color='red', linestyle='--',
                   label=f'Mean: {np.mean(processing_times):.2f}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sound_type_distribution(self, report: Dict, save_path: str = None):
        """Plot sound type distribution (help/warning/danger/other)"""
        sound_types = report["sound_types"]
        
        if not sound_types:
            print("No sound type data to plot")
            return
            
        plt.figure(figsize=(8, 6))
        types = list(sound_types.keys())
        counts = list(sound_types.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        wedges, texts, autotexts = plt.pie(counts, labels=types, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        plt.title("Sound Type Distribution")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, report: Dict, save_dir: str = None):
        """Create a comprehensive dashboard with all visualizations"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("AudioSet Evaluation Dashboard", fontsize=16)
        
        # 1. Class distribution
        class_counts = report["class_distribution"]
        if class_counts:
            classes = list(class_counts.keys())[:10]  # Top 10 classes
            counts = [class_counts[c] for c in classes]
            axes[0, 0].bar(range(len(classes)), counts)
            axes[0, 0].set_title("Top 10 Detected Classes")
            axes[0, 0].set_xticks(range(len(classes)))
            axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
        
        # 2. Confidence distribution
        detailed_results = report["detailed_results"]
        confidences = [r["confidence"] for r in detailed_results if r["success"]]
        if confidences:
            axes[0, 1].hist(confidences, bins=15, alpha=0.7)
            axes[0, 1].set_title("Confidence Distribution")
            axes[0, 1].set_xlabel("Confidence Score")
            axes[0, 1].axvline(np.mean(confidences), color='red', linestyle='--')
        
        # 3. Energy ratio distribution
        energy_ratios = [r["energy_ratio"] for r in detailed_results if r["success"] and r["energy_ratio"] > 0]
        if energy_ratios:
            axes[0, 2].hist(energy_ratios, bins=15, alpha=0.7)
            axes[0, 2].set_title("Energy Ratio Distribution")
            axes[0, 2].set_xlabel("Energy Ratio")
            axes[0, 2].axvline(np.mean(energy_ratios), color='red', linestyle='--')
        
        # 4. Sound type pie chart
        sound_types = report["sound_types"]
        if sound_types:
            types = list(sound_types.keys())
            counts = list(sound_types.values())
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            axes[1, 0].pie(counts, labels=types, colors=colors, autopct='%1.1f%%')
            axes[1, 0].set_title("Sound Type Distribution")
        
        # 5. Processing time distribution
        processing_times = [r.get("processing_time", 0) for r in detailed_results]
        if processing_times:
            axes[1, 1].hist(processing_times, bins=15, alpha=0.7)
            axes[1, 1].set_title("Processing Time Distribution")
            axes[1, 1].set_xlabel("Time (seconds)")
            axes[1, 1].axvline(np.mean(processing_times), color='red', linestyle='--')
        
        # 6. Success rate summary
        summary = report["summary"]
        success_rate = summary["success_rate"]
        axes[1, 2].bar(['Success', 'Failed'], 
                      [success_rate, 1-success_rate], 
                      color=['green', 'red'], alpha=0.7)
        axes[1, 2].set_title(f"Success Rate: {success_rate:.1%}")
        axes[1, 2].set_ylabel("Proportion")
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "evaluation_dashboard.png"), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_html_report(self, report: Dict, save_path: str = None):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AudioSet Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                         background-color: #e8f4f8; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AudioSet Evaluation Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">Total Files: {report['summary']['total_files']}</div>
                <div class="metric">Success Rate: {report['summary']['success_rate']:.1%}</div>
                <div class="metric">Avg Confidence: {report['performance']['average_confidence']:.3f}</div>
                <div class="metric">Avg Energy Ratio: {report['performance']['average_energy_ratio']:.3f}</div>
            </div>
            
            <div class="section">
                <h2>Class Distribution</h2>
                <table>
                    <tr><th>Class</th><th>Count</th></tr>
        """
        
        for class_name, count in report["class_distribution"].items():
            html_content += f"<tr><td>{class_name}</td><td>{count}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Sound Types</h2>
                <table>
                    <tr><th>Type</th><th>Count</th></tr>
        """
        
        for sound_type, count in report["sound_types"].items():
            html_content += f"<tr><td>{sound_type}</td><td>{count}</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML report saved to: {save_path}")
        
        return html_content

def main():
    """Main function for visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results-dir", type=str, default=OUTPUT_DIR,
                       help="Directory containing evaluation results")
    parser.add_argument("--report-file", type=str, default="evaluation_report.json",
                       help="Evaluation report JSON file")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    visualizer = EvaluationVisualizer(args.results_dir)
    
    # Load results
    report = visualizer.load_results(args.report_file)
    
    # Create visualizations
    print("Creating visualizations...")
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Individual plots
    visualizer.plot_class_distribution(report, 
        os.path.join(args.save_dir, "class_distribution.png") if args.save_dir else None)
    visualizer.plot_confidence_distribution(report,
        os.path.join(args.save_dir, "confidence_distribution.png") if args.save_dir else None)
    visualizer.plot_energy_ratio_distribution(report,
        os.path.join(args.save_dir, "energy_ratio_distribution.png") if args.save_dir else None)
    visualizer.plot_processing_time_distribution(report,
        os.path.join(args.save_dir, "processing_time_distribution.png") if args.save_dir else None)
    visualizer.plot_sound_type_distribution(report,
        os.path.join(args.save_dir, "sound_type_distribution.png") if args.save_dir else None)
    
    # Comprehensive dashboard
    visualizer.create_comprehensive_dashboard(report, args.save_dir)
    
    # HTML report
    visualizer.generate_html_report(report,
        os.path.join(args.save_dir, "evaluation_report.html") if args.save_dir else None)
    
    print("Visualization completed!")

if __name__ == "__main__":
    main()
