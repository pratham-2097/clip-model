#!/usr/bin/env python3
"""
Compare VLM model test results and generate a comprehensive comparison report.

Usage:
    python scripts/compare_vlm_models.py --results vlm_test_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_results(results_path: Path) -> List[Dict]:
    """Load test results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def generate_comparison_table(results: List[Dict]) -> str:
    """Generate a markdown comparison table."""
    table = "## Model Comparison Table\n\n"
    table += "| Model | Accuracy | Avg Inference (s) | Memory (GB) | Load Time (s) | Status |\n"
    table += "|-------|----------|-------------------|-------------|---------------|--------|\n"
    
    for result in results:
        model_name = result['model_name']
        accuracy = result['metrics']['accuracy']
        inf_time = result['metrics']['avg_inference_time']
        memory = result['final_memory_gb']
        load_time = result['load_time']
        
        # Status based on criteria
        status = "✅" if accuracy >= 0.80 and inf_time < 2.0 else "⚠️"
        
        table += f"| {model_name} | {accuracy:.2%} | {inf_time:.3f} | {memory:.2f} | {load_time:.2f} | {status} |\n"
    
    return table


def generate_per_condition_metrics(results: List[Dict]) -> str:
    """Generate per-condition accuracy metrics."""
    section = "## Per-Condition Accuracy\n\n"
    
    for result in results:
        section += f"### {result['model_name']}\n\n"
        section += "| Condition | Accuracy | Correct | Total |\n"
        section += "|----------|----------|---------|-------|\n"
        
        per_condition = result['metrics']['per_condition']
        for condition, metrics in per_condition.items():
            if metrics['total'] > 0:
                acc = metrics['correct'] / metrics['total']
                section += f"| {condition.capitalize()} | {acc:.2%} | {metrics['correct']} | {metrics['total']} |\n"
            else:
                section += f"| {condition.capitalize()} | N/A | 0 | 0 |\n"
        
        section += "\n"
    
    return section


def generate_per_object_metrics(results: List[Dict]) -> str:
    """Generate per-object-type accuracy metrics."""
    section = "## Per-Object-Type Accuracy\n\n"
    
    for result in results:
        section += f"### {result['model_name']}\n\n"
        section += "| Object Type | Accuracy | Correct | Total |\n"
        section += "|------------|----------|---------|-------|\n"
        
        per_object = result['metrics']['per_object_type']
        for obj_type, metrics in per_object.items():
            if metrics['total'] > 0:
                acc = metrics['correct'] / metrics['total']
                section += f"| {obj_type} | {acc:.2%} | {metrics['correct']} | {metrics['total']} |\n"
            else:
                section += f"| {obj_type} | N/A | 0 | 0 |\n"
        
        section += "\n"
    
    return section


def generate_recommendation(results: List[Dict]) -> str:
    """Generate final recommendation based on results."""
    section = "## Recommendation\n\n"
    
    # Sort by accuracy (primary) and inference time (secondary)
    sorted_results = sorted(
        results,
        key=lambda x: (x['metrics']['accuracy'], -x['metrics']['avg_inference_time']),
        reverse=True
    )
    
    best = sorted_results[0]
    
    section += f"### 🏆 Best Model: {best['model_name']}\n\n"
    section += f"**Rationale:**\n"
    section += f"- Overall Accuracy: {best['metrics']['accuracy']:.2%}\n"
    section += f"- Inference Time: {best['metrics']['avg_inference_time']:.3f}s per image\n"
    section += f"- Memory Usage: {best['final_memory_gb']:.2f} GB\n"
    
    # Check if meets criteria
    meets_criteria = (
        best['metrics']['accuracy'] >= 0.80 and
        best['metrics']['avg_inference_time'] < 2.0
    )
    
    if meets_criteria:
        section += f"\n✅ **Meets all success criteria for zero-shot performance.**\n"
        section += f"   Proceed to Phase 2: Dataset preparation and fine-tuning.\n"
    else:
        section += f"\n⚠️ **Does not fully meet success criteria.**\n"
        if best['metrics']['accuracy'] < 0.80:
            section += f"   - Accuracy ({best['metrics']['accuracy']:.2%}) below 80% target.\n"
            section += f"   - Consider fine-tuning to improve performance.\n"
        if best['metrics']['avg_inference_time'] >= 2.0:
            section += f"   - Inference time ({best['metrics']['avg_inference_time']:.3f}s) above 2s target.\n"
            section += f"   - Consider quantization for deployment.\n"
    
    # Alternative recommendations
    if len(sorted_results) > 1:
        section += f"\n### Alternative Options\n\n"
        for i, result in enumerate(sorted_results[1:3], 1):  # Top 2 alternatives
            section += f"{i}. **{result['model_name']}**\n"
            section += f"   - Accuracy: {result['metrics']['accuracy']:.2%}\n"
            section += f"   - Inference: {result['metrics']['avg_inference_time']:.3f}s\n"
            section += f"   - Memory: {result['final_memory_gb']:.2f} GB\n\n"
    
    return section


def generate_full_report(results: List[Dict], output_path: Path):
    """Generate full comparison report."""
    report = "# Stage 2 VLM Model Comparison Report\n\n"
    report += "**Generated:** " + str(Path.cwd()) + "\n\n"
    report += "---\n\n"
    
    report += generate_comparison_table(results)
    report += "\n"
    report += generate_per_condition_metrics(results)
    report += "\n"
    report += generate_per_object_metrics(results)
    report += "\n"
    report += generate_recommendation(results)
    
    report += "\n---\n\n"
    report += "## Next Steps\n\n"
    report += "1. Review model performance metrics\n"
    report += "2. Select best model based on accuracy, speed, and resource requirements\n"
    report += "3. Proceed to Phase 2: Dataset preparation\n"
    report += "4. Fine-tune selected model if zero-shot accuracy <85%\n"
    report += "5. Integrate with Stage 1 object detection pipeline\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Comparison report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare VLM model test results")
    parser.add_argument(
        "--results",
        type=str,
        default="vlm_test_results.json",
        help="Path to test results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="STAGE2_MODEL_COMPARISON.md",
        help="Output markdown file for comparison report"
    )
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"❌ Error: Results file not found at {results_path}")
        return
    
    print(f"📊 Loading results from {results_path}...")
    results = load_results(results_path)
    
    print(f"   Found {len(results)} model results\n")
    
    # Generate report
    output_path = Path(args.output)
    generate_full_report(results, output_path)
    
    # Print summary to console
    print("\n📊 Summary:")
    print(f"{'Model':<20} {'Accuracy':<12} {'Inference':<12} {'Memory':<10}")
    print("-" * 60)
    for result in sorted(results, key=lambda x: x['metrics']['accuracy'], reverse=True):
        print(f"{result['model_name']:<20} "
              f"{result['metrics']['accuracy']:>10.2%} "
              f"{result['metrics']['avg_inference_time']:>10.3f}s "
              f"{result['final_memory_gb']:>8.2f}GB")


if __name__ == "__main__":
    main()


