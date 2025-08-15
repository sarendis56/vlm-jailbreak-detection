#!/usr/bin/env python3
"""
Analyze correlation between principled layer selection scores and actual performance metrics.

This script compares the theoretical layer rankings from principled_layer_selection_results.csv
with the empirical performance from multi-run experiments (F1 and AUROC).

Supports both LLaVA and Qwen models with automatic model detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os
import sys

def load_principled_scores(model_type=None):
    """Load the principled layer selection results"""
    # Try different possible paths depending on where script is run from
    base_paths = [
        "results/",  # Run from project directory
        "HiddenDetect/results/"  # Run from parent directory
    ]

    # Model-specific filenames
    if model_type:
        # If specific model type provided, try multiple variations
        if model_type == "qwen":
            model_filenames = [
                "principled_layer_selection_results_qwen25vl_3b.csv",
                "principled_layer_selection_results_qwen25vl_7b.csv",
                "principled_layer_selection_results_qwen25vl.csv",
                "principled_layer_selection_results_qwen.csv"
            ]
        elif model_type == "llava":
            model_filenames = [
                "principled_layer_selection_results_llava_7b.csv",
                "principled_layer_selection_results_llava_13b.csv",
                "principled_layer_selection_results_llava.csv"
            ]
        else:
            model_filenames = [f"principled_layer_selection_results_{model_type}.csv"]
    else:
        # Try all possible model-specific files and fallback
        model_filenames = [
            "principled_layer_selection_results_qwen25vl_3b.csv",
            "principled_layer_selection_results_qwen25vl_7b.csv",
            "principled_layer_selection_results_qwen25vl.csv",
            "principled_layer_selection_results_llava_7b.csv",
            "principled_layer_selection_results_llava_13b.csv",
            "principled_layer_selection_results_llava.csv",
            "principled_layer_selection_results.csv"  # Fallback to old naming
        ]

    scores_path = None
    for base_path in base_paths:
        for filename in model_filenames:
            path = base_path + filename
            if os.path.exists(path):
                scores_path = path
                # Extract model type from filename for later use
                if "qwen" in filename.lower():
                    detected_model = "qwen"
                elif "llava" in filename.lower():
                    detected_model = "llava"
                else:
                    detected_model = "unknown"
                break
        if scores_path:
            break

    if scores_path is None:
        print(f"Error: No principled layer selection results found!")
        print("Looked for files:")
        for base_path in base_paths:
            for filename in model_filenames:
                print(f"  - {base_path + filename}")
        return None, None

    df = pd.read_csv(scores_path)
    print(f"Loaded principled scores for {len(df)} layers from: {scores_path}")
    print(f"Detected model type: {detected_model}")
    return df, detected_model

def find_latest_multi_run_results():
    """Find the latest multi-run results for each method"""
    # Try different possible base directories depending on where script is run from
    possible_base_dirs = [
        "multi_run_results",  # Run from HiddenDetect directory
        "HiddenDetect/multi_run_results"  # Run from parent directory
    ]

    base_dir = None
    for dir_path in possible_base_dirs:
        if os.path.exists(dir_path):
            base_dir = dir_path
            break

    if base_dir is None:
        print(f"Error: multi_run_results directory not found in any of these locations:")
        for dir_path in possible_base_dirs:
            print(f"  - {dir_path}")
        return {}

    methods = ['mcd', 'kcd', 'ml']
    latest_results = {}

    for method in methods:
        # Find all directories for this method
        pattern = f"{base_dir}/{method}_*runs_*"
        dirs = glob.glob(pattern)

        if dirs:
            # Get the latest directory by extracting and comparing timestamps
            def extract_timestamp(dir_path):
                # Extract timestamp from pattern like "ml_qwen_50runs_20250814_181114"
                import re
                match = re.search(r'_(\d{8}_\d{6})$', dir_path)
                return match.group(1) if match else '00000000_000000'

            # Sort by timestamp (newest first)
            latest_dir = sorted(dirs, key=extract_timestamp)[-1]
            latest_results[method] = latest_dir
            print(f"Found latest {method.upper()} results: {latest_dir}")
        else:
            print(f"Warning: No results found for {method}")

    return latest_results

def load_performance_data(results_dirs):
    """Load F1 and AUROC data from multi-run results"""
    all_data = {}

    for method, result_dir in results_dirs.items():
        print(f"\nLoading {method.upper()} data from {result_dir}")

        # Load classification metrics (F1)
        f1_path = f"{result_dir}/aggregated_{method}_classification.csv"
        auroc_path = f"{result_dir}/aggregated_{method}_performance.csv"

        if method == 'ml':
            # For ML method, load individual ML methods (SVM and MLP)
            method_data = {}

            # Detect model type from directory name
            model_type = 'llava'  # default
            if 'qwen' in result_dir.lower():
                model_type = 'qwen'
            elif 'llava' in result_dir.lower():
                model_type = 'llava'

            # Load F1 data
            if os.path.exists(f1_path):
                f1_df = pd.read_csv(f1_path)
                # Filter for COMBINED dataset only
                f1_combined = f1_df[f1_df['Dataset'] == 'COMBINED'].copy()

                # Extract SVM and MLP data separately
                svm_f1 = f1_combined[f1_combined['Method'] == f'ML_{model_type.upper()}_SVM'][['Layer', 'F1_Mean', 'F1_Std']].copy()
                mlp_f1 = f1_combined[f1_combined['Method'] == f'ML_{model_type.upper()}_MLP'][['Layer', 'F1_Mean', 'F1_Std']].copy()

                if not svm_f1.empty:
                    all_data['svm'] = {'f1': svm_f1}
                    print(f"  Loaded SVM F1 data for {len(svm_f1)} layers")

                if not mlp_f1.empty:
                    all_data['mlp'] = {'f1': mlp_f1}
                    print(f"  Loaded MLP F1 data for {len(mlp_f1)} layers")
            else:
                print(f"  Warning: {f1_path} not found")

            # Load AUROC data
            if os.path.exists(auroc_path):
                auroc_df = pd.read_csv(auroc_path)
                # Filter for COMBINED dataset only
                auroc_combined = auroc_df[auroc_df['Dataset'] == 'COMBINED'].copy()

                # Extract SVM and MLP data separately
                svm_auroc = auroc_combined[auroc_combined['Method'] == f'ML_{model_type.upper()}_SVM'][['Layer', 'AUROC_Mean', 'AUROC_Std']].copy()
                mlp_auroc = auroc_combined[auroc_combined['Method'] == f'ML_{model_type.upper()}_MLP'][['Layer', 'AUROC_Mean', 'AUROC_Std']].copy()

                if not svm_auroc.empty:
                    if 'svm' not in all_data:
                        all_data['svm'] = {}
                    all_data['svm']['auroc'] = svm_auroc
                    print(f"  Loaded SVM AUROC data for {len(svm_auroc)} layers")

                if not mlp_auroc.empty:
                    if 'mlp' not in all_data:
                        all_data['mlp'] = {}
                    all_data['mlp']['auroc'] = mlp_auroc
                    print(f"  Loaded MLP AUROC data for {len(mlp_auroc)} layers")
            else:
                print(f"  Warning: {auroc_path} not found")

        else:
            # For MCD/KCD methods, use original logic
            method_data = {}

            # Load F1 data
            if os.path.exists(f1_path):
                f1_df = pd.read_csv(f1_path)
                # Filter for COMBINED dataset only
                f1_combined = f1_df[f1_df['Dataset'] == 'COMBINED'].copy()
                method_data['f1'] = f1_combined[['Layer', 'F1_Mean', 'F1_Std']].copy()
                print(f"  Loaded F1 data for {len(method_data['f1'])} layers")
            else:
                print(f"  Warning: {f1_path} not found")

            # Load AUROC data
            if os.path.exists(auroc_path):
                auroc_df = pd.read_csv(auroc_path)
                # Filter for COMBINED dataset only
                auroc_combined = auroc_df[auroc_df['Dataset'] == 'COMBINED'].copy()
                method_data['auroc'] = auroc_combined[['Layer', 'AUROC_Mean', 'AUROC_Std']].copy()
                print(f"  Loaded AUROC data for {len(method_data['auroc'])} layers")
            else:
                print(f"  Warning: {auroc_path} not found")

            all_data[method] = method_data

    return all_data



def create_f1_correlation_plot(principled_scores, performance_data, model_type="unknown", output_dir=None):
    """Create F1 correlation plot"""

    # Determine output directory based on current working directory
    if output_dir is None:
        if os.path.exists("results"):
            output_dir = "results"  # Run from project directory
        else:
            output_dir = "HiddenDetect/results"  # Run from parent directory

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create single plot for F1
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1_twin = ax1.twinx()

    # Bar chart for Overall Score (left y-axis)
    bars = ax1.bar(principled_scores['Layer'], principled_scores['Overall_Score'],
                   alpha=0.6, color='lightblue', label='Overall Score', width=0.6)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Overall Score', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.0)

    # Set x-axis limits based on aligned data
    max_layer = principled_scores['Layer'].max()
    ax1.set_xlim(-1, max_layer + 1)

    # Line plots for F1 scores (right y-axis)
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (method, data) in enumerate(performance_data.items()):
        if 'f1' in data and not data['f1'].empty:
            f1_data = data['f1'].sort_values('Layer')
            ax1_twin.errorbar(f1_data['Layer'], f1_data['F1_Mean'],
                            yerr=f1_data['F1_Std'],
                            color=colors[i % len(colors)], marker=markers[i % len(markers)],
                            label=f'{method.upper()} F1', linewidth=2, markersize=8,
                            capsize=4, capthick=1.5)

    ax1_twin.set_ylabel('F1 Score', color='red', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.set_ylim(0, 1.0)

    # Add legends
    ax1.legend(loc='upper left', fontsize=11)
    ax1_twin.legend(loc='upper right', fontsize=11)

    # Update title to include model type
    model_title = model_type.upper() if model_type != "unknown" else ""
    title = f'F1 Score vs Overall Score by Layer ({model_title})' if model_title else 'F1 Score vs Overall Score by Layer'
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save as PDF with model-specific name
    if model_type != "unknown":
        output_path = f"{output_dir}/f1_correlation_plot_{model_type}.pdf"
    else:
        output_path = f"{output_dir}/f1_correlation_plot.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"F1 correlation plot saved to: {output_path}")

    plt.close()

def create_auroc_correlation_plot(principled_scores, performance_data, model_type="unknown", output_dir=None):
    """Create AUROC correlation plot"""

    # Determine output directory based on current working directory
    if output_dir is None:
        if os.path.exists("results"):
            output_dir = "results"  # Run from project directory
        else:
            output_dir = "HiddenDetect/results"  # Run from parent directory

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create single plot for AUROC
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    ax2_twin = ax2.twinx()

    # Bar chart for Overall Score (left y-axis)
    bars = ax2.bar(principled_scores['Layer'], principled_scores['Overall_Score'],
                   alpha=0.6, color='lightblue', label='Overall Score', width=0.6)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Overall Score', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1.0)

    # Set x-axis limits based on aligned data
    max_layer = principled_scores['Layer'].max()
    ax2.set_xlim(-1, max_layer + 1)

    # Line plots for AUROC scores (right y-axis)
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (method, data) in enumerate(performance_data.items()):
        if 'auroc' in data and not data['auroc'].empty:
            auroc_data = data['auroc'].sort_values('Layer')
            ax2_twin.errorbar(auroc_data['Layer'], auroc_data['AUROC_Mean'],
                            yerr=auroc_data['AUROC_Std'],
                            color=colors[i % len(colors)], marker=markers[i % len(markers)],
                            label=f'{method.upper()} AUROC', linewidth=2, markersize=8,
                            capsize=4, capthick=1.5)

    ax2_twin.set_ylabel('AUROC', color='red', fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2_twin.set_ylim(0, 1.0)

    # Add legends
    ax2.legend(loc='upper left', fontsize=11)
    ax2_twin.legend(loc='upper right', fontsize=11)

    # Update title to include model type
    model_title = model_type.upper() if model_type != "unknown" else ""
    title = f'AUROC vs Overall Score by Layer ({model_title})' if model_title else 'AUROC vs Overall Score by Layer'
    ax2.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save as PDF with model-specific name
    if model_type != "unknown":
        output_path = f"{output_dir}/auroc_correlation_plot_{model_type}.pdf"
    else:
        output_path = f"{output_dir}/auroc_correlation_plot.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"AUROC correlation plot saved to: {output_path}")

    plt.close()

def compute_correlations(principled_scores, performance_data):
    """Compute correlation coefficients between scores and performance metrics"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    correlations = {}

    for method, data in performance_data.items():
        print(f"\n{method.upper()} Method:")
        method_corr = {}

        # Merge with principled scores
        if 'f1' in data and not data['f1'].empty:
            merged_f1 = pd.merge(principled_scores, data['f1'], on='Layer', how='inner')
            if len(merged_f1) > 1:
                f1_corr = merged_f1['Overall_Score'].corr(merged_f1['F1_Mean'])
                method_corr['f1'] = f1_corr
                print(f"  F1 vs Overall Score correlation: {f1_corr:.4f}")

        if 'auroc' in data and not data['auroc'].empty:
            merged_auroc = pd.merge(principled_scores, data['auroc'], on='Layer', how='inner')
            if len(merged_auroc) > 1:
                auroc_corr = merged_auroc['Overall_Score'].corr(merged_auroc['AUROC_Mean'])
                method_corr['auroc'] = auroc_corr
                print(f"  AUROC vs Overall Score correlation: {auroc_corr:.4f}")

        correlations[method] = method_corr

    return correlations

def print_top_layers_comparison(principled_scores, performance_data, top_n=10):
    """Compare top layers from principled scores vs actual performance"""
    print("\n" + "="*80)
    print(f"TOP {top_n} LAYERS COMPARISON")
    print("="*80)

    # Top layers by principled score
    top_principled = principled_scores.nlargest(top_n, 'Overall_Score')['Layer'].tolist()
    print(f"Top {top_n} layers by Overall Score: {top_principled}")

    # Top layers by performance metrics
    for method, data in performance_data.items():
        print(f"\n{method.upper()} Method:")

        if 'f1' in data and not data['f1'].empty:
            top_f1 = data['f1'].nlargest(top_n, 'F1_Mean')['Layer'].tolist()
            print(f"  Top {top_n} layers by F1: {top_f1}")

            # Calculate overlap
            overlap_f1 = len(set(top_principled) & set(top_f1))
            print(f"  F1 overlap with principled: {overlap_f1}/{top_n} ({overlap_f1/top_n*100:.1f}%)")

        if 'auroc' in data and not data['auroc'].empty:
            top_auroc = data['auroc'].nlargest(top_n, 'AUROC_Mean')['Layer'].tolist()
            print(f"  Top {top_n} layers by AUROC: {top_auroc}")

            # Calculate overlap
            overlap_auroc = len(set(top_principled) & set(top_auroc))
            print(f"  AUROC overlap with principled: {overlap_auroc}/{top_n} ({overlap_auroc/top_n*100:.1f}%)")

def create_summary_table(correlations, performance_data, model_type="unknown", output_dir=None):
    """Create a summary table of the analysis results"""

    # Determine output directory based on current working directory
    if output_dir is None:
        if os.path.exists("results"):
            output_dir = "results"  # Run from project directory
        else:
            output_dir = "HiddenDetect/results"  # Run from parent directory

    # Create summary data
    summary_data = []

    for method, corr_data in correlations.items():
        row = {
            'Method': method.upper(),
            'F1_Correlation': corr_data.get('f1', 'N/A'),
            'AUROC_Correlation': corr_data.get('auroc', 'N/A'),
        }

        # Add performance statistics
        if 'f1' in performance_data[method]:
            f1_data = performance_data[method]['f1']
            row['F1_Mean_Avg'] = f1_data['F1_Mean'].mean()
            row['F1_Mean_Std'] = f1_data['F1_Mean'].std()

        if 'auroc' in performance_data[method]:
            auroc_data = performance_data[method]['auroc']
            row['AUROC_Mean_Avg'] = auroc_data['AUROC_Mean'].mean()
            row['AUROC_Mean_Std'] = auroc_data['AUROC_Mean'].std()

        summary_data.append(row)

    # Create DataFrame and save with model-specific name
    summary_df = pd.DataFrame(summary_data)
    if model_type != "unknown":
        summary_path = f"{output_dir}/layer_correlation_summary_{model_type}.csv"
    else:
        summary_path = f"{output_dir}/layer_correlation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSummary table saved to: {summary_path}")
    print("\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

def main():
    print("="*80)
    print("LAYER PERFORMANCE CORRELATION ANALYSIS")
    print("="*80)
    print("Usage: python analyze_layer_performance_correlation.py [qwen|llava]")
    print("If no model type is specified, will auto-detect from available files.")
    print("="*80)

    # Parse command line arguments for model type
    model_type_arg = None
    if len(sys.argv) > 1:
        model_type_arg = sys.argv[1].lower()
        if model_type_arg not in ['qwen', 'llava']:
            print(f"Warning: Unknown model type '{model_type_arg}'. Will auto-detect from files.")
            model_type_arg = None

    # Load principled layer selection scores
    principled_scores, detected_model = load_principled_scores(model_type_arg)
    if principled_scores is None:
        return

    # Use command line argument if provided, otherwise use detected model, fallback to "unknown"
    model_type = model_type_arg if model_type_arg else (detected_model if detected_model else "unknown")

    # Find latest multi-run results
    results_dirs = find_latest_multi_run_results()
    if not results_dirs:
        print("Error: No multi-run results found!")
        return

    # Load performance data
    performance_data = load_performance_data(results_dirs)

    # Create separate correlation plots
    create_f1_correlation_plot(principled_scores, performance_data, model_type)
    create_auroc_correlation_plot(principled_scores, performance_data, model_type)

    # Compute correlations
    correlations = compute_correlations(principled_scores, performance_data)

    # Compare top layers
    print_top_layers_comparison(principled_scores, performance_data, top_n=10)

    # Create summary table
    create_summary_table(correlations, performance_data, model_type)

    # Determine output directory for final messages
    if os.path.exists("results"):
        results_dir = "results"  # Run from project directory
    else:
        results_dir = "HiddenDetect/results"  # Run from parent directory

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Model type: {model_type.upper()}")
    print("Check the generated plots:")

    # Show model-specific filenames
    if model_type != "unknown":
        print(f"  - F1 correlation: {results_dir}/f1_correlation_plot_{model_type}.pdf")
        print(f"  - AUROC correlation: {results_dir}/auroc_correlation_plot_{model_type}.pdf")
        print(f"Check the summary table: {results_dir}/layer_correlation_summary_{model_type}.csv")
    else:
        print(f"  - F1 correlation: {results_dir}/f1_correlation_plot.pdf")
        print(f"  - AUROC correlation: {results_dir}/auroc_correlation_plot.pdf")
        print(f"Check the summary table: {results_dir}/layer_correlation_summary.csv")

if __name__ == "__main__":
    main()
