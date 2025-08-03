#!/usr/bin/env python3
"""
Multi-seed ablation experiment runner for balanced OOD detection
Runs ablation experiments (no projection) multiple times with different seeds and computes averaged results
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime

def backup_original_file(filepath):
    """Create a backup of the original file"""
    backup_path = str(filepath) + '.backup'
    with open(filepath, 'r') as original:
        content = original.read()
    with open(backup_path, 'w') as backup:
        backup.write(content)
    return backup_path

def restore_from_backup(filepath, backup_path):
    """Restore file from backup"""
    with open(backup_path, 'r') as backup:
        content = backup.read()
    with open(filepath, 'w') as original:
        original.write(content)

def modify_seed_in_content(content, new_seed):
    """Modify the MAIN_SEED variable in file content (centralized seed management)"""
    import re

    # Target the centralized MAIN_SEED variable (both direct and config-based)
    main_seed_pattern = r'MAIN_SEED = \d+'
    replacement = f'MAIN_SEED = {new_seed}'

    # Check if the new centralized seed system is being used
    if re.search(main_seed_pattern, content):
        # Use the new centralized approach - this handles both direct MAIN_SEED and config-based MAIN_SEED
        content = re.sub(main_seed_pattern, replacement, content)
        print(f"  Using centralized MAIN_SEED approach")

        # Count how many replacements were made
        seed_count = len(re.findall(main_seed_pattern.replace(r'\d+', str(new_seed)), content))
        if seed_count > 1:
            print(f"  Updated {seed_count} MAIN_SEED occurrences (including config classes)")
    else:
        # Fallback to old approach for backward compatibility
        print(f"  Warning: MAIN_SEED not found, using legacy seed replacement")

        # Replace individual seed calls (legacy approach)
        content = re.sub(r'random\.seed\(\d+\)', f'random.seed({new_seed})', content)
        content = re.sub(r'np\.random\.seed\(\d+\)', f'np.random.seed({new_seed})', content)
        content = re.sub(r'torch\.manual_seed\(\d+\)', f'torch.manual_seed({new_seed})', content)
        content = re.sub(r'torch\.cuda\.manual_seed\(\d+\)', f'torch.cuda.manual_seed({new_seed})', content)
        content = re.sub(r'torch\.cuda\.manual_seed_all\(\d+\)', f'torch.cuda.manual_seed_all({new_seed})', content)

        # Replace layer seed calculation (any_number + layer_idx pattern)
        content = re.sub(r'layer_seed = \d+ \+ layer_idx', f'layer_seed = {new_seed} + layer_idx', content)

        # Replace the print statement (both literal numbers and variable references)
        content = re.sub(r'Random seeds set for reproducibility \(seed=\d+\)',
                        f'Random seeds set for reproducibility (seed={new_seed})', content)
        # Also handle cases where the print statement uses a variable like {MAIN_SEED}
        content = re.sub(r'Random seeds set for reproducibility \(seed=\{MAIN_SEED\}\)',
                        f'Random seeds set for reproducibility (seed={new_seed})', content)

        # Replace the default parameter in function signature
        content = re.sub(r'random_seed=\d+\)', f'random_seed={new_seed})', content)

    return content

def modify_seed_in_file(filepath, new_seed):
    """Modify the random seed in a Python file"""
    print(f"Updating seed to {new_seed} in {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    modified_content = modify_seed_in_content(content, new_seed)

    # Verify that changes were made
    if content == modified_content:
        # Check if seed is already correct
        import re
        if re.search(f'MAIN_SEED = {new_seed}', content):
            print(f"  MAIN_SEED already set to {new_seed} - no changes needed")
        else:
            print(f"  Warning: No seed replacements made in {filepath}")
            print("  This might indicate the seed patterns were not found or script needs updating")
    else:
        print(f"  Successfully updated seed to {new_seed}")

    with open(filepath, 'w') as f:
        f.write(modified_content)

def run_single_experiment(script_path, seed, run_id, total_runs, working_dir):
    """Run a single experiment with given seed"""
    print(f"\n{'='*80}")
    print(f"RUNNING ABLATION EXPERIMENT {run_id}/{total_runs} WITH SEED {seed}")
    print(f"{'='*80}")

    # Create backup of original file
    backup_path = backup_original_file(script_path)

    try:
        # Modify seed in the script
        modify_seed_in_file(script_path, seed)

        # Run the experiment from the correct working directory
        # Use relative path from working_dir to script
        script_relative = os.path.relpath(script_path, working_dir)

        print(f"Working directory: {working_dir}")
        print(f"Running script: {script_relative}")
        print(f"Using seed: {seed}")

        result = subprocess.run([sys.executable, script_relative],
                              capture_output=True, text=True, cwd=working_dir)

        if result.returncode != 0:
            print(f"ERROR in run {run_id}:")
            print("STDOUT:", result.stdout[-1000:] if result.stdout else "None")  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:] if result.stderr else "None")  # Last 1000 chars
            return False
        else:
            print(f"Run {run_id} completed successfully")
            return True

    except Exception as e:
        print(f"ERROR running experiment {run_id}: {e}")
        return False
    finally:
        # Always restore from backup after each run
        restore_from_backup(script_path, backup_path)
        # Clean up backup file
        try:
            os.remove(backup_path)
        except:
            pass  # Ignore cleanup errors

def load_results_csv(csv_path):
    """Load results from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def aggregate_results_group(results_list, method_name, metric_group, group_name):
    """Aggregate results for a specific group of metrics"""
    if not results_list:
        return None

    # Combine all dataframes
    all_results = pd.concat(results_list, ignore_index=True)

    # Filter out non-numeric values (like "N/A")
    for col in metric_group:
        if col in all_results.columns:
            all_results[col] = pd.to_numeric(all_results[col], errors='coerce')

    # Group and aggregate by Layer and Dataset
    aggregated_results = []
    grouped = all_results.groupby(['Layer', 'Dataset'])

    for (layer, dataset), group in grouped:
        result_row = {
            'Layer': layer,
            'Dataset': dataset,
            'Method': method_name,  # Use the provided method name
            'Group': group_name,
            'Runs': len(group)
        }

        # Compute mean, std, and max/min for each metric
        for col in metric_group:
            if col in group.columns:
                values = group[col].dropna()
                if len(values) > 0:
                    result_row[f'{col}_Mean'] = round(values.mean(), 4)
                    result_row[f'{col}_Std'] = round(values.std() if len(values) > 1 else 0.0, 4)

                    # For FPR, we want minimum (lower is better)
                    # For other metrics, we want maximum (higher is better)
                    if col == 'FPR':
                        result_row[f'{col}_Min'] = round(values.min(), 4)
                    else:
                        result_row[f'{col}_Max'] = round(values.max(), 4)
                else:
                    # Use 0.0000 instead of NaN for missing/non-applicable metrics
                    result_row[f'{col}_Mean'] = 0.0000
                    result_row[f'{col}_Std'] = 0.0000

                    if col == 'FPR':
                        result_row[f'{col}_Min'] = 0.0000
                    else:
                        result_row[f'{col}_Max'] = 0.0000

        aggregated_results.append(result_row)

    df = pd.DataFrame(aggregated_results)

    # Add individual rankings within each group based on primary metric
    # Group 1 (Performance): rank by Accuracy_Mean (higher is better)
    # Group 2 (Classification): rank by F1_Mean (higher is better)
    if group_name == "Performance":
        primary_metric = 'Accuracy_Mean'
        ascending = False  # Higher accuracy is better
    else:  # Classification group
        primary_metric = 'F1_Mean'
        ascending = False  # Higher F1 is better

    # Add ranking for COMBINED results only
    combined_df = df[df['Dataset'] == 'COMBINED'].copy()
    if not combined_df.empty and primary_metric in combined_df.columns:
        combined_df = combined_df.sort_values(primary_metric, ascending=ascending)
        combined_df[f'{group_name}_Rank'] = range(1, len(combined_df) + 1)

        # Merge rankings back to main dataframe
        rank_mapping = dict(zip(combined_df['Layer'], combined_df[f'{group_name}_Rank']))
        df[f'{group_name}_Rank'] = df['Layer'].map(rank_mapping)
        df[f'{group_name}_Rank'] = df[f'{group_name}_Rank'].fillna(0)  # Fill NaN with 0 for non-COMBINED
    else:
        df[f'{group_name}_Rank'] = 0

    return df

def aggregate_results(results_list, method_name):
    """Aggregate results into two focused groups"""
    if not results_list:
        return None, None

    # Define metric groups
    group1_metrics = ['Accuracy', 'AUROC', 'AUPRC']  # Performance metrics
    group2_metrics = ['TPR', 'FPR', 'F1']            # Classification metrics

    # Create aggregated results for each group
    group1_df = aggregate_results_group(results_list, method_name, group1_metrics, "Performance")
    group2_df = aggregate_results_group(results_list, method_name, group2_metrics, "Classification")

    return group1_df, group2_df

def main():
    parser = argparse.ArgumentParser(description='Run multiple ablation experiments with different seeds')
    parser.add_argument('--script', required=True, choices=['kcd', 'mcd', 'both'],
                       help='Which ablation script to run (kcd, mcd, or both)')
    parser.add_argument('--runs', type=int, default=50,
                       help='Number of runs (default: 50)')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='Custom seeds to use (if not provided, will use consecutive seeds starting from 42)')
    parser.add_argument('--output-dir', default='ablation_results',
                       help='Output directory for results (default: ablation_results)')

    # Add usage examples in help
    parser.epilog = """
Examples:
  python run_multiple_ablation.py --script kcd --runs 10
  python run_multiple_ablation.py --script mcd --runs 5 --seeds 42 43 44 45 46
  python run_multiple_ablation.py --script both --runs 20
  python run_multiple_ablation.py --script kcd --runs 30 --output-dir my_ablation_results
"""

    args = parser.parse_args()

    # Determine script path and working directory
    script_dir = Path(__file__).parent  # This is the 'code' directory
    working_dir = script_dir.parent      # This is the parent directory (where experiments should run)

    # Define scripts to run
    scripts_to_run = []
    if args.script == 'kcd':
        scripts_to_run = [('kcd', script_dir / 'balanced_ood_kcd_ablate_projection.py', 'results/balanced_kcd_ablate_projection_results.csv', 'KCD_Ablation_MultiSeed')]
    elif args.script == 'mcd':
        scripts_to_run = [('mcd', script_dir / 'balanced_ood_mcd_ablate_projection.py', 'results/balanced_mcd_ablate_projection_results.csv', 'MCD_Ablation_MultiSeed')]
    else:  # both
        scripts_to_run = [
            ('kcd', script_dir / 'balanced_ood_kcd_ablate_projection.py', 'results/balanced_kcd_ablate_projection_results.csv', 'KCD_Ablation_MultiSeed'),
            ('mcd', script_dir / 'balanced_ood_mcd_ablate_projection.py', 'results/balanced_mcd_ablate_projection_results.csv', 'MCD_Ablation_MultiSeed')
        ]

    # Verify all scripts exist
    for script_name, script_path, _, _ in scripts_to_run:
        if not script_path.exists():
            print(f"Error: Script {script_path} not found!")
            return

    print(f"Script directory: {script_dir}")
    print(f"Working directory: {working_dir}")

    # Determine seeds
    if args.seeds:
        seeds = args.seeds[:args.runs]  # Use provided seeds
        if len(seeds) < args.runs:
            print(f"Warning: Only {len(seeds)} seeds provided, but {args.runs} runs requested")
            args.runs = len(seeds)
    else:
        seeds = list(range(42, 42 + args.runs))  # Use consecutive seeds starting from 42

    # Create output directory (relative to working directory)
    output_dir = working_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Create run-specific subdirectory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"ablation_{args.script}_{args.runs}runs_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print(f"Running {args.runs} ablation experiments with {args.script.upper()} method(s)")
    print(f"Seeds: {seeds}")
    print(f"Results will be saved to: {run_dir}")

    # Run experiments for each script
    all_results = {}

    for script_name, script_path, expected_csv, method_name in scripts_to_run:
        print(f"\n{'='*100}")
        print(f"RUNNING {script_name.upper()} ABLATION EXPERIMENTS")
        print(f"{'='*100}")
        print(f"Script: {script_path}")
        print(f"Expected CSV: {expected_csv}")

        # Store individual run results for this script
        individual_results = []
        successful_runs = 0

        for i, seed in enumerate(seeds, 1):
            success = run_single_experiment(script_path, seed, i, args.runs, working_dir)

            if success:
                # Move the generated CSV to run directory (CSV is generated in working_dir)
                csv_source = working_dir / expected_csv

                # Create results directory if it doesn't exist
                results_dir = working_dir / "results"
                if not results_dir.exists():
                    results_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Created results directory: {results_dir}")

                # Check if the CSV file actually exists
                if not csv_source.exists():
                    print(f"Warning: Expected CSV file not found at {csv_source}")
                    print(f"Looking for alternative locations...")

                    # Try alternative paths
                    alt_paths = [
                        working_dir / expected_csv.split('/')[-1],  # Just filename in working_dir
                        script_dir / expected_csv.split('/')[-1],   # Just filename in script_dir
                        script_dir / expected_csv                   # Full path in script_dir
                    ]

                    for alt_path in alt_paths:
                        if alt_path.exists():
                            csv_source = alt_path
                            print(f"Found CSV at alternative location: {csv_source}")
                            break
                    else:
                        print(f"ERROR: Could not find CSV file for {script_name} run {i}")
                        continue

                # Create destination filename (just the base filename, not the full path)
                csv_filename = expected_csv.split('/')[-1]  # Get just the filename
                csv_dest = run_dir / f"{script_name}_run_{i:02d}_seed_{seed}_{csv_filename}"

                # Move/copy the file
                try:
                    csv_source.rename(csv_dest)
                    print(f"Results moved: {csv_source} -> {csv_dest}")

                    # Load and store results
                    df = load_results_csv(csv_dest)
                    if df is not None:
                        df['Run'] = i
                        df['Seed'] = seed
                        individual_results.append(df)
                        successful_runs += 1
                        print(f"Results processed successfully for {script_name} run {i}")
                    else:
                        print(f"Warning: Could not load CSV data for {script_name} run {i}")
                except Exception as e:
                    print(f"Error moving results file for {script_name} run {i}: {e}")

            print(f"Completed {script_name} {i}/{args.runs} runs ({successful_runs} successful)")

        if successful_runs == 0:
            print(f"ERROR: No successful runs completed for {script_name}!")
            continue

        print(f"\n{'='*80}")
        print(f"AGGREGATING {script_name.upper()} RESULTS FROM {successful_runs} SUCCESSFUL RUNS")
        print(f"{'='*80}")

        # Aggregate results for this script
        group1_df, group2_df = aggregate_results(individual_results, method_name)

        if group1_df is not None and group2_df is not None:
            # Save aggregated results for both groups
            group1_csv = run_dir / f"aggregated_{script_name}_ablation_performance.csv"
            group2_csv = run_dir / f"aggregated_{script_name}_ablation_classification.csv"

            group1_df.to_csv(group1_csv, index=False)
            group2_df.to_csv(group2_csv, index=False)

            print(f"{script_name.upper()} Performance metrics saved: {group1_csv}")
            print(f"{script_name.upper()} Classification metrics saved: {group2_csv}")

            # Store results for final summary
            all_results[script_name] = {
                'method_name': method_name,
                'successful_runs': successful_runs,
                'performance_df': group1_df,
                'classification_df': group2_df,
                'performance_csv': group1_csv,
                'classification_csv': group2_csv
            }
        else:
            print(f"ERROR: Failed to aggregate {script_name} results!")

    # Save experiment metadata
    metadata = {
        'script': args.script,
        'total_runs': args.runs,
        'seeds': seeds,
        'timestamp': timestamp,
        'scripts_run': [script_name for script_name, _, _, _ in scripts_to_run],
        'successful_runs': {script_name: all_results[script_name]['successful_runs']
                           for script_name in all_results.keys()},
        'output_files': {}
    }

    for script_name in all_results.keys():
        metadata['output_files'][script_name] = {
            'performance_metrics': str(all_results[script_name]['performance_csv'].name),
            'classification_metrics': str(all_results[script_name]['classification_csv'].name)
        }

    metadata_file = run_dir / 'ablation_experiment_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Experiment metadata saved: {metadata_file}")

    # Print final summary
    print(f"\n{'='*100}")
    print("ABLATION EXPERIMENT SUMMARY")
    print(f"{'='*100}")
    print(f"Experiment type: {args.script.upper()} ablation (no learned projection)")
    print(f"Total runs per method: {args.runs}")
    print(f"Seeds used: {seeds}")
    print(f"Results directory: {run_dir}")

    for script_name in all_results.keys():
        result = all_results[script_name]
        print(f"\n{script_name.upper()} Results:")
        print(f"  Method: {result['method_name']}")
        print(f"  Successful runs: {result['successful_runs']}/{args.runs}")
        print(f"  Performance metrics: {result['performance_csv']}")
        print(f"  Classification metrics: {result['classification_csv']}")

        # Show top 5 layers for performance
        combined_perf = result['performance_df'][result['performance_df']['Dataset'] == 'COMBINED'].sort_values('Performance_Rank').head(5)
        if not combined_perf.empty:
            print(f"  Top 5 Performance Layers (by Accuracy):")
            for _, row in combined_perf.iterrows():
                print(f"    Layer {int(row['Layer'])}: Accuracy={row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f}")

if __name__ == "__main__":
    main()
