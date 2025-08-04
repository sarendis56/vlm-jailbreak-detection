#!/usr/bin/env python3
"""
Multi-seed experiment runner for balanced OOD detection
Runs experiments multiple times with different seeds and computes averaged results
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
    print(f"RUNNING EXPERIMENT {run_id}/{total_runs} WITH SEED {seed}")
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

    # Group and aggregate - include Method for ML experiments to preserve individual methods
    aggregated_results = []

    if 'ML_' in method_name:
        # For ML experiments, group by Layer, Dataset, and Method to preserve individual ML methods
        grouped = all_results.groupby(['Layer', 'Dataset', 'Method'])

        for (layer, dataset, method), group in grouped:
            result_row = {
                'Layer': layer,
                'Dataset': dataset,
                'Method': method,  # Use the actual method name (ML_LOGISTIC, ML_RIDGE, etc.)
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
    else:
        # For MCD/KCD experiments, group by Layer and Dataset (original behavior)
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
        # For ML experiments, we need to use both Layer and Method for mapping
        if 'ML_' in combined_df['Method'].iloc[0] if len(combined_df) > 0 else False:
            # ML experiments: create mapping using (Layer, Method) tuple
            rank_mapping = dict(zip(zip(combined_df['Layer'], combined_df['Method']), combined_df[f'{group_name}_Rank']))
            df[f'{group_name}_Rank'] = df.apply(lambda row: rank_mapping.get((row['Layer'], row['Method']), 0), axis=1)
        else:
            # MCD/KCD experiments: use Layer only (original behavior)
            rank_mapping = dict(zip(combined_df['Layer'], combined_df[f'{group_name}_Rank']))
            df[f'{group_name}_Rank'] = df['Layer'].map(rank_mapping)
            df[f'{group_name}_Rank'] = df[f'{group_name}_Rank'].fillna(0)  # Fill NaN with 0 for non-COMBINED
    else:
        df[f'{group_name}_Rank'] = 0

    return df

def aggregate_results_ml_learned(results_list, method_name):
    """Special aggregation for learned-ml method with separate MLP/SVM results"""
    if not results_list:
        return None, None, None, None

    # Define metric groups
    group1_metrics = ['Accuracy', 'AUROC', 'AUPRC']  # Performance metrics
    group2_metrics = ['TPR', 'FPR', 'F1']            # Classification metrics

    # Separate MLP and SVM results
    mlp_results = []
    svm_results = []

    for df in results_list:
        mlp_df = df[df['Method'] == 'MLP'].copy()
        svm_df = df[df['Method'] == 'SVM'].copy()

        if not mlp_df.empty:
            # Rename ranking columns for MLP
            if 'MLP_Combined_Rank' in mlp_df.columns:
                mlp_df['Combined_Rank'] = mlp_df['MLP_Combined_Rank']
            if 'MLP_Individual_Rank' in mlp_df.columns:
                mlp_df['Individual_Rank'] = mlp_df['MLP_Individual_Rank']
            mlp_results.append(mlp_df)

        if not svm_df.empty:
            # Rename ranking columns for SVM
            if 'SVM_Combined_Rank' in svm_df.columns:
                svm_df['Combined_Rank'] = svm_df['SVM_Combined_Rank']
            if 'SVM_Individual_Rank' in svm_df.columns:
                svm_df['Individual_Rank'] = svm_df['SVM_Individual_Rank']
            svm_results.append(svm_df)

    # Create aggregated results for each method and group
    mlp_group1_df = aggregate_results_group(mlp_results, f"{method_name}_MLP", group1_metrics, "Performance")
    mlp_group2_df = aggregate_results_group(mlp_results, f"{method_name}_MLP", group2_metrics, "Classification")
    svm_group1_df = aggregate_results_group(svm_results, f"{method_name}_SVM", group1_metrics, "Performance")
    svm_group2_df = aggregate_results_group(svm_results, f"{method_name}_SVM", group2_metrics, "Classification")

    return mlp_group1_df, mlp_group2_df, svm_group1_df, svm_group2_df


def aggregate_results_pca_ml(results_list, method_name):
    """
    Special aggregation for PCA-ML method with PCA_Components dimension

    PCA-ML CSV format:
    Layer, PCA_Components, Dataset, Method, Accuracy, F1, TPR, FPR,
    AUROC, AUPRC, Explained_Variance, Combined_Score, Rank

    This function groups by (Layer, PCA_Components, Dataset) instead of just (Layer, Dataset)
    """
    if not results_list:
        return None, None

    # Define metric groups
    group1_metrics = ['Accuracy', 'AUROC', 'AUPRC']  # Performance metrics
    group2_metrics = ['TPR', 'FPR', 'F1']            # Classification metrics

    # Create aggregated results for each group
    group1_df = aggregate_results_group_pca(results_list, method_name, group1_metrics, "Performance")
    group2_df = aggregate_results_group_pca(results_list, method_name, group2_metrics, "Classification")

    return group1_df, group2_df


def aggregate_results_group_pca(results_list, method_name, metric_group, group_name):
    """Aggregate results for PCA-ML with PCA_Components grouping"""
    if not results_list:
        return None

    # Combine all dataframes
    all_results = pd.concat(results_list, ignore_index=True)

    # Filter out non-numeric values (like "N/A")
    for col in metric_group:
        if col in all_results.columns:
            all_results[col] = pd.to_numeric(all_results[col], errors='coerce')

    # Group by Layer, PCA_Components, Dataset, AND Method for PCA-ML experiments
    # This ensures MLP and SVM results are kept separate
    aggregated_results = []
    grouped = all_results.groupby(['Layer', 'PCA_Components', 'Dataset', 'Method'])

    for (layer, pca_components, dataset, method), group in grouped:
        result_row = {
            'Layer': layer,
            'PCA_Components': pca_components,
            'Dataset': dataset,
            'Method': method,  # Use the actual method (MLP or SVM)
            'Group': group_name,
            'Runs': len(group)
        }

        # Add explained variance if available (take mean across runs)
        if 'Explained_Variance' in group.columns:
            explained_var_values = pd.to_numeric(group['Explained_Variance'], errors='coerce').dropna()
            if len(explained_var_values) > 0:
                result_row['Explained_Variance_Mean'] = round(explained_var_values.mean(), 4)
            else:
                result_row['Explained_Variance_Mean'] = 0.0000

        # Add combined score if available (take mean across runs)
        if 'Combined_Score' in group.columns:
            combined_score_values = pd.to_numeric(group['Combined_Score'], errors='coerce').dropna()
            if len(combined_score_values) > 0:
                result_row['Combined_Score_Mean'] = round(combined_score_values.mean(), 4)
                result_row['Combined_Score_Std'] = round(combined_score_values.std() if len(combined_score_values) > 1 else 0.0, 4)
            else:
                result_row['Combined_Score_Mean'] = 0.0000
                result_row['Combined_Score_Std'] = 0.0000

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

    # Add rankings based on COMBINED results only
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

        # Create mapping using (Layer, PCA_Components, Method) tuple for PCA-ML
        # This ensures rankings are separate for MLP and SVM methods
        rank_mapping = dict(zip(zip(combined_df['Layer'], combined_df['PCA_Components'], combined_df['Method']), combined_df[f'{group_name}_Rank']))
        df[f'{group_name}_Rank'] = df.apply(lambda row: rank_mapping.get((row['Layer'], row['PCA_Components'], row['Method']), 0), axis=1)
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
    parser = argparse.ArgumentParser(description='Run multiple experiments with different seeds')
    parser.add_argument('--script', required=True, choices=['mcd', 'kcd', 'ml', 'learned-ml', 'pca-ml'],
                       help='Which script to run (mcd, kcd, ml, learned-ml, or pca-ml)')
    parser.add_argument('--runs', type=int, default=20,
                       help='Number of runs (default: 20)')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='Custom seeds to use (if not provided, will use consecutive seeds starting from 42)')
    parser.add_argument('--output-dir', default='multi_run_results',
                       help='Output directory for results (default: multi_run_results)')

    # Add usage examples in help
    parser.epilog = """
Examples:
  python run_multiple_experiments.py --script pca-ml --runs 10
  python run_multiple_experiments.py --script pca-ml --runs 5 --seeds 42 43 44 45 46
  python run_multiple_experiments.py --script learned-ml --runs 20
  python run_multiple_experiments.py --script mcd --runs 30 --output-dir my_results
"""
    
    args = parser.parse_args()
    
    # Determine script path and working directory
    script_dir = Path(__file__).parent  # This is the 'code' directory
    working_dir = script_dir.parent      # This is the parent directory (where experiments should run)

    if args.script == 'mcd':
        script_path = script_dir / 'balanced_ood_mcd.py'
        expected_csv = 'results/balanced_mcd_results.csv'
        method_name = 'MCD_k_MultiSeed'
    elif args.script == 'kcd':
        script_path = script_dir / 'balanced_ood_kcd.py'
        expected_csv = 'results/balanced_kcd_results.csv'
        method_name = 'KCD_MultiSeed'
    elif args.script == 'ml':
        script_path = script_dir / 'balanced_ml.py'
        expected_csv = 'results/balanced_ml_results.csv'
        method_name = 'ML_MultiSeed'
    elif args.script == 'learned-ml':
        script_path = script_dir / 'balanced_ood_ml.py'
        expected_csv = 'results/balanced_ood_ml_results.csv'
        method_name = 'ML_Learned_MultiSeed'
    else:  # pca-ml
        script_path = script_dir / 'balanced_pca_ml.py'
        expected_csv = 'results/balanced_pca_ml_results.csv'
        method_name = 'PCA_ML_MultiSeed'

    if not script_path.exists():
        print(f"Error: Script {script_path} not found!")
        return

    print(f"Script directory: {script_dir}")
    print(f"Working directory: {working_dir}")
    print(f"Script path: {script_path}")
    
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
    run_dir = output_dir / f"{args.script}_{args.runs}runs_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    print(f"Running {args.runs} experiments with {args.script.upper()} method")
    print(f"Seeds: {seeds}")
    print(f"Results will be saved to: {run_dir}")
    
    # Store individual run results
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
                    print(f"ERROR: Could not find CSV file for run {i}")
                    continue
            # Create destination filename (just the base filename, not the full path)
            csv_filename = expected_csv.split('/')[-1]  # Get just the filename
            csv_dest = run_dir / f"run_{i:02d}_seed_{seed}_{csv_filename}"

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
                    print(f"Results processed successfully for run {i}")
                else:
                    print(f"Warning: Could not load CSV data for run {i}")
            except Exception as e:
                print(f"Error moving results file for run {i}: {e}")
        
        print(f"Completed {i}/{args.runs} runs ({successful_runs} successful)")
    
    if successful_runs == 0:
        print("ERROR: No successful runs completed!")
        return
    
    print(f"\n{'='*80}")
    print(f"AGGREGATING RESULTS FROM {successful_runs} SUCCESSFUL RUNS")
    print(f"{'='*80}")
    
    # Aggregate results - special handling for learned-ml method
    if args.script == 'learned-ml':
        mlp_group1_df, mlp_group2_df, svm_group1_df, svm_group2_df = aggregate_results_ml_learned(individual_results, method_name)

        if mlp_group1_df is not None and svm_group1_df is not None:
            # Save aggregated results for both methods and groups
            mlp_perf_csv = run_dir / f"aggregated_{args.script}_mlp_performance.csv"
            mlp_class_csv = run_dir / f"aggregated_{args.script}_mlp_classification.csv"
            svm_perf_csv = run_dir / f"aggregated_{args.script}_svm_performance.csv"
            svm_class_csv = run_dir / f"aggregated_{args.script}_svm_classification.csv"

            mlp_group1_df.to_csv(mlp_perf_csv, index=False)
            mlp_group2_df.to_csv(mlp_class_csv, index=False)
            svm_group1_df.to_csv(svm_perf_csv, index=False)
            svm_group2_df.to_csv(svm_class_csv, index=False)

            print(f"MLP Performance metrics saved: {mlp_perf_csv}")
            print(f"MLP Classification metrics saved: {mlp_class_csv}")
            print(f"SVM Performance metrics saved: {svm_perf_csv}")
            print(f"SVM Classification metrics saved: {svm_class_csv}")

            # Save experiment metadata
            metadata = {
                'script': args.script,
                'total_runs': args.runs,
                'successful_runs': successful_runs,
                'seeds': seeds,
                'timestamp': timestamp,
                'method': method_name,
                'output_files': {
                    'mlp_performance_metrics': str(mlp_perf_csv.name),
                    'mlp_classification_metrics': str(mlp_class_csv.name),
                    'svm_performance_metrics': str(svm_perf_csv.name),
                    'svm_classification_metrics': str(svm_class_csv.name)
                }
            }

        metadata_file = run_dir / 'experiment_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Experiment metadata saved: {metadata_file}")

        # Print summary statistics
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY - ML LEARNED FEATURES")
        print(f"{'='*80}")
        print(f"Method: {method_name}")
        print(f"Successful runs: {successful_runs}/{args.runs}")
        print(f"Seeds used: {seeds[:successful_runs]}")
        print(f"Results directory: {run_dir}")
        print(f"MLP Performance metrics: {mlp_perf_csv}")
        print(f"MLP Classification metrics: {mlp_class_csv}")
        print(f"SVM Performance metrics: {svm_perf_csv}")
        print(f"SVM Classification metrics: {svm_class_csv}")

        # Show top 5 layers for each method
        mlp_combined_perf = mlp_group1_df[mlp_group1_df['Dataset'] == 'COMBINED'].sort_values('Performance_Rank').head(5)
        svm_combined_perf = svm_group1_df[svm_group1_df['Dataset'] == 'COMBINED'].sort_values('Performance_Rank').head(5)

        if not mlp_combined_perf.empty:
            print(f"\nTop 5 MLP Performance Layers (by Accuracy):")
            for _, row in mlp_combined_perf.iterrows():
                print(f"  Layer {int(row['Layer'])}: Accuracy={row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f}")

        if not svm_combined_perf.empty:
            print(f"\nTop 5 SVM Performance Layers (by Accuracy):")
            for _, row in svm_combined_perf.iterrows():
                print(f"  Layer {int(row['Layer'])}: Accuracy={row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f}")
        else:
            print("ERROR: Failed to aggregate learned-ml results!")
    else:
        # Regular aggregation for other methods, with special handling for PCA-ML
        if args.script == 'pca-ml':
            group1_df, group2_df = aggregate_results_pca_ml(individual_results, method_name)
        else:
            group1_df, group2_df = aggregate_results(individual_results, method_name)

        if group1_df is not None and group2_df is not None:
            # Save aggregated results for both groups
            group1_csv = run_dir / f"aggregated_{args.script}_performance.csv"
            group2_csv = run_dir / f"aggregated_{args.script}_classification.csv"

            group1_df.to_csv(group1_csv, index=False)
            group2_df.to_csv(group2_csv, index=False)

            print(f"Performance metrics saved: {group1_csv}")
            print(f"Classification metrics saved: {group2_csv}")

            # Save experiment metadata
            metadata = {
                'script': args.script,
                'total_runs': args.runs,
                'successful_runs': successful_runs,
                'seeds': seeds,
                'timestamp': timestamp,
                'method': method_name,
                'output_files': {
                    'performance_metrics': str(group1_csv.name),
                    'classification_metrics': str(group2_csv.name)
                }
            }

            metadata_file = run_dir / 'experiment_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Experiment metadata saved: {metadata_file}")

            # Print summary statistics
            print(f"\n{'='*80}")
            if args.script == 'pca-ml':
                print("EXPERIMENT SUMMARY - PCA + SVM")
            else:
                print("EXPERIMENT SUMMARY")
            print(f"{'='*80}")
            print(f"Method: {method_name}")
            print(f"Successful runs: {successful_runs}/{args.runs}")
            print(f"Seeds used: {seeds[:successful_runs]}")
            print(f"Results directory: {run_dir}")
            print(f"Performance metrics (Accuracy, AUROC, AUPRC): {group1_csv}")
            print(f"Classification metrics (TPR, FPR, F1): {group2_csv}")
            if args.script == 'pca-ml':
                print("Note: Results include PCA_Components, Explained_Variance, and Combined_Score columns")

            # Show top 5 layers for each group
            combined_perf = group1_df[group1_df['Dataset'] == 'COMBINED'].sort_values('Performance_Rank').head(5)
            combined_class = group2_df[group2_df['Dataset'] == 'COMBINED'].sort_values('Classification_Rank').head(5)

            if not combined_perf.empty:
                if args.script == 'pca-ml':
                    # Show results separately for MLP and SVM
                    for method in ['MLP', 'SVM']:
                        method_perf = combined_perf[combined_perf['Method'] == method].head(5)
                        if not method_perf.empty:
                            print(f"\nTop 5 {method} Performance Configurations (by Accuracy):")
                            for _, row in method_perf.iterrows():
                                pca_info = f"PCA-{int(row['PCA_Components'])}" if 'PCA_Components' in row else ""
                                explained_var = f"ExplVar={row['Explained_Variance_Mean']:.3f}" if 'Explained_Variance_Mean' in row else ""
                                combined_score = f"Score={row['Combined_Score_Mean']:.4f}" if 'Combined_Score_Mean' in row else ""
                                print(f"  Layer {int(row['Layer'])} {pca_info}: Accuracy={row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f} ({explained_var}, {combined_score})")
                else:
                    print(f"\nTop 5 Performance Layers (by Accuracy):")
                    for _, row in combined_perf.iterrows():
                        print(f"  Layer {int(row['Layer'])}: Accuracy={row['Accuracy_Mean']:.4f}±{row['Accuracy_Std']:.4f}")

            if not combined_class.empty:
                if args.script == 'pca-ml':
                    # Show results separately for MLP and SVM
                    for method in ['MLP', 'SVM']:
                        method_class = combined_class[combined_class['Method'] == method].head(5)
                        if not method_class.empty:
                            print(f"\nTop 5 {method} Classification Configurations (by F1):")
                            for _, row in method_class.iterrows():
                                pca_info = f"PCA-{int(row['PCA_Components'])}" if 'PCA_Components' in row else ""
                                explained_var = f"ExplVar={row['Explained_Variance_Mean']:.3f}" if 'Explained_Variance_Mean' in row else ""
                                combined_score = f"Score={row['Combined_Score_Mean']:.4f}" if 'Combined_Score_Mean' in row else ""
                                print(f"  Layer {int(row['Layer'])} {pca_info}: F1={row['F1_Mean']:.4f}±{row['F1_Std']:.4f} ({explained_var}, {combined_score})")
                else:
                    print(f"\nTop 5 Classification Layers (by F1):")
                    for _, row in combined_class.iterrows():
                        print(f"  Layer {int(row['Layer'])}: F1={row['F1_Mean']:.4f}±{row['F1_Std']:.4f}")
        else:
            print("ERROR: Failed to aggregate results!")

if __name__ == "__main__":
    main()
