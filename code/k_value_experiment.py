#!/usr/bin/env python3
"""
K-Value Experiment Runner for Balanced OOD KCD

This script runs the balanced_ood_kcd.py with different k values to find the optimal k
for the KCD detector. It automatically modifies the k value, runs the experiment,
and collects results for comparison.

Usage:
    python k_value_experiment.py --k_values 10,25,50,75,100 --output_dir results/k_experiments
    python k_value_experiment.py --k_range 10,100,10 --output_dir results/k_experiments
"""

import argparse
import os
import sys
import subprocess
import json
import csv
import time
from datetime import datetime
import shutil

import re


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run k-value experiments for KCD detector")
    
    # K value specification (mutually exclusive)
    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument('--k_values', type=str,
                        help='Comma-separated list of k values to test (e.g., "10,25,50,75,100")')
    k_group.add_argument('--k_range', type=str,
                        help='K value range as "start,end,step" (e.g., "10,100,10")')

    # Multiple runs for variance reduction
    parser.add_argument('--num_seeds', type=int, default=1,
                       help='Number of different random seeds to train projector with (default: 1)')
    parser.add_argument('--seeds', type=str, default=None,
                       help='Comma-separated list of specific seeds to use (e.g., "42,123,456")')

    # Output and experiment settings
    parser.add_argument('--output_dir', type=str, default='results/k_experiments',
                       help='Directory to store experiment results')
    parser.add_argument('--script_path', type=str, default='code/balanced_ood_kcd.py',
                       help='Path to the balanced_ood_kcd.py script')
    parser.add_argument('--max_workers', type=int, default=1,
                       help='Number of parallel experiments (default: 1 for GPU memory)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be run without actually running experiments')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results (skip completed k values)')
    
    return parser.parse_args()


def get_k_values(args):
    """Get list of k values to test"""
    if args.k_values:
        return [int(k.strip()) for k in args.k_values.split(',')]
    elif args.k_range:
        start, end, step = map(int, args.k_range.split(','))
        return list(range(start, end + 1, step))
    else:
        raise ValueError("Must specify either --k_values or --k_range")


def get_seeds(args):
    """Get list of random seeds to use"""
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(',')]
    else:
        # Generate different seeds based on num_seeds
        base_seed = 42
        return [base_seed + i * 100 for i in range(args.num_seeds)]


def backup_and_modify_script(script_path, new_k_value, new_seed=None):
    """
    Backup original script and modify k value and optionally random seed in place

    Args:
        script_path: Path to script to modify
        new_k_value: New k value to use
        new_seed: New random seed to use (optional)

    Returns:
        backup_path: Path to backup file
    """
    # Create backup
    suffix = f"k{new_k_value}"
    if new_seed is not None:
        suffix += f"_seed{new_seed}"
    backup_path = f"{script_path}.backup_{suffix}"
    shutil.copy2(script_path, backup_path)

    # Read original content
    with open(script_path, 'r') as f:
        content = f.read()

    # Find and replace the k value in KCDDetector initialization
    # First check if the current k value is already the target value
    current_k_match = re.search(r'detector = KCDDetector\(k=(\d+)', content)
    if current_k_match:
        current_k = int(current_k_match.group(1))
        if current_k == new_k_value:
            print(f"  K value is already {new_k_value}, no modification needed")
            # Still write the file to create the backup
            pass
        else:
            print(f"  Changing k from {current_k} to {new_k_value}")

    # Use a simple, direct pattern that we know works
    pattern = r'(detector = KCDDetector\(k=)\d+'
    replacement = f'\\g<1>{new_k_value}'

    # Apply the replacement
    modified_content = re.sub(pattern, replacement, content)

    # Verify the replacement worked (or was already correct)
    if f'k={new_k_value}' not in modified_content:
        # Show some context around potential matches for debugging
        lines = content.split('\n')
        kcd_lines = [f"Line {i+1}: {line}" for i, line in enumerate(lines) if 'KCDDetector' in line]
        context = '\n'.join(kcd_lines) if kcd_lines else "No KCDDetector lines found"
        raise ValueError(f"Could not find KCDDetector k parameter to modify in {script_path}.\nFound KCDDetector lines:\n{context}")

    # Modify random seed if specified
    if new_seed is not None:
        # Find and replace MAIN_SEED = 42
        seed_pattern = r'(MAIN_SEED = )\d+'
        seed_replacement = f'\\g<1>{new_seed}'
        modified_content = re.sub(seed_pattern, seed_replacement, modified_content)

        if f'MAIN_SEED = {new_seed}' not in modified_content:
            print(f"  Warning: Could not modify MAIN_SEED to {new_seed}")
        else:
            print(f"  Changed random seed to {new_seed}")

    # Write modified script
    with open(script_path, 'w') as f:
        f.write(modified_content)

    seed_info = f", seed={new_seed}" if new_seed is not None else ""
    print(f"  Modified script to use k={new_k_value}{seed_info} (backup: {backup_path})")
    return backup_path


def restore_script(script_path, backup_path):
    """Restore original script from backup"""
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, script_path)
        os.remove(backup_path)
        print(f"  Restored original script from backup")


def cleanup_gpu_memory():
    """Try to clean up GPU memory between experiments"""
    try:
        # Try to import torch and clean up GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  🧹 GPU memory cleaned up")
    except ImportError:
        pass  # torch not available
    except Exception as e:
        print(f"  ⚠ GPU cleanup failed: {e}")


def run_experiment(k_value, script_path, output_dir, seed=None, dry_run=False):
    """
    Run experiment with specific k value and optional seed

    Args:
        k_value: K value to test
        script_path: Path to the balanced_ood_kcd.py script
        output_dir: Directory to store results
        seed: Random seed to use (optional)
        dry_run: If True, just print what would be run

    Returns:
        dict: Experiment results or None if failed
    """
    seed_info = f", seed={seed}" if seed is not None else ""
    print(f"\n{'='*60}")
    print(f"Running experiment with k={k_value}{seed_info}")
    print(f"{'='*60}")

    # Create experiment-specific output directory
    exp_suffix = f"k_{k_value}"
    if seed is not None:
        exp_suffix += f"_seed_{seed}"
    exp_dir = os.path.join(output_dir, exp_suffix)
    os.makedirs(exp_dir, exist_ok=True)

    if dry_run:
        modify_info = f"k={k_value}"
        if seed is not None:
            modify_info += f", seed={seed}"
        print(f"[DRY RUN] Would modify: {script_path} to use {modify_info}")
        print(f"[DRY RUN] Would run: python {script_path}")
        print(f"[DRY RUN] Results would be saved to: {exp_dir}")
        return None

    backup_path = None
    try:
        # Backup and modify script with new k value and seed
        backup_path = backup_and_modify_script(script_path, k_value, seed)

        # Run the experiment
        start_time = time.time()
        print(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run the original script (now modified) from the current working directory
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=7200)  # 2 hour timeout

        end_time = time.time()
        duration = end_time - start_time

        # Add a small delay between experiments to help with GPU memory cleanup
        time.sleep(2)
        
        # Save stdout and stderr
        with open(os.path.join(exp_dir, 'stdout.log'), 'w') as f:
            f.write(result.stdout)
        with open(os.path.join(exp_dir, 'stderr.log'), 'w') as f:
            f.write(result.stderr)
        
        # Check if experiment succeeded
        if result.returncode == 0:
            print(f"✓ Experiment completed successfully in {duration:.1f} seconds")
            
            # Try to find and copy the results CSV file
            results_csv = "results/balanced_kcd_results.csv"
            if os.path.exists(results_csv):
                exp_results_csv = os.path.join(exp_dir, "balanced_kcd_results.csv")
                shutil.copy2(results_csv, exp_results_csv)
                print(f"✓ Results saved to: {exp_results_csv}")
                
                # Parse results for summary
                return parse_results_csv(exp_results_csv, k_value, duration, seed)
            else:
                print(f"⚠ Warning: Results CSV not found at {results_csv}")
                result = {"k_value": k_value, "duration": duration, "status": "completed_no_csv"}
                if seed is not None:
                    result["seed"] = seed
                return result
        else:
            # Provide more specific error information
            error_msg = f"Return code {result.returncode}"
            if result.returncode == -11:
                error_msg += " (SIGSEGV - Segmentation fault, likely GPU memory issue)"
            elif result.returncode == -9:
                error_msg += " (SIGKILL - Process killed, likely out of memory)"
            elif result.returncode == -15:
                error_msg += " (SIGTERM - Process terminated)"

            print(f"✗ Experiment failed with {error_msg}")
            print(f"Error output: {result.stderr[:500]}...")

            # Suggest solutions for common issues
            if result.returncode == -11:
                print("  💡 Suggestion: This might be a GPU memory issue. Try:")
                print("     - Reducing batch size in the script")
                print("     - Adding more delay between experiments")
                print("     - Restarting if this happens frequently")

            result_dict = {"k_value": k_value, "duration": duration, "status": "failed", "error": result.stderr, "return_code": result.returncode}
            if seed is not None:
                result_dict["seed"] = seed
            return result_dict

    except subprocess.TimeoutExpired:
        print(f"✗ Experiment timed out after 2 hours")
        result = {"k_value": k_value, "status": "timeout"}
        if seed is not None:
            result["seed"] = seed
        return result
    except Exception as e:
        print(f"✗ Experiment failed with exception: {e}")
        result = {"k_value": k_value, "status": "error", "error": str(e)}
        if seed is not None:
            result["seed"] = seed
        return result
    finally:
        # Restore original script
        if backup_path:
            restore_script(script_path, backup_path)

        # Clean up GPU memory after each experiment
        cleanup_gpu_memory()


def parse_results_csv(csv_path, k_value, duration, seed=None):
    """Parse results CSV and extract key metrics"""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            result = {"k_value": k_value, "duration": duration, "status": "empty_results"}
            if seed is not None:
                result["seed"] = seed
            return result
        
        # Calculate average metrics across all layers and datasets
        accuracies = []
        aurocs = []
        auprcs = []
        f1_scores = []
        
        for row in rows:
            try:
                if row['Accuracy'] != 'nan' and row['Accuracy']:
                    accuracies.append(float(row['Accuracy']))
                if row['AUROC'] != 'nan' and row['AUROC']:
                    aurocs.append(float(row['AUROC']))
                if row['AUPRC'] != 'nan' and row['AUPRC']:
                    auprcs.append(float(row['AUPRC']))
                if row['F1'] != 'nan' and row['F1']:
                    f1_scores.append(float(row['F1']))
            except (ValueError, KeyError):
                continue
        
        result = {
            "k_value": k_value,
            "duration": round(duration, 2),
            "status": "success",
            "num_results": len(rows),
            "avg_accuracy": round(sum(accuracies) / len(accuracies), 4) if accuracies else 0,
            "avg_auroc": round(sum(aurocs) / len(aurocs), 4) if aurocs else 0,
            "avg_auprc": round(sum(auprcs) / len(auprcs), 4) if auprcs else 0,
            "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
            "max_accuracy": round(max(accuracies), 4) if accuracies else 0,
            "max_auroc": round(max(aurocs), 4) if aurocs else 0,
            "max_auprc": round(max(auprcs), 4) if auprcs else 0,
            "max_f1": round(max(f1_scores), 4) if f1_scores else 0
        }
        if seed is not None:
            result["seed"] = seed
        return result
    except Exception as e:
        result = {"k_value": k_value, "duration": duration, "status": "parse_error", "error": str(e)}
        if seed is not None:
            result["seed"] = seed
        return result


def save_experiment_summary(results, output_dir):
    """Save experiment summary to JSON and CSV"""
    # Save detailed JSON
    json_path = os.path.join(output_dir, "experiment_summary.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to: {json_path}")

    # Save summary CSV
    csv_path = os.path.join(output_dir, "k_value_comparison.csv")
    with open(csv_path, 'w', newline='') as f:
        if results:
            # Get all possible fieldnames from all results
            all_fieldnames = set()
            for result in results:
                all_fieldnames.update(result.keys())

            writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
            writer.writeheader()
            writer.writerows(results)
    print(f"✓ Summary CSV saved to: {csv_path}")


def aggregate_results_by_k(results):
    """Aggregate results across multiple seeds for each k value"""
    import numpy as np

    # Group results by k value
    k_groups = {}
    for result in results:
        if result.get('status') != 'success':
            continue
        k_value = result['k_value']
        if k_value not in k_groups:
            k_groups[k_value] = []
        k_groups[k_value].append(result)

    # Aggregate metrics for each k value
    aggregated = []
    for k_value, k_results in k_groups.items():
        if not k_results:
            continue

        # Extract metrics from all seeds for this k value
        metrics = ['avg_accuracy', 'avg_auroc', 'avg_auprc', 'avg_f1', 'max_accuracy', 'max_auroc', 'max_auprc', 'max_f1']
        aggregated_result = {
            'k_value': k_value,
            'num_seeds': len(k_results),
            'status': 'aggregated'
        }

        for metric in metrics:
            values = [r.get(metric, 0) for r in k_results if metric in r]
            if values:
                aggregated_result[f'{metric}_mean'] = round(np.mean(values), 4)
                aggregated_result[f'{metric}_std'] = round(np.std(values), 4)
                aggregated_result[f'{metric}_min'] = round(np.min(values), 4)
                aggregated_result[f'{metric}_max'] = round(np.max(values), 4)

        # Calculate combined score (you can adjust the weights)
        if 'avg_accuracy_mean' in aggregated_result and 'avg_auroc_mean' in aggregated_result:
            combined_score = 0.6 * aggregated_result['avg_accuracy_mean'] + 0.4 * aggregated_result['avg_auroc_mean']
            aggregated_result['combined_score_mean'] = round(combined_score, 4)

            # Combined score std (approximate)
            acc_std = aggregated_result.get('avg_accuracy_std', 0)
            auroc_std = aggregated_result.get('avg_auroc_std', 0)
            combined_std = np.sqrt((0.6 * acc_std)**2 + (0.4 * auroc_std)**2)
            aggregated_result['combined_score_std'] = round(combined_std, 4)

        aggregated.append(aggregated_result)

    return aggregated


def print_results_summary(results, aggregated_results=None):
    """Print a nice summary of results"""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    if aggregated_results:
        print("AGGREGATED RESULTS ACROSS MULTIPLE SEEDS:")
        print(f"{'K Value':<8} {'Seeds':<6} {'Avg Acc (μ±σ)':<15} {'Avg AUROC (μ±σ)':<16} {'Combined (μ±σ)':<15}")
        print("-" * 80)

        # Sort by combined score mean
        aggregated_results.sort(key=lambda x: x.get('combined_score_mean', 0), reverse=True)

        for result in aggregated_results:
            acc_mean = result.get('avg_accuracy_mean', 0)
            acc_std = result.get('avg_accuracy_std', 0)
            auroc_mean = result.get('avg_auroc_mean', 0)
            auroc_std = result.get('avg_auroc_std', 0)
            combined_mean = result.get('combined_score_mean', 0)
            combined_std = result.get('combined_score_std', 0)

            print(f"{result['k_value']:<8} "
                  f"{result['num_seeds']:<6} "
                  f"{acc_mean:.3f}±{acc_std:.3f}    "
                  f"{auroc_mean:.3f}±{auroc_std:.3f}     "
                  f"{combined_mean:.3f}±{combined_std:.3f}")

        # Highlight best performing k value
        if aggregated_results:
            best_result = aggregated_results[0]
            print(f"\n🏆 Best performing k value: {best_result['k_value']} "
                  f"(Combined Score: {best_result.get('combined_score_mean', 0):.4f}±{best_result.get('combined_score_std', 0):.4f})")
    else:
        # Original single-seed summary
        successful_results = [r for r in results if r.get('status') == 'success']

        if not successful_results:
            print("No successful experiments to summarize.")
            return

        # Sort by average accuracy (or another metric)
        successful_results.sort(key=lambda x: x.get('avg_accuracy', 0), reverse=True)

        print(f"{'K Value':<8} {'Avg Acc':<8} {'Max Acc':<8} {'Avg AUROC':<10} {'Max AUROC':<10} {'Duration':<10}")
        print("-" * 70)

        for result in successful_results:
            seed_info = f" (seed {result['seed']})" if 'seed' in result else ""
            print(f"{result['k_value']:<8} "
                  f"{result.get('avg_accuracy', 0):.4f}   "
                  f"{result.get('max_accuracy', 0):.4f}   "
                  f"{result.get('avg_auroc', 0):.4f}     "
                  f"{result.get('max_auroc', 0):.4f}     "
                  f"{result.get('duration', 0):.1f}s{seed_info}")

        # Highlight best performing k value
        best_result = successful_results[0]
        print(f"\n🏆 Best performing k value: {best_result['k_value']} "
              f"(Avg Accuracy: {best_result.get('avg_accuracy', 0):.4f})")


def main():
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.script_path):
        print(f"Error: Script not found at {args.script_path}")
        sys.exit(1)
    
    # Get k values and seeds to test
    k_values = get_k_values(args)
    seeds = get_seeds(args)
    print(f"Testing k values: {k_values}")
    print(f"Using seeds: {seeds}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for resume
    completed_experiments = set()
    if args.resume:
        for k in k_values:
            for seed in seeds:
                exp_suffix = f"k_{k}_seed_{seed}"
                exp_dir = os.path.join(args.output_dir, exp_suffix)
                results_file = os.path.join(exp_dir, "balanced_kcd_results.csv")
                if os.path.exists(results_file):
                    completed_experiments.add((k, seed))
                    print(f"Found existing results for k={k}, seed={seed}, will skip")

    # Generate all k,seed combinations to run
    experiments_to_run = []
    for k in k_values:
        for seed in seeds:
            if (k, seed) not in completed_experiments:
                experiments_to_run.append((k, seed))

    if not experiments_to_run:
        print("All experiments already completed!")
        # Still load existing results for summary
        experiments_to_run = [(k, seed) for k in k_values for seed in seeds]
    else:
        print(f"Will run {len(experiments_to_run)} experiments:")
        for k, seed in experiments_to_run[:5]:  # Show first 5
            print(f"  k={k}, seed={seed}")
        if len(experiments_to_run) > 5:
            print(f"  ... and {len(experiments_to_run) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN MODE] - No experiments will actually be run")

    # Run experiments
    all_results = []

    for i, (k_value, seed) in enumerate(experiments_to_run):
        if (k_value, seed) not in completed_experiments or not args.resume:
            print(f"\nProgress: {i+1}/{len(experiments_to_run)}")
            result = run_experiment(k_value, args.script_path, args.output_dir, seed, args.dry_run)
            if result:
                all_results.append(result)
    
    # Load existing results if resuming
    if args.resume and completed_experiments:
        for k, seed in completed_experiments:
            exp_suffix = f"k_{k}_seed_{seed}"
            exp_dir = os.path.join(args.output_dir, exp_suffix)
            results_file = os.path.join(exp_dir, "balanced_kcd_results.csv")
            if os.path.exists(results_file):
                # Estimate duration as 0 for existing results
                existing_result = parse_results_csv(results_file, k, 0, seed)
                all_results.append(existing_result)

    if not args.dry_run and all_results:
        # Save individual results summary
        save_experiment_summary(all_results, args.output_dir)

        # Aggregate results by k value if multiple seeds
        aggregated_results = None
        if len(seeds) > 1:
            aggregated_results = aggregate_results_by_k(all_results)

            # Save aggregated results
            aggregated_json_path = os.path.join(args.output_dir, "aggregated_results.json")
            with open(aggregated_json_path, 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            print(f"✓ Aggregated results saved to: {aggregated_json_path}")

            # Save aggregated CSV
            aggregated_csv_path = os.path.join(args.output_dir, "aggregated_k_comparison.csv")
            with open(aggregated_csv_path, 'w', newline='') as f:
                if aggregated_results:
                    all_fieldnames = set()
                    for result in aggregated_results:
                        all_fieldnames.update(result.keys())

                    writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
                    writer.writeheader()
                    writer.writerows(aggregated_results)
            print(f"✓ Aggregated CSV saved to: {aggregated_csv_path}")

        # Print summary
        print_results_summary(all_results, aggregated_results)

        print(f"\n✓ All experiments completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
