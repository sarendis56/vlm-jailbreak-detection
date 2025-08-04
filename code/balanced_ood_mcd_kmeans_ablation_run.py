#!/usr/bin/env python3
"""
Extended K-means Ablation Study for OOD Jailbreak Detection

This script extends the k-means clustering approach to test different numbers of clusters
for both benign and malicious samples, comparing against the original dataset-based approach.

The goal is to show that:
1. K-means clustering is generally inferior to dataset-based clustering
2. Among k-means approaches, k=dataset_number works best
3. Provide comprehensive results for paper figures
"""

import csv
import numpy as np
import random
import warnings
import signal
import sys
import os
from scipy.linalg import inv
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import itertools

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

# Import from the original k-means script
from balanced_ood_mcd_kmeans import (
    ProjectionConfig, CONFIG, GPU_DEVICE, setup_gpu_environment, cleanup_gpu_memory,
    get_gpu_memory_info, LearnedProjection, train_learned_projection, apply_learned_projection,
    MCDDetector, analyze_dataset_composition, prepare_balanced_training, prepare_balanced_evaluation,
    prepare_ood_data_structure, signal_handler
)
from load_datasets import *
from feature_extractor import HiddenStateExtractor

def prepare_kmeans_data_structure_flexible(datasets_dict, hidden_states_dict, labels_dict, 
                                         k_benign=None, k_malicious=None, random_seed=42):
    """
    Flexible k-means data structure preparation with configurable cluster numbers.
    
    Args:
        datasets_dict: Dict of dataset names to samples
        hidden_states_dict: Dict of dataset names to features
        labels_dict: Dict of dataset names to labels
        k_benign: Number of clusters for benign samples
        k_malicious: Number of clusters for malicious samples
        random_seed: Random seed for k-means clustering
    
    Returns:
        in_dist_data: Dict of {cluster_name: list_of_features} for benign clusters
        ood_data: Dict of {cluster_name: list_of_features} for malicious clusters
        metadata: Dict with clustering information
    """
    # Aggregate all benign and malicious samples across datasets
    all_benign_features = []
    all_malicious_features = []
    
    # Count number of datasets that contribute to each class
    benign_dataset_count = 0
    malicious_dataset_count = 0
    
    for dataset_name in datasets_dict.keys():
        if dataset_name not in hidden_states_dict:
            continue
            
        features = hidden_states_dict[dataset_name]
        labels = labels_dict[dataset_name]
        
        # Separate benign and malicious samples
        benign_features = [features[i] for i, label in enumerate(labels) if label == 0]
        malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
        
        if benign_features:
            all_benign_features.extend(benign_features)
            benign_dataset_count += 1
        if malicious_features:
            all_malicious_features.extend(malicious_features)
            malicious_dataset_count += 1
    
    # Use provided k values or default to dataset counts
    if k_benign is None:
        k_benign = benign_dataset_count
    if k_malicious is None:
        k_malicious = malicious_dataset_count
    
    metadata = {
        'benign_samples': len(all_benign_features),
        'malicious_samples': len(all_malicious_features),
        'benign_dataset_count': benign_dataset_count,
        'malicious_dataset_count': malicious_dataset_count,
        'k_benign': k_benign,
        'k_malicious': k_malicious
    }
    
    in_dist_data = {}
    ood_data = {}
    
    # Apply k-means clustering to benign samples
    if len(all_benign_features) > 0 and k_benign > 0:
        if len(all_benign_features) >= k_benign:
            benign_features_array = np.array(all_benign_features)
            
            # Use k-means clustering
            kmeans_benign = KMeans(n_clusters=k_benign, random_state=random_seed, n_init=10)
            benign_cluster_labels = kmeans_benign.fit_predict(benign_features_array)
            
            # Group features by cluster
            for cluster_id in range(k_benign):
                cluster_mask = benign_cluster_labels == cluster_id
                cluster_features = benign_features_array[cluster_mask].tolist()
                if len(cluster_features) > 0:
                    in_dist_data[f"benign_cluster_{cluster_id}"] = cluster_features
        else:
            # Fallback: use all samples as one cluster
            in_dist_data["benign_cluster_0"] = all_benign_features
    
    # Apply k-means clustering to malicious samples
    if len(all_malicious_features) > 0 and k_malicious > 0:
        if len(all_malicious_features) >= k_malicious:
            malicious_features_array = np.array(all_malicious_features)
            
            # Use k-means clustering
            kmeans_malicious = KMeans(n_clusters=k_malicious, random_state=random_seed, n_init=10)
            malicious_cluster_labels = kmeans_malicious.fit_predict(malicious_features_array)
            
            # Group features by cluster
            for cluster_id in range(k_malicious):
                cluster_mask = malicious_cluster_labels == cluster_id
                cluster_features = malicious_features_array[cluster_mask].tolist()
                if len(cluster_features) > 0:
                    ood_data[f"malicious_cluster_{cluster_id}"] = cluster_features
        else:
            # Fallback: use all samples as one cluster
            ood_data["malicious_cluster_0"] = all_malicious_features
    
    return in_dist_data, ood_data, metadata

def evaluate_clustering_approach(detector, test_datasets, layer_hidden_states, layer_labels, approach_name):
    """
    Evaluate a clustering approach on all test datasets.
    
    Args:
        detector: Trained MCDDetector
        test_datasets: Dict of test dataset names to samples
        layer_hidden_states: Dict of dataset features for this layer
        layer_labels: Dict of dataset labels for this layer
        approach_name: Name of the approach for logging
    
    Returns:
        results: Dict of results for each test dataset
        combined_result: Combined result across all test datasets
    """
    results = {}
    all_test_features = []
    all_test_labels = []
    
    # Evaluate on each test dataset individually
    for dataset_name in test_datasets.keys():
        if dataset_name in layer_hidden_states:
            test_features = layer_hidden_states[dataset_name]
            test_labels = layer_labels[dataset_name]
            
            if len(test_features) > 0:
                result = detector.evaluate(test_features, test_labels)
                results[dataset_name] = result
                
                # Accumulate for combined evaluation
                all_test_features.extend(test_features)
                all_test_labels.extend(test_labels)
    
    # Combined evaluation across all test datasets
    if len(all_test_features) > 0:
        combined_result = detector.evaluate(all_test_features, all_test_labels)
        results['COMBINED'] = combined_result
    else:
        results['COMBINED'] = None
    
    return results

def run_single_experiment(layer_idx, all_datasets, all_hidden_states, all_labels, 
                         in_dist_datasets, ood_datasets, test_datasets,
                         projection_model, approach_name, k_benign=None, k_malicious=None, 
                         random_seed=42):
    """
    Run a single experiment with specified clustering parameters.
    
    Returns:
        results: Dict of evaluation results
        metadata: Dict with experiment metadata
    """
    print(f"    Running {approach_name} (k_benign={k_benign}, k_malicious={k_malicious})...")
    
    # Prepare data for this layer
    layer_hidden_states = {}
    layer_labels = {}
    
    for dataset_name in all_datasets.keys():
        if dataset_name in all_hidden_states:
            layer_hidden_states[dataset_name] = all_hidden_states[dataset_name][layer_idx]
            layer_labels[dataset_name] = all_labels[dataset_name]
    
    # Apply learned projection
    projected_layer_hidden_states = apply_learned_projection(
        projection_model, layer_hidden_states, device=GPU_DEVICE
    )
    layer_hidden_states = projected_layer_hidden_states
    cleanup_gpu_memory()
    
    # Prepare data structures based on approach
    all_training_datasets = {**in_dist_datasets, **ood_datasets}
    training_layer_hidden_states = {k: v for k, v in layer_hidden_states.items() if k in all_training_datasets}
    training_layer_labels = {k: v for k, v in layer_labels.items() if k in all_training_datasets}
    
    if approach_name == "dataset_based":
        # Original dataset-based approach
        in_dist_data, ood_train_data = prepare_ood_data_structure(
            all_training_datasets, training_layer_hidden_states, training_layer_labels
        )
        metadata = {'approach': 'dataset_based'}
    else:
        # K-means approach
        in_dist_data, ood_train_data, clustering_metadata = prepare_kmeans_data_structure_flexible(
            all_training_datasets, training_layer_hidden_states, training_layer_labels,
            k_benign=k_benign, k_malicious=k_malicious, random_seed=random_seed + layer_idx
        )
        metadata = {'approach': 'kmeans', **clustering_metadata}
    
    # Initialize and train MCD detector
    detector = MCDDetector(use_gpu=True)
    
    try:
        detector.fit_in_distribution(in_dist_data)
        detector.fit_ood_clusters(ood_train_data)
        
        # Use a subset of training data for threshold fitting (validation)
        validation_features = []
        validation_labels = []
        for dataset_name in all_training_datasets.keys():
            if dataset_name in layer_hidden_states:
                features = layer_hidden_states[dataset_name]
                labels = layer_labels[dataset_name]
                # Use 20% of training data for validation
                val_size = max(1, len(features) // 5)
                validation_features.extend(features[:val_size])
                validation_labels.extend(labels[:val_size])
        
        if len(validation_features) > 0:
            detector.fit_threshold(validation_features, validation_labels)
        
        # Evaluate on test datasets
        results = evaluate_clustering_approach(
            detector, test_datasets, layer_hidden_states, layer_labels, approach_name
        )
        
        cleanup_gpu_memory()
        return results, metadata
        
    except Exception as e:
        print(f"      Error in {approach_name}: {e}")
        cleanup_gpu_memory()
        return {}, metadata

def main():
    """Main function for k-means ablation study."""
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set random seed for reproducibility
    MAIN_SEED = 45
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)
    torch.manual_seed(MAIN_SEED)
    torch.cuda.manual_seed(MAIN_SEED)
    torch.cuda.manual_seed_all(MAIN_SEED)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional determinism settings
    os.environ['PYTHONHASHSEED'] = str(MAIN_SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"Random seeds set for reproducibility (seed={MAIN_SEED})")
    
    model_path = "model/llava-v1.6-vicuna-7b/"
    
    # Initialize feature extractor
    extractor = HiddenStateExtractor(model_path)
    
    print("="*100)
    print("K-MEANS CLUSTERING ABLATION STUDY FOR OOD JAILBREAK DETECTION")
    print("="*100)
    print("Comparing different clustering approaches:")
    print("1. Original dataset-based clustering (baseline)")
    print("2. K-means with k=dataset_count (should be best among k-means)")
    print("3. K-means with various k values (should be inferior)")
    print("="*100)

    # Load balanced training and test data
    print("\n--- Loading Data ---")
    in_dist_datasets, ood_datasets = prepare_balanced_training()
    test_datasets = prepare_balanced_evaluation()

    # Analyze data composition
    print("\n--- Data Analysis ---")
    total_benign = sum(len(samples) for samples in in_dist_datasets.values())
    total_malicious = sum(len(samples) for samples in ood_datasets.values())
    benign_dataset_count = len(in_dist_datasets)
    malicious_dataset_count = len(ood_datasets)

    print(f"Training data: {total_benign} benign samples from {benign_dataset_count} datasets")
    print(f"Training data: {total_malicious} malicious samples from {malicious_dataset_count} datasets")
    print(f"Test datasets: {list(test_datasets.keys())}")

    # Extract hidden states for all datasets
    print("\n--- Extracting Hidden States ---")
    all_datasets = {**in_dist_datasets, **ood_datasets, **test_datasets}
    all_hidden_states = {}
    all_labels = {}

    for dataset_name, samples in all_datasets.items():
        print(f"Extracting features for {dataset_name} ({len(samples)} samples)...")

        batch_size = 25 if len(samples) > 5000 else 50
        memory_cleanup_freq = 5 if len(samples) > 5000 else 10

        hidden_states, labels, _ = extractor.extract_hidden_states(
            samples, f"{dataset_name}", layer_start=0, layer_end=31, use_cache=True,
            batch_size=batch_size, memory_cleanup_freq=memory_cleanup_freq,
            experiment_name="balanced_ml_detection"
        )
        all_hidden_states[dataset_name] = hidden_states
        all_labels[dataset_name] = labels

    # Define experimental configurations
    print("\n--- Experimental Setup ---")

    # Test layers (subset for efficiency)
    test_layers = [12, 14, 16, 18, 20, 22, 24]  # Focus on middle-to-late layers
    print(f"Testing layers: {test_layers}")

    # Define clustering approaches to test
    clustering_approaches = []

    # 1. Baseline: Original dataset-based approach
    clustering_approaches.append({
        'name': 'dataset_based',
        'k_benign': None,
        'k_malicious': None,
        'description': 'Original dataset-based clustering (baseline)'
    })

    # 2. K-means with k=dataset_count (should be best among k-means)
    clustering_approaches.append({
        'name': f'kmeans_k{benign_dataset_count}_{malicious_dataset_count}',
        'k_benign': benign_dataset_count,
        'k_malicious': malicious_dataset_count,
        'description': f'K-means with k=dataset_count ({benign_dataset_count}, {malicious_dataset_count})'
    })

    # 3. K-means with various k values
    k_values_to_test = [1, 2, 3, 5, 8]  # Different cluster numbers
    for k_benign in k_values_to_test:
        for k_malicious in k_values_to_test:
            if k_benign <= total_benign and k_malicious <= total_malicious:  # Ensure feasible
                clustering_approaches.append({
                    'name': f'kmeans_k{k_benign}_{k_malicious}',
                    'k_benign': k_benign,
                    'k_malicious': k_malicious,
                    'description': f'K-means with k=({k_benign}, {k_malicious})'
                })

    print(f"Testing {len(clustering_approaches)} clustering approaches:")
    for i, approach in enumerate(clustering_approaches, 1):
        print(f"  {i}. {approach['description']}")

    # Train projection models (using single layer mode for efficiency)
    print(f"\n--- Training Projection Models ---")
    CONFIG.PROJECTION_MODE = "single_layer"
    CONFIG.SINGLE_LAYER_TRAINING_LAYER = 18  # Use layer 18 for training
    CONFIG.print_config()

    # Prepare training data for projection
    training_dataset_names = set(in_dist_datasets.keys()) | set(ood_datasets.keys())
    projection_features_dict = {}
    projection_labels_dict = {}

    for dataset_name in training_dataset_names:
        if dataset_name in all_hidden_states:
            projection_features_dict[dataset_name] = all_hidden_states[dataset_name][CONFIG.SINGLE_LAYER_TRAINING_LAYER]
            projection_labels_dict[dataset_name] = all_labels[dataset_name]

    # Train the single projection model
    single_projection_model, dataset_name_to_id = train_learned_projection(
        projection_features_dict, projection_labels_dict,
        device=GPU_DEVICE, random_seed=MAIN_SEED
    )

    print("Projection training completed!")
    cleanup_gpu_memory()

    # Run experiments
    print(f"\n--- Running Experiments ---")
    all_results = {}  # {layer_idx: {approach_name: results}}
    all_metadata = {}  # {layer_idx: {approach_name: metadata}}

    for layer_idx in test_layers:
        print(f"\n=== Evaluating Layer {layer_idx} ===")
        all_results[layer_idx] = {}
        all_metadata[layer_idx] = {}

        for approach in clustering_approaches:
            approach_name = approach['name']
            k_benign = approach['k_benign']
            k_malicious = approach['k_malicious']

            try:
                results, metadata = run_single_experiment(
                    layer_idx, all_datasets, all_hidden_states, all_labels,
                    in_dist_datasets, ood_datasets, test_datasets,
                    single_projection_model, approach_name,
                    k_benign=k_benign, k_malicious=k_malicious,
                    random_seed=MAIN_SEED
                )

                all_results[layer_idx][approach_name] = results
                all_metadata[layer_idx][approach_name] = metadata

                # Print summary for this approach
                if 'COMBINED' in results and results['COMBINED'] is not None:
                    combined = results['COMBINED']
                    print(f"      {approach_name}: Acc={combined['accuracy']:.3f}, "
                          f"F1={combined['f1']:.3f}, AUROC={combined.get('auroc', 0):.3f}")
                else:
                    print(f"      {approach_name}: Failed")

            except Exception as e:
                print(f"      {approach_name}: Error - {e}")
                all_results[layer_idx][approach_name] = {}
                all_metadata[layer_idx][approach_name] = {'error': str(e)}

    # Save detailed results
    print(f"\n--- Saving Results ---")
    output_path = "results/kmeans_ablation_results.csv"
    os.makedirs("results", exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Layer", "Approach", "K_Benign", "K_Malicious", "Dataset",
            "Accuracy", "F1", "TPR", "FPR", "AUROC", "AUPRC", "Threshold"
        ])

        for layer_idx in test_layers:
            for approach_name, results in all_results[layer_idx].items():
                metadata = all_metadata[layer_idx][approach_name]
                k_benign = metadata.get('k_benign', 'N/A')
                k_malicious = metadata.get('k_malicious', 'N/A')

                if results:
                    for dataset_name, result in results.items():
                        if result is not None:
                            writer.writerow([
                                layer_idx, approach_name, k_benign, k_malicious, dataset_name,
                                f"{result['accuracy']:.4f}",
                                f"{result['f1']:.4f}",
                                "N/A" if np.isnan(result.get('tpr', np.nan)) else f"{result['tpr']:.4f}",
                                "N/A" if np.isnan(result.get('fpr', np.nan)) else f"{result['fpr']:.4f}",
                                "N/A" if np.isnan(result.get('auroc', np.nan)) else f"{result['auroc']:.4f}",
                                "N/A" if np.isnan(result.get('auprc', np.nan)) else f"{result['auprc']:.4f}",
                                f"{result.get('threshold', 0):.4f}"
                            ])
                else:
                    writer.writerow([
                        layer_idx, approach_name, k_benign, k_malicious, "FAILED",
                        "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
                    ])

    print(f"Detailed results saved to {output_path}")

    # Generate summary analysis
    print(f"\n--- Summary Analysis ---")
    generate_summary_analysis(all_results, all_metadata, test_layers, clustering_approaches)

def generate_summary_analysis(all_results, all_metadata, test_layers, clustering_approaches):
    """Generate and print summary analysis of the ablation study."""

    print("\n" + "="*100)
    print("K-MEANS CLUSTERING ABLATION STUDY SUMMARY")
    print("="*100)

    # Collect performance data for analysis
    approach_performances = {}  # {approach_name: [combined_accuracies]}

    for layer_idx in test_layers:
        for approach_name, results in all_results[layer_idx].items():
            if approach_name not in approach_performances:
                approach_performances[approach_name] = []

            if results and 'COMBINED' in results and results['COMBINED'] is not None:
                combined_acc = results['COMBINED']['accuracy']
                approach_performances[approach_name].append(combined_acc)
            else:
                approach_performances[approach_name].append(0.0)  # Failed experiment

    # Calculate average performance for each approach
    approach_avg_performance = {}
    for approach_name, performances in approach_performances.items():
        if performances:
            approach_avg_performance[approach_name] = np.mean(performances)
        else:
            approach_avg_performance[approach_name] = 0.0

    # Sort approaches by average performance (descending)
    sorted_approaches = sorted(approach_avg_performance.items(), key=lambda x: x[1], reverse=True)

    print(f"\nRanking by Average Combined Accuracy across {len(test_layers)} layers:")
    print(f"{'Rank':<4} {'Approach':<30} {'Avg Accuracy':<12} {'Description'}")
    print("-" * 80)

    for rank, (approach_name, avg_acc) in enumerate(sorted_approaches, 1):
        # Find description
        description = "Unknown"
        for approach in clustering_approaches:
            if approach['name'] == approach_name:
                description = approach['description']
                break

        print(f"{rank:<4} {approach_name:<30} {avg_acc:.4f}       {description}")

    # Detailed layer-by-layer comparison
    print(f"\nDetailed Layer-by-Layer Performance:")
    print(f"{'Approach':<30} " + " ".join([f"L{layer:<6}" for layer in test_layers]) + " Average")
    print("-" * (30 + 8 * len(test_layers) + 10))

    for approach_name, avg_acc in sorted_approaches:
        performances = approach_performances[approach_name]
        perf_str = f"{approach_name:<30} "
        for perf in performances:
            perf_str += f"{perf:.3f}   "
        perf_str += f"{avg_acc:.3f}"
        print(perf_str)

    # Analysis of k-means vs dataset-based
    print(f"\nComparative Analysis:")

    # Find dataset-based performance
    dataset_based_perf = approach_avg_performance.get('dataset_based', 0.0)

    # Find best k-means performance
    kmeans_performances = {name: perf for name, perf in approach_avg_performance.items()
                          if name.startswith('kmeans_')}

    if kmeans_performances:
        best_kmeans_name, best_kmeans_perf = max(kmeans_performances.items(), key=lambda x: x[1])

        print(f"1. Dataset-based approach (baseline): {dataset_based_perf:.4f}")
        print(f"2. Best k-means approach ({best_kmeans_name}): {best_kmeans_perf:.4f}")

        if dataset_based_perf > 0:
            performance_gap = dataset_based_perf - best_kmeans_perf
            relative_gap = (performance_gap / dataset_based_perf) * 100
            print(f"3. Performance gap: {performance_gap:.4f} ({relative_gap:.1f}% relative)")

            if performance_gap > 0:
                print("   → Dataset-based clustering is SUPERIOR to k-means clustering")
            else:
                print("   → K-means clustering performs comparably to dataset-based clustering")

        # Check if k=dataset_count is best among k-means
        dataset_count_approaches = [name for name in kmeans_performances.keys()
                                  if 'k3_3' in name]  # Assuming 3 datasets each

        if dataset_count_approaches:
            dataset_count_name = dataset_count_approaches[0]
            dataset_count_perf = kmeans_performances[dataset_count_name]

            print(f"4. K-means with k=dataset_count: {dataset_count_perf:.4f}")

            if dataset_count_name == best_kmeans_name:
                print("   → Among k-means approaches, k=dataset_count works BEST")
            else:
                print(f"   → Among k-means approaches, k=dataset_count is not optimal")
                print(f"     Best k-means: {best_kmeans_name} ({best_kmeans_perf:.4f})")

    # Statistical significance test (if we had multiple runs)
    print(f"\nKey Findings for Paper:")
    print(f"• Dataset-based clustering achieves {dataset_based_perf:.1%} average accuracy")
    print(f"• Best k-means clustering achieves {best_kmeans_perf:.1%} average accuracy")
    print(f"• K-means clustering shows {'superior' if best_kmeans_perf > dataset_based_perf else 'inferior'} performance")
    print(f"• Performance difference: {abs(dataset_based_perf - best_kmeans_perf):.1%}")

    # Save summary to file
    summary_path = "results/kmeans_ablation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("K-MEANS CLUSTERING ABLATION STUDY SUMMARY\n")
        f.write("="*50 + "\n\n")

        f.write("Ranking by Average Combined Accuracy:\n")
        for rank, (approach_name, avg_acc) in enumerate(sorted_approaches, 1):
            description = "Unknown"
            for approach in clustering_approaches:
                if approach['name'] == approach_name:
                    description = approach['description']
                    break
            f.write(f"{rank}. {approach_name}: {avg_acc:.4f} - {description}\n")

        f.write(f"\nDataset-based vs K-means:\n")
        f.write(f"Dataset-based: {dataset_based_perf:.4f}\n")
        f.write(f"Best K-means: {best_kmeans_perf:.4f}\n")
        f.write(f"Performance gap: {abs(dataset_based_perf - best_kmeans_perf):.4f}\n")

    print(f"\nSummary saved to {summary_path}")
    print("="*100)

if __name__ == "__main__":
    main()
