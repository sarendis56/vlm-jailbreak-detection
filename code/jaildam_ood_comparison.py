#!/usr/bin/env python3
"""
JailDAM OOD Comparison Script

This script implements simple OOD detection methods to compare with JailDAM's setup:
- Training: 80% of MM-Vet samples (174 samples, benign only)
- Testing: JailDAM test set (218 MM-Vet + 528 jailbreak samples)
- Methods: KNN distance and Mahalanobis distance (no outlier exposure)
- Models: Llava layer 16 and Flava embeddings
"""

import numpy as np
import random
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from load_datasets import *
from feature_extractor import HiddenStateExtractor
from balanced_flava import FlavaFeatureExtractor

# GPU device setup
GPU_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {GPU_DEVICE}")

class SimpleOODDetector:
    """Simple OOD detector using only benign training data (no outlier exposure)"""
    
    def __init__(self, method='knn', k=5, use_gpu=True):
        self.method = method  # 'knn' or 'mahalanobis'
        self.k = k
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = GPU_DEVICE if self.use_gpu else torch.device('cpu')
        
        # Training data statistics
        self.train_features = None
        self.train_mean = None
        self.train_cov_inv = None
        self.scaler = StandardScaler()
        
        print(f"Initialized {method.upper()} OOD detector (k={k}, GPU={self.use_gpu})")
    
    def fit(self, train_features):
        """Fit the OOD detector on benign training features only"""
        print(f"Training OOD detector on {len(train_features)} benign samples...")
        
        # Standardize features
        train_features_scaled = self.scaler.fit_transform(train_features)
        self.train_features = train_features_scaled
        
        if self.method == 'mahalanobis':
            # Compute mean and covariance for Mahalanobis distance
            self.train_mean = np.mean(train_features_scaled, axis=0)
            
            try:
                # Compute covariance matrix with regularization
                cov_matrix = np.cov(train_features_scaled.T)
                
                # Add regularization to prevent singular matrix
                reg_param = 1e-6
                cov_matrix += reg_param * np.eye(cov_matrix.shape[0])
                
                # Compute inverse
                self.train_cov_inv = np.linalg.inv(cov_matrix)
                print(f"Computed covariance matrix ({cov_matrix.shape[0]}x{cov_matrix.shape[1]})")
                
            except Exception as e:
                print(f"Error computing covariance: {e}, using identity matrix")
                self.train_cov_inv = np.eye(train_features_scaled.shape[1])
        
        print(f"OOD detector training completed")
    
    def _euclidean_distance_batch_gpu(self, X, Y):
        """GPU-accelerated batch Euclidean distance computation"""
        try:
            X_gpu = torch.tensor(X, dtype=torch.float32, device=self.device)
            Y_gpu = torch.tensor(Y, dtype=torch.float32, device=self.device)
            
            if X_gpu.dim() == 1:
                X_gpu = X_gpu.unsqueeze(0)
            if Y_gpu.dim() == 1:
                Y_gpu = Y_gpu.unsqueeze(0)
            
            distances = torch.cdist(X_gpu, Y_gpu, p=2)
            return distances.cpu().numpy()
            
        except Exception as e:
            print(f"GPU distance computation failed: {e}, using CPU")
            return self._euclidean_distance_batch_cpu(X, Y)
    
    def _euclidean_distance_batch_cpu(self, X, Y):
        """CPU batch Euclidean distance computation"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        
        X_expanded = X[:, np.newaxis, :]
        Y_expanded = Y[np.newaxis, :, :]
        distances = np.sqrt(np.sum((X_expanded - Y_expanded) ** 2, axis=2))
        return distances
    
    def _mahalanobis_distance_batch_gpu(self, X, mean, cov_inv):
        """GPU-accelerated batch Mahalanobis distance computation"""
        try:
            X_gpu = torch.tensor(X, dtype=torch.float64, device=self.device)
            mean_gpu = torch.tensor(mean, dtype=torch.float64, device=self.device)
            cov_inv_gpu = torch.tensor(cov_inv, dtype=torch.float64, device=self.device)
            
            if X_gpu.dim() == 1:
                X_gpu = X_gpu.unsqueeze(0)
            
            diff = X_gpu - mean_gpu.unsqueeze(0)
            temp = torch.matmul(diff, cov_inv_gpu)
            distances_squared = torch.sum(temp * diff, dim=1)
            distances = torch.sqrt(torch.clamp(distances_squared, min=0))
            
            return distances.cpu().numpy()
            
        except Exception as e:
            print(f"GPU Mahalanobis computation failed: {e}, using CPU")
            return self._mahalanobis_distance_batch_cpu(X, mean, cov_inv)
    
    def _mahalanobis_distance_batch_cpu(self, X, mean, cov_inv):
        """CPU batch Mahalanobis distance computation"""
        distances = []
        for x in X:
            diff = x - mean
            try:
                result = np.dot(diff, cov_inv)
                result = np.dot(result, diff)
                distances.append(np.sqrt(max(0, result)))
            except:
                distances.append(np.linalg.norm(diff))
        return np.array(distances)
    
    def compute_ood_scores(self, test_features):
        """Compute OOD scores for test features"""
        print(f"Computing OOD scores for {len(test_features)} test samples...")
        
        # Standardize test features using training scaler
        test_features_scaled = self.scaler.transform(test_features)
        
        if self.method == 'knn':
            # KNN-based OOD detection: k-th nearest neighbor distance
            if self.use_gpu:
                distances = self._euclidean_distance_batch_gpu(test_features_scaled, self.train_features)
            else:
                distances = self._euclidean_distance_batch_cpu(test_features_scaled, self.train_features)
            
            # For each test sample, get k-th nearest neighbor distance
            scores = []
            for i in range(len(test_features_scaled)):
                sample_distances = distances[i]
                sorted_distances = np.sort(sample_distances)
                kth_distance = sorted_distances[min(self.k - 1, len(sorted_distances) - 1)]
                scores.append(kth_distance)
            
            return np.array(scores)
            
        elif self.method == 'mahalanobis':
            # Mahalanobis distance-based OOD detection
            if self.use_gpu:
                distances = self._mahalanobis_distance_batch_gpu(test_features_scaled, self.train_mean, self.train_cov_inv)
            else:
                distances = self._mahalanobis_distance_batch_cpu(test_features_scaled, self.train_mean, self.train_cov_inv)
            
            return distances

def load_jaildam_datasets():
    """Load datasets according to JailDAM setup with separate jailbreak datasets"""
    print("Loading JailDAM datasets...")

    # Load MM-Vet (complete dataset for train/test split)
    mmvet_samples = load_mm_vet()
    if not mmvet_samples:
        raise ValueError("Could not load MM-Vet dataset")

    # Ensure we have exactly 218 samples as in JailDAM
    mmvet_samples = mmvet_samples[:218]
    print(f"Loaded {len(mmvet_samples)} MM-Vet samples")

    # Split MM-Vet: 80% for training (174 samples), 20% for testing (44 samples)
    random.seed(42)  # For reproducibility
    random.shuffle(mmvet_samples)

    train_size = int(0.8 * len(mmvet_samples))  # 174 samples
    mmvet_train = mmvet_samples[:train_size]
    mmvet_test = mmvet_samples[train_size:]

    print(f"MM-Vet split: {len(mmvet_train)} train, {len(mmvet_test)} test")

    # Load jailbreak datasets separately for pairwise evaluation
    jailbreak_datasets = {}

    # 1. MM-SafetyBench: 327 samples
    try:
        mm_safety_samples = load_mm_safety_bench_all(max_samples=327)
        jailbreak_datasets["MM-SafetyBench"] = mm_safety_samples
        print(f"Loaded {len(mm_safety_samples)} MM-SafetyBench samples")
    except Exception as e:
        print(f"Could not load MM-SafetyBench: {e}")
        jailbreak_datasets["MM-SafetyBench"] = []

    # 2. FigStep: 49 samples
    try:
        figstep_samples = load_JailBreakV_figstep(max_samples=49)
        jailbreak_datasets["FigStep"] = figstep_samples
        print(f"Loaded {len(figstep_samples)} FigStep samples")
    except Exception as e:
        print(f"Could not load FigStep: {e}")
        jailbreak_datasets["FigStep"] = []

    # 3. JailbreakV-28K llm_transfer_attack: ~75 samples (half of 149)
    try:
        jbv_llm_samples = load_JailBreakV_custom(attack_types=["llm_transfer_attack"], max_samples=75)
        jailbreak_datasets["JailbreakV-LLM"] = jbv_llm_samples
        print(f"Loaded {len(jbv_llm_samples)} JailbreakV LLM transfer samples")
    except Exception as e:
        print(f"Could not load JailbreakV LLM transfer: {e}")
        jailbreak_datasets["JailbreakV-LLM"] = []

    # 4. JailbreakV-28K query_related: ~74 samples (other half of 149)
    try:
        jbv_query_samples = load_JailBreakV_custom(attack_types=["query_related"], max_samples=74)
        jailbreak_datasets["JailbreakV-Query"] = jbv_query_samples
        print(f"Loaded {len(jbv_query_samples)} JailbreakV Query related samples")
    except Exception as e:
        print(f"Could not load JailbreakV Query related: {e}")
        jailbreak_datasets["JailbreakV-Query"] = []

    # Combine all jailbreak samples for overall evaluation
    all_jailbreak_samples = []
    for dataset_name, samples in jailbreak_datasets.items():
        all_jailbreak_samples.extend(samples)

    print(f"Total jailbreak samples: {len(all_jailbreak_samples)}")
    print(f"Total test samples: {len(mmvet_test)} benign + {len(all_jailbreak_samples)} jailbreak = {len(mmvet_test) + len(all_jailbreak_samples)}")

    return mmvet_train, mmvet_test, jailbreak_datasets, all_jailbreak_samples

def extract_embeddings(samples, model_type, dataset_name, extractor):
    """Extract embeddings for a dataset"""
    print(f"Extracting {model_type} embeddings for {dataset_name} ({len(samples)} samples)...")
    
    if model_type == "llava":
        # Extract layer 16 embeddings from Llava
        hidden_states, labels, _ = extractor.extract_hidden_states(
            samples, f"{dataset_name}_jaildam", layer_start=16, layer_end=16,
            use_cache=True, batch_size=25, memory_cleanup_freq=5,
            experiment_name="jaildam_comparison"
        )
        embeddings = np.array(hidden_states[16])
        
    elif model_type == "flava":
        # Extract Flava embeddings
        embeddings, labels = extractor.extract_features(samples, batch_size=16)
        embeddings = np.array(embeddings)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings

def evaluate_pairwise(mmvet_embeddings, jailbreak_embeddings, jailbreak_name, detector):
    """Evaluate MM-Vet vs specific jailbreak dataset"""
    # Combine embeddings and create labels
    test_embeddings = np.vstack([mmvet_embeddings, jailbreak_embeddings])
    test_labels = np.concatenate([
        np.zeros(len(mmvet_embeddings)),  # 0 for MM-Vet
        np.ones(len(jailbreak_embeddings))  # 1 for jailbreak
    ])

    # Compute OOD scores
    ood_scores = detector.compute_ood_scores(test_embeddings)

    # Compute metrics
    auroc = roc_auc_score(test_labels, ood_scores)
    auprc = average_precision_score(test_labels, ood_scores)

    return auroc, auprc

def main():
    print("="*80)
    print("JAILDAM OOD COMPARISON")
    print("="*80)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Load datasets
    mmvet_train, mmvet_test, jailbreak_datasets, all_jailbreak_samples = load_jaildam_datasets()

    # Initialize extractors
    llava_extractor = HiddenStateExtractor("model/llava-v1.6-vicuna-7b/")
    flava_extractor = FlavaFeatureExtractor()

    results = {}

    # Test both models
    for model_type in ["llava", "flava"]:
        print(f"\n{'='*60}")
        print(f"PROCESSING {model_type.upper()} EMBEDDINGS")
        print(f"{'='*60}")

        extractor = llava_extractor if model_type == "llava" else flava_extractor

        # Extract embeddings for training
        train_embeddings = extract_embeddings(mmvet_train, model_type, "mmvet_train", extractor)

        # Extract embeddings for all test datasets
        mmvet_test_embeddings = extract_embeddings(mmvet_test, model_type, "mmvet_test", extractor)

        # Extract embeddings for each jailbreak dataset separately
        jailbreak_embeddings = {}
        for jb_name, jb_samples in jailbreak_datasets.items():
            if len(jb_samples) > 0:
                jb_embeddings = extract_embeddings(jb_samples, model_type, f"{jb_name.lower()}_test", extractor)
                jailbreak_embeddings[jb_name] = jb_embeddings

        # Test both OOD methods
        model_results = {}
        for method in ['knn', 'mahalanobis']:
            print(f"\nTesting {method.upper()} OOD detection...")

            # Initialize and train OOD detector
            detector = SimpleOODDetector(method=method, k=3, use_gpu=True)
            detector.fit(train_embeddings)

            method_results = {}

            # 1. Overall evaluation (including training samples like JailDAM might do)
            print("  Computing overall metrics (including training samples)...")

            # Combine ALL MM-Vet samples (train + test) vs all jailbreak samples
            all_mmvet_embeddings = np.vstack([train_embeddings, mmvet_test_embeddings])
            all_jailbreak_embeddings = np.vstack(list(jailbreak_embeddings.values()))

            overall_test_embeddings = np.vstack([all_mmvet_embeddings, all_jailbreak_embeddings])
            overall_test_labels = np.concatenate([
                np.zeros(len(all_mmvet_embeddings)),  # 0 for MM-Vet
                np.ones(len(all_jailbreak_embeddings))  # 1 for jailbreak
            ])

            overall_ood_scores = detector.compute_ood_scores(overall_test_embeddings)
            overall_auroc = roc_auc_score(overall_test_labels, overall_ood_scores)
            overall_auprc = average_precision_score(overall_test_labels, overall_ood_scores)

            method_results['overall'] = {
                'auroc': overall_auroc,
                'auprc': overall_auprc
            }

            print(f"    Overall (with training) - AUROC: {overall_auroc:.4f}, AUPRC: {overall_auprc:.4f}")

            # 2. Pairwise evaluations: MM-Vet vs each jailbreak dataset
            print("  Computing pairwise metrics...")
            for jb_name, jb_embeddings in jailbreak_embeddings.items():
                auroc, auprc = evaluate_pairwise(all_mmvet_embeddings, jb_embeddings, jb_name, detector)
                method_results[jb_name] = {
                    'auroc': auroc,
                    'auprc': auprc
                }
                print(f"    MM-Vet vs {jb_name:15s} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

            model_results[method] = method_results

        results[model_type] = model_results

    # Print final results
    print("\n" + "="*80)
    print("JAILDAM OOD COMPARISON RESULTS")
    print("="*80)
    print("Training: 174 MM-Vet samples (benign only, no outlier exposure)")
    print("Testing: 218 MM-Vet (including training) + jailbreak samples")
    print("\nOverall Results (including training samples):")

    for model_type in ["llava", "flava"]:
        print(f"\n{model_type.upper()} Embeddings:")
        for method in ['knn', 'mahalanobis']:
            result = results[model_type][method]['overall']
            print(f"  {method.upper():12s} - AUROC: {result['auroc']:.4f}, AUPRC: {result['auprc']:.4f}")

    print("\nPairwise Results (MM-Vet vs each jailbreak dataset):")
    jb_names = ["MM-SafetyBench", "FigStep", "JailbreakV-LLM", "JailbreakV-Query"]

    for model_type in ["llava", "flava"]:
        print(f"\n{model_type.upper()} Embeddings:")
        for method in ['knn', 'mahalanobis']:
            print(f"  {method.upper()}:")
            for jb_name in jb_names:
                if jb_name in results[model_type][method]:
                    result = results[model_type][method][jb_name]
                    print(f"    vs {jb_name:15s} - AUROC: {result['auroc']:.4f}, AUPRC: {result['auprc']:.4f}")

if __name__ == "__main__":
    main()
