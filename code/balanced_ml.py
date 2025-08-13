#!/usr/bin/env python3
"""
Balanced Jailbreak Detection with New Dataset Configuration

This script implements a new balanced dataset division according to good principles:

Training Set (2,000 examples, 1:1 ratio):
- Benign (1,000): Alpaca (500) + MM-Vet (218) + OpenAssistant (282)
- Malicious (1,000): AdvBench (300) + JailbreakV-28K (550, llm_transfer_attack + query_related) + DAN variants (150)

Test Set (1,800 examples, 1:1 ratio):
- Safe (900): XSTest safe (250) + FigTxt safe (300) + VQAv2 (350)
- Unsafe (900): XSTest unsafe (200) + FigTxt unsafe (350) + VAE (200) + JailbreakV-28K (150, figstep attack)

This ensures strict train-test separation and balanced evaluation.
"""

import csv
import numpy as np
import random
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, precision_recall_curve, auc, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

from load_datasets import *
from feature_extractor import HiddenStateExtractor
from generic_classifier import train_generic_classifier, evaluate_generic_classifier
from sklearn.metrics import balanced_accuracy_score

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
class MLConfig:
    """Global configuration for ML experiments"""

    # Random seed for reproducibility
    MAIN_SEED = 42

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        # print("="*80)
        # print("ML CONFIGURATION")
        # print("="*80)
        print(f"Main seed: {cls.MAIN_SEED}")
        # print("="*80)

# Global config instance
CONFIG = MLConfig()

# GPU device setup
GPU_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {GPU_DEVICE}")

class GPULinearClassifier(nn.Module):
    """GPU-accelerated linear classifier"""
    def __init__(self, input_dim, model_type='logistic'):
        super(GPULinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.model_type = model_type

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

def train_gpu_linear_classifier(X_train, y_train, model_type='logistic', epochs=100, lr=1e-3, batch_size=256):
    """Train a GPU-accelerated linear classifier"""
    device = GPU_DEVICE
    input_dim = X_train.shape[1]

    # Convert to tensors and move to GPU
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)

    # Create model
    model = GPULinearClassifier(input_dim, model_type).to(device)

    # Setup optimizer and loss based on model type
    if model_type in ['logistic', 'svm']:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif model_type == 'ridge':
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)  # Higher weight decay for Ridge
    else:  # sgd
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)

    # Create data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            outputs = model(batch_X).squeeze()

            if model_type == 'ridge':
                # For ridge regression, use MSE loss with continuous targets
                loss = criterion(torch.sigmoid(outputs), batch_y)
            else:
                # For logistic/SVM/SGD, use BCE with logits
                loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return model

class ThresholdOptimizer:
    """
    Unified threshold optimization algorithm used across all balanced_* scripts.
    Creates validation set from training data and optimizes threshold using
    balanced accuracy + F1 score with adaptive threshold ranges.
    """

    def __init__(self):
        self.threshold = 0.5  # Default threshold

    def fit_threshold(self, model, X_train, y_train, model_type='linear'):
        """
        Find optimal threshold using validation data sampled from training data.
        This matches the algorithm used in other balanced_* scripts.
        """
        # Create validation set by sampling from training data
        val_features, val_labels = self._create_validation_set(X_train, y_train)

        if len(val_features) == 0 or len(set(val_labels)) < 2:
            print(f"    Warning: Insufficient validation data, using default threshold")
            self.threshold = 0.5
            return 0.5

        # Get prediction scores from the model
        scores = self._get_model_scores(model, val_features, model_type)
        val_labels = np.array(val_labels)

        # Debug: Check score properties
        print(f"    Score statistics: min={np.min(scores):.4f}, max={np.max(scores):.4f}, mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")

        # Check for invalid scores
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            print(f"    ERROR: Invalid scores detected (NaN or Inf), using default threshold")
            self.threshold = 0.5
            return 0.5

        # Check if we have both classes
        unique_labels = np.unique(val_labels)
        if len(unique_labels) < 2:
            print(f"    Warning: Only one class in validation data: {unique_labels}")
            self.threshold = 0.5
            return 0.5

        # Analyze score distributions to detect potential domain shift
        benign_scores = scores[val_labels == 0]
        malicious_scores = scores[val_labels == 1]

        benign_mean, benign_std = np.mean(benign_scores), np.std(benign_scores)
        malicious_mean, malicious_std = np.mean(malicious_scores), np.std(malicious_scores)

        # Calculate separation between distributions
        separation = abs(malicious_mean - benign_mean) / (benign_std + malicious_std + 1e-8)

        print(f"    Validation score separation: {separation:.2f} (higher is better)")

        # Use adaptive threshold range based on score distributions
        if separation > 5.0:  # Well-separated distributions
            # Use narrow range around the midpoint
            midpoint = (benign_mean + malicious_mean) / 2
            range_width = min(benign_std, malicious_std) * 2
            score_range = [midpoint - range_width, midpoint + range_width]
            print(f"    Using narrow threshold range around midpoint: [{score_range[0]:.4f}, {score_range[1]:.4f}]")
        else:  # Overlapping distributions - use wider range
            score_range = np.percentile(scores, [5, 95])
            range_width = score_range[1] - score_range[0]
            score_range[0] -= 0.2 * range_width
            score_range[1] += 0.2 * range_width
            print(f"    Using wide threshold range for overlapping distributions: [{score_range[0]:.4f}, {score_range[1]:.4f}]")

        # Grid search over threshold candidates
        thresholds = np.linspace(score_range[0], score_range[1], 200)  # More granular search
        best_score = 0
        best_threshold = 0.5
        best_f1 = 0
        best_balanced_acc = 0

        for threshold in thresholds:
            y_pred = (scores > threshold).astype(int)
            try:
                # Use balanced accuracy as primary metric (handles class imbalance better)
                balanced_acc = balanced_accuracy_score(val_labels, y_pred)
                f1 = f1_score(val_labels, y_pred, zero_division=0)

                # For domain shift robustness, prioritize balanced accuracy over F1
                combined_score = 0.8 * balanced_acc + 0.2 * f1

                if combined_score > best_score:
                    best_score = combined_score
                    best_threshold = threshold
                    best_f1 = f1
                    best_balanced_acc = balanced_acc
            except:
                continue

        print(f"    Optimal threshold: {best_threshold:.4f} (Balanced Acc: {best_balanced_acc:.4f}, F1: {best_f1:.4f})")

        # Debug: Show final threshold performance
        final_predictions = (scores > best_threshold).astype(int)
        final_accuracy = accuracy_score(val_labels, final_predictions)
        print(f"    Validation performance with optimal threshold: Accuracy={final_accuracy:.4f}")

        self.threshold = best_threshold
        return best_threshold

    def _create_validation_set(self, X_train, y_train, max_samples_per_class=150):
        """Create balanced validation set by sampling from training data"""
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        val_features = []
        val_labels = []

        # Sample from each class (increased sample size for better threshold estimation)
        for class_label in [0, 1]:  # benign and malicious
            class_indices = np.where(y_train == class_label)[0]
            if len(class_indices) > 0:
                sample_size = min(max_samples_per_class, len(class_indices))
                # Use deterministic sampling: take evenly spaced indices for reproducibility
                if sample_size < len(class_indices):
                    sampled_indices = np.linspace(0, len(class_indices)-1, sample_size, dtype=int)
                    selected_indices = class_indices[sampled_indices]
                else:
                    selected_indices = class_indices

                val_features.extend(X_train[selected_indices])
                val_labels.extend([class_label] * len(selected_indices))

        print(f"    Validation set: {len(val_features)} samples ({val_labels.count(0)} benign, {val_labels.count(1)} malicious)")

        # Additional validation checks
        if len(val_features) == 0:
            print(f"    ERROR: No validation features created!")
            return np.array([]), np.array([])

        if len(set(val_labels)) < 2:
            print(f"    ERROR: Validation set has only one class: {set(val_labels)}")
            return np.array([]), np.array([])

        return np.array(val_features), np.array(val_labels)

    def _get_model_scores(self, model, X_val, model_type):
        """Get prediction scores from different model types"""
        if model_type == 'mlp':
            # For MLP models from generic_classifier
            try:
                # Get raw predictions directly from the model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Convert to tensor and get model predictions
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                model.eval()

                with torch.no_grad():
                    # Get raw logits from the model
                    raw_outputs = model(X_val_tensor)
                    if raw_outputs.dim() > 1:
                        raw_outputs = raw_outputs.squeeze()

                    # Convert logits to probabilities using sigmoid
                    probabilities = torch.sigmoid(raw_outputs).cpu().numpy()
                    return probabilities

            except Exception as e:
                print(f"    Warning: Could not get MLP scores ({e}), trying alternative method")
                try:
                    # Fallback: use evaluate_generic_classifier
                    eval_results = evaluate_generic_classifier(
                        model, X_val, np.zeros(len(X_val)), task_type='binary', use_optimal_threshold=False
                    )

                    if 'probabilities' in eval_results:
                        return eval_results['probabilities']
                    else:
                        raise ValueError("No probabilities available")

                except Exception as e2:
                    print(f"    Error: Both MLP score extraction methods failed ({e2})")
                    # Return scores that will lead to default threshold (0.5)
                    return np.full(len(X_val), 0.5)

        else:
            # For linear models (GPU-based) - convert logits to probabilities
            device = GPU_DEVICE
            model.eval()

            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                raw_outputs = model(X_val_tensor).squeeze()

                # Convert raw logits to probabilities using sigmoid
                # This ensures consistent score interpretation across model types
                probabilities = torch.sigmoid(raw_outputs).cpu().numpy()
                return probabilities

def train_linear_classifier(X_train, y_train, model_type='logistic'):
    """Train a linear classifier (now GPU-accelerated)"""
    print(f"    Training GPU-accelerated {model_type.upper()} classifier...")

    # Use GPU implementation for faster training
    model = train_gpu_linear_classifier(X_train, y_train, model_type=model_type,
                                      epochs=100, lr=1e-3, batch_size=256)
    return model

def evaluate_linear_classifier(model, X_test, y_test, threshold=0.5):
    """Evaluate a GPU linear classifier with custom threshold"""
    device = GPU_DEVICE
    model.eval()

    with torch.no_grad():
        # Convert to tensor and move to GPU
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        # Get raw outputs
        raw_outputs = model(X_test_tensor).squeeze()

        # Get probabilities and predictions using custom threshold
        y_proba = torch.sigmoid(raw_outputs).cpu().numpy()
        y_pred = (y_proba > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Calculate TPR, FPR from confusion matrix
    if len(np.unique(y_test)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity/Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

        # Calculate AUROC and AUPRC
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
        auroc = auc(fpr_curve, tpr_curve)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = auc(recall, precision)
    else:
        tpr = float('nan')
        fpr = float('nan')
        auroc = float('nan')
        auprc = float('nan')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'tpr': tpr,
        'fpr': fpr,
        'auroc': auroc,
        'auprc': auprc,
        'predictions': y_pred,
        'probabilities': y_proba
    }

def evaluate_mlp_classifier_with_threshold(model, X_test, y_test, threshold=0.5):
    """Evaluate MLP classifier with custom threshold"""
    # Get predictions using the generic classifier's evaluate function
    eval_results = evaluate_generic_classifier(
        model, X_test, y_test, task_type='binary', use_optimal_threshold=False
    )

    # If we have probabilities, re-calculate predictions with custom threshold
    if 'probabilities' in eval_results:
        y_proba = eval_results['probabilities']
        y_pred = (y_proba > threshold).astype(int)

        # Recalculate metrics with new predictions
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Calculate TPR, FPR from confusion matrix
        if len(np.unique(y_test)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            # Keep original AUROC and AUPRC (threshold-independent)
            auroc = eval_results.get('auroc', float('nan'))
            auprc = eval_results.get('auprc', float('nan'))
        else:
            tpr = float('nan')
            fpr = float('nan')
            auroc = float('nan')
            auprc = float('nan')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'auroc': auroc,
            'auprc': auprc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    else:
        # Fallback to original results if no probabilities available
        return eval_results

def create_balanced_training_set():
    """
    Create balanced training set (2,000 examples, 1:1 ratio)
    
    Benign (1,000):
    - Alpaca: 500 examples (text instruction-following)
    - MM-Vet: 218 examples (multimodal reasoning)
    - OpenAssistant: 282 examples (high-quality assistant responses)
    
    Malicious (1,000):
    - AdvBench: 450 examples (harmful prompts)
    - JailbreakV-28K: 400 examples (multimodal attacks)
    - DAN variants: 150 examples ("Do Anything Now" variants)
    """
    print("Creating balanced training set...")
    
    benign_samples = []
    malicious_samples = []
    
    # Benign samples
    try:
        # Alpaca - 500 examples
        alpaca_samples = load_alpaca(max_samples=500)
        benign_samples.extend(alpaca_samples)
        print(f"Added {len(alpaca_samples)} Alpaca samples")
    except Exception as e:
        print(f"Could not load Alpaca: {e}")
    
    try:
        # MM-Vet - 218 examples (use first 218 for training)
        mmvet_samples = load_mm_vet()
        mmvet_subset = mmvet_samples[:218] if len(mmvet_samples) >= 218 else mmvet_samples
        benign_samples.extend(mmvet_subset)
        print(f"Added {len(mmvet_subset)} MM-Vet samples (reserved first 218 for training)")
    except Exception as e:
        print(f"Could not load MM-Vet: {e}")
    
    try:
        # OpenAssistant - 282 examples
        openassistant_samples = load_openassistant(max_samples=282)
        benign_samples.extend(openassistant_samples)
        print(f"Added {len(openassistant_samples)} OpenAssistant samples")
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")

    print("-"*80)
    
    # Malicious samples
    try:
        # AdvBench - 300 examples
        advbench_samples = load_advbench(max_samples=300)
        malicious_samples.extend(advbench_samples)
        print(f"Added {len(advbench_samples)} AdvBench samples")
    except Exception as e:
        print(f"Could not load AdvBench: {e}")
    
    try:
        # JailbreakV-28K - 550 samples (using llm_transfer_attack and query_related for training)
        print("  Loading JailbreakV-28K with controlled attack type distribution...")

        # Load llm_transfer_attack samples (275 samples)
        llm_attack_samples = load_JailBreakV_custom(
            attack_types=["llm_transfer_attack"],
            max_samples=275
        )

        # Load query_related samples (275 samples)
        query_related_samples = load_JailBreakV_custom(
            attack_types=["query_related"],
            max_samples=275
        )

        # Combine both attack types
        jbv_samples = []
        if llm_attack_samples:
            jbv_samples.extend(llm_attack_samples)
            print(f"    Loaded {len(llm_attack_samples)} llm_transfer_attack samples")
        if query_related_samples:
            jbv_samples.extend(query_related_samples)
            print(f"    Loaded {len(query_related_samples)} query_related samples")

        malicious_samples.extend(jbv_samples)
        print(f"Added {len(jbv_samples)} JailbreakV-28K samples ({len(llm_attack_samples)} llm_transfer + {len(query_related_samples)} query_related for training)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K: {e}")
    
    try:
        # DAN variants - 150 examples
        dan_samples = load_dan_prompts(max_samples=150)
        malicious_samples.extend(dan_samples)
        print(f"Added {len(dan_samples)} DAN variant samples")
    except Exception as e:
        print(f"Could not load DAN prompts: {e}")
    
    # Target: 1,000 benign and 1,000 malicious samples
    target_benign = 1000
    target_malicious = 1000

    # Adjust benign samples to target
    if len(benign_samples) < target_benign:
        print(f"Warning: Only {len(benign_samples)} benign samples available, target was {target_benign}")

    # Adjust malicious samples to target
    if len(malicious_samples) < target_malicious:
        print(f"Warning: Only {len(malicious_samples)} malicious samples available, target was {target_malicious}")

    print(f"Training set: {len(benign_samples)} benign, {len(malicious_samples)} malicious")
    return benign_samples, malicious_samples

def create_balanced_test_set():
    print("-"*80)
    print("Creating balanced test set...")
    
    safe_samples = []
    unsafe_samples = []
    
    # Safe samples
    try:
        # XSTest safe - 250 examples
        xstest_samples = load_XSTest()
        xstest_safe = [s for s in xstest_samples if s['toxicity'] == 0]
        xstest_safe_subset = random.sample(xstest_safe, min(250, len(xstest_safe)))
        safe_samples.extend(xstest_safe_subset)
        print(f"Added {len(xstest_safe_subset)} XSTest safe samples")
    except Exception as e:
        print(f"Could not load XSTest safe: {e}")
    
    try:
        # FigTxt safe - 300 examples
        figtxt_samples = load_FigTxt()
        figtxt_safe = [s for s in figtxt_samples if s['toxicity'] == 0]
        figtxt_safe_subset = random.sample(figtxt_safe, min(300, len(figtxt_safe)))
        safe_samples.extend(figtxt_safe_subset)
        print(f"Added {len(figtxt_safe_subset)} FigTxt safe samples")
    except Exception as e:
        print(f"Could not load FigTxt safe: {e}")
    
    try:
        # VQAv2 - 350 examples
        vqav2_samples = load_vqav2(max_samples=350)
        safe_samples.extend(vqav2_samples)
        print(f"Added {len(vqav2_samples)} VQAv2 samples")
    except Exception as e:
        print(f"Could not load VQAv2: {e}")

    print("-"*80)
    
    # Unsafe samples
    try:
        # XSTest unsafe - 200 examples
        xstest_samples = load_XSTest()
        xstest_unsafe = [s for s in xstest_samples if s['toxicity'] == 1]
        xstest_unsafe_subset = random.sample(xstest_unsafe, min(200, len(xstest_unsafe)))
        unsafe_samples.extend(xstest_unsafe_subset)
        print(f"Added {len(xstest_unsafe_subset)} XSTest unsafe samples")
    except Exception as e:
        print(f"Could not load XSTest unsafe: {e}")
    
    try:
        # FigTxt unsafe - 350 examples
        figtxt_samples = load_FigTxt()
        figtxt_unsafe = [s for s in figtxt_samples if s['toxicity'] == 1]
        figtxt_unsafe_subset = random.sample(figtxt_unsafe, min(350, len(figtxt_unsafe)))
        unsafe_samples.extend(figtxt_unsafe_subset)
        print(f"Added {len(figtxt_unsafe_subset)} FigTxt unsafe samples")
    except Exception as e:
        print(f"Could not load FigTxt unsafe: {e}")
    
    try:
        # VAE - 200 examples
        vae_samples = load_adversarial_img()
        if len(vae_samples) >= 200:
            vae_subset = random.sample(vae_samples, 200)
        else:
            # exception
            raise Exception(f"Too few VAE samples: {len(vae_samples)}")
        unsafe_samples.extend(vae_subset)
        print(f"Added {len(vae_subset)} VAE samples")
    except Exception as e:
        print(f"Could not load VAE: {e}")
    
    try:
        # JailbreakV-28K - 150 samples (figstep attack for testing)
        jbv_test_samples = load_JailBreakV_figstep(max_samples=150)
        unsafe_samples.extend(jbv_test_samples)
        print(f"Added {len(jbv_test_samples)} JailbreakV-28K samples (figstep attack for testing)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K for testing: {e}")
    
    # Target: 1000 safe and 1000 unsafe samples
    target_safe = 1000
    target_unsafe = 1000

    # Adjust safe samples to target
    if len(safe_samples) > target_safe:
        safe_samples = random.sample(safe_samples, target_safe)
    elif len(safe_samples) < target_safe:
        print(f"Warning: Only {len(safe_samples)} safe samples available, target was {target_safe}")

    # Adjust unsafe samples to target
    if len(unsafe_samples) > target_unsafe:
        unsafe_samples = random.sample(unsafe_samples, target_unsafe)
    elif len(unsafe_samples) < target_unsafe:
        print(f"Warning: Only {len(unsafe_samples)} unsafe samples available, target was {target_unsafe}")

    print(f"Test set: {len(safe_samples)} safe, {len(unsafe_samples)} unsafe")
    return safe_samples, unsafe_samples

def prepare_balanced_datasets_organized():
    """
    Load balanced datasets in organized format for dataset-specific caching.
    Returns training and test datasets as dictionaries with dataset names as keys.
    This matches the approach used in other balanced_* scripts.

    IMPORTANT: Train/Test Separation for JailbreakV-28K:
    - Training: Uses llm_transfer_attack (275) + query_related (275) = 550 samples total
    - Testing: Uses figstep attack only (150 samples)
    This ensures different attack types between training and testing for robust evaluation.
    Controlled distribution ensures exactly 275 samples from each training attack type.
    """
    print("Loading balanced datasets in organized format...")
    print("Note: JailbreakV-28K uses different attack types for train/test separation:")
    print("  - Training: llm_transfer_attack (275) + query_related (275) = 550 total")
    print("  - Testing: figstep attack only (150 samples)")

    # === TRAINING DATASETS ===
    training_datasets = {}

    # Benign training data
    print("Loading benign training data...")

    # 1. Alpaca - 500 samples
    try:
        alpaca_samples = load_alpaca(max_samples=500)
        if alpaca_samples:
            training_datasets["Alpaca"] = alpaca_samples
            print(f"  Loaded {len(alpaca_samples)} Alpaca samples")
    except Exception as e:
        print(f"Could not load Alpaca: {e}")

    # 2. MM-Vet - 218 samples
    try:
        mmvet_samples = load_mm_vet()
        if mmvet_samples:
            # Limit to 218 samples and ensure they are benign
            mmvet_benign = [s for s in mmvet_samples if s.get('toxicity', 0) == 0][:218]
            if mmvet_benign:
                training_datasets["MM-Vet"] = mmvet_benign
                print(f"  Loaded {len(mmvet_benign)} MM-Vet samples")
    except Exception as e:
        print(f"Could not load MM-Vet: {e}")

    # 3. OpenAssistant - 282 samples
    try:
        openassistant_samples = load_openassistant(max_samples=282)
        if openassistant_samples:
            training_datasets["OpenAssistant"] = openassistant_samples
            print(f"  Loaded {len(openassistant_samples)} OpenAssistant samples")
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")

    # Malicious training data
    print("Loading malicious training data...")

    # 1. AdvBench - 300 samples
    try:
        advbench_samples = load_advbench(max_samples=300)
        if advbench_samples:
            training_datasets["AdvBench"] = advbench_samples
            print(f"  Loaded {len(advbench_samples)} AdvBench samples")
    except Exception as e:
        print(f"Could not load AdvBench: {e}")

    # 2. JailbreakV-28K - 550 samples (using llm_transfer_attack and query_related for training)
    try:
        # Use separate calls for controlled distribution to ensure we know exactly how many
        # samples come from each attack type (275 from each attack type)
        print("  Loading JailbreakV-28K with controlled attack type distribution...")

        # Load llm_transfer_attack samples (275 samples)
        llm_attack_samples = load_JailBreakV_custom(
            attack_types=["llm_transfer_attack"],
            max_samples=275
        )

        # Load query_related samples (275 samples)
        query_related_samples = load_JailBreakV_custom(
            attack_types=["query_related"],
            max_samples=275
        )

        # Combine both attack types
        jbv_samples = []
        if llm_attack_samples:
            jbv_samples.extend(llm_attack_samples)
            print(f"    Loaded {len(llm_attack_samples)} llm_transfer_attack samples")
        if query_related_samples:
            jbv_samples.extend(query_related_samples)
            print(f"    Loaded {len(query_related_samples)} query_related samples")

        if jbv_samples:
            training_datasets["JailbreakV-28K"] = jbv_samples
            print(f"  Total JailbreakV-28K training samples: {len(jbv_samples)} ({len(llm_attack_samples)} llm_transfer + {len(query_related_samples)} query_related)")
        else:
            print("  Warning: No JailbreakV-28K samples loaded")

    except Exception as e:
        print(f"Could not load JailbreakV-28K: {e}")

    # 3. DAN variants - 150 samples
    try:
        dan_samples = load_dan_prompts(max_samples=150)  # Use the correct function name
        if dan_samples:
            training_datasets["DAN"] = dan_samples
            print(f"  Loaded {len(dan_samples)} DAN samples")
    except Exception as e:
        print(f"Could not load DAN variants: {e}")

    # === TEST DATASETS ===
    test_datasets = {}

    # Safe test data
    print("Loading safe test data...")

    # 1. XSTest safe - 250 samples
    try:
        xstest_samples = load_XSTest()  # Use the correct function name
        if xstest_samples:
            xstest_safe = [s for s in xstest_samples if s.get('toxicity', 0) == 0][:250]
            if xstest_safe:
                test_datasets["XSTest_safe"] = xstest_safe
                print(f"  Loaded {len(xstest_safe)} XSTest safe samples")
    except Exception as e:
        print(f"Could not load XSTest: {e}")

    # 2. FigTxt safe - 300 samples
    try:
        figtxt_samples = load_FigTxt()  # Use the correct function name
        if figtxt_samples:
            figtxt_safe = [s for s in figtxt_samples if s.get('toxicity', 0) == 0][:300]
            if figtxt_safe:
                test_datasets["FigTxt_safe"] = figtxt_safe
                print(f"  Loaded {len(figtxt_safe)} FigTxt safe samples")
    except Exception as e:
        print(f"Could not load FigTxt: {e}")

    # 3. VQAv2 - 350 samples
    try:
        vqav2_samples = load_vqav2(max_samples=350)
        if vqav2_samples:
            test_datasets["VQAv2"] = vqav2_samples
            print(f"  Loaded {len(vqav2_samples)} VQAv2 samples")
    except Exception as e:
        print(f"Could not load VQAv2: {e}")

    # Unsafe test data
    print("Loading unsafe test data...")

    # 1. XSTest unsafe - 200 samples
    try:
        if 'xstest_samples' in locals():
            xstest_unsafe = [s for s in xstest_samples if s.get('toxicity', 0) == 1][:200]
            if xstest_unsafe:
                test_datasets["XSTest_unsafe"] = xstest_unsafe
                print(f"  Loaded {len(xstest_unsafe)} XSTest unsafe samples")
    except Exception as e:
        print(f"Could not load XSTest unsafe: {e}")

    # 2. FigTxt unsafe - 350 samples
    try:
        if 'figtxt_samples' in locals():
            figtxt_unsafe = [s for s in figtxt_samples if s.get('toxicity', 0) == 1][:350]
            if figtxt_unsafe:
                test_datasets["FigTxt_unsafe"] = figtxt_unsafe
                print(f"  Loaded {len(figtxt_unsafe)} FigTxt unsafe samples")
    except Exception as e:
        print(f"Could not load FigTxt unsafe: {e}")

    # 3. VAE - 200 samples
    try:
        vae_samples = load_adversarial_img()  # Use the correct function name
        if vae_samples:
            # Limit to 200 samples
            vae_limited = vae_samples[:200]
            test_datasets["VAE"] = vae_limited
            print(f"  Loaded {len(vae_limited)} VAE samples")
    except Exception as e:
        print(f"Could not load VAE: {e}")

    # 4. JailbreakV-28K test - 150 samples (using figstep attack for testing)
    try:
        # Use figstep attack for testing to ensure different attack type than training
        jbv_test_samples = load_JailBreakV_figstep(max_samples=150)
        if jbv_test_samples:
            test_datasets["JailbreakV-28K_test"] = jbv_test_samples
            print(f"  Loaded {len(jbv_test_samples)} JailbreakV-28K test samples (figstep attack)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K test: {e}")

    return training_datasets, test_datasets

def save_layer_rankings(results, layers, model_types):
    """Save layer rankings for each method based on AUROC performance"""

    # Create rankings for each method
    method_rankings = {}

    for model_type in model_types:
        # Get AUROC scores for this method across all layers
        layer_aurocs = []
        for layer_idx in layers:
            result = results[layer_idx][model_type]
            auroc = result['auroc']
            # Handle NaN values by assigning a very low score
            if np.isnan(auroc):
                auroc = 0.0
            layer_aurocs.append((layer_idx, auroc))

        # Sort by AUROC (descending)
        layer_aurocs.sort(key=lambda x: x[1], reverse=True)
        method_rankings[model_type] = layer_aurocs

    # Save rankings to CSV
    rankings_path = "results/balanced_ml_results_ranking.csv"
    with open(rankings_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Method", "Rank", "Layer", "AUROC", "Accuracy", "F1", "TPR", "FPR", "AUPRC"])

        for model_type in model_types:
            for rank, (layer_idx, auroc) in enumerate(method_rankings[model_type], 1):
                result = results[layer_idx][model_type]

                # Format values
                auroc_str = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
                auprc_str = "N/A" if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"
                f1_str = f"{result['f1']:.4f}"
                tpr_str = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
                fpr_str = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"

                writer.writerow([
                    model_type.upper(),
                    rank,
                    layer_idx,
                    auroc_str,
                    f"{result['accuracy']:.4f}",
                    f1_str,
                    tpr_str,
                    fpr_str,
                    auprc_str
                ])

    print(f"Layer rankings saved to {rankings_path}")

    # Print rankings summary
    print("\n" + "="*80)
    print("LAYER RANKINGS BY METHOD (Based on AUROC)")
    print("="*80)

    for model_type in model_types:
        print(f"\n{model_type.upper()} Method - Top 5 Layers:")
        print("  Rank | Layer | AUROC  | Accuracy | F1     | TPR    | FPR    | AUPRC")
        print("  -----|-------|--------|----------|--------|--------|--------|--------")

        for rank, (layer_idx, _) in enumerate(method_rankings[model_type][:5], 1):
            result = results[layer_idx][model_type]
            auroc_str = "N/A   " if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
            auprc_str = "N/A   " if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"
            f1_str = f"{result['f1']:.4f}"
            tpr_str = "N/A   " if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
            fpr_str = "N/A   " if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"

            print(f"   {rank:2d}  |  {layer_idx:2d}   | {auroc_str} |  {result['accuracy']:.4f}  | {f1_str} | {tpr_str} | {fpr_str} | {auprc_str}")

    # Cross-method comparison: which method performs best at each layer
    print(f"\n" + "="*100)
    print("BEST METHOD FOR EACH LAYER (Based on AUROC)")
    print("="*100)
    print("Layer | Best Method | AUROC  | Accuracy | F1     | TPR    | FPR    | Runner-up Method | AUROC")
    print("------|-------------|--------|----------|--------|--------|--------|------------------|--------")

    for layer_idx in layers:
        # Get all methods' performance for this layer
        layer_performance = []
        for model_type in model_types:
            result = results[layer_idx][model_type]
            auroc = result['auroc'] if not np.isnan(result['auroc']) else 0.0
            layer_performance.append((model_type, auroc, result['accuracy'], result['f1'], result['tpr'], result['fpr']))

        # Sort by AUROC
        layer_performance.sort(key=lambda x: x[1], reverse=True)

        best_method, best_auroc, best_acc, best_f1, best_tpr, best_fpr = layer_performance[0]
        runner_method, runner_auroc, _, _, _, _ = layer_performance[1]

        best_auroc_str = "N/A   " if best_auroc == 0.0 else f"{best_auroc:.4f}"
        runner_auroc_str = "N/A   " if runner_auroc == 0.0 else f"{runner_auroc:.4f}"
        best_f1_str = f"{best_f1:.4f}"
        best_tpr_str = "N/A   " if np.isnan(best_tpr) else f"{best_tpr:.4f}"
        best_fpr_str = "N/A   " if np.isnan(best_fpr) else f"{best_fpr:.4f}"

        print(f" {layer_idx:2d}   | {best_method.upper():11s} | {best_auroc_str} |  {best_acc:.4f}  | {best_f1_str} | {best_tpr_str} | {best_fpr_str} | {runner_method.upper():16s} | {runner_auroc_str}")

def main():
    model_path = "model/llava-v1.6-vicuna-7b/"

    print("="*80)
    print("BALANCED JAILBREAK DETECTION WITH NEW DATASET CONFIGURATION")
    print("="*80)

    # Set random seed for reproducibility (use centralized seed)
    MAIN_SEED = CONFIG.MAIN_SEED
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)

    # Additional determinism settings
    os.environ['PYTHONHASHSEED'] = str(MAIN_SEED)

    CONFIG.print_config()
    print(f"Random seeds set for reproducibility (seed={MAIN_SEED})")
    
    # Load datasets using the same organized approach as other balanced_* scripts
    training_datasets, test_datasets = prepare_balanced_datasets_organized()

    # Initialize feature extractor
    extractor = HiddenStateExtractor(model_path)

    # Extract hidden states for all datasets individually (with dataset-specific caching)
    print("\n--- Extracting Hidden States with Dataset-Specific Caching ---")
    all_datasets = {**training_datasets, **test_datasets}
    all_hidden_states = {}
    all_labels = {}

    for dataset_name, samples in all_datasets.items():
        if len(samples) > 0:  # Only process non-empty datasets
            print(f"Extracting features for {dataset_name} ({len(samples)} samples)...")

            # Use smaller batch sizes for large datasets to manage memory
            batch_size = 25 if len(samples) > 5000 else 50
            memory_cleanup_freq = 5 if len(samples) > 5000 else 10

            hidden_states, labels, _ = extractor.extract_hidden_states(
                samples, f"{dataset_name}", layer_start=0, layer_end=31, use_cache=True,
                batch_size=batch_size, memory_cleanup_freq=memory_cleanup_freq,
                experiment_name="balanced_ml_detection"
            )
            all_hidden_states[dataset_name] = hidden_states
            all_labels[dataset_name] = labels
        else:
            print(f"Skipping {dataset_name} - no samples found")

    # Combine training and test data from individual datasets
    print("\n--- Combining Individual Dataset Features ---")
    train_hidden_states_dict = {}
    test_hidden_states_dict = {}
    train_labels = []
    test_labels = []

    # Initialize layer dictionaries
    for layer_idx in range(32):
        train_hidden_states_dict[layer_idx] = []
        test_hidden_states_dict[layer_idx] = []

    # Combine training datasets
    training_dataset_names = ["Alpaca", "MM-Vet", "OpenAssistant", "AdvBench", "JailbreakV-28K", "DAN"]
    for dataset_name in training_dataset_names:
        if dataset_name in all_hidden_states:
            dataset_labels = all_labels[dataset_name]
            train_labels.extend(dataset_labels)

            for layer_idx in range(32):
                layer_features = all_hidden_states[dataset_name][layer_idx]
                train_hidden_states_dict[layer_idx].extend(layer_features)

    # Combine test datasets
    test_dataset_names = ["XSTest_safe", "FigTxt_safe", "VQAv2", "XSTest_unsafe", "FigTxt_unsafe", "VAE", "JailbreakV-28K_test"]
    for dataset_name in test_dataset_names:
        if dataset_name in all_hidden_states:
            dataset_labels = all_labels[dataset_name]
            test_labels.extend(dataset_labels)

            for layer_idx in range(32):
                layer_features = all_hidden_states[dataset_name][layer_idx]
                test_hidden_states_dict[layer_idx].extend(layer_features)

    # Convert to numpy arrays
    for layer_idx in range(32):
        train_hidden_states_dict[layer_idx] = np.array(train_hidden_states_dict[layer_idx])
        test_hidden_states_dict[layer_idx] = np.array(test_hidden_states_dict[layer_idx])

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    print(f"Combined training features: {len(train_labels)} samples ({np.sum(train_labels == 0)} benign, {np.sum(train_labels == 1)} malicious)")
    print(f"Combined test features: {len(test_labels)} samples ({np.sum(test_labels == 0)} safe, {np.sum(test_labels == 1)} unsafe)")
    
    # Train and evaluate models for each layer and model type
    results = {}
    layers = list(range(0, 32))  # layers 0-31
    model_types = ['mlp', 'logistic', 'ridge', 'svm', 'sgd']

    print("\n--- Training and Evaluating Models ---")
    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")

        X_train = np.array(train_hidden_states_dict[layer_idx])
        y_train = np.array(train_labels)
        X_test = np.array(test_hidden_states_dict[layer_idx])
        y_test = np.array(test_labels)

        input_dim = X_train.shape[1]
        results[layer_idx] = {}

        for model_type in model_types:
            print(f"  Training {model_type.upper()} for layer {layer_idx}...")

            if model_type == 'mlp':
                # Train MLP using generic classifier WITHOUT threshold optimization
                model = train_generic_classifier(
                    X_train, y_train,
                    input_dim=input_dim, output_dim=1,
                    hidden_dims=[128, 16], dropout_rate=0.3,
                    epochs=30, batch_size=32, lr=1e-3,
                    task_type='binary', verbose=False, optimize_threshold=False
                )

                # Use unified threshold optimization algorithm
                print(f"    Optimizing threshold for {model_type.upper()}...")
                threshold_optimizer = ThresholdOptimizer()
                optimal_threshold = threshold_optimizer.fit_threshold(model, X_train, y_train, model_type='mlp')

                # Evaluate MLP model on test set using optimal threshold
                eval_results = evaluate_mlp_classifier_with_threshold(
                    model, X_test, y_test, threshold=optimal_threshold
                )
                eval_results['threshold'] = optimal_threshold

            else:
                # Train linear classifier
                model = train_linear_classifier(X_train, y_train, model_type)

                # Use unified threshold optimization algorithm
                print(f"    Optimizing threshold for {model_type.upper()}...")
                threshold_optimizer = ThresholdOptimizer()
                optimal_threshold = threshold_optimizer.fit_threshold(model, X_train, y_train, model_type='linear')

                # Evaluate linear classifier using optimal threshold
                eval_results = evaluate_linear_classifier(model, X_test, y_test, threshold=optimal_threshold)
                eval_results['threshold'] = optimal_threshold

            results[layer_idx][model_type] = eval_results

            # Handle NaN values for display
            f1_str = f"{eval_results['f1']:.4f}"
            tpr_str = "N/A" if np.isnan(eval_results['tpr']) else f"{eval_results['tpr']:.4f}"
            fpr_str = "N/A" if np.isnan(eval_results['fpr']) else f"{eval_results['fpr']:.4f}"
            auroc_str = "N/A" if np.isnan(eval_results['auroc']) else f"{eval_results['auroc']:.4f}"
            auprc_str = "N/A" if np.isnan(eval_results['auprc']) else f"{eval_results['auprc']:.4f}"

            print(f"    {model_type.upper()} - Accuracy: {eval_results['accuracy']:.4f}, F1: {f1_str}, TPR: {tpr_str}, FPR: {fpr_str}, AUROC: {auroc_str}, AUPRC: {auprc_str}, Thresh: {eval_results['threshold']:.4f}")

    # Calculate total dataset sizes
    total_train_size = len(train_labels)
    total_test_size = len(test_labels)

    # Save results and generate summary
    save_results_and_summary(results, layers, model_types, total_train_size, total_test_size)

def save_results_and_summary(results, layers, model_types, train_size, test_size):
    """Save results to CSV and print summary"""
    # Save results to CSV
    output_path = "results/balanced_ml_results.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Layer", "Model_Type", "Test_Accuracy", "Test_F1", "Test_TPR", "Test_FPR", "Test_AUROC", "Test_AUPRC", "Threshold", "Train_Size", "Test_Size"])

        for layer_idx in layers:
            for model_type in model_types:
                result = results[layer_idx][model_type]

                # Handle NaN values for CSV
                f1_val = f"{result['f1']:.4f}"
                tpr_val = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
                fpr_val = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"
                auroc_val = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
                auprc_val = "N/A" if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"
                threshold_val = f"{result['threshold']:.4f}"

                writer.writerow([
                    layer_idx,
                    model_type.upper(),
                    f"{result['accuracy']:.4f}",
                    f1_val,
                    tpr_val,
                    fpr_val,
                    auroc_val,
                    auprc_val,
                    threshold_val,
                    train_size,
                    test_size
                ])

    print(f"\nResults saved to {output_path}")

    # Save layer rankings for each method
    save_layer_rankings(results, layers, model_types)

    # Print summary - find best performing model across all layers and types
    best_performance = []
    for layer_idx in layers:
        for model_type in model_types:
            result = results[layer_idx][model_type]
            best_performance.append((layer_idx, model_type, result['accuracy']))

    best_performance.sort(key=lambda x: x[2], reverse=True)
    best_layer, best_model, best_accuracy = best_performance[0]

    print("\n" + "="*80)
    print("BALANCED JAILBREAK DETECTION SUMMARY")
    print("="*80)
    print(f"Dataset Configuration: Balanced train-test split with strict separation")
    print(f"Training Set: {train_size} samples (1:1 benign:malicious ratio)")
    print(f"Test Set: {test_size} samples (1:1 safe:unsafe ratio)")
    print(f"Best Overall Performance: Layer {best_layer} with {best_model.upper()} (Accuracy: {best_accuracy:.4f})")
    print(f"Layers analyzed: {len(layers)} (layers {min(layers)}-{max(layers)})")
    print(f"Models compared: {', '.join([m.upper() for m in model_types])}")

    print(f"\nTop 10 Performing Combinations:")
    for i, (layer_idx, model_type, accuracy) in enumerate(best_performance[:10], 1):
        result = results[layer_idx][model_type]
        auroc_str = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
        f1_str = f"{result['f1']:.4f}"
        tpr_str = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
        fpr_str = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"
        print(f"  {i:2d}. Layer {layer_idx} {model_type.upper()}: Acc={accuracy:.4f}, F1={f1_str}, TPR={tpr_str}, FPR={fpr_str}, AUROC={auroc_str}")

    # Print model type comparison
    print(f"\nModel Type Performance Summary:")
    model_avg_performance = {}
    for model_type in model_types:
        accuracies = [results[layer_idx][model_type]['accuracy'] for layer_idx in layers]
        aurocs = [results[layer_idx][model_type]['auroc'] for layer_idx in layers if not np.isnan(results[layer_idx][model_type]['auroc'])]
        model_avg_performance[model_type] = np.mean(accuracies)

        auroc_avg = np.mean(aurocs) if aurocs else 0.0
        auroc_std = np.std(aurocs) if aurocs else 0.0

        print(f"  {model_type.upper()}: Avg Accuracy = {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}, Avg AUROC = {auroc_avg:.4f} ± {auroc_std:.4f}")

    print("\nDetailed layer rankings and cross-method comparisons are shown above.")

    # Save results to CSV for multi-run compatibility
    output_path = "results/balanced_ml_results.csv"
    os.makedirs("results", exist_ok=True)

    # Calculate rankings based on accuracy (primary metric for ML methods)
    # Create list of (layer, model_type, accuracy) for ranking
    performance_list = []
    for layer_idx in layers:
        for model_type in model_types:
            result = results[layer_idx][model_type]
            performance_list.append((layer_idx, model_type, result['accuracy']))

    # Sort by accuracy (descending) for ranking
    performance_list.sort(key=lambda x: x[2], reverse=True)

    # Create ranking mappings
    combined_ranks = {}  # (layer, model) -> rank
    individual_ranks = {}  # (layer, model) -> rank

    # Combined ranking (overall ranking across all layer-model combinations)
    for rank, (layer_idx, model_type, _) in enumerate(performance_list, 1):
        combined_ranks[(layer_idx, model_type)] = rank

    # Individual ranking per layer (rank models within each layer)
    for layer_idx in layers:
        layer_performance = [(model_type, results[layer_idx][model_type]['accuracy'])
                           for model_type in model_types]
        layer_performance.sort(key=lambda x: x[1], reverse=True)

        for rank, (model_type, _) in enumerate(layer_performance, 1):
            individual_ranks[(layer_idx, model_type)] = rank

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Layer', 'Dataset', 'Method', 'Accuracy', 'F1', 'TPR', 'FPR', 'AUROC', 'AUPRC', 'Threshold', 'Combined_Rank', 'Individual_Rank']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for layer_idx in layers:
            for model_type in model_types:
                result = results[layer_idx][model_type]
                writer.writerow({
                    'Layer': layer_idx,
                    'Dataset': 'COMBINED',  # ML script uses combined balanced dataset
                    'Method': f'ML_{model_type.upper()}',
                    'Accuracy': f"{result['accuracy']:.6f}",
                    'F1': f"{result['f1']:.6f}",
                    'TPR': f"{result['tpr']:.6f}" if not np.isnan(result['tpr']) else "N/A",
                    'FPR': f"{result['fpr']:.6f}" if not np.isnan(result['fpr']) else "N/A",
                    'AUROC': f"{result['auroc']:.6f}" if not np.isnan(result['auroc']) else "N/A",
                    'AUPRC': f"{result.get('auprc', 0.0):.6f}",  # Default to 0 if not available
                    'Threshold': f"{result.get('threshold', 0.5):.6f}",  # Default threshold
                    'Combined_Rank': combined_ranks[(layer_idx, model_type)],
                    'Individual_Rank': individual_ranks[(layer_idx, model_type)]
                })

    print(f"Results saved to: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
