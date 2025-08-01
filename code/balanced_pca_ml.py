import csv
import numpy as np
import random
import warnings
import signal
import sys
import os
from scipy.linalg import inv
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from scipy.spatial.distance import cdist

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

from load_datasets import *
from feature_extractor import HiddenStateExtractor

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
class PCAConfig:
    """Global configuration for PCA projection methods"""

    # Random seed management (centralized for run_multiple_experiments.py compatibility)
    MAIN_SEED = 42  # Main seed for reproducibility - modified by run_multiple_experiments.py

    # PCA projection mode
    # "single_layer": Train one PCA on layer 16, use for all layers
    # "layer_specific": Train separate PCA for each layer
    PROJECTION_MODE = "layer_specific"  # Change this to switch modes

    # Single layer mode settings
    SINGLE_LAYER_TRAINING_LAYER = 18  # Which layer to use for training PCA

    # PCA settings
    INPUT_DIM = 4096
    # Test multiple PCA component numbers (reduced for faster GPU processing)
    PCA_COMPONENTS = [64, 128, 256, 512]  # Different dimensionalities to test

    # Classifier settings - GPU-only options
    CLASSIFIERS_TO_TEST = ["MLP", "SVM"]  # Test both MLP and SVM on GPU
    # CLASSIFIERS_TO_TEST = ["MLP"]       # Test only MLP
    # CLASSIFIERS_TO_TEST = ["SVM"]       # Test only SVM

    # SVM settings (reduced for GPU efficiency)
    SVM_KERNEL_OPTIONS = ['rbf', 'linear']  # Reduced for faster GPU training
    SVM_C_OPTIONS = [0.1, 1, 10]           # Reduced grid
    SVM_GAMMA_OPTIONS = ['scale', 'auto']   # Reduced grid

    @classmethod
    def set_seed(cls, seed):
        """Set the main seed for reproducibility"""
        cls.MAIN_SEED = seed
        print(f"PCAConfig: Main seed set to {seed}")

    @classmethod
    def get_layer_seed(cls, layer_idx):
        """Get layer-specific seed for deterministic but different seeds per layer"""
        return cls.MAIN_SEED + layer_idx

    @classmethod
    def get_svm_seed(cls, layer_idx, n_components):
        """Get SVM-specific seed for deterministic training"""
        return cls.MAIN_SEED + layer_idx + n_components

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*80)
        print("PCA PROJECTION CONFIGURATION")
        print("="*80)
        print(f"Main seed: {cls.MAIN_SEED}")
        print(f"Mode: {cls.PROJECTION_MODE}")
        if cls.PROJECTION_MODE == "single_layer":
            print(f"Training layer: {cls.SINGLE_LAYER_TRAINING_LAYER}")
        print(f"Input dimension: {cls.INPUT_DIM}")
        print(f"PCA components to test: {cls.PCA_COMPONENTS}")
        print(f"GPU classifiers to test: {cls.CLASSIFIERS_TO_TEST}")
        print(f"SVM kernels: {cls.SVM_KERNEL_OPTIONS}")
        print(f"SVM C values: {cls.SVM_C_OPTIONS}")
        print(f"SVM gamma values: {cls.SVM_GAMMA_OPTIONS}")
        print("="*80)

# Global config instance
CONFIG = PCAConfig()

def setup_gpu_environment():
    if torch.cuda.is_available():
        # Use both GPUs if available
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA devices")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")

        # Set memory fraction to avoid OOM (RTX 4090 has 24GB)
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory for better utilization

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Enable memory pool for faster allocation
        torch.cuda.empty_cache()

        # Set up multi-GPU if available
        if device_count > 1:
            print(f"Multi-GPU setup: Using {device_count} GPUs")
            return [torch.device(f'cuda:{i}') for i in range(device_count)]
        else:
            return torch.device('cuda:0')  # Primary GPU
    else:
        print("CUDA not available, falling back to CPU")
        return torch.device('cpu')

# Initialize GPU
GPU_DEVICES = setup_gpu_environment()
# Primary device for single operations
GPU_DEVICE = GPU_DEVICES[0] if isinstance(GPU_DEVICES, list) else GPU_DEVICES
# Check if we have multiple GPUs
MULTI_GPU = isinstance(GPU_DEVICES, list) and len(GPU_DEVICES) > 1

def cleanup_gpu_memory():
    """Clean up GPU memory across all available GPUs"""
    if torch.cuda.is_available():
        # Clean up all GPUs
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Force garbage collection
        import gc
        gc.collect()

def get_gpu_memory_info():
    """Get GPU memory usage information for all GPUs"""
    if torch.cuda.is_available():
        info_lines = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                info_lines.append(f"GPU {i}: {allocated:.1f}GB/{total:.1f}GB allocated, {reserved:.1f}GB reserved")
        return "\n".join(info_lines)
    return "GPU not available"

def monitor_gpu_usage(operation_name="Operation"):
    """Monitor and log GPU usage before/after operations"""
    if torch.cuda.is_available():
        print(f"  {operation_name} - GPU Memory Status:")
        print(f"    {get_gpu_memory_info()}")

def optimize_gpu_batch_size(data_size, feature_dim, base_batch_size=1000):
    """Dynamically optimize batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return min(base_batch_size, data_size)

    # Estimate memory usage per sample (in bytes)
    # Feature tensor (float32) + gradients + intermediate activations
    memory_per_sample = feature_dim * 4 * 3  # Conservative estimate

    # Get available GPU memory (use 70% to be safe)
    available_memory = torch.cuda.get_device_properties(0).total_memory * 0.7

    # Calculate optimal batch size
    optimal_batch_size = int(available_memory / memory_per_sample)
    optimal_batch_size = min(optimal_batch_size, data_size, base_batch_size * 4)  # Cap at 4x base
    optimal_batch_size = max(optimal_batch_size, 32)  # Minimum batch size

    return optimal_batch_size

def gpu_standardize_features(X, device=None):
    """GPU-accelerated feature standardization"""
    if device is None:
        device = GPU_DEVICE

    if not torch.cuda.is_available():
        # Fallback to CPU sklearn StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler

    # Convert to GPU tensor
    X_tensor = torch.FloatTensor(X).to(device)

    # Compute mean and std on GPU
    mean = torch.mean(X_tensor, dim=0, keepdim=True)
    std = torch.std(X_tensor, dim=0, keepdim=True)

    # Avoid division by zero
    std = torch.where(std == 0, torch.ones_like(std), std)

    # Standardize
    X_standardized = (X_tensor - mean) / std

    # Return CPU numpy array and parameters for later use
    return X_standardized.cpu().numpy(), {'mean': mean.cpu().numpy(), 'std': std.cpu().numpy()}

def gpu_apply_standardization(X, scaler_params, device=None):
    """Apply pre-computed standardization parameters on GPU"""
    if device is None:
        device = GPU_DEVICE

    if not torch.cuda.is_available() or 'mean' not in scaler_params:
        # Fallback to sklearn
        return scaler_params.transform(X) if hasattr(scaler_params, 'transform') else X

    # Convert to GPU tensor
    X_tensor = torch.FloatTensor(X).to(device)
    mean_tensor = torch.FloatTensor(scaler_params['mean']).to(device)
    std_tensor = torch.FloatTensor(scaler_params['std']).to(device)

    # Apply standardization
    X_standardized = (X_tensor - mean_tensor) / std_tensor

    return X_standardized.cpu().numpy()

def gpu_batch_distance_computation(X1, X2, metric='euclidean', device=None):
    """GPU-accelerated batch distance computation"""
    if device is None:
        device = GPU_DEVICE

    if not torch.cuda.is_available():
        # CPU fallback
        return cdist(X1, X2, metric=metric)  # type: ignore

    # Convert to GPU tensors
    X1_tensor = torch.FloatTensor(X1).to(device)
    X2_tensor = torch.FloatTensor(X2).to(device)

    if metric == 'euclidean':
        # Efficient batch Euclidean distance computation
        distances = torch.cdist(X1_tensor, X2_tensor, p=2)
    elif metric == 'cosine':
        # Cosine distance computation
        X1_norm = F.normalize(X1_tensor, p=2, dim=1)
        X2_norm = F.normalize(X2_tensor, p=2, dim=1)
        cosine_sim = torch.mm(X1_norm, X2_norm.t())
        distances = 1 - cosine_sim
    else:
        # Fallback to CPU for unsupported metrics
        return cdist(X1, X2, metric=metric)  # type: ignore  # type: ignore

    return distances.cpu().numpy()

class GPUPCAProjection:
    """
    GPU-accelerated PCA-based dimensionality reduction from 4096 to various target dimensions.
    Uses PyTorch for GPU acceleration when available.
    """

    def __init__(self, n_components=256, random_state=42, use_gpu=True):
        self.n_components = n_components
        self.random_state = random_state
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = GPU_DEVICE if self.use_gpu else torch.device('cpu')
        self.is_fitted = False

        # GPU tensors for PCA components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.scaler_params = None

    def fit(self, X):
        """Fit PCA on training data with GPU acceleration"""
        print(f"    Fitting GPU-accelerated PCA with {self.n_components} components on {X.shape[0]} samples...")

        if self.use_gpu:
            self._fit_gpu(X)
        else:
            self._fit_cpu(X)

        self.is_fitted = True
        return self

    def _fit_gpu(self, X):
        """GPU-accelerated PCA fitting"""
        # GPU standardization
        X_scaled, self.scaler_params = gpu_standardize_features(X, self.device)

        # Convert to GPU tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Center the data
        self.mean_ = torch.mean(X_tensor, dim=0, keepdim=True)
        X_centered = X_tensor - self.mean_

        # Compute covariance matrix on GPU
        n_samples = X_tensor.shape[0]
        cov_matrix = torch.mm(X_centered.t(), X_centered) / (n_samples - 1)

        # Eigendecomposition on GPU
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components
        self.components_ = eigenvectors[:, :self.n_components].t()  # Shape: (n_components, n_features)
        selected_eigenvalues = eigenvalues[:self.n_components]

        # Compute explained variance ratio
        total_variance = torch.sum(eigenvalues)
        self.explained_variance_ratio_ = selected_eigenvalues / total_variance

        # Report results
        explained_var_sum = self.explained_variance_ratio_[:5].sum().item() if len(self.explained_variance_ratio_) >= 5 else self.explained_variance_ratio_.sum().item()
        cumulative_var = self.explained_variance_ratio_.sum().item()

        print(f"    GPU PCA explained variance: {explained_var_sum:.3f} (first 5 components)")
        print(f"    GPU PCA cumulative variance: {cumulative_var:.3f} (all {self.n_components} components)")

    def _fit_cpu(self, X):
        """CPU fallback PCA fitting"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        pca.fit(X_scaled)

        # Store results
        self.scaler_params = scaler
        self.mean_ = torch.FloatTensor(pca.mean_).to(self.device)
        self.components_ = torch.FloatTensor(pca.components_).to(self.device)
        self.explained_variance_ratio_ = torch.FloatTensor(pca.explained_variance_ratio_).to(self.device)

        print(f"    CPU PCA explained variance: {pca.explained_variance_ratio_[:5].sum():.3f} (first 5 components)")
        print(f"    CPU PCA cumulative variance: {pca.explained_variance_ratio_.sum():.3f} (all {self.n_components} components)")

    def transform(self, X):
        """Transform data using fitted PCA"""
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")

        if self.use_gpu:
            return self._transform_gpu(X)
        else:
            return self._transform_cpu(X)

    def _transform_gpu(self, X):
        """GPU-accelerated transformation"""
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA not fitted yet")

        # Apply standardization
        X_scaled = gpu_apply_standardization(X, self.scaler_params, self.device)

        # Convert to GPU tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Center and project
        X_centered = X_tensor - self.mean_
        X_pca = torch.mm(X_centered, self.components_.t())

        return X_pca.cpu().numpy()

    def _transform_cpu(self, X):
        """CPU transformation"""
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA not fitted yet")

        if hasattr(self.scaler_params, 'transform'):
            # sklearn StandardScaler
            X_scaled = self.scaler_params.transform(X)  # type: ignore
        else:
            # GPU scaler params
            X_scaled = gpu_apply_standardization(X, self.scaler_params, self.device)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        X_centered = X_tensor - self.mean_
        X_pca = torch.mm(X_centered, self.components_.t())
        return X_pca.cpu().numpy()

    def fit_transform(self, X):
        """Fit PCA and transform data"""
        return self.fit(X).transform(X)

    def get_explained_variance_ratio(self):
        """Get explained variance ratio"""
        if not self.is_fitted or self.explained_variance_ratio_ is None:
            return None
        return self.explained_variance_ratio_.cpu().numpy()

    def get_n_components(self):
        """Get number of components"""
        return self.n_components


# Keep CPU version as fallback
class PCAProjection:
    """CPU fallback PCA implementation"""

    def __init__(self, n_components=256, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("PCA must be fitted before transform")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_explained_variance_ratio(self):
        if not self.is_fitted:
            return None
        return self.pca.explained_variance_ratio_

    def get_n_components(self):
        return self.n_components


def train_pca_projections(features_dict, labels_dict, n_components_list=None, random_seed=42):
    """
    Train PCA projections for different component numbers

    Args:
        features_dict: Dict of {dataset_name: features_array}
        labels_dict: Dict of {dataset_name: labels_array}
        n_components_list: List of component numbers to test
        random_seed: Random seed for reproducibility

    Returns:
        pca_models: Dict of {n_components: PCAProjection}
        dataset_name_to_id: Mapping of dataset names to IDs
    """
    if n_components_list is None:
        n_components_list = CONFIG.PCA_COMPONENTS

    print(f"Training PCA projections with components: {n_components_list}")

    # Prepare training data
    all_features = []
    all_dataset_labels = []
    all_toxicity_labels = []
    dataset_name_to_id = {}

    dataset_id = 0
    for dataset_name, features in features_dict.items():
        if dataset_name not in labels_dict:
            continue

        features_array = np.array(features)
        labels_array = np.array(labels_dict[dataset_name])

        if len(features_array) != len(labels_array):
            print(f"Warning: Feature-label mismatch for {dataset_name}: {len(features_array)} vs {len(labels_array)}")
            continue

        # Assign dataset ID
        if dataset_name not in dataset_name_to_id:
            dataset_name_to_id[dataset_name] = dataset_id
            dataset_id += 1

        all_features.append(features_array)
        all_dataset_labels.extend([dataset_name_to_id[dataset_name]] * len(features_array))
        all_toxicity_labels.extend(labels_array.tolist())

        print(f"  {dataset_name}: {len(features_array)} samples, dataset_id={dataset_name_to_id[dataset_name]}")

    # Convert to arrays
    all_features = np.vstack(all_features)
    all_dataset_labels = np.array(all_dataset_labels)
    all_toxicity_labels = np.array(all_toxicity_labels)

    print(f"Total training samples: {len(all_features)}")
    print(f"Feature shape: {all_features.shape}")
    print(f"Unique datasets: {len(dataset_name_to_id)}")
    print(f"Toxicity distribution: {np.bincount(all_toxicity_labels)}")

    # Train PCA models for different component numbers
    pca_models = {}

    for n_components in n_components_list:
        print(f"\n--- Training PCA with {n_components} components ---")

        # Ensure we don't exceed the number of features or samples
        max_components = min(n_components, all_features.shape[1], all_features.shape[0] - 1)
        if max_components != n_components:
            print(f"  Adjusting components from {n_components} to {max_components}")

        # Use GPU-accelerated PCA when available
        if torch.cuda.is_available():
            pca_model = GPUPCAProjection(n_components=max_components, random_state=random_seed, use_gpu=True)
            print(f"    Using GPU-accelerated PCA")
        else:
            pca_model = PCAProjection(n_components=max_components, random_state=random_seed)
            print(f"    Using CPU PCA (GPU not available)")

        pca_model.fit(all_features)
        pca_models[n_components] = pca_model

        print(f"  PCA with {max_components} components trained successfully!")

    return pca_models, dataset_name_to_id


def apply_pca_projection(pca_model, features_dict):
    """
    Apply PCA projection to transform features

    Args:
        pca_model: Trained PCA model
        features_dict: Dict of {dataset_name: features_array}

    Returns:
        projected_features_dict: Dict of {dataset_name: projected_features_array}
    """
    projected_features_dict = {}

    print(f"Applying PCA projection with {pca_model.get_n_components()} components...")

    for dataset_name, features in features_dict.items():
        features_array = np.array(features)
        print(f"  Projecting {dataset_name}: {features_array.shape} -> ", end="")

        # Apply PCA transformation
        projected_features = pca_model.transform(features_array)
        projected_features_dict[dataset_name] = projected_features

        print(f"{projected_features.shape}")

    return projected_features_dict

def apply_learned_projection(model, features_dict, device=None):
    """
    Apply the learned projection to transform features from 4096 to 256 dimensions
    Optimized for multi-GPU usage and larger batch sizes

    Args:
        model: Trained projection model
        features_dict: Dict of {dataset_name: features_array}
        device: GPU device

    Returns:
        projected_features_dict: Dict of {dataset_name: projected_features_array}
    """
    if device is None:
        device = GPU_DEVICE

    model.eval()
    projected_features_dict = {}

    # Use DataParallel for multi-GPU if available
    if MULTI_GPU and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"  Using {torch.cuda.device_count()} GPUs for projection")

    with torch.no_grad():
        for dataset_name, features in features_dict.items():
            features_array = np.array(features)

            # Use larger batch sizes for better GPU utilization
            batch_size = min(2000, len(features_array))  # Adaptive batch size
            projected_features = []

            # Process in batches with progress indication for large datasets
            num_batches = (len(features_array) + batch_size - 1) // batch_size

            for i in range(0, len(features_array), batch_size):
                batch_features = features_array[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_features).to(device)

                # Apply projection
                batch_projected = model(batch_tensor)
                projected_features.append(batch_projected.cpu().numpy())

                # Clean up GPU memory periodically
                if i % (batch_size * 5) == 0:  # Every 5 batches
                    torch.cuda.empty_cache()

            # Combine batches efficiently
            projected_features = np.vstack(projected_features)
            projected_features_dict[dataset_name] = projected_features

    # Final cleanup
    torch.cuda.empty_cache()
    return projected_features_dict


def gpu_accelerated_data_preparation(train_features, train_labels, device=None):
    """
    GPU-accelerated data preparation and augmentation
    """
    if device is None:
        device = GPU_DEVICE

    if not torch.cuda.is_available():
        return train_features, train_labels

    # Convert to GPU tensors for processing
    features_tensor = torch.FloatTensor(train_features).to(device)
    labels_tensor = torch.LongTensor(train_labels).to(device)

    # GPU-based data augmentation (add small noise for regularization)
    if len(train_features) < 5000:  # Only for smaller datasets
        noise_std = 0.01
        noise = torch.randn_like(features_tensor) * noise_std
        augmented_features = features_tensor + noise

        # Combine original and augmented data
        features_tensor = torch.cat([features_tensor, augmented_features], dim=0)
        labels_tensor = torch.cat([labels_tensor, labels_tensor], dim=0)

    # Shuffle on GPU
    indices = torch.randperm(features_tensor.size(0), device=device)
    features_tensor = features_tensor[indices]
    labels_tensor = labels_tensor[indices]

    # Return CPU numpy arrays
    return features_tensor.cpu().numpy(), labels_tensor.cpu().numpy()

def compute_optimal_shrinkage(X, sample_cov):
    """Compute optimal shrinkage intensity using Ledoit-Wolf formula"""
    n, d = X.shape

    if n <= 1:
        return 1.0  # Full shrinkage for very small samples

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute trace of sample covariance
    trace_cov = np.trace(sample_cov)

    # Estimate optimal shrinkage intensity
    # This is a simplified version of the Ledoit-Wolf formula
    if n < d:
        # When n < d, use stronger shrinkage
        lambda_opt = min(1.0, (d - n) / (d + n))
    else:
        # Standard Ledoit-Wolf shrinkage estimation
        # Simplified formula for computational efficiency
        lambda_opt = min(1.0, (d / n) * 0.1)

    return lambda_opt

def enhanced_ledoit_wolf_covariance(X, min_samples=50):
    """Enhanced Ledoit-Wolf covariance estimator with minimum sample size validation"""
    n, d = X.shape

    if n <= 1:
        raise ValueError(f"Cannot compute covariance with only {n} sample(s). Need at least 2 samples.")

    if n < min_samples:
        raise ValueError(
            f"Cluster has only {n} samples, which is below the minimum required {min_samples} samples. "
            f"This is insufficient for reliable covariance estimation. "
            f"Consider: (1) collecting more data, (2) using fewer/broader clusters, or "
            f"(3) reducing min_samples if you understand the risks."
        )

    # Step 1: Compute sample covariance
    sample_cov = np.cov(X.T, bias=False)

    # Step 2: Compute shrinkage target (scaled identity)
    trace_cov = np.trace(sample_cov)
    target = (trace_cov / d) * np.eye(d)

    # Step 3: Enhanced shrinkage intensity (more conservative for small samples)
    lambda_opt = compute_optimal_shrinkage(X, sample_cov)
    if n < min_samples:
        lambda_opt = max(0.2, lambda_opt)  # Minimum 20% shrinkage for small samples

    # Step 4: Shrunk estimator
    shrunk_cov = (1 - lambda_opt) * sample_cov + lambda_opt * target

    return shrunk_cov

def ledoit_wolf_covariance(X):
    """Original Ledoit-Wolf covariance estimator"""
    return enhanced_ledoit_wolf_covariance(X, min_samples=50)

def enhanced_ledoit_wolf_covariance_gpu(X, min_samples=50, device=None):
    """GPU-accelerated Enhanced Ledoit-Wolf covariance estimator with minimum sample size validation"""
    if device is None:
        device = GPU_DEVICE

    n, d = X.shape
    # print(f"    GPU covariance computation: {n} samples, {d} dimensions")

    if n <= 1:
        raise ValueError(f"Cannot compute covariance with only {n} sample(s). Need at least 2 samples.")

    if n < min_samples:
        raise ValueError(
            f"Cluster has only {n} samples, which is below the minimum required {min_samples} samples. "
            f"This is insufficient for reliable covariance estimation. "
            f"Consider: (1) collecting more data, (2) using fewer/broader clusters, or "
            f"(3) reducing min_samples if you understand the risks."
        )

    # Convert to GPU tensor (use float64 for numerical consistency with CPU)
    X_gpu = torch.tensor(X, dtype=torch.float64, device=device)

    # GPU covariance computation
    X_centered = X_gpu - torch.mean(X_gpu, dim=0, keepdim=True)
    sample_cov_gpu = torch.matmul(X_centered.T, X_centered) / (n - 1)

    # Shrinkage target (scaled identity)
    trace_cov = torch.trace(sample_cov_gpu)
    target_gpu = (trace_cov / d) * torch.eye(d, device=device)

    # Enhanced shrinkage intensity (match CPU implementation exactly)
    # Convert back to CPU for shrinkage calculation to ensure consistency
    sample_cov_cpu = sample_cov_gpu.cpu().numpy()
    X_cpu = X_gpu.cpu().numpy()
    lambda_opt = compute_optimal_shrinkage(X_cpu, sample_cov_cpu)
    if n < min_samples:
        lambda_opt = max(0.2, lambda_opt)  # Minimum 20% shrinkage for small samples

    # Shrunk estimator
    shrunk_cov_gpu = (1 - lambda_opt) * sample_cov_gpu + lambda_opt * target_gpu

    # Move back to CPU as numpy array
    return shrunk_cov_gpu.cpu().numpy()

def compute_matrix_inverse_gpu(matrix, device=None):
    """GPU-accelerated matrix inversion with fallback"""
    if device is None:
        device = GPU_DEVICE

    try:
        # Convert to GPU tensor (use float64 for numerical consistency)
        matrix_gpu = torch.tensor(matrix, dtype=torch.float64, device=device)

        # GPU matrix inversion using Cholesky decomposition for stability
        try:
            # Try Cholesky first (faster for positive definite matrices)
            L = torch.linalg.cholesky(matrix_gpu)
            inv_gpu = torch.cholesky_inverse(L)
        except:
            # Fallback to general inverse
            inv_gpu = torch.linalg.inv(matrix_gpu)

        # Move back to CPU
        return inv_gpu.cpu().numpy()

    except Exception as e:
        print(f"    GPU matrix inversion failed: {e}, using CPU pseudo-inverse")
        return np.linalg.pinv(matrix)

def mahalanobis_distance_batch_gpu(X, mean, cov_inv, device=None):
    """GPU-accelerated batch Mahalanobis distance computation"""
    if device is None:
        device = GPU_DEVICE

    try:
        # Convert inputs to GPU tensors (use float64 for numerical consistency)
        X_gpu = torch.tensor(X, dtype=torch.float64, device=device)
        mean_gpu = torch.tensor(mean, dtype=torch.float64, device=device)
        cov_inv_gpu = torch.tensor(cov_inv, dtype=torch.float64, device=device)

        # Compute differences (broadcasting)
        if X_gpu.dim() == 1:
            X_gpu = X_gpu.unsqueeze(0)  # Add batch dimension

        diff = X_gpu - mean_gpu.unsqueeze(0)  # Shape: (batch_size, feature_dim)

        # Batch Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        # result = diff @ cov_inv @ diff.T, but we want diagonal elements
        temp = torch.matmul(diff, cov_inv_gpu)  # Shape: (batch_size, feature_dim)
        distances_squared = torch.sum(temp * diff, dim=1)  # Element-wise multiply and sum
        distances = torch.sqrt(torch.clamp(distances_squared, min=0))

        # Move back to CPU
        return distances.cpu().numpy()

    except Exception as e:
        print(f"    GPU Mahalanobis distance failed: {e}, using CPU fallback")
        # CPU fallback
        if X.ndim == 1:
            X = X.reshape(1, -1)

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

class GPUMLPClassifier(nn.Module):
    """GPU-accelerated MLP classifier for jailbreak detection"""

    def __init__(self, input_dim=256, hidden_dim=128, dropout=0.3):
        super(GPUMLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


class MLPClassifierWrapper:
    """GPU-accelerated MLP classifier wrapper for jailbreak detection on learned features"""

    def __init__(self, input_dim=256, hidden_dim=128, random_state=42, use_gpu=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = GPU_DEVICE if self.use_gpu else torch.device('cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.gpu_model = None

    def fit(self, X, y):
        """Train the MLP classifier with GPU acceleration"""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        if self.use_gpu and len(X_scaled) > 1000:  # Use GPU for larger datasets
            print(f"    Training GPU MLP classifier on {self.device}")
            self._fit_gpu(X_scaled, y)
        else:
            print(f"    Training CPU MLP classifier")
            self._fit_cpu(X_scaled, y)

    def _fit_gpu(self, X, y):
        """GPU-accelerated training"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # Initialize model
        self.gpu_model = GPUMLPClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.gpu_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        self.gpu_model.train()
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 20

        for epoch in range(200):  # Max epochs
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.gpu_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                clip_grad_norm_(self.gpu_model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                break

        self.gpu_model.eval()

    def _fit_cpu(self, X, y):
        """CPU fallback training"""
        # Create MLP with small architecture suitable for 256-dim features
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        )
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions"""
        if self.gpu_model is not None:
            return self._predict_gpu(X)
        elif self.model is not None:
            return self._predict_cpu(X)
        else:
            raise ValueError("Model not trained yet")

    def _predict_gpu(self, X):
        """GPU prediction"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.gpu_model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()

    def _predict_cpu(self, X):
        """CPU prediction"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.gpu_model is not None:
            return self._predict_proba_gpu(X)
        elif self.model is not None:
            return self._predict_proba_cpu(X)
        else:
            raise ValueError("Model not trained yet")

    def _predict_proba_gpu(self, X):
        """GPU probability prediction"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.gpu_model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

    def _predict_proba_cpu(self, X):
        """CPU probability prediction"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        scores = probabilities[:, 1]  # Probability of malicious class

        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, zero_division=0)

        # Calculate metrics
        unique_classes = np.unique(y)
        if len(unique_classes) > 1:
            try:
                tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                fpr_curve, tpr_curve, _ = roc_curve(y, scores)
                auroc = auc(fpr_curve, tpr_curve)
                precision, recall, _ = precision_recall_curve(y, scores)
                auprc = auc(recall, precision)
            except:
                tpr = fpr = auroc = auprc = float('nan')
        else:
            tpr = fpr = auroc = auprc = float('nan')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'auroc': auroc,
            'auprc': auprc,
            'predictions': predictions,
            'scores': scores
        }


class GPUFastClassifierWrapper:
    """Ultra-fast GPU-native classifier replacing SVM for massive speedup"""

    def __init__(self, random_state=42, use_gpu=True):
        self.random_state = random_state
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = GPU_DEVICE if self.use_gpu else torch.device('cpu')
        self.scaler_params = None
        self.model = None

        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

    def fit(self, X, y):
        """Train ultra-fast GPU classifier"""
        print(f"    Training GPU-native classifier with {'GPU' if self.use_gpu else 'CPU'} acceleration...")

        # GPU-accelerated standardization
        if self.use_gpu:
            X_scaled, self.scaler_params = gpu_standardize_features(X, self.device)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scaler_params = scaler

        if self.use_gpu:
            self._fit_gpu(X_scaled, y)
        else:
            self._fit_cpu_fallback(X_scaled, y)

    def _fit_gpu(self, X, y):
        """GPU-native training - much faster than SVM GridSearch"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        input_dim = X.shape[1]

        # Simple but effective architecture for binary classification
        self.model = nn.Sequential(
            nn.Linear(input_dim, min(128, input_dim * 2)),
            nn.BatchNorm1d(min(128, input_dim * 2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(min(128, input_dim * 2), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        ).to(self.device)

        # Fast training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=min(256, len(X)), shuffle=True)

        # Fast training (much faster than GridSearchCV)
        self.model.train()
        for epoch in range(50):  # Much fewer epochs than MLP, still effective
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Early stopping for very fast convergence
            if epoch > 10 and total_loss < 0.01:
                break

        self.model.eval()
        print(f"    GPU classifier training completed in {epoch+1} epochs")

    def _fit_cpu_fallback(self, X, y):
        """CPU fallback using simple logistic regression (much faster than SVM)"""
        from sklearn.linear_model import LogisticRegression

        # Use simple logistic regression instead of expensive SVM GridSearch
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=200,
            C=1.0,  # Fixed hyperparameter - no grid search needed
            solver='lbfgs'
        )
        self.model.fit(X, y)
        print(f"    CPU logistic regression training completed")

    def predict(self, X):
        """Make predictions with GPU-accelerated preprocessing"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.use_gpu:
            X_scaled = gpu_apply_standardization(X, self.scaler_params, self.device)
            return self._predict_gpu(X_scaled)
        else:
            X_scaled = self.scaler_params.transform(X)  # type: ignore
            return self.model.predict(X_scaled)

    def _predict_gpu(self, X_scaled):
        """GPU prediction"""
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """Get prediction probabilities with GPU-accelerated preprocessing"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.use_gpu:
            X_scaled = gpu_apply_standardization(X, self.scaler_params, self.device)
            return self._predict_proba_gpu(X_scaled)
        else:
            X_scaled = self.scaler_params.transform(X)  # type: ignore
            return self.model.predict_proba(X_scaled)

    def _predict_proba_gpu(self, X_scaled):
        """GPU probability prediction"""
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions = self.predict(X)
        scores = self.predict_proba(X)[:, 1]  # Probability of positive class

        accuracy = accuracy_score(y, predictions)

        # Handle binary classification metrics
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            auroc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y, scores)
            auprc = auc(recall, precision)
            f1 = f1_score(y, predictions)

            # Calculate TPR and FPR
            tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            auroc = float('nan')
            auprc = float('nan')
            f1 = f1_score(y, predictions, average='weighted')
            tpr_val = float('nan')
            fpr_val = float('nan')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr_val,
            'fpr': fpr_val,
            'auroc': auroc,
            'auprc': auprc,
            'predictions': predictions,
            'scores': scores,
            'best_params': 'GPU_native_classifier'  # No hyperparameter search needed
        }


class GPULinearSVM(nn.Module):
    """GPU-native Linear SVM implementation using PyTorch"""

    def __init__(self, input_dim, C=1.0):
        super(GPULinearSVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.C = C

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)

    def hinge_loss(self, outputs, targets):
        """Hinge loss for SVM: max(0, 1 - y * f(x))"""
        # Convert targets from {0, 1} to {-1, 1}
        targets = 2 * targets.float() - 1
        # Hinge loss
        loss = torch.mean(torch.clamp(1 - targets * outputs.squeeze(), min=0))
        # L2 regularization
        l2_reg = 0.5 * torch.sum(self.linear.weight ** 2)
        return loss + l2_reg / self.C


class GPUNativeSVMWrapper:
    """GPU-native SVM implementation - much faster than sklearn SVM"""

    def __init__(self, random_state=42, C=1.0):
        self.random_state = random_state
        self.C = C
        self.device = GPU_DEVICE
        self.scaler_params = None
        self.model = None

        # Set random seeds
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

    def fit(self, X, y):
        """Train GPU-native linear SVM"""
        print(f"    Training GPU-native Linear SVM (C={self.C})...")

        # GPU-accelerated standardization
        X_scaled, self.scaler_params = gpu_standardize_features(X, self.device)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        input_dim = X.shape[1]

        # Initialize GPU-native SVM
        self.model = GPULinearSVM(input_dim, C=self.C).to(self.device)

        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)

        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=min(256, len(X)), shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(100):  # More epochs for SVM convergence
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.model.hinge_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Early stopping for convergence
            if epoch > 20 and total_loss < 0.001:
                break

        self.model.eval()
        print(f"    GPU-native SVM training completed in {epoch+1} epochs")

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = gpu_apply_standardization(X, self.scaler_params, self.device)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs.squeeze() > 0).long()

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """Get prediction probabilities using sigmoid"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = gpu_apply_standardization(X, self.scaler_params, self.device)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Convert SVM scores to probabilities using sigmoid
            probs_pos = torch.sigmoid(outputs.squeeze())
            probs_neg = 1 - probs_pos
            probabilities = torch.stack([probs_neg, probs_pos], dim=1)

        return probabilities.cpu().numpy()

    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions = self.predict(X)
        scores = self.predict_proba(X)[:, 1]  # Probability of positive class

        accuracy = accuracy_score(y, predictions)

        # Handle binary classification metrics
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            auroc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y, scores)
            auprc = auc(recall, precision)
            f1 = f1_score(y, predictions)

            # Calculate TPR and FPR
            tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            auroc = float('nan')
            auprc = float('nan')
            f1 = f1_score(y, predictions, average='weighted')
            tpr_val = float('nan')
            fpr_val = float('nan')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr_val,
            'fpr': fpr_val,
            'auroc': auroc,
            'auprc': auprc,
            'predictions': predictions,
            'scores': scores,
            'best_params': f'GPU_native_SVM_C_{self.C}'
        }


class GPUSVMClassifierWrapper:
    """Original GPU-accelerated SVM classifier (kept for compatibility)"""

    def __init__(self, random_state=42, use_gpu=True):
        self.random_state = random_state
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = GPU_DEVICE if self.use_gpu else torch.device('cpu')
        self.scaler_params = None
        self.model = None

    def fit(self, X, y):
        """Train the SVM classifier with GPU-accelerated preprocessing"""
        print(f"    Training SVM with {'GPU' if self.use_gpu else 'CPU'} acceleration...")

        # GPU-accelerated standardization
        if self.use_gpu:
            X_scaled, self.scaler_params = gpu_standardize_features(X, self.device)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scaler_params = scaler

        # Reduced parameter grid for faster training on GPU
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }

        svm = SVC(random_state=self.random_state, probability=True)

        # Use smaller CV for faster training, more CPU cores for GPU systems
        n_jobs = -1 if self.use_gpu else min(4, os.cpu_count() or 1)
        self.model = GridSearchCV(
            svm, param_grid, cv=3, scoring='f1', n_jobs=n_jobs, verbose=0
        )

        self.model.fit(X_scaled, y)
        print(f"    SVM training completed with best params: {self.model.best_params_}")

    def predict(self, X):
        """Make predictions with GPU-accelerated preprocessing"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.use_gpu:
            X_scaled = gpu_apply_standardization(X, self.scaler_params, self.device)
        else:
            X_scaled = self.scaler_params.transform(X)  # type: ignore

        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities with GPU-accelerated preprocessing"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.use_gpu:
            X_scaled = gpu_apply_standardization(X, self.scaler_params, self.device)
        else:
            X_scaled = self.scaler_params.transform(X)  # type: ignore

        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions = self.predict(X)
        scores = self.predict_proba(X)[:, 1]  # Probability of positive class

        accuracy = accuracy_score(y, predictions)

        # Handle binary classification metrics
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            auroc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y, scores)
            auprc = auc(recall, precision)
            f1 = f1_score(y, predictions)

            # Calculate TPR and FPR
            tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            auroc = float('nan')
            auprc = float('nan')
            f1 = f1_score(y, predictions, average='weighted')
            tpr_val = float('nan')
            fpr_val = float('nan')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr_val,
            'fpr': fpr_val,
            'auroc': auroc,
            'auprc': auprc,
            'predictions': predictions,
            'scores': scores,
            'best_params': self.model.best_params_ if hasattr(self.model, 'best_params_') else None
        }


class SVMClassifierWrapper:
    """CPU fallback SVM classifier for jailbreak detection on learned features"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        """Train the SVM classifier with grid search"""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Grid search for best hyperparameters
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }

        svm = SVC(random_state=self.random_state, probability=True)

        # Use smaller CV for faster training
        self.model = GridSearchCV(
            svm, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0
        )

        self.model.fit(X_scaled, y)

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        scores = probabilities[:, 1]  # Probability of malicious class

        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, zero_division=0)

        # Calculate metrics
        unique_classes = np.unique(y)
        if len(unique_classes) > 1:
            try:
                tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                fpr_curve, tpr_curve, _ = roc_curve(y, scores)
                auroc = auc(fpr_curve, tpr_curve)
                precision, recall, _ = precision_recall_curve(y, scores)
                auprc = auc(recall, precision)
            except:
                tpr = fpr = auroc = auprc = float('nan')
        else:
            tpr = fpr = auroc = auprc = float('nan')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'tpr': tpr,
            'fpr': fpr,
            'auroc': auroc,
            'auprc': auprc,
            'predictions': predictions,
            'scores': scores,
            'best_params': self.model.best_params_ if hasattr(self.model, 'best_params_') else None
        }


def analyze_dataset_composition(dataset, dataset_name):
    """Analyze and report dataset composition"""
    total = len(dataset)
    benign = sum(1 for s in dataset if s['toxicity'] == 0)
    malicious = sum(1 for s in dataset if s['toxicity'] == 1)

    has_images = sum(1 for s in dataset if s.get('img') is not None)
    text_only = total - has_images

    print(f"{dataset_name}: {total} samples")
    print(f"  Classes: {benign} benign, {malicious} malicious")
    print(f"  Modality: {has_images} multimodal, {text_only} text-only")

    if benign == 0:
        print(f"  WARNING: {dataset_name} contains ONLY malicious samples!")
    elif malicious == 0:
        print(f"  WARNING: {dataset_name} contains ONLY benign samples!")

    return {'total': total, 'benign': benign, 'malicious': malicious, 'multimodal': has_images}


def prepare_ood_data_structure(datasets_dict, hidden_states_dict, labels_dict):
    in_dist_data = {}
    ood_data = {}
    
    for dataset_name in datasets_dict.keys():
        if dataset_name not in hidden_states_dict:
            continue
            
        features = hidden_states_dict[dataset_name]
        labels = labels_dict[dataset_name]
        
        # Separate benign and malicious samples
        benign_features = [features[i] for i, label in enumerate(labels) if label == 0]
        malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
        
        if benign_features:
            in_dist_data[f"{dataset_name}_benign"] = benign_features
        if malicious_features:
            ood_data[f"{dataset_name}_malicious"] = malicious_features
    
    return in_dist_data, ood_data


def signal_handler(sig, frame):
    print('\nInterrupted by user. Exiting...')
    sys.exit(0)

def prepare_balanced_training():
    """
    Load balanced training data according to the balanced dataset configuration:
    Training Set (2,000 examples, 1:1 ratio):
    - Benign (1,000): Alpaca (500) + MM-Vet (218) + OpenAssistant (282)
    - Malicious (1,000): AdvBench (300) + JailbreakV-28K (550) + DAN variants (150)
    """
    print("Loading balanced training data...")
    benign_training = {}
    malicious_training = {}

    # === BENIGN TRAINING DATA (1,000 total) ===
    print("Loading benign training data (1,000 samples)...")

    # 1. Alpaca - 500 samples
    try:
        alpaca_samples = load_alpaca(max_samples=500)
        if alpaca_samples:
            benign_training["Alpaca"] = alpaca_samples
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
                benign_training["MM-Vet"] = mmvet_benign
                print(f"  Loaded {len(mmvet_benign)} MM-Vet samples")
    except Exception as e:
        print(f"Could not load MM-Vet: {e}")

    # 3. OpenAssistant - 282 samples
    try:
        openassistant_samples = load_openassistant(max_samples=282)
        if openassistant_samples:
            benign_training["OpenAssistant"] = openassistant_samples
            print(f"  Loaded {len(openassistant_samples)} OpenAssistant samples")
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")

    # === MALICIOUS TRAINING DATA (1,000 total) ===
    print("Loading malicious training data (1,000 samples)...")

    # 1. AdvBench - 300 samples
    try:
        advbench_samples = load_advbench(max_samples=300)
        if advbench_samples:
            malicious_training["AdvBench"] = advbench_samples
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
            malicious_training["JailbreakV-28K"] = jbv_samples
            print(f"  Total JailbreakV-28K training samples: {len(jbv_samples)} ({len(llm_attack_samples)} llm_transfer + {len(query_related_samples)} query_related)")
        else:
            print("  Warning: No JailbreakV-28K samples loaded")

    except Exception as e:
        print(f"Could not load JailbreakV-28K: {e}")

    # 3. DAN Prompts - 150 samples
    try:
        dan_samples = load_dan_prompts(max_samples=150)
        if dan_samples:
            malicious_training["DAN"] = dan_samples
            print(f"  Loaded {len(dan_samples)} DAN samples")
    except Exception as e:
        print(f"Could not load DAN Prompts: {e}")

    return benign_training, malicious_training

def prepare_balanced_evaluation():
    """
    Prepare balanced evaluation datasets according to the balanced dataset configuration:
    Test Set (1,800 examples, 1:1 ratio):
    - Safe (900): XSTest safe (250) + FigTxt safe (300) + VQAv2 (350)
    - Unsafe (900): XSTest unsafe (200) + FigTxt unsafe (350) + VAE (200) + JailbreakV-28K (150)
    """
    print("Loading balanced evaluation datasets...")
    test_datasets = {}

    # === SAFE TEST DATA (900 total) ===
    print("Loading safe test data (900 samples)...")

    # 1. XSTest safe - 250 samples
    try:
        xstest_samples = load_XSTest()
        if xstest_samples:
            xstest_safe = [s for s in xstest_samples if s.get('toxicity', 0) == 0][:250]
            if xstest_safe:
                test_datasets["XSTest_safe"] = xstest_safe
                print(f"  Loaded {len(xstest_safe)} XSTest safe samples")
    except Exception as e:
        print(f"Could not load XSTest: {e}")

    # 2. XSTest unsafe - 200 samples
    try:
        if 'xstest_samples' in locals():
            xstest_unsafe = [s for s in xstest_samples if s.get('toxicity', 0) == 1][:200]
            if xstest_unsafe:
                test_datasets["XSTest_unsafe"] = xstest_unsafe
                print(f"  Loaded {len(xstest_unsafe)} XSTest unsafe samples")
    except Exception as e:
        print(f"Could not load XSTest unsafe: {e}")

    # 3. FigTxt safe - 300 samples
    try:
        figtxt_samples = load_FigTxt()
        if figtxt_samples:
            figtxt_safe = [s for s in figtxt_samples if s.get('toxicity', 0) == 0][:300]
            if figtxt_safe:
                test_datasets["FigTxt_safe"] = figtxt_safe
                print(f"  Loaded {len(figtxt_safe)} FigTxt safe samples")
    except Exception as e:
        print(f"Could not load FigTxt: {e}")

    # 4. FigTxt unsafe - 350 samples
    try:
        if 'figtxt_samples' in locals():
            figtxt_unsafe = [s for s in figtxt_samples if s.get('toxicity', 0) == 1][:350]
            if figtxt_unsafe:
                test_datasets["FigTxt_unsafe"] = figtxt_unsafe
                print(f"  Loaded {len(figtxt_unsafe)} FigTxt unsafe samples")
    except Exception as e:
        print(f"Could not load FigTxt unsafe: {e}")

    # 5. VQAv2 - 350 safe samples
    try:
        vqav2_samples = load_vqav2(max_samples=350)
        if vqav2_samples:
            test_datasets["VQAv2"] = vqav2_samples
            print(f"  Loaded {len(vqav2_samples)} VQAv2 samples")
    except Exception as e:
        print(f"Could not load VQAv2: {e}")

    # === UNSAFE TEST DATA (additional) ===
    print("Loading additional unsafe test data...")

    # 6. VAE (adversarial visual attacks) - 200 samples
    try:
        vae_samples = load_adversarial_img()
        if vae_samples:
            # Limit to 200 samples
            vae_limited = vae_samples[:200]
            test_datasets["VAE"] = vae_limited
            print(f"  Loaded {len(vae_limited)} VAE samples")
    except Exception as e:
        print(f"Could not load VAE: {e}")

    # 7. JailbreakV-28K test samples - 150 samples (figstep attack only)
    try:
        jbv_test_samples = load_JailBreakV_figstep(max_samples=150)
        if jbv_test_samples:
            test_datasets["JailbreakV-28K_test"] = jbv_test_samples
            print(f"  Loaded {len(jbv_test_samples)} JailbreakV-28K test samples (figstep attack)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K test samples: {e}")

    return test_datasets

def main():
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    # Set random seed for reproducibility using centralized config
    MAIN_SEED = CONFIG.MAIN_SEED  # Use seed from PCAConfig
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)
    torch.manual_seed(MAIN_SEED)
    torch.cuda.manual_seed(MAIN_SEED)
    torch.cuda.manual_seed_all(MAIN_SEED)

    # Ensure deterministic behavior for overall pipeline
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Additional determinism settings
    os.environ['PYTHONHASHSEED'] = str(MAIN_SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CuBLAS operations
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Random seeds set for reproducibility (seed={MAIN_SEED})")

    model_path = "model/llava-v1.6-vicuna-7b/"

    # Initialize feature extractor
    extractor = HiddenStateExtractor(model_path)

    print("="*80)
    print("BALANCED JAILBREAK DETECTION USING ML MODELS ON LEARNED FEATURES")
    print("="*80)
    print("Approach: Train ML models directly on learned 256-dimensional features")
    print("          Layer-specific projections: 4096 -> 256 dimensions per layer")
    print("          Multi-objective contrastive loss for optimal feature learning")
    print("          ML Models: Small MLP and SVM for each layer")
    print("          Compare performance with lower-dimensional features vs raw 4096-dim")
    print("="*80)
    print("Training Set (2,000 examples, 1:1 ratio):")
    print("  - Benign (1,000): Alpaca (500) + MM-Vet (218) + OpenAssistant (282)")
    print("  - Malicious (1,000): AdvBench (300) + JailbreakV-28K (550) + DAN variants (150)")
    print("Test Set (1,800 examples, 1:1 ratio):")
    print("  - Safe (900): XSTest safe (250) + FigTxt safe (300) + VQAv2 (350)")
    print("  - Unsafe (900): XSTest unsafe (200) + FigTxt unsafe (350) + VAE (200) + JailbreakV-28K (150)")
    print("="*80)

    # Load balanced training data
    in_dist_datasets, ood_datasets = prepare_balanced_training()

    # Analyze training data composition
    print("\n--- Training Data Analysis ---")
    total_benign = 0
    total_malicious = 0

    print("In-Distribution (Benign) Training Data:")
    for dataset_name, samples in in_dist_datasets.items():
        analyze_dataset_composition(samples, dataset_name)
        total_benign += len(samples)

    print("\nOOD (Malicious) Training Data:")
    for dataset_name, samples in ood_datasets.items():
        analyze_dataset_composition(samples, dataset_name)
        total_malicious += len(samples)

    print(f"\nTraining Data Summary:")
    print(f"  Total Benign (In-Distribution): {total_benign:,} samples")
    print(f"  Total Malicious (OOD): {total_malicious:,} samples")
    print(f"  Ratio (Benign:Malicious): {total_benign/max(total_malicious,1):.1f}:1")
    print("-"*80)

    # Load balanced evaluation data
    test_datasets = prepare_balanced_evaluation()

    print("\n--- Evaluation Data Analysis ---")
    for dataset_name, samples in test_datasets.items():
        analyze_dataset_composition(samples, dataset_name)

    # Extract hidden states for all datasets (with caching)
    print("\n--- Extracting Hidden States ---")
    all_datasets = {**in_dist_datasets, **ood_datasets, **test_datasets}
    all_hidden_states = {}
    all_labels = {}

    for dataset_name, samples in all_datasets.items():
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

    layers = list(range(0, 32))  # layers 0-31
    layer_results = {}

    # === TRAIN PCA PROJECTIONS ===
    CONFIG.print_config()

    # Store PCA models for different component numbers
    pca_models_by_layer = {}
    dataset_name_to_id = None

    # Only use training datasets (in_dist_datasets and ood_datasets)
    training_dataset_names = set(in_dist_datasets.keys()) | set(ood_datasets.keys())

    print(f"Training datasets: {list(training_dataset_names)}")
    print("Test datasets will NOT be used for PCA training")

    if CONFIG.PROJECTION_MODE == "single_layer":
        print(f"\n=== Training Single PCA (Layer {CONFIG.SINGLE_LAYER_TRAINING_LAYER}) ===")
        print(f"Will use this PCA for all layers")

        # Prepare training data for the single training layer
        projection_features_dict = {}
        projection_labels_dict = {}

        for dataset_name in training_dataset_names:
            if dataset_name in all_hidden_states:
                projection_features_dict[dataset_name] = all_hidden_states[dataset_name][CONFIG.SINGLE_LAYER_TRAINING_LAYER]
                projection_labels_dict[dataset_name] = all_labels[dataset_name]

        # Train PCA models with different component numbers
        single_pca_models, dataset_name_to_id = train_pca_projections(
            projection_features_dict,
            projection_labels_dict,
            random_seed=CONFIG.MAIN_SEED
        )

        # Use the same PCA models for all layers
        for layer_idx in layers:
            pca_models_by_layer[layer_idx] = single_pca_models

        print(f"Single PCA training completed! Using for all {len(layers)} layers.")

    elif CONFIG.PROJECTION_MODE == "layer_specific":
        print(f"\n=== Training Layer-Specific PCA ===")
        print("Training separate PCA models for each layer (0-31)")

        for layer_idx in layers:
            print(f"\n--- Training PCA for Layer {layer_idx} ---")

            # Prepare training data for this layer's PCA
            projection_features_dict = {}
            projection_labels_dict = {}

            for dataset_name in training_dataset_names:
                if dataset_name in all_hidden_states:
                    projection_features_dict[dataset_name] = all_hidden_states[dataset_name][layer_idx]
                    projection_labels_dict[dataset_name] = all_labels[dataset_name]

            # Train PCA models for this layer with layer-specific seed
            layer_seed = CONFIG.get_layer_seed(layer_idx)  # Different seed for each layer but deterministic
            layer_pca_models, layer_dataset_name_to_id = train_pca_projections(
                projection_features_dict,
                projection_labels_dict,
                random_seed=layer_seed
            )

            # Store the trained models
            pca_models_by_layer[layer_idx] = layer_pca_models

            # Use dataset mapping from first layer (should be consistent across layers)
            if dataset_name_to_id is None:
                dataset_name_to_id = layer_dataset_name_to_id

            print(f"Layer {layer_idx} PCA training completed!")

        print(f"\nAll {len(pca_models_by_layer)} layer-specific PCA models trained successfully!")

    else:
        raise ValueError(f"Unknown projection mode: {CONFIG.PROJECTION_MODE}. Use 'single_layer' or 'layer_specific'")

    # Results storage for different PCA components
    all_results = {}

    for layer_idx in layers:
        print(f"\n=== Evaluating Layer {layer_idx} ===")
        monitor_gpu_usage(f"Layer {layer_idx} Start")

        # Prepare data for this layer
        layer_hidden_states = {}
        layer_labels = {}

        for dataset_name in all_datasets.keys():
            if dataset_name in all_hidden_states:
                layer_hidden_states[dataset_name] = all_hidden_states[dataset_name][layer_idx]
                layer_labels[dataset_name] = all_labels[dataset_name]

        # Test different PCA component numbers
        layer_results = {}

        for n_components in CONFIG.PCA_COMPONENTS:
            print(f"\n--- Testing PCA with {n_components} components ---")

            # Get the appropriate PCA model
            pca_model = pca_models_by_layer[layer_idx][n_components]

            # Apply PCA projection to transform features
            print(f"  Applying PCA projection to layer {layer_idx} features...")
            projected_layer_hidden_states = apply_pca_projection(
                pca_model,
                layer_hidden_states
            )

            # Use projected features
            current_layer_states = projected_layer_hidden_states
            monitor_gpu_usage(f"Layer {layer_idx} PCA-{n_components} Complete")

            # Prepare training data for SVM models
            print(f"    Preparing training data for SVM (PCA-{n_components})...")

            # Collect training features and labels
            train_features = []
            train_labels = []

            # Add benign samples from in-distribution datasets
            for dataset_name in in_dist_datasets.keys():
                if dataset_name in projected_layer_hidden_states:
                    features = projected_layer_hidden_states[dataset_name]
                    labels = layer_labels[dataset_name]

                    # Get benign samples
                    for i, label in enumerate(labels):
                        if label == 0:  # benign
                            train_features.append(features[i])
                            train_labels.append(0)

            # Add malicious samples from OOD datasets
            for dataset_name in ood_datasets.keys():
                if dataset_name in projected_layer_hidden_states:
                    features = projected_layer_hidden_states[dataset_name]
                    labels = layer_labels[dataset_name]

                    # Get malicious samples
                    for i, label in enumerate(labels):
                        if label == 1:  # malicious
                            train_features.append(features[i])
                            train_labels.append(1)

            if len(train_features) == 0:
                print(f"    Skipping PCA-{n_components} - no training data")
                continue

            # Convert to numpy arrays
            train_features = np.array(train_features)
            train_labels = np.array(train_labels)

            print(f"    Training data: {len(train_features)} samples ({np.sum(train_labels == 0)} benign, {np.sum(train_labels == 1)} malicious)")
            print(f"    Feature dimension after PCA: {train_features.shape[1]}")

            # Train multiple GPU classifiers (MLP and/or SVM)
            classifier_results_dict = {}

            for classifier_type in CONFIG.CLASSIFIERS_TO_TEST:
                classifier_seed = CONFIG.get_svm_seed(layer_idx, n_components)

                if classifier_type == "MLP":
                    print(f"    Training GPU MLP classifier for PCA-{n_components}...")
                    classifier_model = GPUFastClassifierWrapper(random_state=classifier_seed, use_gpu=True)
                elif classifier_type == "SVM":
                    print(f"    Training GPU-native SVM for PCA-{n_components}...")
                    # Test different C values for SVM
                    best_svm_model = None
                    best_svm_score = 0

                    for C_val in [0.1, 1.0, 10.0]:  # Quick grid search on GPU
                        svm_model = GPUNativeSVMWrapper(random_state=classifier_seed, C=C_val)
                        svm_model.fit(train_features, train_labels)

                        # Quick validation on training data (for speed)
                        train_pred = svm_model.predict(train_features)
                        train_acc = accuracy_score(train_labels, train_pred)

                        if train_acc > best_svm_score:
                            best_svm_score = train_acc
                            best_svm_model = svm_model

                    classifier_model = best_svm_model
                    print(f"    Best SVM C value found with training accuracy: {best_svm_score:.4f}")
                else:
                    continue

            # === COMBINED EVALUATION (Real-world scenario) ===
            print(f"    Computing combined evaluation for PCA-{n_components}...")

            # Combine all test features and labels for unified evaluation
            combined_test_features = []
            combined_test_labels = []
            dataset_boundaries = {}  # Track which samples belong to which dataset
            current_idx = 0

            for test_dataset_name in test_datasets.keys():
                if test_dataset_name in projected_layer_hidden_states:
                    test_features = projected_layer_hidden_states[test_dataset_name]
                    test_labels_data = layer_labels[test_dataset_name]

                    dataset_boundaries[test_dataset_name] = (current_idx, current_idx + len(test_features))
                    combined_test_features.extend(test_features)
                    combined_test_labels.extend(test_labels_data)
                    current_idx += len(test_features)

            if combined_test_features:
                # Convert to numpy arrays
                combined_test_features = np.array(combined_test_features)
                combined_test_labels = np.array(combined_test_labels)

                print(f"    Combined test set: {len(combined_test_features)} samples ({np.sum(combined_test_labels == 0)} benign, {np.sum(combined_test_labels == 1)} malicious)")

                # Evaluate each classifier type
                pca_performance = {
                    'n_components': n_components,
                    'explained_variance': pca_model.get_explained_variance_ratio().sum() if pca_model.get_explained_variance_ratio() is not None else 0.0
                }

                def format_metric(value):
                    return "N/A" if np.isnan(value) else f"{value:.4f}"

                for classifier_type in CONFIG.CLASSIFIERS_TO_TEST:
                    classifier_seed = CONFIG.get_svm_seed(layer_idx, n_components)

                    # Get the appropriate classifier
                    if classifier_type == "MLP":
                        classifier_model = GPUFastClassifierWrapper(random_state=classifier_seed, use_gpu=True)
                        classifier_model.fit(train_features, train_labels)
                        method_key = 'COMBINED_MLP'
                        print(f"    Evaluating GPU MLP for PCA-{n_components}...")
                    elif classifier_type == "SVM":
                        # Use the best SVM model found earlier
                        best_svm_model = None
                        best_svm_score = 0

                        for C_val in [0.1, 1.0, 10.0]:
                            svm_model = GPUNativeSVMWrapper(random_state=classifier_seed, C=C_val)
                            svm_model.fit(train_features, train_labels)
                            train_pred = svm_model.predict(train_features)
                            train_acc = accuracy_score(train_labels, train_pred)

                            if train_acc > best_svm_score:
                                best_svm_score = train_acc
                                best_svm_model = svm_model

                        classifier_model = best_svm_model
                        method_key = 'COMBINED_SVM'
                        print(f"    Evaluating GPU-native SVM for PCA-{n_components}...")
                    else:
                        continue

                    # Evaluate classifier
                    classifier_results = classifier_model.evaluate(combined_test_features, combined_test_labels)

                    # Display results
                    print(f"    {classifier_type} RESULTS (PCA-{n_components}): Acc={classifier_results['accuracy']:.4f}, F1={format_metric(classifier_results['f1'])}, "
                          f"TPR={format_metric(classifier_results['tpr'])}, FPR={format_metric(classifier_results['fpr'])}, "
                          f"AUROC={format_metric(classifier_results['auroc'])}, AUPRC={format_metric(classifier_results['auprc'])}")

                    if 'best_params' in classifier_results and classifier_results['best_params']:
                        print(f"    {classifier_type} Config: {classifier_results['best_params']}")

                    # Store results
                    pca_performance[method_key] = classifier_results

                # === INDIVIDUAL DATASET ANALYSIS (for debugging) ===
                print(f"    Individual dataset analysis for PCA-{n_components}:")
                for test_dataset_name, (start_idx, end_idx) in dataset_boundaries.items():
                    dataset_features = combined_test_features[start_idx:end_idx]
                    dataset_labels = combined_test_labels[start_idx:end_idx]

                    # Evaluate each classifier type on individual datasets
                    for classifier_type in CONFIG.CLASSIFIERS_TO_TEST:
                        classifier_seed = CONFIG.get_svm_seed(layer_idx, n_components)

                        # Get the appropriate classifier (retrain for individual evaluation)
                        if classifier_type == "MLP":
                            classifier_model = GPUFastClassifierWrapper(random_state=classifier_seed, use_gpu=True)
                            classifier_model.fit(train_features, train_labels)
                            suffix = "_MLP"
                        elif classifier_type == "SVM":
                            # Use best SVM configuration
                            best_svm_model = None
                            best_svm_score = 0

                            for C_val in [0.1, 1.0, 10.0]:
                                svm_model = GPUNativeSVMWrapper(random_state=classifier_seed, C=C_val)
                                svm_model.fit(train_features, train_labels)
                                train_pred = svm_model.predict(train_features)
                                train_acc = accuracy_score(train_labels, train_pred)

                                if train_acc > best_svm_score:
                                    best_svm_score = train_acc
                                    best_svm_model = svm_model

                            classifier_model = best_svm_model
                            suffix = "_SVM"
                        else:
                            continue

                        # Evaluate on individual dataset
                        dataset_classifier_results = classifier_model.evaluate(dataset_features, dataset_labels)

                        print(f"      {test_dataset_name:15s}: {classifier_type} Acc={dataset_classifier_results['accuracy']:.4f}")

                        # Store individual dataset results
                        pca_performance[f"{test_dataset_name}{suffix}"] = dataset_classifier_results
            else:
                print(f"    No test data available for PCA-{n_components} evaluation")
                pca_performance = {'n_components': n_components}

            # Store results for this PCA component number
            layer_results[n_components] = pca_performance

        # Store results for this layer
        all_results[layer_idx] = layer_results

    # Calculate ranking based on PCA + Classifier results (MLP and SVM)
    pca_classifier_scores = []

    for layer_idx in layers:
        if layer_idx in all_results:
            layer_data = all_results[layer_idx]

            for n_components in CONFIG.PCA_COMPONENTS:
                if n_components in layer_data:
                    # Check for MLP, SVM, and legacy results
                    for method_key in ['COMBINED_MLP', 'COMBINED_SVM', 'COMBINED_GPU_CLASSIFIER']:
                        if method_key in layer_data[n_components]:
                            result = layer_data[n_components][method_key]
                            accuracy = result['accuracy']
                            auroc = result['auroc'] if not np.isnan(result['auroc']) else 0.0
                            auprc = result['auprc'] if not np.isnan(result['auprc']) else 0.0
                            f1 = result['f1'] if not np.isnan(result['f1']) else 0.0
                            explained_var = layer_data[n_components].get('explained_variance', 0.0)

                            # Combined score prioritizing accuracy, F1, and AUROC
                            combined_score = 0.4 * accuracy + 0.3 * f1 + 0.2 * auroc + 0.1 * auprc

                            # Determine method name for CSV
                            if method_key == 'COMBINED_MLP':
                                method_name = 'MLP'
                            elif method_key == 'COMBINED_SVM':
                                method_name = 'SVM'
                            else:
                                method_name = 'GPU_CLASSIFIER'

                            pca_classifier_scores.append((
                                layer_idx, n_components, accuracy, f1, auroc, auprc,
                                explained_var, combined_score, method_name
                            ))

    # Sort by combined score (real-world performance)
    pca_classifier_scores.sort(key=lambda x: x[7], reverse=True)

    # Save results to CSV
    output_path = "results/balanced_pca_ml_results.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Layer", "PCA_Components", "Dataset", "Method", "Accuracy", "F1", "TPR", "FPR",
            "AUROC", "AUPRC", "Explained_Variance", "Combined_Score", "Rank"
        ])

        # Write results for each layer and PCA component combination
        for rank, (layer_idx, n_components, accuracy, f1, auroc, auprc, explained_var, combined_score, method_name) in enumerate(pca_classifier_scores, 1):
            if layer_idx in all_results and n_components in all_results[layer_idx]:
                layer_data = all_results[layer_idx][n_components]

                # Check for the specific method result that matches this ranking entry
                method_key = f'COMBINED_{method_name}' if method_name in ['MLP', 'SVM'] else 'COMBINED_GPU_CLASSIFIER'

                if method_key in layer_data:
                    result = layer_data[method_key]

                    # Handle NaN values for CSV
                    f1_val = "N/A" if np.isnan(result['f1']) else f"{result['f1']:.4f}"
                    tpr_val = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
                    fpr_val = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"
                    auroc_val = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
                    auprc_val = "N/A" if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"

                    writer.writerow([
                        layer_idx,
                        n_components,
                        "COMBINED",
                        method_name,
                        f"{result['accuracy']:.4f}",
                        f1_val,
                        tpr_val,
                        fpr_val,
                        auroc_val,
                        auprc_val,
                        f"{explained_var:.4f}",
                        f"{combined_score:.4f}",
                        rank
                    ])

                # Also write individual dataset results
                for dataset_name, dataset_result in layer_data.items():
                    if dataset_name.endswith('_SVM') and dataset_name != 'COMBINED_SVM':
                        dataset_clean = dataset_name[:-4]  # Remove _SVM suffix

                        f1_val = "N/A" if np.isnan(dataset_result['f1']) else f"{dataset_result['f1']:.4f}"
                        tpr_val = "N/A" if np.isnan(dataset_result['tpr']) else f"{dataset_result['tpr']:.4f}"
                        fpr_val = "N/A" if np.isnan(dataset_result['fpr']) else f"{dataset_result['fpr']:.4f}"
                        auroc_val = "N/A" if np.isnan(dataset_result['auroc']) else f"{dataset_result['auroc']:.4f}"
                        auprc_val = "N/A" if np.isnan(dataset_result['auprc']) else f"{dataset_result['auprc']:.4f}"

                        writer.writerow([
                            layer_idx,
                            n_components,
                            dataset_clean,
                            "SVM",
                            f"{dataset_result['accuracy']:.4f}",
                            f1_val,
                            tpr_val,
                            fpr_val,
                            auroc_val,
                            auprc_val,
                            f"{explained_var:.4f}",
                            "N/A",  # Combined score only for COMBINED results
                            "N/A"   # Rank only for COMBINED results
                        ])


    print(f"\nResults saved to {output_path}")

    # Print summary results
    print("\n" + "="*120)
    print("BALANCED JAILBREAK DETECTION SUMMARY (PCA + SVM)")
    print("="*120)
    print("Training Configuration (2,000 samples, 1:1 ratio):")
    print(f"  In-Distribution Datasets: {list(in_dist_datasets.keys())}")
    print(f"  OOD Datasets: {list(ood_datasets.keys())}")
    print("Test Configuration (1,800 samples, 1:1 ratio):")
    print(f"  Test Datasets: {list(test_datasets.keys())}")
    print(f"  PCA Components Tested: {CONFIG.PCA_COMPONENTS}")
    print("-"*120)

    # PCA + SVM PERFORMANCE RANKING
    print(f"\n{'PCA + SVM CLASSIFIER PERFORMANCE RANKING':<120}")
    print(f"{'Rank':<5} {'Layer':<6} {'PCA':<5} {'Accuracy':<10} {'F1':<8} {'TPR':<8} {'FPR':<8} {'AUROC':<10} {'AUPRC':<10} {'ExplVar':<8} {'Combined':<10}")
    print("-" * 120)

    for rank, (layer_idx, n_components, accuracy, f1, auroc, auprc, explained_var, combined_score, method_name) in enumerate(pca_classifier_scores[:20], 1):  # Show top 20
        acc_str = f"{accuracy:.3f}"
        f1_str = f"{f1:.3f}" if not np.isnan(f1) else "N/A"
        auroc_str = f"{auroc:.3f}" if not np.isnan(auroc) else "N/A"
        auprc_str = f"{auprc:.3f}" if not np.isnan(auprc) else "N/A"
        explained_str = f"{explained_var:.3f}"
        combined_str = f"{combined_score:.3f}"

        # Get TPR and FPR from the actual results
        if layer_idx in all_results and n_components in all_results[layer_idx] and 'COMBINED_SVM' in all_results[layer_idx][n_components]:
            result = all_results[layer_idx][n_components]['COMBINED_SVM']
            tpr_str = f"{result['tpr']:.3f}" if not np.isnan(result['tpr']) else "N/A"
            fpr_str = f"{result['fpr']:.3f}" if not np.isnan(result['fpr']) else "N/A"
        else:
            tpr_str = "N/A"
            fpr_str = "N/A"

        print(f"{rank:<5} {layer_idx:<6} {n_components:<5} {acc_str:<10} {f1_str:<8} {tpr_str:<8} {fpr_str:<8} {auroc_str:<10} {auprc_str:<10} {explained_str:<8} {combined_str:<10}")

    # BEST PERFORMANCE ANALYSIS
    print(f"\n{'BEST PERFORMANCE ANALYSIS':<100}")
    print("-" * 100)

    if pca_classifier_scores:
        best_layer, best_components, best_accuracy, best_f1, best_auroc, best_auprc, best_explained_var, best_combined_score, best_method = pca_classifier_scores[0]

        print(f"BEST OVERALL PERFORMANCE:")
        print(f"  Layer: {best_layer}")
        print(f"  PCA Components: {best_components}")
        print(f"  Accuracy: {best_accuracy:.4f}")
        print(f"  F1 Score: {best_f1:.4f}" if not np.isnan(best_f1) else "  F1 Score: N/A")
        print(f"  AUROC: {best_auroc:.4f}" if not np.isnan(best_auroc) else "  AUROC: N/A")
        print(f"  AUPRC: {best_auprc:.4f}" if not np.isnan(best_auprc) else "  AUPRC: N/A")
        print(f"  Explained Variance: {best_explained_var:.4f}")
        print(f"  Combined Score: {best_combined_score:.4f}")

        # Show SVM parameters for best model
        if best_layer in all_results and best_components in all_results[best_layer] and 'COMBINED_SVM' in all_results[best_layer][best_components]:
            best_result = all_results[best_layer][best_components]['COMBINED_SVM']
            if 'best_params' in best_result and best_result['best_params']:
                print(f"  Best SVM Parameters: {best_result['best_params']}")

        # Analysis by PCA components
        print(f"\nPERFORMANCE BY PCA COMPONENTS:")
        component_performance = {}
        for layer_idx, n_components, accuracy, f1, auroc, auprc, explained_var, combined_score, method_name in pca_classifier_scores:
            if n_components not in component_performance:
                component_performance[n_components] = []
            component_performance[n_components].append(combined_score)

        for n_components in sorted(component_performance.keys()):
            scores = component_performance[n_components]
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            print(f"  {n_components} components: Avg={avg_score:.3f}, Max={max_score:.3f} (across {len(scores)} layers)")

    print("\n" + "="*120)
    print("PCA + SVM ANALYSIS COMPLETE")
    print("="*120)
    print("- Balanced 1:1 training and test ratios for robust evaluation")
    print("- Training: Alpaca (500) + MM-Vet (218) + OpenAssistant (282) vs AdvBench (300) + JailbreakV-28K (550) + DAN (150)")
    print("- Testing: XSTest + FigTxt + VQAv2 (safe) vs XSTest + FigTxt + VAE + JailbreakV-28K (unsafe)")
    print("- PCA dimensionality reduction applied before SVM classification")
    print(f"- PCA components tested: {CONFIG.PCA_COMPONENTS}")
    print("- Results saved to: results/balanced_pca_ml_results.csv")


if __name__ == "__main__":
    main()
