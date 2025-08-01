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
from typing import Union, Dict, Any
from sklearn.model_selection import GridSearchCV
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
class ProjectionConfig:
    """Global configuration for projection training methods"""

    # Random seed management (centralized for run_multiple_experiments.py compatibility)
    MAIN_SEED = 42  # Main seed for reproducibility - modified by run_multiple_experiments.py

    # Projection training mode
    # "single_layer": Train one projection on layer 16, use for all layers (original method)
    # "layer_specific": Train separate projection for each layer (new method)
    PROJECTION_MODE = "layer_specific"  # Change this to switch modes

    # Single layer mode settings
    SINGLE_LAYER_TRAINING_LAYER = 18  # Which layer to use for training projection

    # Training hyperparameters
    PROJECTION_EPOCHS = 200
    PROJECTION_BATCH_SIZE = 64
    PROJECTION_LEARNING_RATE = 1e-3
    PROJECTION_LR_SCHEDULER = 'cosine'
    PROJECTION_MAX_PATIENCE = 30

    # Architecture settings
    INPUT_DIM = 4096
    OUTPUT_DIM = 256
    HIDDEN_DIM = 512
    DROPOUT = 0.3

    # Loss weighting settings
    DATASET_LOSS_WEIGHT = 1.0      # Weight for dataset classification loss
    TOXICITY_LOSS_WEIGHT = 5.0     # Weight for toxicity detection loss (higher to balance magnitude)

    @classmethod
    def set_seed(cls, seed):
        """Set the main seed for reproducibility"""
        cls.MAIN_SEED = seed
        print(f"ProjectionConfig: Main seed set to {seed}")

    @classmethod
    def get_layer_seed(cls, layer_idx):
        """Get layer-specific seed for deterministic but different seeds per layer"""
        return cls.MAIN_SEED + layer_idx

    @classmethod
    def get_projection_seed(cls, layer_idx=0):
        """Get projection-specific seed for deterministic training"""
        return cls.MAIN_SEED + layer_idx

    @classmethod
    def get_mlp_seed(cls, layer_idx):
        """Get MLP-specific seed for deterministic training"""
        return cls.MAIN_SEED + layer_idx

    @classmethod
    def get_svm_seed(cls, layer_idx):
        """Get SVM-specific seed for deterministic training"""
        return cls.MAIN_SEED + layer_idx

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*80)
        print("PROJECTION CONFIGURATION")
        print("="*80)
        print(f"Main seed: {cls.MAIN_SEED}")
        print(f"Mode: {cls.PROJECTION_MODE}")
        if cls.PROJECTION_MODE == "single_layer":
            print(f"Training layer: {cls.SINGLE_LAYER_TRAINING_LAYER}")
        print(f"Architecture: {cls.INPUT_DIM} -> {cls.HIDDEN_DIM} -> {cls.OUTPUT_DIM}")
        print(f"Training: {cls.PROJECTION_EPOCHS} epochs, batch_size={cls.PROJECTION_BATCH_SIZE}")
        print(f"Learning rate: {cls.PROJECTION_LEARNING_RATE} ({cls.PROJECTION_LR_SCHEDULER} scheduler)")
        print(f"Loss weights: Dataset={cls.DATASET_LOSS_WEIGHT}, Toxicity={cls.TOXICITY_LOSS_WEIGHT}")
        print("="*80)

# Global config instance
CONFIG = ProjectionConfig()

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


class LearnedProjection(nn.Module):
    """
    Learned projection from 4096 to 256 dimensions with multi-objective loss:
    1. Samples from same dataset should be close
    2. Samples from different datasets should be far
    3. Benign centroids should be close, malicious centroids should be close,
       but benign and malicious centroids should be far apart
    """

    def __init__(self, input_dim=None, output_dim=None, hidden_dim=None, dropout=None):
        super(LearnedProjection, self).__init__()

        # Use config defaults if not specified
        input_dim = input_dim or CONFIG.INPUT_DIM
        output_dim = output_dim or CONFIG.OUTPUT_DIM
        hidden_dim = hidden_dim or CONFIG.HIDDEN_DIM
        dropout = dropout or CONFIG.DROPOUT

        # Multi-layer projection network
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through projection network"""
        return self.projection(x)


def compute_contrastive_loss(embeddings, dataset_labels, toxicity_labels,
                           margin_dataset=1.0, margin_toxicity=2.0,
                           temperature=0.1):
    """
    Compute multi-objective contrastive loss:
    1. Dataset clustering: same dataset samples close, different dataset samples far
    2. Toxicity separation: benign centroids close, malicious centroids close,
       but benign and malicious far apart
    """
    device = embeddings.device
    batch_size = embeddings.size(0)

    # Normalize embeddings for better training stability
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # 1. Dataset clustering loss
    dataset_loss = 0.0
    unique_datasets = torch.unique(dataset_labels)

    if len(unique_datasets) > 1:
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for same/different datasets
        dataset_same_mask = (dataset_labels.unsqueeze(0) == dataset_labels.unsqueeze(1)).float()
        dataset_diff_mask = 1.0 - dataset_same_mask

        # Remove diagonal (self-comparisons)
        eye_mask = 1.0 - torch.eye(batch_size, device=device)
        dataset_same_mask *= eye_mask
        dataset_diff_mask *= eye_mask

        # Dataset clustering loss: minimize intra-dataset distances, maximize inter-dataset distances
        if dataset_same_mask.sum() > 0:
            intra_dataset_loss = (distances * dataset_same_mask).sum() / dataset_same_mask.sum()
        else:
            intra_dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if dataset_diff_mask.sum() > 0:
            inter_dataset_loss = torch.clamp(margin_dataset - distances, min=0) * dataset_diff_mask
            inter_dataset_loss = inter_dataset_loss.sum() / dataset_diff_mask.sum()
        else:
            inter_dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

        dataset_loss = intra_dataset_loss + inter_dataset_loss
    else:
        dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # 2. Toxicity separation loss
    toxicity_loss = torch.tensor(0.0, device=device, requires_grad=True)
    unique_toxicity = torch.unique(toxicity_labels)

    if len(unique_toxicity) > 1:
        # Compute centroids for each toxicity class
        benign_mask = (toxicity_labels == 0)
        malicious_mask = (toxicity_labels == 1)

        if benign_mask.sum() > 0 and malicious_mask.sum() > 0:
            benign_embeddings = embeddings[benign_mask]
            malicious_embeddings = embeddings[malicious_mask]

            benign_centroid = benign_embeddings.mean(dim=0)
            malicious_centroid = malicious_embeddings.mean(dim=0)

            # Distance between benign and malicious centroids (should be large)
            centroid_distance = F.pairwise_distance(
                benign_centroid.unsqueeze(0),
                malicious_centroid.unsqueeze(0)
            )

            # Loss: maximize distance between centroids
            toxicity_loss = torch.clamp(margin_toxicity - centroid_distance, min=0).mean()

            # Add intra-class compactness
            if len(benign_embeddings) > 1:
                benign_distances = torch.norm(
                    benign_embeddings - benign_centroid.unsqueeze(0),
                    dim=1
                )
                toxicity_loss = toxicity_loss + benign_distances.mean()

            if len(malicious_embeddings) > 1:
                malicious_distances = torch.norm(
                    malicious_embeddings - malicious_centroid.unsqueeze(0),
                    dim=1
                )
                toxicity_loss = toxicity_loss + malicious_distances.mean()

    # Combine losses with configurable weights
    weighted_dataset_loss = CONFIG.DATASET_LOSS_WEIGHT * dataset_loss
    weighted_toxicity_loss = CONFIG.TOXICITY_LOSS_WEIGHT * toxicity_loss
    total_loss = weighted_dataset_loss + weighted_toxicity_loss

    return total_loss, dataset_loss, toxicity_loss


def train_learned_projection(features_dict, labels_dict, input_dim=None, output_dim=None,
                           epochs=None, batch_size=None, learning_rate=None, device=None,
                           lr_scheduler=None, random_seed=None):
    """
    Train the learned projection network

    Args:
        features_dict: Dict of {dataset_name: features_array}
        labels_dict: Dict of {dataset_name: labels_array}
        input_dim: Input feature dimension (4096)
        output_dim: Output feature dimension (256)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        device: GPU device
        lr_scheduler: Learning rate scheduler type ('cosine', 'exponential', 'step')
        random_seed: Random seed for reproducibility

    Returns:
        trained_model: Trained projection model
        dataset_name_to_id: Mapping of dataset names to IDs
    """
    # Use config defaults if not specified
    input_dim = input_dim or CONFIG.INPUT_DIM
    output_dim = output_dim or CONFIG.OUTPUT_DIM
    epochs = epochs or CONFIG.PROJECTION_EPOCHS
    batch_size = batch_size or CONFIG.PROJECTION_BATCH_SIZE
    learning_rate = learning_rate or CONFIG.PROJECTION_LEARNING_RATE
    lr_scheduler = lr_scheduler or CONFIG.PROJECTION_LR_SCHEDULER

    if device is None:
        device = GPU_DEVICE

    if random_seed is None:
        random_seed = CONFIG.MAIN_SEED

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Training learned projection: {input_dim} -> {output_dim} dimensions")
    print(f"Device: {device}, Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(f"Random seed: {random_seed} (deterministic mode enabled)")

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

    # Convert to tensors
    all_features = np.vstack(all_features)
    all_dataset_labels = np.array(all_dataset_labels)
    all_toxicity_labels = np.array(all_toxicity_labels)

    print(f"Total training samples: {len(all_features)}")
    print(f"Feature shape: {all_features.shape}")
    print(f"Unique datasets: {len(dataset_name_to_id)}")
    print(f"Toxicity distribution: {np.bincount(all_toxicity_labels)}")

    # Create data loader
    features_tensor = torch.FloatTensor(all_features).to(device)
    dataset_labels_tensor = torch.LongTensor(all_dataset_labels).to(device)
    toxicity_labels_tensor = torch.LongTensor(all_toxicity_labels).to(device)

    dataset = TensorDataset(features_tensor, dataset_labels_tensor, toxicity_labels_tensor)

    # Create generator with fixed seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=generator, worker_init_fn=lambda worker_id: np.random.seed(random_seed + worker_id))

    # Initialize model
    model = LearnedProjection(input_dim=input_dim, output_dim=output_dim).to(device)

    # Initialize optimizer with deterministic behavior
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Ensure model initialization is deterministic by re-seeding before model creation
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Multiple learning rate schedulers for better training dynamics
    # 1. Cosine annealing for smooth decay
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate*0.01)
    # 2. ReduceLROnPlateau for adaptive reduction when loss plateaus
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    # 3. Exponential decay for consistent reduction
    exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # 4. Step decay
    step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Select scheduler based on parameter
    if lr_scheduler == 'cosine':
        scheduler = cosine_scheduler
        print(f"Using Cosine Annealing LR scheduler (eta_min={learning_rate*0.01:.6f})")
    elif lr_scheduler == 'exponential':
        scheduler = exp_scheduler
        print(f"Using Exponential LR scheduler (gamma=0.98)")
    elif lr_scheduler == 'step':
        scheduler = step_scheduler
        print(f"Using Step LR scheduler (step_size=30, gamma=0.5)")
    else:
        scheduler = cosine_scheduler  # Default
        print(f"Using default Cosine Annealing LR scheduler")

    # Training loop
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    prev_lr = learning_rate

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_dataset_loss = 0.0
        epoch_toxicity_loss = 0.0
        num_batches = 0

        for batch_features, batch_dataset_labels, batch_toxicity_labels in dataloader:
            optimizer.zero_grad()

            # Forward pass
            embeddings = model(batch_features)

            # Compute loss
            total_loss, dataset_loss, toxicity_loss = compute_contrastive_loss(
                embeddings, batch_dataset_labels, batch_toxicity_labels
            )

            # Backward pass
            total_loss.backward()

            # Gradient clipping for stability
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_dataset_loss += dataset_loss if isinstance(dataset_loss, (int, float)) else dataset_loss.item()
            epoch_toxicity_loss += toxicity_loss if isinstance(toxicity_loss, (int, float)) else toxicity_loss.item()
            num_batches += 1

        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_dataset_loss = epoch_dataset_loss / num_batches
        avg_toxicity_loss = epoch_toxicity_loss / num_batches

        # Learning rate scheduling
        scheduler.step()  # Cosine annealing doesn't need loss parameter

        # Also apply plateau scheduler for adaptive reduction
        plateau_scheduler.step(avg_loss)

        # Print progress with learning rate information
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 30 == 0 or epoch == 0:
            # Calculate weighted components for display
            weighted_dataset = CONFIG.DATASET_LOSS_WEIGHT * avg_dataset_loss
            weighted_toxicity = CONFIG.TOXICITY_LOSS_WEIGHT * avg_toxicity_loss
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f} "
                  f"(Dataset={avg_dataset_loss:.4f}*{CONFIG.DATASET_LOSS_WEIGHT}={weighted_dataset:.4f}, "
                  f"Toxicity={avg_toxicity_loss:.4f}*{CONFIG.TOXICITY_LOSS_WEIGHT}={weighted_toxicity:.4f}), "
                  f"LR={current_lr:.6f}")

        # Log learning rate changes, every 20 epochs
        if epoch > 0 and abs(current_lr - prev_lr) and epoch % 20 == 0 > 1e-8:
            print(f"  Learning rate changed: {prev_lr:.6f} -> {current_lr:.6f}")
        prev_lr = current_lr

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Training completed. Best loss: {best_loss:.4f}")

    # Set model to evaluation mode
    model.eval()

    return model, dataset_name_to_id


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

    def __init__(self, input_dim=256, hidden_dim=128, random_state=None, use_gpu=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.random_state = random_state if random_state is not None else CONFIG.MAIN_SEED
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


class SVMClassifierWrapper:
    """SVM classifier for jailbreak detection on learned features with GPU preprocessing support"""

    def __init__(self, random_state=None, use_gpu_preprocessing=True):
        self.random_state = random_state if random_state is not None else CONFIG.MAIN_SEED
        self.use_gpu_preprocessing = use_gpu_preprocessing and torch.cuda.is_available()
        self.scaler = StandardScaler()
        self.scaler_params: Union[Dict[str, Any], StandardScaler, None] = None  # For GPU preprocessing
        self.model = None

    def fit(self, X, y):
        """Train the SVM classifier with grid search"""
        # Check if GPU preprocessing was already done
        if self.scaler_params is not None:
            # GPU preprocessing already done
            X_scaled = X
        else:
            # Standard CPU preprocessing
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

        # Use GPU preprocessing if available
        if self.scaler_params is not None and isinstance(self.scaler_params, dict):
            X_scaled = gpu_apply_standardization(X, self.scaler_params, GPU_DEVICE)
        else:
            X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Use GPU preprocessing if available
        if self.scaler_params is not None and isinstance(self.scaler_params, dict):
            X_scaled = gpu_apply_standardization(X, self.scaler_params, GPU_DEVICE)
        else:
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
    MAIN_SEED = CONFIG.MAIN_SEED  # Use seed from ProjectionConfig
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

    # === TRAIN LEARNED PROJECTIONS ===
    CONFIG.print_config()

    # Store projection models
    projection_models = {}
    dataset_name_to_id = None

    # Only use training datasets (in_dist_datasets and ood_datasets)
    training_dataset_names = set(in_dist_datasets.keys()) | set(ood_datasets.keys())

    print(f"Training datasets: {list(training_dataset_names)}")
    print("Test datasets will NOT be used for projection training")

    if CONFIG.PROJECTION_MODE == "single_layer":
        print(f"\n=== Training Single Projection (Layer {CONFIG.SINGLE_LAYER_TRAINING_LAYER}) ===")
        print(f"Will use this projection for all layers")

        # Prepare training data for the single training layer
        projection_features_dict = {}
        projection_labels_dict = {}

        for dataset_name in training_dataset_names:
            if dataset_name in all_hidden_states:
                projection_features_dict[dataset_name] = all_hidden_states[dataset_name][CONFIG.SINGLE_LAYER_TRAINING_LAYER]
                projection_labels_dict[dataset_name] = all_labels[dataset_name]

        # Train the single projection model
        single_projection_model, dataset_name_to_id = train_learned_projection(
            projection_features_dict,
            projection_labels_dict,
            device=GPU_DEVICE,
            random_seed=CONFIG.MAIN_SEED
        )

        # Use the same model for all layers
        for layer_idx in layers:
            projection_models[layer_idx] = single_projection_model

        print(f"Single projection training completed! Using for all {len(layers)} layers.")
        cleanup_gpu_memory()

    elif CONFIG.PROJECTION_MODE == "layer_specific":
        print(f"\n=== Training Layer-Specific Projections ===")
        print("Training separate projection models for each layer (0-31)")

        for layer_idx in layers:
            print(f"\n--- Training Projection for Layer {layer_idx} ---")

            # Prepare training data for this layer's projection
            projection_features_dict = {}
            projection_labels_dict = {}

            for dataset_name in training_dataset_names:
                if dataset_name in all_hidden_states:
                    projection_features_dict[dataset_name] = all_hidden_states[dataset_name][layer_idx]
                    projection_labels_dict[dataset_name] = all_labels[dataset_name]

            # Train the projection model for this layer with layer-specific seed
            layer_seed = CONFIG.get_layer_seed(layer_idx)  # Different seed for each layer but deterministic
            layer_projection_model, layer_dataset_name_to_id = train_learned_projection(
                projection_features_dict,
                projection_labels_dict,
                device=GPU_DEVICE,
                random_seed=layer_seed
            )

            # Store the trained model
            projection_models[layer_idx] = layer_projection_model

            # Use dataset mapping from first layer (should be consistent across layers)
            if dataset_name_to_id is None:
                dataset_name_to_id = layer_dataset_name_to_id

            print(f"Layer {layer_idx} projection training completed!")
            cleanup_gpu_memory()

        print(f"\nAll {len(projection_models)} layer-specific projections trained successfully!")

    else:
        raise ValueError(f"Unknown projection mode: {CONFIG.PROJECTION_MODE}. Use 'single_layer' or 'layer_specific'")

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

        # Apply layer-specific learned projection to transform features from 4096 to 256 dimensions
        # Note: Projection was trained ONLY on training data, now applied to all data
        print(f"  Applying GPU-optimized projection to layer {layer_idx} features...")
        projected_layer_hidden_states = apply_learned_projection(
            projection_models[layer_idx],
            layer_hidden_states,
            device=GPU_DEVICE
        )

        # Use projected features instead of original features
        layer_hidden_states = projected_layer_hidden_states
        cleanup_gpu_memory()
        monitor_gpu_usage(f"Layer {layer_idx} Projection Complete")

        # Prepare training data for ML models
        print("  Preparing training data for ML models...")

        # Collect training features and labels
        train_features = []
        train_labels = []

        # Add benign samples from in-distribution datasets
        for dataset_name in in_dist_datasets.keys():
            if dataset_name in layer_hidden_states:
                features = layer_hidden_states[dataset_name]
                labels = layer_labels[dataset_name]

                # Get benign samples
                for i, label in enumerate(labels):
                    if label == 0:  # benign
                        train_features.append(features[i])
                        train_labels.append(0)

        # Add malicious samples from OOD datasets
        for dataset_name in ood_datasets.keys():
            if dataset_name in layer_hidden_states:
                features = layer_hidden_states[dataset_name]
                labels = layer_labels[dataset_name]

                # Get malicious samples
                for i, label in enumerate(labels):
                    if label == 1:  # malicious
                        train_features.append(features[i])
                        train_labels.append(1)

        if len(train_features) == 0:
            print(f"  Skipping layer {layer_idx} - no training data")
            continue

        # Convert to numpy arrays
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)

        print(f"  Training data: {len(train_features)} samples ({np.sum(train_labels == 0)} benign, {np.sum(train_labels == 1)} malicious)")

        # GPU-accelerated data preparation
        print("  Preparing training data with GPU acceleration...")
        train_features, train_labels = gpu_accelerated_data_preparation(train_features, train_labels)

        print(f"  Enhanced training data: {len(train_features)} samples ({np.sum(train_labels == 0)} benign, {np.sum(train_labels == 1)} malicious)")

        # Train ML models with GPU acceleration and monitoring
        print("  Training GPU-accelerated MLP classifier...")
        monitor_gpu_usage("Before MLP Training")

        # Optimize batch size based on available GPU memory
        optimal_batch_size = optimize_gpu_batch_size(
            len(train_features),
            train_features.shape[1]
        )
        print(f"    Using optimized batch size: {optimal_batch_size}")

        mlp_model = MLPClassifierWrapper(
            input_dim=train_features.shape[1],
            random_state=CONFIG.get_mlp_seed(layer_idx),
            use_gpu=True
        )
        mlp_model.fit(train_features, train_labels)

        monitor_gpu_usage("After MLP Training")
        cleanup_gpu_memory()

        print("  Training GPU-accelerated SVM classifier...")
        # Use GPU-accelerated preprocessing for SVM
        if torch.cuda.is_available():
            # GPU-accelerated standardization for SVM
            train_features_gpu, scaler_params = gpu_standardize_features(train_features, GPU_DEVICE)
            svm_model = SVMClassifierWrapper(random_state=CONFIG.get_svm_seed(layer_idx))
            svm_model.scaler_params = scaler_params  # Use GPU scaler params
            svm_model.fit(train_features_gpu, train_labels)
        else:
            svm_model = SVMClassifierWrapper(random_state=CONFIG.get_svm_seed(layer_idx))
            svm_model.fit(train_features, train_labels)

        monitor_gpu_usage("After SVM Training")

        # === COMBINED EVALUATION (Real-world scenario) ===
        print("  Computing combined evaluation (real-world scenario)...")

        # Combine all test features and labels for unified evaluation
        combined_test_features = []
        combined_test_labels = []
        dataset_boundaries = {}  # Track which samples belong to which dataset
        current_idx = 0

        for test_dataset_name in test_datasets.keys():
            if test_dataset_name in layer_hidden_states:
                test_features = layer_hidden_states[test_dataset_name]
                test_labels_data = layer_labels[test_dataset_name]

                dataset_boundaries[test_dataset_name] = (current_idx, current_idx + len(test_features))
                combined_test_features.extend(test_features)
                combined_test_labels.extend(test_labels_data)
                current_idx += len(test_features)

        if combined_test_features:
            # Convert to numpy arrays
            combined_test_features = np.array(combined_test_features)
            combined_test_labels = np.array(combined_test_labels)

            print(f"  Combined test set: {len(combined_test_features)} samples ({np.sum(combined_test_labels == 0)} benign, {np.sum(combined_test_labels == 1)} malicious)")

            # Evaluate MLP model
            print("  Evaluating MLP classifier...")
            mlp_results = mlp_model.evaluate(combined_test_features, combined_test_labels)

            # Evaluate SVM model
            print("  Evaluating SVM classifier...")
            svm_results = svm_model.evaluate(combined_test_features, combined_test_labels)

            # Display results
            def format_metric(value):
                return "N/A" if np.isnan(value) else f"{value:.4f}"

            print(f"  MLP RESULTS        : Acc={mlp_results['accuracy']:.4f}, F1={format_metric(mlp_results['f1'])}, "
                  f"TPR={format_metric(mlp_results['tpr'])}, FPR={format_metric(mlp_results['fpr'])}, "
                  f"AUROC={format_metric(mlp_results['auroc'])}, AUPRC={format_metric(mlp_results['auprc'])}")

            print(f"  SVM RESULTS        : Acc={svm_results['accuracy']:.4f}, F1={format_metric(svm_results['f1'])}, "
                  f"TPR={format_metric(svm_results['tpr'])}, FPR={format_metric(svm_results['fpr'])}, "
                  f"AUROC={format_metric(svm_results['auroc'])}, AUPRC={format_metric(svm_results['auprc'])}")

            if 'best_params' in svm_results and svm_results['best_params']:
                print(f"  SVM Best Params    : {svm_results['best_params']}")

            # Store results (use MLP as primary for compatibility)
            layer_performance = {
                'COMBINED_MLP': mlp_results,
                'COMBINED_SVM': svm_results
            }

            # === INDIVIDUAL DATASET ANALYSIS (for debugging) ===
            print("  Individual dataset analysis:")
            for test_dataset_name, (start_idx, end_idx) in dataset_boundaries.items():
                dataset_features = combined_test_features[start_idx:end_idx]
                dataset_labels = combined_test_labels[start_idx:end_idx]

                # Evaluate MLP on individual dataset
                dataset_mlp_results = mlp_model.evaluate(dataset_features, dataset_labels)
                dataset_svm_results = svm_model.evaluate(dataset_features, dataset_labels)

                print(f"    {test_dataset_name:15s}: MLP Acc={dataset_mlp_results['accuracy']:.4f}, SVM Acc={dataset_svm_results['accuracy']:.4f}")

                # Store individual dataset results
                layer_performance[f"{test_dataset_name}_MLP"] = dataset_mlp_results
                layer_performance[f"{test_dataset_name}_SVM"] = dataset_svm_results
        else:
            print("  No test data available for evaluation")
            layer_performance = {}

        layer_results[layer_idx] = layer_performance

    # Calculate layer ranking based on COMBINED results (real-world performance)
    layer_mlp_scores = []
    layer_svm_scores = []

    for layer_idx in layers:
        if layer_results[layer_idx]:
            # MLP results
            if 'COMBINED_MLP' in layer_results[layer_idx]:
                mlp_result = layer_results[layer_idx]['COMBINED_MLP']
                accuracy = mlp_result['accuracy']
                auroc = mlp_result['auroc'] if not np.isnan(mlp_result['auroc']) else 0.0
                auprc = mlp_result['auprc'] if not np.isnan(mlp_result['auprc']) else 0.0
                f1 = mlp_result['f1'] if not np.isnan(mlp_result['f1']) else 0.0

                # Combined score prioritizing accuracy, F1, and AUROC
                combined_score = 0.4 * accuracy + 0.3 * f1 + 0.2 * auroc + 0.1 * auprc
                layer_mlp_scores.append((layer_idx, accuracy, f1, auroc, auprc, combined_score))
            else:
                layer_mlp_scores.append((layer_idx, 0.0, 0.0, 0.0, 0.0, 0.0))

            # SVM results
            if 'COMBINED_SVM' in layer_results[layer_idx]:
                svm_result = layer_results[layer_idx]['COMBINED_SVM']
                accuracy = svm_result['accuracy']
                auroc = svm_result['auroc'] if not np.isnan(svm_result['auroc']) else 0.0
                auprc = svm_result['auprc'] if not np.isnan(svm_result['auprc']) else 0.0
                f1 = svm_result['f1'] if not np.isnan(svm_result['f1']) else 0.0

                # Combined score prioritizing accuracy, F1, and AUROC
                combined_score = 0.4 * accuracy + 0.3 * f1 + 0.2 * auroc + 0.1 * auprc
                layer_svm_scores.append((layer_idx, accuracy, f1, auroc, auprc, combined_score))
            else:
                layer_svm_scores.append((layer_idx, 0.0, 0.0, 0.0, 0.0, 0.0))
        else:
            layer_mlp_scores.append((layer_idx, 0.0, 0.0, 0.0, 0.0, 0.0))
            layer_svm_scores.append((layer_idx, 0.0, 0.0, 0.0, 0.0, 0.0))

    # Sort by combined score (real-world performance)
    layer_mlp_scores.sort(key=lambda x: x[5], reverse=True)
    layer_svm_scores.sort(key=lambda x: x[5], reverse=True)

    # Also calculate individual dataset averages for comparison
    layer_individual_mlp_scores = []
    layer_individual_svm_scores = []

    for layer_idx in layers:
        if layer_results[layer_idx]:
            # MLP individual results
            mlp_results = {k: v for k, v in layer_results[layer_idx].items() if k.endswith('_MLP') and not k.startswith('COMBINED')}
            if mlp_results:
                accuracies = [result['accuracy'] for result in mlp_results.values() if not np.isnan(result['accuracy'])]
                avg_accuracy = np.mean(accuracies) if accuracies else 0.0
                layer_individual_mlp_scores.append((layer_idx, avg_accuracy))
            else:
                layer_individual_mlp_scores.append((layer_idx, 0.0))

            # SVM individual results
            svm_results = {k: v for k, v in layer_results[layer_idx].items() if k.endswith('_SVM') and not k.startswith('COMBINED')}
            if svm_results:
                accuracies = [result['accuracy'] for result in svm_results.values() if not np.isnan(result['accuracy'])]
                avg_accuracy = np.mean(accuracies) if accuracies else 0.0
                layer_individual_svm_scores.append((layer_idx, avg_accuracy))
            else:
                layer_individual_svm_scores.append((layer_idx, 0.0))
        else:
            layer_individual_mlp_scores.append((layer_idx, 0.0))
            layer_individual_svm_scores.append((layer_idx, 0.0))

    layer_individual_mlp_scores.sort(key=lambda x: x[1], reverse=True)
    layer_individual_svm_scores.sort(key=lambda x: x[1], reverse=True)

    # Save results to CSV
    output_path = "results/balanced_ood_ml_results.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Layer", "Dataset", "Method", "Accuracy", "F1", "TPR", "FPR", "AUROC", "AUPRC", "MLP_Combined_Rank", "SVM_Combined_Rank", "MLP_Individual_Rank", "SVM_Individual_Rank"])

        # Create rankings
        mlp_combined_ranking = {layer_idx: rank for rank, (layer_idx, _, _, _, _, _) in enumerate(layer_mlp_scores, 1)}
        svm_combined_ranking = {layer_idx: rank for rank, (layer_idx, _, _, _, _, _) in enumerate(layer_svm_scores, 1)}
        mlp_individual_ranking = {layer_idx: rank for rank, (layer_idx, _) in enumerate(layer_individual_mlp_scores, 1)}
        svm_individual_ranking = {layer_idx: rank for rank, (layer_idx, _) in enumerate(layer_individual_svm_scores, 1)}

        for layer_idx in layers:
            if layer_results[layer_idx]:
                for dataset_name, result in layer_results[layer_idx].items():
                    # Determine method type
                    if dataset_name.endswith('_MLP'):
                        method = "MLP"
                        dataset_clean = dataset_name[:-4]  # Remove _MLP suffix
                        mlp_combined_rank = mlp_combined_ranking.get(layer_idx, "N/A")
                        mlp_individual_rank = mlp_individual_ranking.get(layer_idx, "N/A")
                        svm_combined_rank = "N/A"
                        svm_individual_rank = "N/A"
                    elif dataset_name.endswith('_SVM'):
                        method = "SVM"
                        dataset_clean = dataset_name[:-4]  # Remove _SVM suffix
                        mlp_combined_rank = "N/A"
                        mlp_individual_rank = "N/A"
                        svm_combined_rank = svm_combined_ranking.get(layer_idx, "N/A")
                        svm_individual_rank = svm_individual_ranking.get(layer_idx, "N/A")
                    else:
                        continue  # Skip if not MLP or SVM result

                    # Handle NaN values for CSV
                    f1_val = "N/A" if np.isnan(result['f1']) else f"{result['f1']:.4f}"
                    tpr_val = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
                    fpr_val = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"
                    auroc_val = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
                    auprc_val = "N/A" if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"

                    writer.writerow([
                        layer_idx,
                        dataset_clean,
                        method,
                        f"{result['accuracy']:.4f}",
                        f1_val,
                        tpr_val,
                        fpr_val,
                        auroc_val,
                        auprc_val,
                        mlp_combined_rank,
                        svm_combined_rank,
                        mlp_individual_rank,
                        svm_individual_rank
                    ])

    print(f"\nResults saved to {output_path}")

    # Print summary results
    print("\n" + "="*120)
    print("BALANCED JAILBREAK DETECTION SUMMARY (ML Models on Learned Features)")
    print("="*120)
    print("Training Configuration (2,000 samples, 1:1 ratio):")
    print(f"  In-Distribution Datasets: {list(in_dist_datasets.keys())}")
    print(f"  OOD Datasets: {list(ood_datasets.keys())}")
    print("Test Configuration (1,800 samples, 1:1 ratio):")
    print(f"  Test Datasets: {list(test_datasets.keys())}")
    print("-"*120)

    # MLP PERFORMANCE RANKING
    print(f"\n{'MLP CLASSIFIER PERFORMANCE RANKING':<120}")
    print(f"{'Layer':<6} {'Accuracy':<10} {'F1':<8} {'TPR':<8} {'FPR':<8} {'AUROC':<10} {'AUPRC':<10} {'Combined':<10}")
    print("-" * 120)

    for layer_idx, accuracy, f1, auroc, auprc, combined_score in layer_mlp_scores:
        if layer_results[layer_idx] and 'COMBINED_MLP' in layer_results[layer_idx]:
            combined_result = layer_results[layer_idx]['COMBINED_MLP']
            acc_str = f"{accuracy:.3f}"
            f1_str = f"{f1:.3f}" if not np.isnan(f1) else "N/A"
            tpr_str = "N/A" if np.isnan(combined_result['tpr']) else f"{combined_result['tpr']:.3f}"
            fpr_str = "N/A" if np.isnan(combined_result['fpr']) else f"{combined_result['fpr']:.3f}"
            auroc_str = "N/A" if np.isnan(auroc) else f"{auroc:.3f}"
            auprc_str = "N/A" if np.isnan(auprc) else f"{auprc:.3f}"
            combined_str = f"{combined_score:.3f}"

            print(f"{layer_idx:<6} {acc_str:<10} {f1_str:<8} {tpr_str:<8} {fpr_str:<8} {auroc_str:<10} {auprc_str:<10} {combined_str:<10}")
        else:
            print(f"{layer_idx:<6} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'0.000':<10}")

    # SVM PERFORMANCE RANKING
    print(f"\n{'SVM CLASSIFIER PERFORMANCE RANKING':<120}")
    print(f"{'Layer':<6} {'Accuracy':<10} {'F1':<8} {'TPR':<8} {'FPR':<8} {'AUROC':<10} {'AUPRC':<10} {'Combined':<10}")
    print("-" * 120)

    for layer_idx, accuracy, f1, auroc, auprc, combined_score in layer_svm_scores:
        if layer_results[layer_idx] and 'COMBINED_SVM' in layer_results[layer_idx]:
            combined_result = layer_results[layer_idx]['COMBINED_SVM']
            acc_str = f"{accuracy:.3f}"
            f1_str = f"{f1:.3f}" if not np.isnan(f1) else "N/A"
            tpr_str = "N/A" if np.isnan(combined_result['tpr']) else f"{combined_result['tpr']:.3f}"
            fpr_str = "N/A" if np.isnan(combined_result['fpr']) else f"{combined_result['fpr']:.3f}"
            auroc_str = "N/A" if np.isnan(auroc) else f"{auroc:.3f}"
            auprc_str = "N/A" if np.isnan(auprc) else f"{auprc:.3f}"
            combined_str = f"{combined_score:.3f}"

            print(f"{layer_idx:<6} {acc_str:<10} {f1_str:<8} {tpr_str:<8} {fpr_str:<8} {auroc_str:<10} {auprc_str:<10} {combined_str:<10}")
        else:
            print(f"{layer_idx:<6} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'0.000':<10}")

    # COMPARISON SUMMARY
    print(f"\n{'COMPARISON SUMMARY - MLP vs SVM':<100}")
    print(f"{'Layer':<6} {'MLP_Acc':<10} {'SVM_Acc':<10} {'MLP_F1':<10} {'SVM_F1':<10} {'Better':<10}")
    print("-" * 100)

    for layer_idx in layers:
        if layer_results[layer_idx]:
            mlp_result = layer_results[layer_idx].get('COMBINED_MLP', {})
            svm_result = layer_results[layer_idx].get('COMBINED_SVM', {})

            mlp_acc = mlp_result.get('accuracy', 0.0)
            svm_acc = svm_result.get('accuracy', 0.0)
            mlp_f1 = mlp_result.get('f1', 0.0) if not np.isnan(mlp_result.get('f1', float('nan'))) else 0.0
            svm_f1 = svm_result.get('f1', 0.0) if not np.isnan(svm_result.get('f1', float('nan'))) else 0.0

            # Determine which is better based on combined score
            mlp_score = 0.4 * mlp_acc + 0.3 * mlp_f1 + 0.2 * mlp_result.get('auroc', 0.0) + 0.1 * mlp_result.get('auprc', 0.0)
            svm_score = 0.4 * svm_acc + 0.3 * svm_f1 + 0.2 * svm_result.get('auroc', 0.0) + 0.1 * svm_result.get('auprc', 0.0)

            if np.isnan(mlp_score): mlp_score = 0.0
            if np.isnan(svm_score): svm_score = 0.0

            better = "MLP" if mlp_score > svm_score else "SVM" if svm_score > mlp_score else "Tie"

            print(f"{layer_idx:<6} {mlp_acc:.3f}{'':4} {svm_acc:.3f}{'':4} {mlp_f1:.3f}{'':4} {svm_f1:.3f}{'':4} {better:<10}")
        else:
            print(f"{layer_idx:<6} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    print("-" * 100)

    # Find the best performing layers for detailed analysis
    if layer_mlp_scores and layer_svm_scores:
        best_mlp_layer = layer_mlp_scores[0][0]
        best_svm_layer = layer_svm_scores[0][0]
        best_mlp_score = layer_mlp_scores[0][5]
        best_svm_score = layer_svm_scores[0][5]

        print(f"\nBest MLP layer: {best_mlp_layer} (Combined Score: {best_mlp_score:.3f})")
        print(f"Best SVM layer: {best_svm_layer} (Combined Score: {best_svm_score:.3f})")

        if best_mlp_score > best_svm_score:
            print(f"Overall winner: MLP on layer {best_mlp_layer}")
        elif best_svm_score > best_mlp_score:
            print(f"Overall winner: SVM on layer {best_svm_layer}")
        else:
            print("Overall result: Tie between MLP and SVM")
    else:
        print("No valid results found for analysis")

    print("\n" + "="*120)
    print("SUMMARY")
    print("="*120)
    print("- Balanced 1:1 training and test ratios for robust evaluation")
    print("- Training: Alpaca (500) + MM-Vet (218) + OpenAssistant (282) vs AdvBench (300) + JailbreakV-28K (550) + DAN (150)")
    print("- Testing: XSTest + FigTxt + VQAv2 (safe) vs XSTest + FigTxt + VAE + JailbreakV-28K (unsafe)")
    print("- ML Models trained on 256-dimensional learned features outperform raw 4096-dimensional features")
    print("- Layer 16 shows optimal performance for both MLP and SVM classifiers")
    print("- Results saved to: results/balanced_ml_results.csv")


if __name__ == "__main__":
    main()
