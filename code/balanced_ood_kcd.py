import csv
import numpy as np
import random
import warnings
import signal
import sys
import os
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

from load_datasets import *

# Smart import handling based on command line arguments and available dependencies
import sys

def detect_model_from_args():
    """Detect model type from command line arguments"""
    if len(sys.argv) > 1:
        model_arg = sys.argv[1].lower()
        if model_arg == 'qwen':
            return 'qwen'
        elif model_arg == 'llava':
            return 'llava'
    return 'llava'  # Default

# Determine which model we're using
REQUESTED_MODEL = detect_model_from_args()

# Import appropriate dependencies based on requested model
if REQUESTED_MODEL == 'qwen':
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        from feature_extractor_qwen import HiddenStateExtractor
        QWEN_AVAILABLE = True
        LLAVA_AVAILABLE = False
        print("Using Qwen model configuration")
    except ImportError as e:
        print(f"Error: Qwen dependencies not available: {e}")
        print("Please install: pip install qwen-vl-utils transformers>=4.37.0")
        sys.exit(1)
else:
    try:
        from feature_extractor import HiddenStateExtractor
        LLAVA_AVAILABLE = True
        QWEN_AVAILABLE = False
        print("Using LLaVA model configuration")
    except ImportError as e:
        print(f"Error: LLaVA dependencies not available: {e}")
        print("Please install LLaVA dependencies")
        sys.exit(1)

def get_model_specific_extractor(model_path, model_type):
    """Get the appropriate feature extractor based on model type"""
    # Since we've already imported the correct HiddenStateExtractor based on model type,
    # we can just use it directly
    return HiddenStateExtractor(model_path)

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
class ProjectionConfig:
    """Global configuration for projection training methods"""

    # Projection training mode
    # "single_layer": Train one projection on layer 16, use for all layers (original method)
    # "layer_specific": Train separate projection for each layer (new method)
    PROJECTION_MODE = "layer_specific"  # Change this to switch modes

    # Single layer mode settings
    SINGLE_LAYER_TRAINING_LAYER = 16  # Which layer to use for training projection

    # Training hyperparameters
    PROJECTION_EPOCHS = 200
    PROJECTION_BATCH_SIZE = 64
    PROJECTION_LEARNING_RATE = 1e-3
    PROJECTION_LR_SCHEDULER = 'cosine'
    PROJECTION_MAX_PATIENCE = 15

    # Architecture settings
    INPUT_DIM = 4096
    OUTPUT_DIM = 256
    HIDDEN_DIM = 512
    DROPOUT = 0.3

    # Loss weighting settings
    DATASET_LOSS_WEIGHT = 1.0      # Weight for dataset classification loss
    TOXICITY_LOSS_WEIGHT = 5.0     # Weight for toxicity detection loss (higher to balance magnitude)

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*80)
        print("PROJECTION CONFIGURATION")
        print("="*80)
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
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        return torch.device('cuda:0')  # Primary GPU
    else:
        print("CUDA not available, falling back to CPU")
        return torch.device('cpu')

GPU_DEVICE = setup_gpu_environment()

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_info():
    """Get GPU memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
    return "GPU not available"


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
                           margin_dataset=1.0, margin_toxicity=2.0):
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
    dataset_loss = torch.tensor(0.0, device=device, requires_grad=True)
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
                           lr_scheduler=None, random_seed=42):
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

    # Create deterministic generator for DataLoader
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=generator, worker_init_fn=lambda worker_id: np.random.seed(random_seed + worker_id))

    # Initialize model
    model = LearnedProjection(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

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
    max_patience = 15
    prev_lr = learning_rate  # Initialize for learning rate change tracking

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # Log learning rate changes
        if epoch > 0 and abs(current_lr - prev_lr) > 1e-8 and epoch % 30 == 0:
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

    print("Applying learned projection to features...")

    with torch.no_grad():
        for dataset_name, features in features_dict.items():
            features_array = np.array(features)
            print(f"  Projecting {dataset_name}: {features_array.shape} -> ", end="")

            # Convert to tensor and project in batches to manage memory
            batch_size = 1000
            projected_features = []

            for i in range(0, len(features_array), batch_size):
                batch_features = features_array[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_features).to(device)

                # Apply projection
                batch_projected = model(batch_tensor)
                projected_features.append(batch_projected.cpu().numpy())

                # Clean up GPU memory
                del batch_tensor, batch_projected
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine batches
            projected_features = np.vstack(projected_features)
            projected_features_dict[dataset_name] = projected_features

            print(f"{projected_features.shape}")

    return projected_features_dict

def euclidean_distance_batch_gpu(X, Y, device=None):
    """GPU-accelerated batch Euclidean distance computation between X and Y"""
    if device is None:
        device = GPU_DEVICE

    try:
        # Convert inputs to GPU tensors (use float32 for memory efficiency)
        X_gpu = torch.tensor(X, dtype=torch.float32, device=device)
        Y_gpu = torch.tensor(Y, dtype=torch.float32, device=device)

        # Ensure proper dimensions
        if X_gpu.dim() == 1:
            X_gpu = X_gpu.unsqueeze(0)  # Add batch dimension
        if Y_gpu.dim() == 1:
            Y_gpu = Y_gpu.unsqueeze(0)  # Add batch dimension

        # Compute pairwise Euclidean distances
        # X: (n_samples, n_features), Y: (n_training, n_features)
        # Result: (n_samples, n_training)
        distances = torch.cdist(X_gpu, Y_gpu, p=2)

        # Move back to CPU
        return distances.cpu().numpy()

    except Exception as e:
        print(f"    GPU Euclidean distance failed: {e}, using CPU fallback")
        # CPU fallback
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        distances = []
        for x in X:
            row_distances = []
            for y in Y:
                dist = np.linalg.norm(x - y)
                row_distances.append(dist)
            distances.append(row_distances)

        return np.array(distances)

class KCDDetector:
    """
    Contrastive K-Nearest Neighbors (KNN) based jailbreak detection.

    This implements an improved KNN algorithm that uses BOTH benign and malicious samples:
    1. Uses L2-normalized features (critical for performance)
    2. Computes contrastive score: distance_to_malicious - distance_to_benign
    3. Leverages both classes for better decision boundary learning
    4. Enhanced with GPU acceleration for distance computation
    """

    def __init__(self, k, use_gpu=True, normalization=True):
        """
        Initialize KCD detector.

        Args:
            k: Number of nearest neighbors (50 for small datasets like ours)
            use_gpu: Whether to use GPU acceleration for distance computation
            normalization: Whether to L2-normalize features (critical - always True)
        """
        self.k = k
        self.normalization = normalization
        self.threshold = 0.0
        self.benign_features = None
        self.malicious_features = None
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if self.use_gpu:
            print(f"    KCDDetector initialized with GPU acceleration on {GPU_DEVICE}")
        else:
            print("    KCDDetector initialized with CPU computation")

        print(f"    Hyperparameters: k={self.k}, normalization={self.normalization}")

    def _normalize_features(self, features):
        """L2 normalize features - critical for KNN performance"""
        if not self.normalization:
            return features

        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # L2 normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms

    def _euclidean_distance_batch(self, X, Y):
        """Batch Euclidean distance computation with GPU acceleration"""
        if self.use_gpu:
            return euclidean_distance_batch_gpu(X, Y)
        else:
            # CPU batch computation using broadcasting
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if Y.ndim == 1:
                Y = Y.reshape(1, -1)

            # Compute pairwise distances using broadcasting
            # X: (n_samples, n_features), Y: (n_training, n_features)
            # Result: (n_samples, n_training)
            X_expanded = X[:, np.newaxis, :]  # (n_samples, 1, n_features)
            Y_expanded = Y[np.newaxis, :, :]  # (1, n_training, n_features)
            distances = np.sqrt(np.sum((X_expanded - Y_expanded) ** 2, axis=2))
            return distances

    def fit_training_data(self, benign_data, malicious_data):
        """
        Fit KCD detector with both benign and malicious training data.

        Args:
            benign_data: Dict of {dataset_name: list_of_features} for benign samples
            malicious_data: Dict of {dataset_name: list_of_features} for malicious samples
        """
        print("Fitting KCD detector with both benign and malicious training data...")

        # Process benign data
        all_benign_features = []
        benign_total = 0

        for dataset_name, features in benign_data.items():
            if len(features) > 0:
                all_benign_features.extend(features)
                benign_total += len(features)
                print(f"  Benign {dataset_name}: {len(features)} samples")
            else:
                print(f"  Warning: Benign {dataset_name} has no samples, skipping")

        # Process malicious data
        all_malicious_features = []
        malicious_total = 0

        for dataset_name, features in malicious_data.items():
            if len(features) > 0:
                all_malicious_features.extend(features)
                malicious_total += len(features)
                print(f"  Malicious {dataset_name}: {len(features)} samples")
            else:
                print(f"  Warning: Malicious {dataset_name} has no samples, skipping")

        if benign_total == 0:
            raise ValueError("No benign training data provided!")
        if malicious_total == 0:
            raise ValueError("No malicious training data provided!")

        # Convert to numpy arrays and normalize
        self.benign_features = np.array(all_benign_features)
        self.benign_features = self._normalize_features(self.benign_features)

        self.malicious_features = np.array(all_malicious_features)
        self.malicious_features = self._normalize_features(self.malicious_features)

        print(f"  Total benign training samples: {len(self.benign_features)}")
        print(f"  Total malicious training samples: {len(self.malicious_features)}")
        print(f"  Feature dimension: {self.benign_features.shape[1]}")

    def compute_kcd_score(self, x):
        """
        Compute KCD outlier score for a sample.

        Score = distance_to_k_nearest_malicious - distance_to_k_nearest_benign

        Intuition:
        - If sample is benign: distance_to_benign is small, distance_to_malicious is large → negative score
        - If sample is malicious: distance_to_benign is large, distance_to_malicious is small → positive score

        Higher scores indicate more likely to be malicious (jailbreak).
        """
        if self.benign_features is None or self.malicious_features is None:
            raise ValueError("Detector not fitted! Call fit_training_data() first.")

        # Normalize test sample
        x_normalized = self._normalize_features(x.reshape(1, -1))

        # Compute distances to benign training samples
        benign_distances = self._euclidean_distance_batch(x_normalized, self.benign_features)
        benign_distances = benign_distances.flatten()
        benign_distances_sorted = np.sort(benign_distances)
        kth_benign_distance = benign_distances_sorted[min(self.k - 1, len(benign_distances_sorted) - 1)]

        # Compute distances to malicious training samples
        malicious_distances = self._euclidean_distance_batch(x_normalized, self.malicious_features)
        malicious_distances = malicious_distances.flatten()
        malicious_distances_sorted = np.sort(malicious_distances)
        kth_malicious_distance = malicious_distances_sorted[min(self.k - 1, len(malicious_distances_sorted) - 1)]

        # Contrastive score: closer to malicious samples = higher score
        contrastive_score = kth_benign_distance - kth_malicious_distance

        return contrastive_score

    def fit_threshold(self, validation_features, validation_labels):
        """Find optimal threshold using validation data"""
        # Compute KCD scores for validation data
        _, scores = self.predict(validation_features)
        validation_labels = np.array(validation_labels)

        # Check if we have both classes
        unique_labels = np.unique(validation_labels)
        if len(unique_labels) < 2:
            print(f"Warning: Only one class in validation data: {unique_labels}")
            self.threshold = 0.0
            return 0.0

        # Find threshold that maximizes balanced accuracy
        from sklearn.metrics import f1_score, balanced_accuracy_score

        # Analyze score distributions
        benign_scores = scores[validation_labels == 0]
        malicious_scores = scores[validation_labels == 1]

        benign_mean, benign_std = np.mean(benign_scores), np.std(benign_scores)
        malicious_mean, malicious_std = np.mean(malicious_scores), np.std(malicious_scores)

        # Calculate separation between distributions
        separation = abs(malicious_mean - benign_mean) / (benign_std + malicious_std + 1e-8)

        print(f"  Validation score separation: {separation:.2f} (higher is better)")
        print(f"  Benign scores: mean={benign_mean:.4f}, std={benign_std:.4f}")
        print(f"  Malicious scores: mean={malicious_mean:.4f}, std={malicious_std:.4f}")

        # Use adaptive threshold range based on score distributions
        if separation > 2.0:  # Well-separated distributions
            # Use narrow range around the midpoint
            midpoint = (benign_mean + malicious_mean) / 2
            range_width = min(benign_std, malicious_std) * 2
            score_range = [midpoint - range_width, midpoint + range_width]
            print(f"  Using narrow threshold range around midpoint: [{score_range[0]:.4f}, {score_range[1]:.4f}]")
        else:  # Overlapping distributions - use wider range
            score_range = np.percentile(scores, [5, 95])
            range_width = score_range[1] - score_range[0]
            score_range[0] -= 0.2 * range_width
            score_range[1] += 0.2 * range_width
            print(f"  Using wide threshold range for overlapping distributions: [{score_range[0]:.4f}, {score_range[1]:.4f}]")

        thresholds = np.linspace(score_range[0], score_range[1], 200)
        best_score = 0
        best_threshold = 0.0
        best_f1 = 0
        best_balanced_acc = 0

        for threshold in thresholds:
            y_pred = (scores > threshold).astype(int)
            try:
                balanced_acc = balanced_accuracy_score(validation_labels, y_pred)
                f1 = f1_score(validation_labels, y_pred, zero_division=0)
                combined_score = 0.8 * balanced_acc + 0.2 * f1

                if combined_score > best_score:
                    best_score = combined_score
                    best_threshold = threshold
                    best_f1 = f1
                    best_balanced_acc = balanced_acc
            except:
                continue

        print(f"  Optimal threshold: {best_threshold:.4f} (Balanced Acc: {best_balanced_acc:.4f}, F1: {best_f1:.4f})")

        self.threshold = best_threshold
        return best_threshold

    def predict(self, features):
        """Predict whether samples are OOD (jailbreak) using KCD"""
        if len(features) > 100:  # Use batch processing for larger datasets
            scores = self._compute_kcd_scores_batch(features)
        else:
            scores = np.array([self.compute_kcd_score(x) for x in features])
        predictions = (scores > self.threshold).astype(int)
        return predictions, scores

    def _compute_kcd_scores_batch(self, features):
        """Batch KCD score computation with GPU acceleration"""
        features = np.array(features)

        # Normalize all test features
        features_normalized = self._normalize_features(features)

        # Compute distances to benign training samples (batch)
        benign_distances = self._euclidean_distance_batch(features_normalized, self.benign_features)
        # benign_distances shape: (n_test_samples, n_benign_training_samples)

        # Compute distances to malicious training samples (batch)
        malicious_distances = self._euclidean_distance_batch(features_normalized, self.malicious_features)
        # malicious_distances shape: (n_test_samples, n_malicious_training_samples)

        # For each test sample, compute contrastive score
        scores = []
        for i in range(len(features_normalized)):
            # k-th nearest benign distance
            benign_sample_distances = benign_distances[i]
            benign_distances_sorted = np.sort(benign_sample_distances)
            kth_benign_distance = benign_distances_sorted[min(self.k - 1, len(benign_distances_sorted) - 1)]

            # k-th nearest malicious distance
            malicious_sample_distances = malicious_distances[i]
            malicious_distances_sorted = np.sort(malicious_sample_distances)
            kth_malicious_distance = malicious_distances_sorted[min(self.k - 1, len(malicious_distances_sorted) - 1)]

            # Contrastive score
            contrastive_score = kth_benign_distance - kth_malicious_distance
            scores.append(contrastive_score)

        return np.array(scores)
    
    def evaluate(self, test_features, test_labels):
        """Evaluate the detector on test data"""
        predictions, scores = self.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions)

        # Calculate F1 score
        f1 = f1_score(test_labels, predictions, zero_division=0)

        # Calculate TPR, FPR from confusion matrix and AUROC/AUPRC
        unique_classes = np.unique(test_labels)
        if len(unique_classes) > 1:
            try:
                # Calculate TPR, FPR from confusion matrix
                tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Sensitivity/Recall)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

                # Calculate AUROC and AUPRC
                fpr_curve, tpr_curve, _ = roc_curve(test_labels, scores)
                auroc = auc(fpr_curve, tpr_curve)
                precision, recall, _ = precision_recall_curve(test_labels, scores)
                auprc = auc(recall, precision)
            except:
                tpr = float('nan')
                fpr = float('nan')
                auroc = float('nan')
                auprc = float('nan')
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
            'predictions': predictions,
            'scores': scores,
            'threshold': self.threshold
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


def prepare_knn_data_structure(datasets_dict, hidden_states_dict, labels_dict):
    benign_data = {}
    malicious_data = {}

    for dataset_name in datasets_dict.keys():
        if dataset_name not in hidden_states_dict:
            continue

        features = hidden_states_dict[dataset_name]
        labels = labels_dict[dataset_name]

        # Separate benign and malicious samples
        benign_features = [features[i] for i, label in enumerate(labels) if label == 0]
        malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]

        if benign_features:
            benign_data[f"{dataset_name}_benign"] = benign_features
        if malicious_features:
            malicious_data[f"{dataset_name}_malicious"] = malicious_features

    return benign_data, malicious_data


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

    # Support both LLaVA and Qwen models
    import sys

    # Check command line arguments for model specification
    if len(sys.argv) > 1:
        model_arg = sys.argv[1].lower()
        if model_arg == 'qwen':
            model_path = "./model/Qwen2.5-VL-3B-Instruct"
            model_type = 'qwen'
        elif model_arg == 'llava':
            model_path = "model/llava-v1.6-vicuna-7b/"
            model_type = 'llava'
        else:
            print(f"Unknown model type: {model_arg}. Use 'llava' or 'qwen'")
            sys.exit(1)
    else:
        # Default to LLaVA for backward compatibility
        model_path = "model/llava-v1.6-vicuna-7b/"
        model_type = 'llava'

    # Set random seed for reproducibility (use consistent seed)
    MAIN_SEED = 42  # Match the seed used elsewhere in the script
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

    # Initialize model-specific feature extractor
    extractor = get_model_specific_extractor(model_path, model_type)

    # Get the layer range for this model
    layer_start, layer_end = extractor.get_default_layer_range()
    num_layers = layer_end - layer_start + 1
    print(f"Model layer range: {layer_start}-{layer_end} ({num_layers} layers)")

    print("="*80)
    print(f"BALANCED OOD-BASED JAILBREAK DETECTION USING KCD WITH LAYER-SPECIFIC PROJECTIONS ({model_type.upper()})")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    print("Approach: Use BOTH benign and malicious prompts for training")
    print("          Compute contrastive score: distance_to_malicious - distance_to_benign")
    print("          Apply L2 normalization and Euclidean distance")
    print("          Enhanced with layer-specific projections: 4096 -> 256 dimensions per layer")
    print("          Multi-objective contrastive loss for optimal feature learning")
    print("          Enhanced with GPU acceleration for distance computation")
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
            samples, f"{dataset_name}", use_cache=True,
            batch_size=batch_size, memory_cleanup_freq=memory_cleanup_freq,
            experiment_name="balanced_ml_detection"
        )
        all_hidden_states[dataset_name] = hidden_states
        all_labels[dataset_name] = labels

    layers = list(range(layer_start, layer_end + 1))  # Use dynamic layer range
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
            random_seed=MAIN_SEED
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
            layer_seed = MAIN_SEED + layer_idx  # Different seed for each layer but deterministic
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

        # Prepare data for this layer
        layer_hidden_states = {}
        layer_labels = {}

        for dataset_name in all_datasets.keys():
            if dataset_name in all_hidden_states:
                layer_hidden_states[dataset_name] = all_hidden_states[dataset_name][layer_idx]
                layer_labels[dataset_name] = all_labels[dataset_name]

        # Apply layer-specific learned projection to transform features from 4096 to 256 dimensions
        # Note: Projection was trained ONLY on training data, now applied to all data
        print(f"  Applying layer-specific projection to layer {layer_idx} features...")
        projected_layer_hidden_states = apply_learned_projection(
            projection_models[layer_idx],
            layer_hidden_states,
            device=GPU_DEVICE
        )

        # Use projected features instead of original features
        layer_hidden_states = projected_layer_hidden_states
        cleanup_gpu_memory()

        # Prepare benign training data for KCD
        benign_data, _ = prepare_knn_data_structure(
            in_dist_datasets,
            {k: v for k, v in layer_hidden_states.items() if k in in_dist_datasets},
            {k: v for k, v in layer_labels.items() if k in in_dist_datasets}
        )

        detector = KCDDetector(k=50, use_gpu=True, normalization=True)

        # Prepare malicious training data from OOD datasets
        malicious_data = {}
        for dataset_name in ood_datasets.keys():
            if dataset_name in layer_hidden_states:
                features = layer_hidden_states[dataset_name]
                labels = layer_labels[dataset_name]

                # Get malicious samples (should be most/all samples in OOD datasets)
                malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
                if malicious_features:
                    malicious_data[f"{dataset_name}_malicious"] = malicious_features

        # Fit KCD detector with both benign and malicious training data
        print(f"  {get_gpu_memory_info()}")
        try:
            detector.fit_training_data(benign_data, malicious_data)
            cleanup_gpu_memory()
        except ValueError as e:
            print(f"  Skipping layer {layer_idx} - {e}")
            continue

        # Debug: Check if detector was fitted
        if detector.benign_features is None or detector.malicious_features is None:
            print(f"  Skipping layer {layer_idx} - detector not fitted properly")
            continue

        print(f"  KCD detector fitted with {len(detector.benign_features)} benign + {len(detector.malicious_features)} malicious samples, k={detector.k}")

        # Create validation set from training data for threshold optimization
        val_features = []
        val_labels = []

        # Sample from benign training data
        for _, features in benign_data.items():
            sample_size = min(100, len(features))
            if sample_size > 0:
                sampled_features = random.sample(features, sample_size)
                val_features.extend(sampled_features)
                val_labels.extend([0] * len(sampled_features))

        # Sample from malicious training data (from OOD datasets)
        for dataset_name in ood_datasets.keys():
            if dataset_name in layer_hidden_states:
                features = layer_hidden_states[dataset_name]
                labels = layer_labels[dataset_name]

                # Get malicious samples
                malicious_features = [features[i] for i, label in enumerate(labels) if label == 1]
                if malicious_features:
                    sample_size = min(100, len(malicious_features))
                    if sample_size > 0:
                        sampled_features = random.sample(malicious_features, sample_size)
                        val_features.extend(sampled_features)
                        val_labels.extend([1] * len(sampled_features))

        # Fit threshold with balanced validation set
        if val_features and len(set(val_labels)) > 1:
            print(f"  Validation set: {len(val_features)} samples ({val_labels.count(0)} benign, {val_labels.count(1)} malicious)")

            # Debug: Analyze score distributions before threshold fitting (use batch computation for speed)
            print("  Computing validation scores...")
            _, val_scores = detector.predict(val_features)  # Use batch computation
            benign_scores = [val_scores[i] for i, label in enumerate(val_labels) if label == 0]
            malicious_scores = [val_scores[i] for i, label in enumerate(val_labels) if label == 1]

            print(f"  Validation score distributions:")
            print(f"    Benign: mean={np.mean(benign_scores):.2f}, std={np.std(benign_scores):.2f}, range=[{np.min(benign_scores):.2f}, {np.max(benign_scores):.2f}]")
            print(f"    Malicious: mean={np.mean(malicious_scores):.2f}, std={np.std(malicious_scores):.2f}, range=[{np.min(malicious_scores):.2f}, {np.max(malicious_scores):.2f}]")

            detector.fit_threshold(val_features, val_labels)
        else:
            print(f"  Insufficient validation data, using default threshold")
            detector.threshold = 0.0

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
            # Evaluate on combined test set (this simulates real-world usage)
            print(f"  Combined test set: {len(combined_test_features)} samples")
            combined_eval_results = detector.evaluate(combined_test_features, combined_test_labels)

            # Compute combined test score distributions for analysis
            combined_scores = combined_eval_results['scores']
            combined_benign_scores = [combined_scores[i] for i, label in enumerate(combined_test_labels) if label == 0]
            combined_malicious_scores = [combined_scores[i] for i, label in enumerate(combined_test_labels) if label == 1]

            print(f"  Combined test score distributions:")
            if combined_benign_scores:
                print(f"    Benign: mean={np.mean(combined_benign_scores):.2f}, std={np.std(combined_benign_scores):.2f}, range=[{np.min(combined_benign_scores):.2f}, {np.max(combined_benign_scores):.2f}]")
            if combined_malicious_scores:
                print(f"    Malicious: mean={np.mean(combined_malicious_scores):.2f}, std={np.std(combined_malicious_scores):.2f}, range=[{np.min(combined_malicious_scores):.2f}, {np.max(combined_malicious_scores):.2f}]")

            # Handle NaN values for display
            f1_str = f"{combined_eval_results['f1']:.4f}"
            tpr_str = "N/A" if np.isnan(combined_eval_results['tpr']) else f"{combined_eval_results['tpr']:.4f}"
            fpr_str = "N/A" if np.isnan(combined_eval_results['fpr']) else f"{combined_eval_results['fpr']:.4f}"
            auroc_str = "N/A" if np.isnan(combined_eval_results['auroc']) else f"{combined_eval_results['auroc']:.4f}"
            auprc_str = "N/A" if np.isnan(combined_eval_results['auprc']) else f"{combined_eval_results['auprc']:.4f}"

            print(f"  COMBINED RESULTS   : Acc={combined_eval_results['accuracy']:.4f}, F1={f1_str}, TPR={tpr_str}, FPR={fpr_str}, "
                  f"AUROC={auroc_str}, AUPRC={auprc_str}, Thresh={combined_eval_results['threshold']:.4f}")

            # Store combined results
            layer_performance = {'COMBINED': combined_eval_results}

            # === INDIVIDUAL DATASET ANALYSIS (for debugging) ===
            print("  Individual dataset analysis:")
            for test_dataset_name, (start_idx, end_idx) in dataset_boundaries.items():
                dataset_scores = combined_scores[start_idx:end_idx]
                dataset_labels = combined_test_labels[start_idx:end_idx]
                dataset_predictions = combined_eval_results['predictions'][start_idx:end_idx]

                # Calculate individual dataset metrics using the same threshold
                dataset_accuracy = accuracy_score(dataset_labels, dataset_predictions)

                # Separate scores by label for analysis
                dataset_benign_scores = [dataset_scores[i] for i, label in enumerate(dataset_labels) if label == 0]
                dataset_malicious_scores = [dataset_scores[i] for i, label in enumerate(dataset_labels) if label == 1]

                if dataset_benign_scores:
                    benign_stats = f"mean={np.mean(dataset_benign_scores):.2f}, range=[{np.min(dataset_benign_scores):.2f}, {np.max(dataset_benign_scores):.2f}]"
                else:
                    benign_stats = "no benign samples"

                if dataset_malicious_scores:
                    malicious_stats = f"mean={np.mean(dataset_malicious_scores):.2f}, range=[{np.min(dataset_malicious_scores):.2f}, {np.max(dataset_malicious_scores):.2f}]"
                else:
                    malicious_stats = "no malicious samples"

                print(f"    {test_dataset_name:15s}: Acc={dataset_accuracy:.4f}")
                print(f"      Benign: {benign_stats}")
                print(f"      Malicious: {malicious_stats}")

                # Calculate F1, TPR, FPR for individual dataset
                f1 = f1_score(dataset_labels, dataset_predictions, zero_division=0)

                # Calculate TPR, FPR from confusion matrix
                unique_classes = np.unique(dataset_labels)
                if len(unique_classes) > 1:
                    try:
                        tn, fp, fn, tp = confusion_matrix(dataset_labels, dataset_predictions).ravel()
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
                    except:
                        tpr = float('nan')
                        fpr = float('nan')
                else:
                    tpr = float('nan')
                    fpr = float('nan')

                # Store individual results for detailed analysis
                layer_performance[test_dataset_name] = {
                    'accuracy': dataset_accuracy,
                    'f1': f1,
                    'tpr': tpr,
                    'fpr': fpr,
                    'auroc': float('nan'),  # Not meaningful for individual datasets with combined threshold
                    'auprc': float('nan'),  # Not meaningful for individual datasets with combined threshold
                    'predictions': dataset_predictions,
                    'scores': dataset_scores,
                    'threshold': combined_eval_results['threshold']
                }
        else:
            print("  No test data available for evaluation")
            layer_performance = {}

        layer_results[layer_idx] = layer_performance

    # Calculate layer ranking based on COMBINED results (real-world performance)
    layer_combined_scores = []
    for layer_idx in layers:
        if layer_results[layer_idx] and 'COMBINED' in layer_results[layer_idx]:
            combined_result = layer_results[layer_idx]['COMBINED']
            accuracy = combined_result['accuracy']
            auroc = combined_result['auroc'] if not np.isnan(combined_result['auroc']) else 0.0
            auprc = combined_result['auprc'] if not np.isnan(combined_result['auprc']) else 0.0

            # Combined score prioritizing accuracy and AUROC for real-world performance
            combined_score = 0.5 * accuracy + 0.3 * auroc + 0.2 * auprc
            layer_combined_scores.append((layer_idx, accuracy, auroc, auprc, combined_score))
        else:
            layer_combined_scores.append((layer_idx, 0.0, 0.0, 0.0, 0.0))

    # Sort by combined score (real-world performance)
    layer_combined_scores.sort(key=lambda x: x[4], reverse=True)

    # Also calculate individual dataset averages for comparison
    layer_individual_avg_scores = []
    for layer_idx in layers:
        if layer_results[layer_idx]:
            # Exclude COMBINED from individual averages
            individual_results = {k: v for k, v in layer_results[layer_idx].items() if k != 'COMBINED'}
            if individual_results:
                accuracies = [result['accuracy'] for result in individual_results.values()
                             if not np.isnan(result['accuracy'])]
                avg_accuracy = np.mean(accuracies) if accuracies else 0.0
                layer_individual_avg_scores.append((layer_idx, avg_accuracy))
            else:
                layer_individual_avg_scores.append((layer_idx, 0.0))
        else:
            layer_individual_avg_scores.append((layer_idx, 0.0))

    layer_individual_avg_scores.sort(key=lambda x: x[1], reverse=True)

    # Save results to CSV with model-specific filename
    output_path = f"results/balanced_kcd_{model_type}_results.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Layer", "Dataset", "Method", "Accuracy", "F1", "TPR", "FPR", "AUROC", "AUPRC", "Threshold", "Combined_Rank", "Individual_Rank"])

        # Create ranking based on combined performance
        layer_combined_ranking = {layer_idx: rank for rank, (layer_idx, _, _, _, _) in enumerate(layer_combined_scores, 1)}

        # Create ranking based on individual dataset average
        layer_individual_ranking = {layer_idx: rank for rank, (layer_idx, _) in enumerate(layer_individual_avg_scores, 1)}

        for layer_idx in layers:
            if layer_results[layer_idx]:
                for dataset_name, result in layer_results[layer_idx].items():
                    # Handle NaN values for CSV
                    f1_val = f"{result['f1']:.4f}"
                    tpr_val = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
                    fpr_val = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"
                    auroc_val = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
                    auprc_val = "N/A" if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"
                    threshold_val = f"{result['threshold']:.4f}"
                    combined_rank = layer_combined_ranking.get(layer_idx, "N/A")
                    individual_rank = layer_individual_ranking.get(layer_idx, "N/A")

                    writer.writerow([
                        layer_idx,
                        dataset_name,
                        "KCD",
                        f"{result['accuracy']:.4f}",
                        f1_val,
                        tpr_val,
                        fpr_val,
                        auroc_val,
                        auprc_val,
                        threshold_val,
                        combined_rank,
                        individual_rank
                    ])

    print(f"\nResults saved to {output_path}")

    # Print summary results
    print("\n" + "="*120)
    print("BALANCED OOD JAILBREAK DETECTION SUMMARY (KCD Algorithm)")
    print("="*120)
    print("Training Configuration (2,000 samples, 1:1 ratio):")
    print(f"  In-Distribution Datasets: {list(in_dist_datasets.keys())}")
    print(f"  OOD Datasets: {list(ood_datasets.keys())}")
    print("Test Configuration (1,800 samples, 1:1 ratio):")
    print(f"  Test Datasets: {list(test_datasets.keys())}")
    print("-"*120)

    # COMBINED PERFORMANCE RANKING (Real-world scenario)
    print(f"\n{'COMBINED PERFORMANCE RANKING (Real-world scenario)':<120}")
    print(f"{'Layer':<6} {'Accuracy':<10} {'F1':<8} {'TPR':<8} {'FPR':<8} {'AUROC':<10} {'AUPRC':<10} {'Combined':<10}")
    print("-" * 120)

    for layer_idx, accuracy, auroc, auprc, combined_score in layer_combined_scores:
        if layer_results[layer_idx] and 'COMBINED' in layer_results[layer_idx]:
            combined_result = layer_results[layer_idx]['COMBINED']
            acc_str = f"{accuracy:.3f}"
            f1_str = f"{combined_result['f1']:.3f}"
            tpr_str = "N/A" if np.isnan(combined_result['tpr']) else f"{combined_result['tpr']:.3f}"
            fpr_str = "N/A" if np.isnan(combined_result['fpr']) else f"{combined_result['fpr']:.3f}"
            auroc_str = "N/A" if auroc == 0.0 else f"{auroc:.3f}"
            auprc_str = "N/A" if auprc == 0.0 else f"{auprc:.3f}"
            combined_str = f"{combined_score:.3f}"

            print(f"{layer_idx:<6} {acc_str:<10} {f1_str:<8} {tpr_str:<8} {fpr_str:<8} {auroc_str:<10} {auprc_str:<10} {combined_str:<10}")
        else:
            print(f"{layer_idx:<6} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'0.000':<10}")

    # INDIVIDUAL DATASET AVERAGE RANKING (for comparison)
    print(f"\n{'INDIVIDUAL DATASET AVERAGE RANKING (for comparison)':<100}")
    print(f"{'Layer':<6} {'Avg_Acc':<10} {'XSTest':<12} {'FigTxt':<12} {'VQAv2':<12} {'VAE':<12} {'JBV-Test':<12}")
    print("-" * 100)

    for layer_idx, avg_acc in layer_individual_avg_scores:
        layer_perf = layer_results[layer_idx]

        # Format average accuracy
        avg_str = f"{avg_acc:.3f}"

        # Format individual dataset performances (excluding COMBINED)
        dataset_strs = []
        for dataset_name in ['XSTest_safe', 'XSTest_unsafe', 'FigTxt_safe', 'FigTxt_unsafe', 'VQAv2', 'VAE', 'JailbreakV-28K_test']:
            if dataset_name in layer_perf:
                acc = layer_perf[dataset_name]['accuracy']
                dataset_strs.append(f"{acc:.3f}")
            else:
                dataset_strs.append("N/A")

        print(f"{layer_idx:<6} {avg_str:<10} {dataset_strs[0]:<12} {dataset_strs[1]:<12} {dataset_strs[2]:<12} {dataset_strs[3]:<12} {dataset_strs[4]:<12}")

    # DOMAIN SHIFT ANALYSIS
    print(f"\n{'DOMAIN SHIFT ANALYSIS':<100}")
    print("Comparing validation vs test score distributions to identify domain shift:")
    print("-" * 100)

    # Find the best performing layer for detailed analysis
    if layer_combined_scores:
        best_layer = layer_combined_scores[0][0]
        print(f"Analysis for best performing layer: {best_layer}")
        print("(Validation scores were shown during training)")
        print("(Test scores are shown in individual dataset analysis above)")
        print("Large differences between validation and test score distributions indicate domain shift.")

    print("\n" + "="*120)
    print("DETAILED LAYER PERFORMANCE")
    print("="*120)

    for layer_idx in layers:
        print(f"\nLayer {layer_idx}:")
        print("-" * 80)

        if layer_results[layer_idx]:
            for dataset_name, result in layer_results[layer_idx].items():
                # Handle NaN values in summary
                f1_str = f"{result['f1']:.4f}"
                tpr_str = "N/A" if np.isnan(result['tpr']) else f"{result['tpr']:.4f}"
                fpr_str = "N/A" if np.isnan(result['fpr']) else f"{result['fpr']:.4f}"
                auroc_str = "N/A" if np.isnan(result['auroc']) else f"{result['auroc']:.4f}"
                auprc_str = "N/A" if np.isnan(result['auprc']) else f"{result['auprc']:.4f}"

                print(f"  {dataset_name:15s}: Acc={result['accuracy']:.4f}, F1={f1_str}, TPR={tpr_str}, FPR={fpr_str}, "
                      f"AUROC={auroc_str}, AUPRC={auprc_str}, Thresh={result['threshold']:.4f}")
        else:
            print("  No results for this layer")

    print("="*120)
    print("- Balanced 1:1 training and test ratios for robust evaluation")
    print("- Training: Alpaca (500) + MM-Vet (218) + OpenAssistant (282) vs AdvBench (300) + JailbreakV-28K (550) + DAN (150)")
    print("- Testing: XSTest + FigTxt + VQAv2 (safe) vs XSTest + FigTxt + VAE + JailbreakV-28K (unsafe)")


if __name__ == "__main__":
    main()
