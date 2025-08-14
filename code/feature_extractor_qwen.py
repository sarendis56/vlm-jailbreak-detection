import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os
import tempfile
from tqdm import tqdm
from generic_classifier import FeatureCache
import gc
import psutil

# Import Qwen dependencies
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class HiddenStateExtractor:
    """Qwen-specific hidden state extractor with caching support"""
    
    def __init__(self, model_path, cache_dir="cache"):
        self.model_path = model_path
        self.cache = FeatureCache(cache_dir)
        self.model = None
        self.processor = None
        self.tokenizer = None

    def _get_memory_info(self):
        """Get current memory usage information"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        else:
            gpu_memory = gpu_memory_cached = gpu_max_memory = 0

        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        return {
            'gpu_allocated': gpu_memory,
            'gpu_cached': gpu_memory_cached,
            'gpu_max': gpu_max_memory,
            'cpu_used': cpu_memory
        }

    def _print_memory_info(self, prefix=""):
        """Print current memory usage"""
        info = self._get_memory_info()
        print(f"{prefix}GPU: {info['gpu_allocated']:.2f}GB allocated, {info['gpu_cached']:.2f}GB cached, {info['gpu_max']:.2f}GB max")

    def _get_model_layers(self):
        """Get the number of layers in the model"""
        if self.model is None:
            self._load_model()

        # For Qwen2.5-VL models, get the number of layers from the language model
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
            num_layers = len(self.model.language_model.model.layers)
        else:
            # Fallback: try to get from config
            if hasattr(self.model.config, 'num_hidden_layers'):
                num_layers = self.model.config.num_hidden_layers
            else:
                # Default fallback based on model name
                if '3B' in self.model_path or '3b' in self.model_path:
                    num_layers = 36
                elif '7B' in self.model_path or '7b' in self.model_path:
                    num_layers = 28
                else:
                    num_layers = 32

        print(f"Detected {num_layers} layers in model")
        return num_layers

    def get_default_layer_range(self):
        """Get the default layer range for this model (all layers)"""
        num_layers = self._get_model_layers()
        # For Qwen, the model returns embedding layer + transformer layers
        # So if model has 36 transformer layers, we get 37 hidden states (0-36)
        return 0, num_layers  # Include the extra embedding layer

    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError as e:
                print(f"Warning: CUDA cleanup failed: {e}")

    def _load_model(self):
        """Load Qwen model only when needed"""
        if self.model is None:
            print(f"Loading Qwen model from {self.model_path}...")

            # Clear any existing GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load with more memory-efficient settings
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            # Load processor
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Get tokenizer from processor
            self.tokenizer = self.processor.tokenizer

            # Set model to eval mode to save memory
            self.model.eval()

            print(f"Model loaded successfully. Device: {self.model.device}")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def _prepare_qwen_messages(self, sample):
        """Prepare messages for Qwen2.5-VL based on sample format with memory optimization"""
        content = []

        # Handle different image formats with memory optimization
        if sample.get("img") is not None:
            img_data = sample["img"]

            # Case 1: File path(s) as string
            if isinstance(img_data, str):
                if "," in img_data:
                    # Multiple comma-separated paths
                    image_files = [path.strip() for path in img_data.split(",")]
                    for img_path in image_files:
                        if not img_path.startswith(("http://", "https://", "file://")):
                            img_path = f"file://{os.path.abspath(img_path)}"
                        content.append({"type": "image", "image": img_path})
                else:
                    # Single path
                    if not img_data.startswith(("http://", "https://", "file://")):
                        img_data = f"file://{os.path.abspath(img_data)}"
                    content.append({"type": "image", "image": img_data})

            # Case 2: Binary image data - save to temp file and clean up immediately
            elif isinstance(img_data, bytes):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(img_data)
                    tmp_path = tmp_file.name
                content.append({"type": "image", "image": f"file://{tmp_path}"})
                # Note: temp file will be cleaned up by OS eventually

            # Case 3: List of images
            elif isinstance(img_data, list):
                for item in img_data:
                    if isinstance(item, str):
                        if not item.startswith(("http://", "https://", "file://")):
                            item = f"file://{os.path.abspath(item)}"
                        content.append({"type": "image", "image": item})
                    elif isinstance(item, bytes):
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                            tmp_file.write(item)
                            tmp_path = tmp_file.name
                        content.append({"type": "image", "image": f"file://{tmp_path}"})

        # Add text
        content.append({"type": "text", "text": sample["txt"]})

        # Return messages in the format expected by Qwen2.5-VL
        messages = [{"role": "user", "content": content}]
        return messages

    def extract_hidden_states(self, dataset, dataset_name, layer_start=None, layer_end=None,
                            use_cache=True, label_key='toxicity', batch_size=50,
                            memory_cleanup_freq=5, experiment_name=None):
        """Extract hidden states from Qwen model"""
        
        # Set default layer ranges if not provided
        if layer_start is None or layer_end is None:
            default_start, default_end = self.get_default_layer_range()
            if layer_start is None:
                layer_start = default_start
            if layer_end is None:
                layer_end = default_end
            print(f"Using automatic layer range: {layer_start}-{layer_end}")

        # Check cache first
        layer_range = (layer_start, layer_end)
        dataset_size = len(dataset)

        if use_cache and self.cache.exists(dataset_name, self.model_path, layer_range, dataset_size, experiment_name):
            print(f"Loading cached features for {dataset_name}...")
            return self.cache.load(dataset_name, self.model_path, layer_range, dataset_size, experiment_name)

        # Load model if not already loaded
        self._load_model()

        # Extract features
        all_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
        labels = []

        pbar = tqdm(dataset, desc=f"Extracting features from {dataset_name}", unit="sample")
        for sample_idx, sample in enumerate(pbar):
            try:
                # Periodic memory cleanup and monitoring
                if sample_idx % memory_cleanup_freq == 0:
                    self._aggressive_cleanup()
                    # if sample_idx % 20 == 0:  # Print memory info every 20 samples
                    #     self._print_memory_info(f"Sample {sample_idx}: ")

                # Prepare messages for Qwen2.5-VL
                messages = self._prepare_qwen_messages(sample)

                # Process the messages using the processor
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                vision_info = process_vision_info(messages)
                image_inputs = vision_info[0] if len(vision_info) > 0 else None
                video_inputs = vision_info[1] if len(vision_info) > 1 else None
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)

                # Truncate if sequence is too long - use conservative limit to prevent NaN
                seq_len = inputs.input_ids.shape[1]
                max_length = 4096  # Conservative limit to prevent NaN issues
                if seq_len > max_length:
                    print(f"Warning: Truncating sequence from {seq_len} to {max_length} tokens")
                    inputs.input_ids = inputs.input_ids[:, :max_length]
                    if hasattr(inputs, 'attention_mask'):
                        inputs.attention_mask = inputs.attention_mask[:, :max_length]

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                    # Check if any hidden states contain NaN - if so, skip this sample
                    has_nan_in_any_layer = False
                    for hidden_state in outputs.hidden_states:
                        if torch.isnan(hidden_state).any() or torch.isinf(hidden_state).any():
                            has_nan_in_any_layer = True
                            break

                    if has_nan_in_any_layer:
                        print(f"Warning: NaN/Inf detected in model outputs for sample {sample_idx}, skipping...")
                        # Add zero vectors for all layers
                        for layer_idx in range(layer_start, layer_end+1):
                            if len(all_hidden_states[layer_idx]) > 0:
                                zero_vector = np.zeros_like(all_hidden_states[layer_idx][0])
                            else:
                                zero_vector = np.zeros(2048)
                            all_hidden_states[layer_idx].append(zero_vector)
                    else:
                        # Extract hidden states for specified layers
                        for layer_idx in range(layer_start, layer_end+1):
                            if layer_idx < len(outputs.hidden_states):
                                hidden_state = outputs.hidden_states[layer_idx]
                                # Use last token representation and immediately move to CPU
                                last_token_hidden = hidden_state[:, -1, :].cpu().numpy().flatten()
                                all_hidden_states[layer_idx].append(last_token_hidden)
                                # Clear the hidden state from GPU memory
                                del hidden_state
                            else:
                                print(f"Warning: Layer {layer_idx} not found in outputs (only {len(outputs.hidden_states)} layers available)")
                                # Add zero vector for missing layers
                                if len(all_hidden_states[layer_idx]) > 0:
                                    zero_vector = np.zeros_like(all_hidden_states[layer_idx][0])
                                else:
                                    zero_vector = np.zeros(2048)  # Qwen2.5-VL-3B hidden size
                                all_hidden_states[layer_idx].append(zero_vector)

                # Extract label
                labels.append(sample.get(label_key, 0))

                # Explicit cleanup of large tensors
                del outputs, inputs
                if 'text' in locals():
                    del text
                if 'image_inputs' in locals() and image_inputs is not None:
                    del image_inputs
                if 'video_inputs' in locals() and video_inputs is not None:
                    del video_inputs

                # More frequent cleanup for Qwen (every sample)
                if sample_idx % 1 == 0:  # Clean up after every sample
                    self._aggressive_cleanup()

                # Update progress
                pbar.set_postfix({
                    'processed': f"{len(labels)}/{len(dataset)}",
                    'type': "📷" if sample.get('img') is not None else "📝"
                })

            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")

                # Emergency cleanup on error
                self._aggressive_cleanup()

                # Add zero vectors for failed samples
                for layer_idx in range(layer_start, layer_end+1):
                    if layer_idx in all_hidden_states and len(all_hidden_states[layer_idx]) > 0:
                        zero_vector = np.zeros_like(all_hidden_states[layer_idx][0])
                        all_hidden_states[layer_idx].append(zero_vector)
                    else:
                        # If no previous samples, create a default zero vector
                        zero_vector = np.zeros(2048)  # Qwen2.5-VL-3B hidden size
                        all_hidden_states[layer_idx].append(zero_vector)
                labels.append(0)  # Default label for failed samples

        # Convert to numpy arrays
        for layer_idx in range(layer_start, layer_end+1):
            all_hidden_states[layer_idx] = np.array(all_hidden_states[layer_idx])

        # Cache results
        if use_cache:
            metadata = {
                'dataset_size': len(dataset),
                'label_key': label_key,
                'processed_samples': len(labels),
                'layer_start': layer_start,
                'layer_end': layer_end
            }
            self.cache.save(dataset_name, self.model_path, layer_range, all_hidden_states, 
                          labels, metadata, dataset_size, experiment_name)

        self._aggressive_cleanup()
        return all_hidden_states, labels, len(labels)
