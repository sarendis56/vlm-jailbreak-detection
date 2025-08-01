import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import re
from tqdm import tqdm
from generic_classifier import FeatureCache
import gc
import psutil

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

class HiddenStateExtractor:
    """Unified hidden state extractor with caching support"""
    
    def __init__(self, model_path, cache_dir="cache"):
        self.model_path = model_path
        self.cache = FeatureCache(cache_dir)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.model_name = None

    def _get_memory_info(self):
        """Get current memory usage information"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
        else:
            gpu_memory = gpu_memory_cached = 0

        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        return {
            'gpu_allocated': gpu_memory,
            'gpu_cached': gpu_memory_cached,
            'cpu_used': cpu_memory
        }

    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup with error handling"""
        # Force garbage collection first
        gc.collect()

        # Clear GPU cache with error handling
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError as e:
                print(f"Warning: CUDA cleanup failed: {e}")
                # Try basic cleanup
                try:
                    torch.cuda.empty_cache()
                except:
                    print("Warning: Even basic CUDA cleanup failed")

            # Additional cleanup for CUDA
            try:
                torch.cuda.ipc_collect()
            except:
                pass

    def _extract_hidden_states_chunked(self, dataset, dataset_name, layer_start, layer_end,
                                     use_cache, label_key, batch_size, memory_cleanup_freq,
                                     experiment_name, chunk_size):
        """Process large datasets in chunks with individual chunk caching"""
        total_samples = len(dataset)
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        layer_range = (layer_start, layer_end)

        print(f"Processing {total_samples} samples in {num_chunks} chunks of {chunk_size}")

        # Check if we can load from individual chunk caches
        all_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
        all_labels = []
        chunks_to_process = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_name = f"{dataset_name}_chunk_{chunk_idx}"
            chunk_size_actual = end_idx - start_idx

            # Check if this chunk is already cached and complete
            if use_cache and self.cache.exists(chunk_name, self.model_path, layer_range, chunk_size_actual, experiment_name):
                print(f"Found cached chunk {chunk_idx + 1}/{num_chunks} ({chunk_name}), validating...")
                try:
                    chunk_hidden_states, chunk_labels, chunk_metadata = self.cache.load(chunk_name, self.model_path, layer_range, chunk_size_actual, experiment_name)

                    # Validate that the cached chunk is complete
                    expected_samples = chunk_size_actual
                    actual_samples = len(chunk_labels)

                    if actual_samples == expected_samples:
                        print(f"✓ Cached chunk {chunk_idx + 1} is complete ({actual_samples}/{expected_samples} samples)")

                        # Accumulate results from cached chunk
                        for layer_idx in range(layer_start, layer_end+1):
                            if isinstance(chunk_hidden_states[layer_idx], np.ndarray):
                                all_hidden_states[layer_idx].extend(chunk_hidden_states[layer_idx].tolist())
                            else:
                                all_hidden_states[layer_idx].extend(chunk_hidden_states[layer_idx])
                        all_labels.extend(chunk_labels)

                        print(f"Loaded chunk {chunk_idx + 1} from cache. Total processed: {len(all_labels)}")
                        continue
                    else:
                        print(f"✗ Cached chunk {chunk_idx + 1} is incomplete ({actual_samples}/{expected_samples} samples)")
                        print(f"  Removing incomplete cache and reprocessing chunk {chunk_idx + 1}")
                        # Remove the incomplete cache file
                        cache_key = self.cache._get_cache_key(chunk_name, self.model_path, layer_range, chunk_size_actual, experiment_name)
                        cache_path = self.cache._get_cache_path(cache_key)
                        import os
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                            print(f"  Removed incomplete cache file: {cache_path}")

                except Exception as e:
                    print(f"✗ Error validating cached chunk {chunk_idx + 1}: {e}")
                    print(f"  Removing corrupted cache and reprocessing chunk {chunk_idx + 1}")
                    # Remove the corrupted cache file
                    try:
                        cache_key = self.cache._get_cache_key(chunk_name, self.model_path, layer_range, chunk_size_actual, experiment_name)
                        cache_path = self.cache._get_cache_path(cache_key)
                        import os
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                            print(f"  Removed corrupted cache file: {cache_path}")
                    except:
                        pass

            # Add to chunks that need processing
            chunks_to_process.append((chunk_idx, start_idx, end_idx, chunk_name))

        # Process remaining chunks
        for chunk_idx, start_idx, end_idx, chunk_name in chunks_to_process:
            chunk_data = dataset[start_idx:end_idx]
            chunk_size_actual = len(chunk_data)

            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (samples {start_idx}-{end_idx-1})")

            try:
                # Process this chunk
                chunk_hidden_states, chunk_labels, _ = self._extract_chunk(
                    chunk_data, chunk_name, layer_start, layer_end, label_key, batch_size, memory_cleanup_freq
                )

                # Save this chunk to cache immediately
                if use_cache:
                    metadata = {
                        'chunk_idx': chunk_idx,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'processed_samples': len(chunk_labels)
                    }
                    self.cache.save(chunk_name, self.model_path, layer_range, chunk_hidden_states, chunk_labels,
                                  metadata, chunk_size_actual, experiment_name)
                    print(f"Chunk {chunk_idx + 1} cached successfully")

                # Accumulate results
                for layer_idx in range(layer_start, layer_end+1):
                    all_hidden_states[layer_idx].extend(chunk_hidden_states[layer_idx])
                all_labels.extend(chunk_labels)

                print(f"Chunk {chunk_idx + 1} completed successfully. Total processed: {len(all_labels)}")

                # Clear chunk data from memory
                del chunk_hidden_states, chunk_labels

            except Exception as e:
                error_msg = str(e)
                print(f"Error processing chunk {chunk_idx + 1}: {e}")

                # Check for CUDA device-side assert errors (usually fatal)
                if "device-side assert triggered" in error_msg:
                    print("FATAL: CUDA device-side assert detected!")
                    print("This usually indicates corrupted GPU state that requires process restart.")
                    print("Recommendation: Restart the Python process and try again.")
                    print("The completed chunks are saved and will be reused on restart.")
                    raise RuntimeError("CUDA device-side assert triggered - process restart required")

                print("Performing emergency cleanup and continuing...")
                self._aggressive_cleanup()
                continue

            # Cleanup between chunks
            self._aggressive_cleanup()

        # Convert to numpy arrays
        for layer_idx in range(layer_start, layer_end+1):
            all_hidden_states[layer_idx] = np.array(all_hidden_states[layer_idx])

        # Cache the final combined results
        if use_cache:
            metadata = {
                'total_chunks': num_chunks,
                'chunk_size': chunk_size,
                'processed_samples': len(all_labels)
            }
            self.cache.save(dataset_name, self.model_path, layer_range, all_hidden_states, all_labels,
                          metadata, len(all_labels), experiment_name)
            print(f"Final combined results cached for {dataset_name}")

        return all_hidden_states, all_labels, len(all_labels)

    def _extract_chunk(self, chunk_data, chunk_name, layer_start, layer_end,
                      label_key, batch_size, memory_cleanup_freq):
        """Extract hidden states for a single chunk"""
        all_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
        labels = []

        total_samples = len(chunk_data)

        with tqdm(total=total_samples, desc=f"Extracting hidden states for {chunk_name}") as pbar:
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_data = chunk_data[batch_start:batch_end]

                batch_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
                batch_labels = []

                for idx, sample in enumerate(batch_data):
                    try:
                        # Process sample (same logic as main function)
                        if sample['img'] == None:
                            prompt = self._construct_conv_prompt(sample)
                            input_ids = (
                                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                                .unsqueeze(0)
                            )
                            # Truncate if sequence is too long
                            max_length = getattr(self.model.config, 'max_position_embeddings', 4096)
                            if input_ids.shape[1] > max_length:
                                print(f"Truncating sequence from {input_ids.shape[1]} to {max_length} tokens")
                                input_ids = input_ids[:, :max_length]
                            input_ids = input_ids.cuda()

                            with torch.no_grad():
                                outputs = self.model(input_ids, images=None, image_sizes=None, output_hidden_states=True)
                        else:
                            prompt = self._construct_conv_prompt(sample)
                            input_ids = (
                                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                                .unsqueeze(0)
                            )
                            # Truncate if sequence is too long
                            max_length = getattr(self.model.config, 'max_position_embeddings', 4096)
                            if input_ids.shape[1] > max_length:
                                print(f"Truncating sequence from {input_ids.shape[1]} to {max_length} tokens")
                                input_ids = input_ids[:, :max_length]
                            input_ids = input_ids.cuda()

                            images_tensor, images_size = self._prepare_imgs_tensor_both_cases(sample)
                            if images_tensor is None:
                                continue
                            with torch.no_grad():
                                outputs = self.model(input_ids, images=images_tensor, image_sizes=images_size, output_hidden_states=True)

                        # Extract hidden states for specified layers
                        for layer_idx in range(layer_start, layer_end+1):
                            hidden_state = outputs.hidden_states[layer_idx]
                            pooled_state = torch.mean(hidden_state, dim=1).squeeze().cpu().numpy()
                            batch_hidden_states[layer_idx].append(pooled_state)

                        batch_labels.append(sample[label_key])

                        # Cleanup
                        del outputs, input_ids
                        if 'images_tensor' in locals() and images_tensor is not None:
                            del images_tensor

                        pbar.update(1)

                    except Exception as e:
                        error_msg = str(e)
                        print(f"Error processing sample {batch_start + idx}: {e}")

                        # Check for CUDA device-side assert errors (usually fatal)
                        if "device-side assert triggered" in error_msg:
                            print("FATAL: CUDA device-side assert detected in chunk processing!")
                            print("This usually indicates corrupted GPU state that requires process restart.")
                            raise RuntimeError("CUDA device-side assert triggered - process restart required")

                        self._aggressive_cleanup()
                        continue

                # Add batch results to main results
                for layer_idx in range(layer_start, layer_end+1):
                    all_hidden_states[layer_idx].extend(batch_hidden_states[layer_idx])
                labels.extend(batch_labels)

                # Memory cleanup
                if (batch_start // batch_size + 1) % memory_cleanup_freq == 0:
                    self._aggressive_cleanup()

        return all_hidden_states, labels, len(labels)

    def _load_model(self):
        """Load model only when needed"""
        if self.model is None:
            print(f"Loading model from {self.model_path}...")
            self.model_name = get_model_name_from_path(self.model_path)        
            kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16    
            }
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=self.model_name,
                **kwargs
            )
    
    def _find_conv_mode(self, model_name):
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"  
        return conv_mode    
        
    def _adjust_query_for_images(self, qs):   
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        return qs

    def _construct_conv_prompt(self, sample):        
        conv = conv_templates[self._find_conv_mode(self.model_name)].copy()  
        if (sample['img'] != None):     
            qs = self._adjust_query_for_images(sample['txt'])
        else:
            qs = sample['txt']
        conv.append_message(conv.roles[0], qs)  
        conv.append_message(conv.roles[1], None)       
        prompt = conv.get_prompt()
        return prompt

    def _load_image(self, image_file):
        try:
            if image_file.startswith("http") or image_file.startswith("https"):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_file).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            return None

    def _load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self._load_image(image_file)
            if image is not None:  # Only add valid images
                out.append(image)
        return out
    
    def _load_image_from_bytes(self, image_data):          
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _load_images_from_bytes(self, image_data_list):
        valid_images = []
        for data in image_data_list:
            image = self._load_image_from_bytes(data)
            if image is not None:  # Only add valid images
                valid_images.append(image)
        return valid_images

    def _prepare_imgs_tensor_both_cases(self, sample):
        try:
            if isinstance(sample['img'], str):
                image_files_path = sample['img'].split(",")
                img_prompt = self._load_images(image_files_path)
            elif isinstance(sample['img'], bytes):
                img_prompt = [self._load_image_from_bytes(sample['img'])]
            elif isinstance(sample['img'], list):
                if all(isinstance(item, bytes) for item in sample['img']):
                    img_prompt = self._load_images_from_bytes(sample['img'])
                else:
                    raise ValueError("List contains non-bytes data.")
            else:
                raise ValueError("Unsupported data type in sample['img']. "
                                "Expected str, bytes, or list of bytes.")
            # Filter out None values before processing
            valid_images = [img for img in img_prompt if img is not None]
            if not valid_images:
                print("No valid images found in sample")
                return None, None

            images_size = [img.size for img in valid_images]
            images_tensor = process_images(valid_images, self.image_processor, self.model.config)

            # Ensure images_tensor is a tensor before calling .to()
            if isinstance(images_tensor, list):
                print(f"Warning: process_images returned list instead of tensor")
                return None, None

            images_tensor = images_tensor.to(self.model.device, dtype=torch.float16)
            return images_tensor, images_size
        except Exception as e:
            print(f"Error preparing image tensors: {e}")
            return None, None

    def extract_hidden_states(self, dataset, dataset_name, layer_start=16, layer_end=32,
                            use_cache=True, label_key='toxicity', batch_size=10, memory_cleanup_freq=5, experiment_name=None,
                            chunk_size=5000, save_intermediate=True):
        """
        Extract hidden states with improved memory management for large datasets

        Args:
            dataset: List of samples
            dataset_name: Name for caching
            layer_start, layer_end: Layer range to extract
            use_cache: Whether to use caching
            label_key: Key to extract labels ('toxicity' or 'category_id')
            batch_size: Process in batches to manage memory
            memory_cleanup_freq: Clean GPU memory every N samples
            experiment_name: Optional experiment name for cache organization
        """
        layer_range = (layer_start, layer_end)
        dataset_size = len(dataset)

        # Check cache first
        if use_cache and self.cache.exists(dataset_name, self.model_path, layer_range, dataset_size, experiment_name):
            print(f"Loading cached features for {dataset_name} (size: {dataset_size}, layers: {layer_start}-{layer_end})...")
            return self.cache.load(dataset_name, self.model_path, layer_range, dataset_size, experiment_name)

        # Load model if not cached
        self._load_model()

        # For large datasets, process in chunks and save intermediate results
        total_samples = len(dataset)
        if save_intermediate and total_samples > chunk_size:
            print(f"Large dataset detected ({total_samples} samples). Processing in chunks of {chunk_size}...")
            return self._extract_hidden_states_chunked(dataset, dataset_name, layer_start, layer_end,
                                                     use_cache, label_key, batch_size, memory_cleanup_freq,
                                                     experiment_name, chunk_size)

        # Extract hidden states for all samples with memory management
        all_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
        labels = []

        # Process in batches to manage memory
        pbar = tqdm(total=total_samples, desc=f"Extracting hidden states for {dataset_name}", unit="sample")

        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch = dataset[batch_start:batch_end]

            # Process batch
            batch_hidden_states = {i: [] for i in range(layer_start, layer_end+1)}
            batch_labels = []

            for idx, sample in enumerate(batch):
                try:
                    # Clear GPU cache periodically and monitor memory
                    if (batch_start + idx) % memory_cleanup_freq == 0:
                        self._aggressive_cleanup()
                        mem_info = self._get_memory_info()
                        if mem_info['gpu_allocated'] > 10:  # More than 10GB
                            print(f"High GPU memory usage: {mem_info['gpu_allocated']:.1f}GB allocated, "
                                  f"{mem_info['gpu_cached']:.1f}GB cached")

                    # Process sample
                    if sample['img'] == None:
                        prompt = self._construct_conv_prompt(sample)
                        input_ids = (
                            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                            .unsqueeze(0)
                        )
                        # Truncate if sequence is too long (max 4096 tokens for most models)
                        max_length = getattr(self.model.config, 'max_position_embeddings', 4096)
                        if input_ids.shape[1] > max_length:
                            print(f"Truncating sequence from {input_ids.shape[1]} to {max_length} tokens")
                            input_ids = input_ids[:, :max_length]
                        input_ids = input_ids.cuda()

                        with torch.no_grad():
                            outputs = self.model(input_ids, images=None, image_sizes=None, output_hidden_states=True)
                    else:
                        prompt = self._construct_conv_prompt(sample)
                        input_ids = (
                            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                            .unsqueeze(0)
                        )
                        # Truncate if sequence is too long (max 4096 tokens for most models)
                        max_length = getattr(self.model.config, 'max_position_embeddings', 4096)
                        if input_ids.shape[1] > max_length:
                            print(f"Truncating sequence from {input_ids.shape[1]} to {max_length} tokens")
                            input_ids = input_ids[:, :max_length]
                        input_ids = input_ids.cuda()

                        images_tensor, images_size = self._prepare_imgs_tensor_both_cases(sample)
                        if images_tensor is None:
                            continue
                        with torch.no_grad():
                            outputs = self.model(input_ids, images=images_tensor, image_sizes=images_size, output_hidden_states=True)

                    # Extract last token hidden states for each layer and immediately move to CPU
                    for layer_idx in range(layer_start, layer_end+1):
                        hidden_state = outputs.hidden_states[layer_idx + 1]  # +1 because hidden_states[0] is input embeddings
                        last_token_hidden = hidden_state[:, -1, :].cpu().numpy().flatten()  # Get last token, flatten, move to CPU
                        batch_hidden_states[layer_idx].append(last_token_hidden)

                    batch_labels.append(sample[label_key])

                    # Clear outputs from GPU memory immediately
                    del outputs
                    del input_ids
                    if 'images_tensor' in locals() and images_tensor is not None:
                        del images_tensor

                    # Update progress
                    processed = len(labels) + len(batch_labels)
                    has_image = "📷" if sample.get('img') is not None else "📝"
                    label_value = sample[label_key]
                    pbar.set_postfix({
                        'processed': f"{processed}/{total_samples}",
                        'type': has_image,
                        label_key: label_value,
                        'batch': f"{batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1}"
                    })
                    pbar.update(1)

                except Exception as e:
                    print(f"Error processing sample {batch_start + idx}: {e}")
                    # Aggressive cleanup on error
                    self._aggressive_cleanup()

                    # Check if it's a CUDA memory error
                    if "CUDA" in str(e) or "memory" in str(e).lower():
                        print("CUDA memory error detected. Performing emergency cleanup...")
                        # Emergency cleanup
                        if 'outputs' in locals():
                            del outputs
                        if 'input_ids' in locals():
                            del input_ids
                        if 'images_tensor' in locals() and images_tensor is not None:
                            del images_tensor
                        self._aggressive_cleanup()

                        # Reduce batch size for remaining samples
                        if batch_size > 10:
                            print(f"Reducing batch size from {batch_size} to {batch_size//2}")
                            batch_size = batch_size // 2

                    continue

            # Add batch results to main results
            for layer_idx in range(layer_start, layer_end+1):
                all_hidden_states[layer_idx].extend(batch_hidden_states[layer_idx])
            labels.extend(batch_labels)

            # Clear batch data
            del batch_hidden_states
            del batch_labels

            # Aggressive memory cleanup after each batch
            self._aggressive_cleanup()

            # print(f"Completed batch {batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1}, "
                #   f"processed {len(labels)}/{total_samples} samples")

        pbar.close()

        # Cache the results
        if use_cache:
            metadata = {
                'dataset_size': len(dataset),
                'label_key': label_key,
                'processed_samples': len(labels)
            }
            self.cache.save(dataset_name, self.model_path, layer_range, all_hidden_states, labels,
                          metadata, dataset_size, experiment_name)

        # Final cleanup
        self._aggressive_cleanup()

        print(f"Feature extraction completed. Final memory usage:")
        final_mem = self._get_memory_info()
        print(f"  GPU: {final_mem['gpu_allocated']:.1f}GB allocated, {final_mem['gpu_cached']:.1f}GB cached")
        print(f"  CPU: {final_mem['cpu_used']:.1f}GB used")

        return all_hidden_states, labels, {}

    def clear_cache(self):
        """Clear all cached features"""
        self.cache.clear()
