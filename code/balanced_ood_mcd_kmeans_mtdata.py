#!/usr/bin/env python3
"""
Test the balanced OOD MCD k-means method against SafeMTData multi-turn jailbreaking dataset.
This script integrates SafeMTData with the existing detection framework.
"""

import os
import sys
import json
import torch
import numpy as np
import random
import warnings
import hashlib
import pickle
from datasets import load_dataset
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import csv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")

# Add LLaVA to path
sys.path.append('src/LLaVA')

# Import existing modules
from load_datasets import *
from feature_extractor import HiddenStateExtractor
from balanced_ood_mcd_kmeans import (
    MCDDetector, prepare_kmeans_data_structure,
    prepare_balanced_training, train_learned_projection, apply_learned_projection,
    CONFIG, GPU_DEVICE, cleanup_gpu_memory
)
from balanced_ood_mcd import prepare_balanced_evaluation

def build_dynamic_multiturn_conversation_for_detection(multi_turn_queries, tokenizer, model, model_name):
    """Build multi-turn conversation with CONSISTENT hidden state extraction position"""
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    def find_conv_mode(model_name):
        if "llama-2" in model_name.lower():
            return "llava_llama_2"
        elif "mistral" in model_name.lower():
            return "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            return "chatml_direct"
        elif "v1" in model_name.lower():
            return "llava_v1"
        elif "mpt" in model_name.lower():
            return "mpt"
        else:
            return "llava_v0"

    conv_mode = find_conv_mode(model_name)
    conv = conv_templates[conv_mode].copy()

    # Build conversation step by step with real LLaVA responses for ALL turns EXCEPT the final one
    for turn_idx, query in enumerate(multi_turn_queries[:-1]):  # Process all but the last turn
        # Add user message to conversation
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)

        # Get current prompt
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Truncate if too long
        max_length = getattr(model.config, 'max_position_embeddings', 4096)
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]

        # Generate response from LLaVA
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=256,  # Shorter responses for intermediate turns
                use_cache=True,
                stopping_criteria=None
            )

        # Decode response
        input_token_len = input_ids.shape[1]
        if output_ids.shape[1] >= input_token_len:
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        else:
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        outputs = outputs.strip()

        # Update conversation with LLaVA's actual response
        conv.messages[-1][-1] = outputs

        print(f"\n--- Turn {turn_idx + 1} ---")
        print(f"User: {query}")
        print(f"LLaVA: {outputs}")

    # For the FINAL turn: Add user query but NO assistant response (same as single-turn)
    final_query = multi_turn_queries[-1]
    conv.append_message(conv.roles[0], final_query)
    conv.append_message(conv.roles[1], None)  # CRITICAL: No response, same as single-turn

    print(f"\n--- Turn {len(multi_turn_queries)} (FINAL - No Response Added) ---")
    print(f"User: {final_query}")
    print(f"Assistant: [EMPTY - Hidden state extracted here, same as single-turn]")

    # Return the conversation object ready for feature extraction
    return conv

def get_safemtdata_cache_path():
    """Get the cache directory path for SafeMTData responses"""
    cache_dir = "cache/safemtdata_responses"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_sample_hash(sample):
    """Generate a unique hash for a SafeMTData sample"""
    # Create a deterministic hash based on the sample content
    content = json.dumps(sample, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def cache_safemtdata_response(sample_hash, response_data):
    """Cache a SafeMTData response to avoid re-querying the model"""
    cache_dir = get_safemtdata_cache_path()
    cache_file = os.path.join(cache_dir, f"{sample_hash}.pkl")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(response_data, f)

def load_cached_safemtdata_response(sample_hash):
    """Load a cached SafeMTData response if it exists"""
    cache_dir = get_safemtdata_cache_path()
    cache_file = os.path.join(cache_dir, f"{sample_hash}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def load_safemtdata_for_adaptability_test(tokenizer, model, model_name, num_training=200, num_test=50):
    """Load SafeMTData for adaptability test: training + test with NO overlap"""
    print(f"Loading SafeMTData for adaptability test: {num_training} training + {num_test} test samples...")

    try:
        # Load Attack_600 dataset
        attack_600 = load_dataset("SafeMTData/SafeMTData", 'Attack_600')["Attack_600"]
        print(f"Loaded Attack_600: {len(attack_600)} samples")

        # Ensure we have enough samples
        total_needed = num_training + num_test
        if len(attack_600) < total_needed:
            print(f"Warning: Only {len(attack_600)} samples available, need {total_needed}")
            num_training = min(num_training, len(attack_600))
            num_test = min(num_test, len(attack_600) - num_training)
            print(f"Adjusted to: {num_training} training + {num_test} test")

        # Split into discrete sets: training [0:num_training], test [num_training:num_training+num_test]
        attack_600_list = list(attack_600)
        training_samples = attack_600_list[:num_training]
        test_samples = attack_600_list[num_training:num_training + num_test]
        
        print(f"Split SafeMTData: {len(training_samples)} training, {len(test_samples)} test, {len(attack_600) - total_needed} unused")

        # Process training samples
        print(f"\n--- Processing {len(training_samples)} SafeMTData TRAINING samples ---")
        safemt_training = []
        for idx, sample in enumerate(training_samples):

            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                # Check cache first
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)
                
                if cached_response is not None:
                    # Using cached response
                    final_conversation_text = cached_response
                else:
                    print(f"  Generating new response for sample {idx + 1}")
                    # Build REAL multi-turn conversation with CONSISTENT hidden state extraction
                    final_conv = build_dynamic_multiturn_conversation_for_detection(
                        multi_turn_queries, tokenizer, model, model_name
                    )
                    # Get the final conversation text for feature extraction
                    final_conversation_text = final_conv.get_prompt()
                    # Cache the response
                    cache_safemtdata_response(sample_hash, final_conversation_text)

                safemt_sample = {
                    "txt": final_conversation_text,  # REAL multi-turn conversation with LLaVA responses
                    "img": None,  # Text-only
                    "toxicity": 1,  # All SafeMTData samples are malicious
                    "category": sample.get('category', 'multi_turn_jailbreak'),
                    "original_id": sample.get('id', -1),
                    "num_turns": len(multi_turn_queries),
                    "final_query": multi_turn_queries[-1]  # Keep track of the final harmful query
                }
                safemt_training.append(safemt_sample)

        # Process test samples  
        print(f"\n--- Processing {len(test_samples)} SafeMTData TEST samples ---")
        safemt_test = []
        for idx, sample in enumerate(test_samples):

            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                # Check cache first
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)
                
                if cached_response is not None:
                    # Using cached response
                    final_conversation_text = cached_response
                else:
                    print(f"  Generating new response for test sample {idx + 1}")
                    # Build REAL multi-turn conversation with CONSISTENT hidden state extraction
                    final_conv = build_dynamic_multiturn_conversation_for_detection(
                        multi_turn_queries, tokenizer, model, model_name
                    )
                    # Get the final conversation text for feature extraction
                    final_conversation_text = final_conv.get_prompt()
                    # Cache the response
                    cache_safemtdata_response(sample_hash, final_conversation_text)

                safemt_sample = {
                    "txt": final_conversation_text,  # REAL multi-turn conversation with LLaVA responses
                    "img": None,  # Text-only
                    "toxicity": 1,  # All SafeMTData samples are malicious
                    "category": sample.get('category', 'multi_turn_jailbreak'),
                    "original_id": sample.get('id', -1),
                    "num_turns": len(multi_turn_queries),
                    "final_query": multi_turn_queries[-1]  # Keep track of the final harmful query
                }
                safemt_test.append(safemt_sample)

        print(f"\nConverted SafeMTData: {len(safemt_training)} training + {len(safemt_test)} test samples")
        print("Sample conversation structure:")
        if safemt_training:
            sample_conv = safemt_training[0]['txt']
            print(f"First 500 chars: {sample_conv[:500]}...")

        return safemt_training, safemt_test

    except Exception as e:
        print(f"Error loading SafeMTData: {e}")
        return [], []

def load_safemtdata_for_variable_training_experiment(tokenizer, model, model_name, num_training=0, num_test=100, random_seed=42):
    """Load SafeMTData for variable training experiment with fixed test set"""
    print(f"Loading SafeMTData for variable training experiment (seed={random_seed}): {num_training} training + {num_test} test samples...")

    try:
        # Load Attack_600 dataset
        attack_600 = load_dataset("SafeMTData/SafeMTData", 'Attack_600')["Attack_600"]
        print(f"Loaded Attack_600: {len(attack_600)} samples")

        # Convert to list and set random seed for reproducible sampling
        attack_600_list = list(attack_600)
        random.seed(random_seed)
        
        # Shuffle the dataset to get different samples for each run
        shuffled_samples = attack_600_list.copy()
        random.shuffle(shuffled_samples)
        
        # Always use first 100 samples for test (consistent across all experiments)
        test_samples = shuffled_samples[:num_test]
        
        # Sample training data from the remaining samples
        remaining_samples = shuffled_samples[num_test:]
        if num_training > 0:
            if len(remaining_samples) < num_training:
                print(f"Warning: Only {len(remaining_samples)} samples available for training, need {num_training}")
                num_training = len(remaining_samples)
            training_samples = remaining_samples[:num_training]
        else:
            training_samples = []
        
        print(f"Sampled SafeMTData (seed={random_seed}): {len(training_samples)} training, {len(test_samples)} test")

        # Check if we need to process any samples (i.e., if any are not cached)
        samples_needing_processing = []
        all_samples = training_samples + test_samples
        
        for sample in all_samples:
            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)
                if cached_response is None:
                    samples_needing_processing.append(sample)
        
        # Only load model if there are samples that need processing
        model_loaded = False
        if samples_needing_processing and (tokenizer is None or model is None):
            print(f"Loading LLaVA model and tokenizer for SafeMTData processing ({len(samples_needing_processing)} uncached samples)...")
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_name,
                model_base=None,
                model_name=get_model_name_from_path(model_name),
                device=GPU_DEVICE
            )
            print(f"Model loaded on device: {GPU_DEVICE}")
            model_loaded = True
            
            # Clean up after loading
            cleanup_gpu_memory()
        elif not samples_needing_processing:
            print("All samples are cached, skipping model loading")

        # Process training samples with caching
        safemt_training = []
        if training_samples:
            print(f"\n--- Processing {len(training_samples)} SafeMTData TRAINING samples ---")
            for idx, sample in enumerate(training_samples):

                multi_turn_queries = sample['multi_turn_queries']
                if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                    # Check cache first
                    sample_hash = get_sample_hash(sample)
                    cached_response = load_cached_safemtdata_response(sample_hash)
                    
                    if cached_response is not None:
                        # Using cached response
                        final_conversation_text = cached_response
                    else:
                        print(f"  Generating new response for training sample {idx + 1}")
                        # Build REAL multi-turn conversation with CONSISTENT hidden state extraction
                        final_conv = build_dynamic_multiturn_conversation_for_detection(
                            multi_turn_queries, tokenizer, model, model_name
                        )
                        # Get the final conversation text for feature extraction
                        final_conversation_text = final_conv.get_prompt()
                        # Cache the response
                        cache_safemtdata_response(sample_hash, final_conversation_text)

                    safemt_sample = {
                        "txt": final_conversation_text,
                        "img": None,
                        "toxicity": 1,
                        "category": sample.get('category', 'multi_turn_jailbreak'),
                        "original_id": sample.get('id', -1),
                        "num_turns": len(multi_turn_queries),
                        "final_query": multi_turn_queries[-1]
                    }
                    safemt_training.append(safemt_sample)

        # Process test samples with caching (consistent across all runs)
        print(f"\n--- Processing {len(test_samples)} SafeMTData TEST samples ---")
        safemt_test = []
        for idx, sample in enumerate(test_samples):

            multi_turn_queries = sample['multi_turn_queries']
            if isinstance(multi_turn_queries, list) and len(multi_turn_queries) > 0:
                # Check cache first
                sample_hash = get_sample_hash(sample)
                cached_response = load_cached_safemtdata_response(sample_hash)
                
                if cached_response is not None:
                    # Using cached response
                    final_conversation_text = cached_response
                else:
                    print(f"  Generating new response for test sample {idx + 1}")
                    # Build REAL multi-turn conversation with CONSISTENT hidden state extraction
                    final_conv = build_dynamic_multiturn_conversation_for_detection(
                        multi_turn_queries, tokenizer, model, model_name
                    )
                    # Get the final conversation text for feature extraction
                    final_conversation_text = final_conv.get_prompt()
                    # Cache the response
                    cache_safemtdata_response(sample_hash, final_conversation_text)

                safemt_sample = {
                    "txt": final_conversation_text,
                    "img": None,
                    "toxicity": 1,
                    "category": sample.get('category', 'multi_turn_jailbreak'),
                    "original_id": sample.get('id', -1),
                    "num_turns": len(multi_turn_queries),
                    "final_query": multi_turn_queries[-1]
                }
                safemt_test.append(safemt_sample)

        print(f"\nConverted SafeMTData (seed={random_seed}): {len(safemt_training)} training + {len(safemt_test)} test samples")
        
        # Clean up model memory after SafeMTData processing (only if we loaded it)
        if model_loaded and 'model' in locals():
            del model, tokenizer, image_processor
            cleanup_gpu_memory()
            print("Cleaned up SafeMTData model from memory")

        return safemt_training, safemt_test

    except Exception as e:
        print(f"Error loading SafeMTData: {e}")
        # Clean up on error too (only if we loaded the model)
        if 'model_loaded' in locals() and model_loaded and 'model' in locals():
            del model, tokenizer, image_processor
            cleanup_gpu_memory()
        return [], []

def create_adaptability_datasets(tokenizer, model, model_name):
    """Create training and test datasets for adaptability experiment"""
    print("Creating adaptability experiment datasets...")

    # Load original training and test datasets (baseline)
    print("Loading original datasets...")
    benign_training, malicious_training = prepare_balanced_training()
    original_test_datasets = prepare_balanced_evaluation()

    # Load SafeMTData with training/test split (NO overlap)
    print("Loading SafeMTData with training/test split...")
    safemt_training, safemt_test = load_safemtdata_for_adaptability_test(
        tokenizer, model, model_name, num_training=200, num_test=50
    )

    # EXPAND malicious training by adding SafeMTData training samples
    expanded_malicious_training = malicious_training.copy()
    if safemt_training:
        expanded_malicious_training["SafeMTData_Training"] = safemt_training
        print(f"EXPANDED malicious training: added {len(safemt_training)} SafeMTData samples")

    # EXPAND test datasets by adding SafeMTData test samples 
    expanded_test_datasets = original_test_datasets.copy()
    if safemt_test:
        expanded_test_datasets["SafeMTData_Test"] = safemt_test
        print(f"EXPANDED test datasets: added {len(safemt_test)} SafeMTData samples")

    print(f"\nAdaptability experiment datasets:")
    print(f"TRAINING:")
    for dataset_name, samples in benign_training.items():
        print(f"  Benign - {dataset_name}: {len(samples)} samples")
    for dataset_name, samples in expanded_malicious_training.items():
        dataset_type = "Multi-turn" if "SafeMTData" in dataset_name else "Single-turn"
        print(f"  Malicious - {dataset_name}: {len(samples)} samples [{dataset_type}]")
    
    total_benign = sum(len(samples) for samples in benign_training.values())
    total_malicious = sum(len(samples) for samples in expanded_malicious_training.values())
    print(f"  TOTAL TRAINING: {total_benign} benign + {total_malicious} malicious = {total_benign + total_malicious}")
    
    print(f"TEST:")
    for dataset_name, samples in expanded_test_datasets.items():
        dataset_type = "Multi-turn" if "SafeMTData" in dataset_name else "Single-turn"
        toxicity_info = f"({sum(1 for s in samples if s.get('toxicity', 0) == 1)} malicious, {sum(1 for s in samples if s.get('toxicity', 0) == 0)} benign)"
        print(f"  {dataset_name}: {len(samples)} samples {toxicity_info} [{dataset_type}]")

    return benign_training, expanded_malicious_training, expanded_test_datasets

def analyze_safemtdata_results_detailed(detector, test_features, test_labels, dataset_name, layer_idx):
    """Analyze SafeMTData results with detailed per-sample scoring"""
    print(f"\n=== SafeMTData Multi-turn Attack Analysis (Layer {layer_idx}) ===")

    # Get detailed predictions and scores
    predictions, scores = detector.predict(test_features)
    threshold = detector.threshold

    print(f"Detection threshold: {threshold:.4f}")
    print(f"Total samples: {len(test_features)}")
    print(f"All samples should be malicious (label=1): {set(test_labels)}")
    print(f"Prediction logic: score > {threshold:.4f} → malicious (1), score ≤ {threshold:.4f} → benign (0)")

    print(f"\n--- Per-Sample Analysis ---")
    print(f"{'Sample':<8} {'Score':<12} {'Prediction':<12} {'Label':<8} {'Correct':<8} {'Status'}")
    print("-" * 70)

    correct_count = 0
    for i, (score, pred, label) in enumerate(zip(scores, predictions, test_labels)):
        correct = (pred == label)
        if correct:
            correct_count += 1

        status = "✅ DETECTED" if (pred == 1 and label == 1) else "❌ MISSED" if (pred == 0 and label == 1) else "⚠️  OTHER"

        print(f"{i+1:<8} {score:<12.4f} {pred:<12} {label:<8} {correct:<8} {status}")

    accuracy = correct_count / len(test_labels)

    # Calculate detailed metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n--- Classification Report ---")

    # Check if we have both classes in the data
    unique_labels = set(test_labels)
    unique_preds = set(predictions)

    if len(unique_labels) == 1 and len(unique_preds) == 1:
        # Only one class present - SafeMTData only has malicious samples
        if 1 in unique_labels and 1 in unique_preds:
            print("All samples are malicious and all were correctly detected as malicious.")
            print("Perfect detection: 100% accuracy on malicious samples.")
        else:
            print("Unexpected single class scenario.")
    else:
        # Normal case with multiple classes
        available_classes = sorted(list(unique_labels.union(unique_preds)))
        target_names = ['Benign' if c == 0 else 'Malicious' for c in available_classes]
        print(classification_report(test_labels, predictions, labels=available_classes, target_names=target_names))

    print(f"\n--- Confusion Matrix ---")
    cm = confusion_matrix(test_labels, predictions)

    # Handle different matrix sizes
    if cm.shape == (1, 1):
        print(f"Single class confusion matrix:")
        if test_labels[0] == 1:  # All malicious
            print(f"Malicious samples correctly detected: {cm[0,0]}")
        else:  # All benign
            print(f"Benign samples correctly detected: {cm[0,0]}")
    else:
        print(f"Confusion Matrix:")
        print(f"                Predicted")
        print(f"Actual    Benign  Malicious")
        print(f"Benign    {cm[0,0]:<6}  {cm[0,1]:<6}")
        print(f"Malicious {cm[1,0]:<6}  {cm[1,1]:<6}")

    # Score distribution analysis
    print(f"\n--- Score Distribution Analysis ---")
    print(f"Score statistics:")
    print(f"  Min score: {min(scores):.4f}")
    print(f"  Max score: {max(scores):.4f}")
    print(f"  Mean score: {sum(scores)/len(scores):.4f}")
    print(f"  Threshold: {threshold:.4f}")

    # Count how many scores are above/below threshold
    above_threshold = sum(1 for s in scores if s > threshold)
    below_threshold = len(scores) - above_threshold

    print(f"  Scores above threshold (predicted malicious): {above_threshold}")
    print(f"  Scores below threshold (predicted benign): {below_threshold}")

    return {
        'accuracy': accuracy,
        'scores': scores,
        'predictions': predictions,
        'labels': test_labels,
        'threshold': threshold
    }

def main():
    """Main function to test SafeMTData with balanced OOD MCD k-means"""
    print("="*80)
    print("ADAPTABILITY TEST: TRAINING WITH SAFEMTDATA TO DETECT MULTI-TURN ATTACKS")
    print("="*80)
    print("Experiment: Add 200 SafeMTData samples to malicious training (1000→1200)")
    print("            Test on 50 different SafeMTData samples (no overlap)")
    print("            Evaluate adaptability to multi-turn jailbreaking patterns")
    print("="*80)
    
    # Set random seed for reproducibility
    SEED = 45
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    model_path = "model/llava-v1.6-vicuna-7b/"

    # Load LLaVA model for dynamic multi-turn conversations
    print("\n--- Loading LLaVA Model for Dynamic Multi-turn Conversations ---")
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path, model_base=None, model_name=model_name, **kwargs
    )
    print(f"Model loaded: {model_name}")

    # Initialize feature extractor
    extractor = HiddenStateExtractor(model_path)

    # Load adaptability experiment datasets (expanded training + test)
    print("\n--- Loading Adaptability Experiment Data ---")
    benign_training, expanded_malicious_training, test_datasets = create_adaptability_datasets(tokenizer, model, model_name)
    training_datasets = {**benign_training, **expanded_malicious_training}
    
    # Print dataset summary
    print(f"\nDataset Summary:")
    print(f"Training datasets: {list(training_datasets.keys())}")
    print(f"Test datasets: {list(test_datasets.keys())}")
    
    # Test on optimal layer (16) based on previous results
    optimal_layer = 16
    print(f"\n--- Testing on Optimal Layer {optimal_layer} ---")
    
    # Extract hidden states for all datasets (with caching) - same approach as balanced_ood_mcd_kmeans.py
    print("Extracting hidden states for all datasets...")
    all_datasets = {**training_datasets, **test_datasets}
    all_hidden_states = {}
    all_labels = {}

    for dataset_name, samples in all_datasets.items():
        print(f"Extracting features for {dataset_name} ({len(samples)} samples)...")
        hidden_states, labels, _ = extractor.extract_hidden_states(
            samples, dataset_name=dataset_name,
            layer_start=optimal_layer, layer_end=optimal_layer,
            use_cache=True, experiment_name="safemtdata_test"
        )
        all_hidden_states[dataset_name] = hidden_states[optimal_layer]
        all_labels[dataset_name] = labels

    # Separate training and test data
    training_hidden_states = {k: v for k, v in all_hidden_states.items() if k in training_datasets}
    training_labels = {k: v for k, v in all_labels.items() if k in training_datasets}

    test_hidden_states = {k: v for k, v in all_hidden_states.items() if k in test_datasets}
    test_labels = {k: v for k, v in all_labels.items() if k in test_datasets}
    
    # CRITICAL: Train and apply learned projection (same as original script)
    print("Training learned projection for optimal layer...")
    
    # Train learned projection using training data (same as original)
    projection_model, dataset_name_to_id = train_learned_projection(
        training_hidden_states, training_labels,
        device=GPU_DEVICE, random_seed=SEED + optimal_layer
    )
    
    print("Applying learned projection to all datasets...")
    
    # Apply projection to all hidden states (training + test)
    all_projected_hidden_states = apply_learned_projection(
        projection_model, all_hidden_states, device=GPU_DEVICE
    )
    
    # Update hidden states with projected versions
    training_hidden_states = {k: v for k, v in all_projected_hidden_states.items() if k in training_datasets}
    test_hidden_states = {k: v for k, v in all_projected_hidden_states.items() if k in test_datasets}
    
    cleanup_gpu_memory()
    
    # Prepare data structure using optimal k-means configuration (8 benign, 1 malicious)
    print("Preparing k-means data structure with optimal configuration...")

    # Separate benign and malicious datasets for k-means clustering
    benign_datasets = {k: v for k, v in training_datasets.items() if k in benign_training}
    malicious_datasets = {k: v for k, v in training_datasets.items() if k in expanded_malicious_training}

    benign_hidden_states = {k: v for k, v in training_hidden_states.items() if k in benign_training}
    malicious_hidden_states = {k: v for k, v in training_hidden_states.items() if k in expanded_malicious_training}

    benign_labels = {k: v for k, v in training_labels.items() if k in benign_training}
    malicious_labels = {k: v for k, v in training_labels.items() if k in expanded_malicious_training}

    # Combine for k-means clustering
    all_training_datasets = {**benign_datasets, **malicious_datasets}
    all_training_hidden_states = {**benign_hidden_states, **malicious_hidden_states}
    all_training_labels = {**benign_labels, **malicious_labels}

    in_dist_data, ood_data = prepare_kmeans_data_structure(
        all_training_datasets, all_training_hidden_states, all_training_labels,
        random_seed=SEED + optimal_layer, k_benign=8, k_malicious=1  # Layer-specific seed like original
    )
    
    # Train MCD detector
    print("Training MCD detector...")
    detector = MCDDetector(use_gpu=True)
    detector.fit_in_distribution(in_dist_data)
    detector.fit_ood_clusters(ood_data)
    
    # Create validation data for threshold optimization (SAME AS ORIGINAL SCRIPT)
    print("Preparing validation data...")
    val_features = []
    val_labels = []

    # Sample from in-distribution data (increase sample size for better threshold estimation)
    # Use deterministic sampling based on indices for reproducibility
    for _, features in in_dist_data.items():
        sample_size = min(100, len(features))
        if sample_size > 0:
            # Use deterministic sampling: take evenly spaced indices
            indices = np.linspace(0, len(features)-1, sample_size, dtype=int)
            sampled_features = [features[i] for i in indices]
            val_features.extend(sampled_features)
            val_labels.extend([0] * len(sampled_features))

    # Sample from OOD data (increase sample size for better threshold estimation)
    for _, features in ood_data.items():
        sample_size = min(100, len(features))
        if sample_size > 0:
            # Use deterministic sampling: take evenly spaced indices
            indices = np.linspace(0, len(features)-1, sample_size, dtype=int)
            sampled_features = [features[i] for i in indices]
            val_features.extend(sampled_features)
            val_labels.extend([1] * len(sampled_features))

    print(f"Validation set: {len(val_features)} samples ({val_labels.count(0)} benign, {val_labels.count(1)} malicious)")

    # Optimize threshold (SAME AS ORIGINAL SCRIPT)
    print("Optimizing detection threshold...")
    detector.fit_threshold(val_features, val_labels)
    
    # Test on all datasets with comprehensive analysis
    print("Testing on all datasets with comprehensive analysis...")

    all_results = {}
    safemt_detailed_result = None

    for dataset_name in test_datasets.keys():
        features = test_hidden_states[dataset_name]
        labels = test_labels[dataset_name]

        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR {dataset_name}")
        print(f"{'='*80}")

        # Get basic evaluation metrics
        result = detector.evaluate(features, labels)
        all_results[dataset_name] = result
        print(f"Basic metrics: Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, AUROC={result['auroc']:.4f}")

        # Detailed per-sample analysis for SafeMTData Test (adaptability evaluation)
        if "SafeMTData_Test" in dataset_name:
            print(f"--- ADAPTABILITY ANALYSIS: MULTI-TURN DETECTION AFTER TRAINING ---")
            safemt_detailed_result = analyze_safemtdata_results_detailed(detector, features, labels, dataset_name, optimal_layer)
        else:
            # Brief analysis for other datasets
            dataset_type = "Multi-turn" if "SafeMTData" in dataset_name else "Single-turn"
            toxicity_dist = f"({sum(labels)} malicious, {len(labels) - sum(labels)} benign)"
            print(f"Dataset type: {dataset_type}, Distribution: {toxicity_dist}")

    # Summary comparison table
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE RESULTS COMPARISON")
    print(f"{'='*100}")
    print(f"{'Dataset':<25} {'Accuracy':<10} {'F1':<8} {'AUROC':<8} {'TPR':<8} {'FPR':<8} {'Type':<15} {'Notes'}")
    print("-" * 100)

    for dataset_name, result in all_results.items():
        dataset_type = "Multi-turn" if "SafeMTData" in dataset_name else "Single-turn"
        notes = "NEW" if "SafeMTData" in dataset_name else "Original"
        print(f"{dataset_name:<25} {result['accuracy']:<10.4f} {result['f1']:<8.4f} "
              f"{result['auroc']:<8.4f} {result['tpr']:<8.4f} {result['fpr']:<8.4f} {dataset_type:<15} {notes}")

    # COMBINED EVALUATION (like in original script)
    print(f"\n{'='*100}")
    print(f"COMBINED EVALUATION (All Test Datasets Together)")
    print(f"{'='*100}")

    # Combine all test features and labels
    combined_features = []
    combined_labels = []
    dataset_boundaries = {}
    current_idx = 0

    for dataset_name in test_datasets.keys():
        if dataset_name in test_hidden_states:
            features = test_hidden_states[dataset_name]
            labels = test_labels[dataset_name]

            dataset_boundaries[dataset_name] = (current_idx, current_idx + len(features))
            combined_features.extend(features)
            combined_labels.extend(labels)
            current_idx += len(features)

    print(f"Combined dataset: {len(combined_features)} samples")
    print(f"  Benign: {combined_labels.count(0)} samples")
    print(f"  Malicious: {combined_labels.count(1)} samples")
    print(f"  Ratio (Benign:Malicious): {combined_labels.count(0)/max(combined_labels.count(1),1):.2f}:1")

    # Evaluate on combined dataset
    combined_result = detector.evaluate(combined_features, combined_labels)

    print(f"\nCOMBINED METRICS:")
    print(f"  Accuracy: {combined_result['accuracy']:.4f}")
    print(f"  F1 Score: {combined_result['f1']:.4f}")
    print(f"  True Positive Rate (TPR): {combined_result['tpr']:.4f}")
    print(f"  False Positive Rate (FPR): {combined_result['fpr']:.4f}")
    print(f"  AUROC: {combined_result['auroc']:.4f}")
    print(f"  AUPRC: {combined_result['auprc']:.4f}")
    print(f"  Threshold: {detector.threshold:.4f}")

    # Score distribution analysis
    print(f"\nSCORE DISTRIBUTION ANALYSIS:")
    predictions, scores = detector.predict(combined_features)

    benign_scores = [scores[i] for i, label in enumerate(combined_labels) if label == 0]
    malicious_scores = [scores[i] for i, label in enumerate(combined_labels) if label == 1]

    if benign_scores:
        print(f"  Benign scores: mean={np.mean(benign_scores):.4f}, std={np.std(benign_scores):.4f}, "
              f"range=[{np.min(benign_scores):.4f}, {np.max(benign_scores):.4f}]")
    if malicious_scores:
        print(f"  Malicious scores: mean={np.mean(malicious_scores):.4f}, std={np.std(malicious_scores):.4f}, "
              f"range=[{np.min(malicious_scores):.4f}, {np.max(malicious_scores):.4f}]")

    # Threshold analysis
    print(f"  Threshold: {detector.threshold:.4f}")
    above_threshold = sum(1 for s in scores if s > detector.threshold)
    below_threshold = len(scores) - above_threshold
    print(f"  Samples above threshold (predicted malicious): {above_threshold}")
    print(f"  Samples below threshold (predicted benign): {below_threshold}")

    # Distribution separation
    if benign_scores and malicious_scores:
        separation = abs(np.mean(malicious_scores) - np.mean(benign_scores)) / (np.std(benign_scores) + np.std(malicious_scores) + 1e-8)
        print(f"  Score separation: {separation:.4f} (higher is better)")

        if separation < 2.0:
            print("  ⚠️  WARNING: Low score separation indicates overlapping distributions")
        elif separation > 5.0:
            print("  ✅ GOOD: High score separation indicates well-separated distributions")
        else:
            print("  ✅ MODERATE: Reasonable score separation")

    # Analyze threshold compatibility
    print(f"\n{'='*100}")
    print(f"THRESHOLD COMPATIBILITY ANALYSIS")
    print(f"{'='*100}")
    print(f"Threshold: {detector.threshold:.4f}")

    high_performance = [name for name, result in all_results.items() if result['accuracy'] > 0.8]
    moderate_performance = [name for name, result in all_results.items() if 0.6 <= result['accuracy'] <= 0.8]
    low_performance = [name for name, result in all_results.items() if result['accuracy'] < 0.6]

    print(f"High Performance (>80% accuracy): {len(high_performance)} datasets")
    for name in high_performance:
        print(f"  ✅ {name}: {all_results[name]['accuracy']:.1%}")

    print(f"Moderate Performance (60-80% accuracy): {len(moderate_performance)} datasets")
    for name in moderate_performance:
        print(f"  ⚠️  {name}: {all_results[name]['accuracy']:.1%}")

    print(f"Low Performance (<60% accuracy): {len(low_performance)} datasets")
    for name in low_performance:
        print(f"  ❌ {name}: {all_results[name]['accuracy']:.1%}")
    

def aggregate_and_report_results(all_results, training_amounts, results_dir="results"):
    """
    Aggregate results by training size, print summary, write CSV, and plot trends.
    """
    os.makedirs(results_dir, exist_ok=True)
    summary = []
    for num_training in training_amounts:
        runs = [r for r in all_results if r['num_training'] == num_training and 'error' not in r]
        if not runs:
            continue
        avg_combined_acc = np.mean([r['combined_accuracy'] for r in runs])
        std_combined_acc = np.std([r['combined_accuracy'] for r in runs])
        avg_safemt_acc = np.mean([r['safemt_accuracy'] for r in runs])
        std_safemt_acc = np.std([r['safemt_accuracy'] for r in runs])
        avg_auroc = np.mean([r['combined_auroc'] for r in runs])
        std_auroc = np.std([r['combined_auroc'] for r in runs])
        summary.append({
            'num_training': num_training,
            'avg_combined_acc': avg_combined_acc,
            'std_combined_acc': std_combined_acc,
            'avg_safemt_acc': avg_safemt_acc,
            'std_safemt_acc': std_safemt_acc,
            'avg_auroc': avg_auroc,
            'std_auroc': std_auroc
        })
    # Print summary table
    print("\nAggregated Results (mean ± std):")
    print(f"{'Train#':>7}  {'Overall Acc':>12}  {'SafeMT Acc':>12}  {'AUROC':>10}")
    for row in summary:
        print(f"{row['num_training']:7d}  {row['avg_combined_acc']:.4f} ± {row['std_combined_acc']:.4f}  "
              f"{row['avg_safemt_acc']:.4f} ± {row['std_safemt_acc']:.4f}  "
              f"{row['avg_auroc']:.4f} ± {row['std_auroc']:.4f}")
    # Write CSV
    csv_path = os.path.join(results_dir, "safemt_aggregated.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["num_training", "avg_combined_acc", "std_combined_acc", "avg_safemt_acc", "std_safemt_acc", "avg_auroc", "std_auroc"])
        for row in summary:
            writer.writerow([
                row['num_training'], row['avg_combined_acc'], row['std_combined_acc'],
                row['avg_safemt_acc'], row['std_safemt_acc'], row['avg_auroc'], row['std_auroc']
            ])
    print(f"\nAggregated results written to: {csv_path}")
    # Plot trends
    x = [row['num_training'] for row in summary]
    y_acc = [row['avg_combined_acc'] for row in summary]
    y_acc_std = [row['std_combined_acc'] for row in summary]
    y_mt = [row['avg_safemt_acc'] for row in summary]
    y_mt_std = [row['std_safemt_acc'] for row in summary]
    y_auroc = [row['avg_auroc'] for row in summary]
    y_auroc_std = [row['std_auroc'] for row in summary]
    plt.figure(figsize=(8,6))
    plt.errorbar(x, y_acc, yerr=y_acc_std, label='Overall Accuracy', marker='o', capsize=3)
    plt.errorbar(x, y_mt, yerr=y_mt_std, label='SafeMTData Accuracy', marker='s', capsize=3)
    plt.errorbar(x, y_auroc, yerr=y_auroc_std, label='AUROC', marker='^', capsize=3)
    # Add horizontal dashed lines for baselines
    plt.axhline(0.9221, color='tab:blue', linestyle='--', linewidth=1.5)
    plt.axhline(0.9817, color='tab:green', linestyle='--', linewidth=1.5)
    plt.xlabel('Number of SafeMTData Training Samples')
    plt.ylabel('Score')
    plt.title('Detection Performance vs. SafeMTData Training Size')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "safemt_trends.png")
    plt.savefig(fig_path)
    print(f"Trend plot saved to: {fig_path}")
    plt.close()

def run_variable_training_experiment(tokenizer, model, model_name):
    """Run the variable training data experiment: 0 to 200 training samples, 5 runs each"""
    print("="*80)
    print("VARIABLE TRAINING DATA EXPERIMENT: SAFEMTDATA ADAPTABILITY ANALYSIS")
    print("="*80)
    
    # Experimental configuration
    training_amounts = list(range(0,51, 5))
    random_seeds = [42, 123, 456, 789, 999]
    
    # Results storage
    all_results = []
    
    # Load original datasets (consistent across all experiments)
    print("\nLoading original datasets...")
    benign_training, malicious_training = prepare_balanced_training()
    original_test_datasets = prepare_balanced_evaluation()
    
    # Initialize feature extractor ONCE for all experiments
    print("Initializing feature extractor...")
    extractor = HiddenStateExtractor(model_name)
    
    # Pre-extract features for all original datasets (they're consistent across runs)
    print("Pre-extracting features for original datasets...")
    original_training_datasets = {**benign_training, **malicious_training}
    original_all_datasets = {**original_training_datasets, **original_test_datasets}
    
    original_hidden_states = {}
    original_labels = {}
    
    for dataset_name, samples in original_all_datasets.items():
        print(f"Pre-extracting features for {dataset_name} ({len(samples)} samples)...")
        
        # Use smaller batch sizes for large datasets to manage memory
        batch_size = 25 if len(samples) > 5000 else 50
        memory_cleanup_freq = 5 if len(samples) > 5000 else 10
        
        hidden_states, labels, _ = extractor.extract_hidden_states(
            samples, f"{dataset_name}", layer_start=16, layer_end=16, use_cache=True,
            batch_size=batch_size, memory_cleanup_freq=memory_cleanup_freq,
            experiment_name="safemtdata_variable_training"
        )
        original_hidden_states[dataset_name] = hidden_states
        original_labels[dataset_name] = labels
        
        # Clean up GPU memory after each dataset
        cleanup_gpu_memory()
    
    # CRITICAL: Manually unload the model from HiddenStateExtractor to free GPU memory
    print("Unloading model from HiddenStateExtractor to free GPU memory...")
    if hasattr(extractor, 'model') and extractor.model is not None:
        del extractor.model
    if hasattr(extractor, 'tokenizer') and extractor.tokenizer is not None:
        del extractor.tokenizer
    if hasattr(extractor, 'image_processor') and extractor.image_processor is not None:
        del extractor.image_processor
    extractor.model = None
    extractor.tokenizer = None
    extractor.image_processor = None
    del extractor
    cleanup_gpu_memory()
    print("Model unloaded, GPU memory freed")
    
    for num_training in training_amounts:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {num_training} SAFEMTDATA TRAINING SAMPLES")
        print(f"{'='*60}")
        
        run_results = []
        
        for run_idx, seed in enumerate(random_seeds):
            print(f"\n--- Run {run_idx + 1}/10 (seed={seed}) ---")
            
            # Set global random seeds for this run to ensure reproducibility and variation
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            try:
                # Load SafeMTData with current configuration (only data, not features)
                safemt_training, safemt_test = load_safemtdata_for_variable_training_experiment(
                    tokenizer, model, model_name, num_training=num_training, num_test=100, random_seed=seed
                )
                
                # Extract features ONLY for SafeMTData (original features already extracted)
                all_hidden_states = original_hidden_states.copy()
                all_labels = original_labels.copy()
                
                # Create a new extractor for SafeMTData features (since we unloaded the previous one)
                safemt_extractor = None
                if safemt_training or safemt_test:
                    print("Creating new extractor for SafeMTData feature extraction...")
                    safemt_extractor = HiddenStateExtractor(model_name)
                
                # Extract features for SafeMTData training samples (if any)
                if safemt_training:
                    print(f"Extracting features for SafeMTData_Training ({len(safemt_training)} samples)...")
                    hidden_states, labels, _ = safemt_extractor.extract_hidden_states(
                        safemt_training, "SafeMTData_Training", layer_start=16, layer_end=16, use_cache=True,
                        batch_size=25, memory_cleanup_freq=5,
                        experiment_name="safemtdata_variable_training"
                    )
                    all_hidden_states["SafeMTData_Training"] = hidden_states
                    all_labels["SafeMTData_Training"] = labels
                    cleanup_gpu_memory()
                
                # Extract features for SafeMTData test samples
                if safemt_test:
                    print(f"Extracting features for SafeMTData_Test ({len(safemt_test)} samples)...")
                    hidden_states, labels, _ = safemt_extractor.extract_hidden_states(
                        safemt_test, "SafeMTData_Test", layer_start=16, layer_end=16, use_cache=True,
                        batch_size=25, memory_cleanup_freq=5,
                        experiment_name="safemtdata_variable_training"
                    )
                    all_hidden_states["SafeMTData_Test"] = hidden_states
                    all_labels["SafeMTData_Test"] = labels
                    cleanup_gpu_memory()
                
                # Clean up SafeMTData extractor
                if safemt_extractor is not None:
                    if hasattr(safemt_extractor, 'model') and safemt_extractor.model is not None:
                        del safemt_extractor.model
                    if hasattr(safemt_extractor, 'tokenizer') and safemt_extractor.tokenizer is not None:
                        del safemt_extractor.tokenizer
                    if hasattr(safemt_extractor, 'image_processor') and safemt_extractor.image_processor is not None:
                        del safemt_extractor.image_processor
                    del safemt_extractor
                    cleanup_gpu_memory()
                
                # Create dataset configurations
                expanded_malicious_training = malicious_training.copy()
                if safemt_training:
                    expanded_malicious_training["SafeMTData_Training"] = safemt_training
                
                test_datasets = original_test_datasets.copy()
                if safemt_test:
                    test_datasets["SafeMTData_Test"] = safemt_test
                
                training_datasets = {**benign_training, **expanded_malicious_training}
                
                print(f"Training datasets: {len(benign_training)} benign + {len(expanded_malicious_training)} malicious")
                print(f"Test datasets: {len(test_datasets)} total")
                
                # Separate training and test data
                training_hidden_states = {k: v for k, v in all_hidden_states.items() if k in training_datasets}
                training_labels = {k: v for k, v in all_labels.items() if k in training_datasets}
                
                test_hidden_states = {k: v for k, v in all_hidden_states.items() if k in test_datasets}
                test_labels = {k: v for k, v in all_labels.items() if k in test_datasets}
                
                # Apply learned projections (critical for performance!)
                print("Training learned projections...")
                
                # Prepare data for projection training (combine all training datasets)
                projection_features_dict = {}
                projection_labels_dict = {}
                
                for dataset_name in training_datasets.keys():
                    if dataset_name in training_hidden_states and dataset_name in training_labels:
                        features = training_hidden_states[dataset_name]
                        labels = training_labels[dataset_name]
                        
                        # Extract features from dict structure (features are likely {layer_16: array})
                        if isinstance(features, dict):
                            # Use layer 16 features (the layer we extracted)
                            if 16 in features:
                                features_array = features[16]
                            else:
                                # Fallback: use first available layer
                                features_array = list(features.values())[0]
                        else:
                            features_array = features
                        
                        # Ensure features are numpy arrays
                        if isinstance(features_array, list):
                            features_array = np.array(features_array)
                        
        
                        
                        projection_features_dict[dataset_name] = features_array
                        projection_labels_dict[dataset_name] = labels
                
                projection_model, dataset_name_to_id = train_learned_projection(
                    projection_features_dict, projection_labels_dict, random_seed=seed
                )
                
                print("Applying learned projections...")
                
                # Extract arrays from dict structure for apply_learned_projection
                training_features_for_projection = {}
                test_features_for_projection = {}
                
                for dataset_name, features in training_hidden_states.items():
                    if isinstance(features, dict) and 16 in features:
                        training_features_for_projection[dataset_name] = features[16]
                    else:
                        training_features_for_projection[dataset_name] = features
                        
                for dataset_name, features in test_hidden_states.items():
                    if isinstance(features, dict) and 16 in features:
                        test_features_for_projection[dataset_name] = features[16]
                    else:
                        test_features_for_projection[dataset_name] = features
                
                projected_training_hidden_states = apply_learned_projection(projection_model, training_features_for_projection)
                projected_test_hidden_states = apply_learned_projection(projection_model, test_features_for_projection)
                
                # Prepare k-means data structure
                print("Preparing k-means clustering...")
                in_dist_data, ood_data = prepare_kmeans_data_structure(
                    training_datasets, projected_training_hidden_states, training_labels,
                    random_seed=seed, k_benign=8, k_malicious=1
                )
                
                # Train MCD detector
                print("Training MCD detector...")
                detector = MCDDetector(use_gpu=True)
                detector.fit_in_distribution(in_dist_data)
                detector.fit_ood_clusters(ood_data)
                
                # Create validation data for threshold optimization
                print("Optimizing threshold...")
                val_features = []
                val_labels = []
                
                # Sample from clusters (same as original script)
                for cluster_data in in_dist_data.values():
                    if len(cluster_data) > 0:
                        sample_size = min(100, len(cluster_data))
                        indices = np.linspace(0, len(cluster_data)-1, sample_size, dtype=int)
                        sampled_features = [cluster_data[i] for i in indices]
                        val_features.extend(sampled_features)
                        val_labels.extend([0] * len(sampled_features))
                
                for cluster_data in ood_data.values():
                    if len(cluster_data) > 0:
                        sample_size = min(100, len(cluster_data))
                        indices = np.linspace(0, len(cluster_data)-1, sample_size, dtype=int)
                        sampled_features = [cluster_data[i] for i in indices]
                        val_features.extend(sampled_features)
                        val_labels.extend([1] * len(sampled_features))
                
                # Fit threshold
                if val_features and len(set(val_labels)) > 1:
                    detector.fit_threshold(val_features, val_labels)
                else:
                    detector.threshold = 0.0
                
                # Evaluate on all test datasets
                print("Evaluating performance...")
                
                # Combined evaluation
                all_test_features = []
                all_test_labels = []
                for dataset_name, features in projected_test_hidden_states.items():
                    labels = test_labels[dataset_name]
                    all_test_features.extend(features)
                    all_test_labels.extend(labels)
                
                combined_predictions, combined_scores = detector.predict(all_test_features)
                
                combined_accuracy = accuracy_score(all_test_labels, combined_predictions)
                combined_f1 = f1_score(all_test_labels, combined_predictions)
                combined_auroc = roc_auc_score(all_test_labels, combined_scores)
                
                # SafeMTData specific evaluation
                safemt_accuracy = 0.0
                if "SafeMTData_Test" in projected_test_hidden_states:
                    safemt_features = projected_test_hidden_states["SafeMTData_Test"]
                    safemt_labels = test_labels["SafeMTData_Test"]
                    safemt_predictions, safemt_scores = detector.predict(safemt_features)
                    safemt_accuracy = accuracy_score(safemt_labels, safemt_predictions)
                
                # Store results
                result = {
                    'num_training': num_training,
                    'run': run_idx + 1,
                    'seed': seed,
                    'combined_accuracy': combined_accuracy,
                    'combined_f1': combined_f1,
                    'combined_auroc': combined_auroc,
                    'safemt_accuracy': safemt_accuracy,
                    'threshold': detector.threshold
                }
                run_results.append(result)
                
                print(f"Results: Combined Acc={combined_accuracy:.4f}, SafeMT Acc={safemt_accuracy:.4f}, AUROC={combined_auroc:.4f}")
                
                # Clean up large variables and GPU memory after each run
                del projected_training_hidden_states, projected_test_hidden_states
                del training_hidden_states, test_hidden_states, training_labels, test_labels
                del detector, projection_model, in_dist_data, ood_data
                del all_test_features, all_test_labels, combined_scores, combined_predictions
                if 'safemt_features' in locals():
                    del safemt_features, safemt_scores, safemt_predictions
                cleanup_gpu_memory()
                
            except Exception as e:
                print(f"Error in run {run_idx + 1}: {e}")
                
                # Clean up memory even on error
                cleanup_gpu_memory()
                
                result = {
                    'num_training': num_training,
                    'run': run_idx + 1,
                    'seed': seed,
                    'combined_accuracy': 0.0,
                    'combined_f1': 0.0,
                    'combined_auroc': 0.0,
                    'safemt_accuracy': 0.0,
                    'threshold': 0.0,
                    'error': str(e)
                }
                run_results.append(result)
        
        all_results.extend(run_results)
        
        # Print summary for this training amount
        if run_results:
            avg_combined_acc = np.mean([r['combined_accuracy'] for r in run_results])
            avg_safemt_acc = np.mean([r['safemt_accuracy'] for r in run_results])
            avg_auroc = np.mean([r['combined_auroc'] for r in run_results])
            
            std_combined_acc = np.std([r['combined_accuracy'] for r in run_results])
            std_safemt_acc = np.std([r['safemt_accuracy'] for r in run_results])
            std_auroc = np.std([r['combined_auroc'] for r in run_results])
            
            print(f"\n*** SUMMARY FOR {num_training} TRAINING SAMPLES ***")
            print(f"Combined Accuracy: {avg_combined_acc:.4f} ± {std_combined_acc:.4f}")
            print(f"SafeMTData Accuracy: {avg_safemt_acc:.4f} ± {std_safemt_acc:.4f}")
            print(f"AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
        
        # Clean up after each training amount
        cleanup_gpu_memory()
    
    # Save all results
    results_file = "results/safemt_results.json"
    os.makedirs("results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    for num_training in training_amounts:
        training_results = [r for r in all_results if r['num_training'] == num_training]
        if training_results:
            avg_combined_acc = np.mean([r['combined_accuracy'] for r in training_results])
            avg_safemt_acc = np.mean([r['safemt_accuracy'] for r in training_results])
            print(f"{num_training:3d} training: Combined={avg_combined_acc:.3f}, SafeMT={avg_safemt_acc:.3f}")
    
    # Aggregate, output CSV, and plot
    aggregate_and_report_results(all_results, training_amounts, results_dir="results")
    # Final cleanup
    del original_hidden_states, original_labels
    cleanup_gpu_memory()
    
    return all_results

if __name__ == "__main__":
    # Set model path - HiddenStateExtractor will load the model when needed
    model_path = "model/llava-v1.6-vicuna-7b/"
    
    print("Preparing variable training experiment...")
    print("Note: LLaVA model will be loaded by HiddenStateExtractor when needed")
    
    # For SafeMTData processing, we need tokenizer and model
    # We'll load them lazily inside the SafeMTData functions to avoid memory issues
    
    # Run the variable training experiment
    run_variable_training_experiment(None, None, model_path)
