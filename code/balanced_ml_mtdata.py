#!/usr/bin/env python3
"""
Test the balanced ML method against SafeMTData multi-turn jailbreaking dataset.
This script integrates SafeMTData with the existing ML detection framework.
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
from balanced_ml import (
    prepare_balanced_datasets_organized, train_gpu_linear_classifier, 
    evaluate_linear_classifier, MLConfig, ThresholdOptimizer
)

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
                device='cuda'
            )
            print(f"Model loaded on device: cuda")
            model_loaded = True
            
            # Clean up after loading
            torch.cuda.empty_cache()
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
            torch.cuda.empty_cache()
            print("Cleaned up SafeMTData model from memory")

        return safemt_training, safemt_test

    except Exception as e:
        print(f"Error loading SafeMTData: {e}")
        # Clean up on error too (only if we loaded the model)
        if 'model_loaded' in locals() and model_loaded and 'model' in locals():
            del model, tokenizer, image_processor
            torch.cuda.empty_cache()
        return [], []

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
    csv_path = os.path.join(results_dir, "safemt_ml_aggregated.csv")
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
    
    # Add horizontal dashed lines for baselines (you can adjust these based on your baseline results)
    plt.axhline(0.8856, color='tab:blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline Accuracy')
    plt.axhline(0.982, color='tab:green', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline AUROC')
    plt.axhline(0.9185, color='tab:orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Highest Accuracy for MCD')
    
    plt.xlabel('Number of SafeMTData Training Samples')
    plt.ylabel('Score')
    plt.title('ML Detection Performance vs. SafeMTData Training Size')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "safemt_ml_trends.png")
    plt.savefig(fig_path)
    print(f"Trend plot saved to: {fig_path}")
    plt.close()

def run_variable_training_experiment(model_path, optimal_layer=16):
    """Run the variable training data experiment: 0 to 50 training samples, 5 runs each"""
    print("="*80)
    print("VARIABLE TRAINING DATA EXPERIMENT: SAFEMTDATA ADAPTABILITY ANALYSIS (ML)")
    print("="*80)
    
    # Experimental configuration
    training_amounts = list(range(0, 51, 5))
    random_seeds = [42, 123, 456, 789, 999]
    
    # Results storage
    all_results = []
    
    # Load original datasets (consistent across all experiments)
    print("\nLoading original datasets...")
    training_datasets, test_datasets = prepare_balanced_datasets_organized()
    
    # Initialize feature extractor ONCE for all experiments
    print("Initializing feature extractor...")
    extractor = HiddenStateExtractor(model_path)
    
    # Pre-extract features for all original datasets (they're consistent across runs)
    print("Pre-extracting features for original datasets...")
    original_all_datasets = {**training_datasets, **test_datasets}
    
    original_hidden_states = {}
    original_labels = {}
    
    for dataset_name, samples in original_all_datasets.items():
        print(f"Pre-extracting features for {dataset_name} ({len(samples)} samples)...")
        
        # Use smaller batch sizes for large datasets to manage memory
        batch_size = 25 if len(samples) > 5000 else 50
        memory_cleanup_freq = 5 if len(samples) > 5000 else 10
        
        hidden_states, labels, _ = extractor.extract_hidden_states(
            samples, f"{dataset_name}", layer_start=optimal_layer, layer_end=optimal_layer, use_cache=True,
            batch_size=batch_size, memory_cleanup_freq=memory_cleanup_freq,
            experiment_name="safemtdata_variable_training"
        )
        original_hidden_states[dataset_name] = hidden_states
        original_labels[dataset_name] = labels
        
        # Clean up GPU memory after each dataset
        torch.cuda.empty_cache()
    
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
    torch.cuda.empty_cache()
    print("Model unloaded, GPU memory freed")
    
    for num_training in training_amounts:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {num_training} SAFEMTDATA TRAINING SAMPLES")
        print(f"{'='*60}")
        
        run_results = []
        
        for run_idx, seed in enumerate(random_seeds):
            print(f"\n--- Run {run_idx + 1}/5 (seed={seed}) ---")
            
            # Set global random seeds for this run to ensure reproducibility and variation
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            try:
                # Load SafeMTData with current configuration (only data, not features)
                safemt_training, safemt_test = load_safemtdata_for_variable_training_experiment(
                    None, None, model_path, num_training=num_training, num_test=100, random_seed=seed
                )
                
                # Extract features ONLY for SafeMTData (original features already extracted)
                all_hidden_states = original_hidden_states.copy()
                all_labels = original_labels.copy()
                
                # Create a new extractor for SafeMTData features (since we unloaded the previous one)
                safemt_extractor = None
                if safemt_training or safemt_test:
                    print("Creating new extractor for SafeMTData feature extraction...")
                    safemt_extractor = HiddenStateExtractor(model_path)
                
                # Extract features for SafeMTData training samples (if any)
                if safemt_training:
                    print(f"Extracting features for SafeMTData_Training ({len(safemt_training)} samples)...")
                    hidden_states, labels, _ = safemt_extractor.extract_hidden_states(
                        safemt_training, "SafeMTData_Training", layer_start=optimal_layer, layer_end=optimal_layer, use_cache=True,
                        batch_size=25, memory_cleanup_freq=5,
                        experiment_name="safemtdata_variable_training"
                    )
                    all_hidden_states["SafeMTData_Training"] = hidden_states
                    all_labels["SafeMTData_Training"] = labels
                    torch.cuda.empty_cache()
                
                # Extract features for SafeMTData test samples
                if safemt_test:
                    print(f"Extracting features for SafeMTData_Test ({len(safemt_test)} samples)...")
                    hidden_states, labels, _ = safemt_extractor.extract_hidden_states(
                        safemt_test, "SafeMTData_Test", layer_start=optimal_layer, layer_end=optimal_layer, use_cache=True,
                        batch_size=25, memory_cleanup_freq=5,
                        experiment_name="safemtdata_variable_training"
                    )
                    all_hidden_states["SafeMTData_Test"] = hidden_states
                    all_labels["SafeMTData_Test"] = labels
                    torch.cuda.empty_cache()
                
                # Clean up SafeMTData extractor
                if safemt_extractor is not None:
                    if hasattr(safemt_extractor, 'model') and safemt_extractor.model is not None:
                        del safemt_extractor.model
                    if hasattr(safemt_extractor, 'tokenizer') and safemt_extractor.tokenizer is not None:
                        del safemt_extractor.tokenizer
                    if hasattr(safemt_extractor, 'image_processor') and safemt_extractor.image_processor is not None:
                        del safemt_extractor.image_processor
                    del safemt_extractor
                    torch.cuda.empty_cache()
                
                # Create dataset configurations
                expanded_malicious_training = {}
                # Add original malicious training datasets
                for dataset_name in ["AdvBench", "JailbreakV-28K", "DAN"]:
                    if dataset_name in training_datasets:
                        expanded_malicious_training[dataset_name] = training_datasets[dataset_name]
                
                if safemt_training:
                    expanded_malicious_training["SafeMTData_Training"] = safemt_training
                
                test_datasets_expanded = test_datasets.copy()
                if safemt_test:
                    test_datasets_expanded["SafeMTData_Test"] = safemt_test
                
                # Prepare training data
                train_features = []
                train_labels = []
                
                # Add benign training data
                for dataset_name in ["Alpaca", "MM-Vet", "OpenAssistant"]:
                    if dataset_name in all_hidden_states:
                        features = all_hidden_states[dataset_name][optimal_layer]
                        labels = all_labels[dataset_name]
                        train_features.extend(features)
                        train_labels.extend(labels)
                
                # Add malicious training data (including SafeMTData if present)
                for dataset_name in expanded_malicious_training.keys():
                    if dataset_name in all_hidden_states:
                        features = all_hidden_states[dataset_name][optimal_layer]
                        labels = all_labels[dataset_name]
                        train_features.extend(features)
                        train_labels.extend(labels)
                
                # Prepare test data
                test_features = []
                test_labels = []
                
                for dataset_name in test_datasets_expanded.keys():
                    if dataset_name in all_hidden_states:
                        features = all_hidden_states[dataset_name][optimal_layer]
                        labels = all_labels[dataset_name]
                        test_features.extend(features)
                        test_labels.extend(labels)
                
                # Convert to numpy arrays
                train_features = np.array(train_features)
                train_labels = np.array(train_labels)
                test_features = np.array(test_features)
                test_labels = np.array(test_labels)
                
                print(f"Training: {len(train_features)} samples ({np.sum(train_labels == 0)} benign, {np.sum(train_labels == 1)} malicious)")
                print(f"Testing: {len(test_features)} samples ({np.sum(test_labels == 0)} benign, {np.sum(test_labels == 1)} malicious)")
                
                # Train ML classifier
                print("Training logistic regression classifier...")
                model = train_gpu_linear_classifier(train_features, train_labels, model_type='logistic', epochs=100, lr=1e-3)
                
                # Optimize threshold
                print("Optimizing threshold...")
                threshold_optimizer = ThresholdOptimizer()
                optimal_threshold = threshold_optimizer.fit_threshold(model, train_features, train_labels, model_type='linear')
                
                # Evaluate on test set
                print("Evaluating on test set...")
                test_result = evaluate_linear_classifier(model, test_features, test_labels, threshold=optimal_threshold)
                
                combined_accuracy = test_result['accuracy']
                combined_f1 = test_result['f1']
                combined_auroc = test_result['auroc']
                
                # SafeMTData specific evaluation
                safemt_accuracy = 0.0
                if "SafeMTData_Test" in all_hidden_states:
                    safemt_features = np.array(all_hidden_states["SafeMTData_Test"][optimal_layer])
                    safemt_labels = np.array(all_labels["SafeMTData_Test"])
                    safemt_result = evaluate_linear_classifier(model, safemt_features, safemt_labels, threshold=optimal_threshold)
                    safemt_accuracy = safemt_result['accuracy']
                
                # Store results
                result = {
                    'num_training': num_training,
                    'run': run_idx + 1,
                    'seed': seed,
                    'combined_accuracy': combined_accuracy,
                    'combined_f1': combined_f1,
                    'combined_auroc': combined_auroc,
                    'safemt_accuracy': safemt_accuracy,
                    'threshold': optimal_threshold
                }
                run_results.append(result)
                
                print(f"Results: Combined Acc={combined_accuracy:.4f}, SafeMT Acc={safemt_accuracy:.4f}, AUROC={combined_auroc:.4f}")
                
                # Clean up large variables and GPU memory after each run
                del model, train_features, train_labels, test_features, test_labels
                if 'safemt_features' in locals():
                    del safemt_features, safemt_labels
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in run {run_idx + 1}: {e}")
                
                # Clean up memory even on error
                torch.cuda.empty_cache()
                
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
        torch.cuda.empty_cache()
    
    # Save all results
    results_file = "results/safemt_ml_results.json"
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
    torch.cuda.empty_cache()
    
    return all_results

if __name__ == "__main__":
    # Set model path - HiddenStateExtractor will load the model when needed
    model_path = "model/llava-v1.6-vicuna-7b/"
    
    print("Preparing variable training experiment for ML classifiers...")
    print("Note: LLaVA model will be loaded by HiddenStateExtractor when needed")
    
    # Set random seed for reproducibility
    SEED = 45
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Run the variable training experiment
    run_variable_training_experiment(model_path, optimal_layer=16)
