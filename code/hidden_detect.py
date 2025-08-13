import torch
import torch.nn as nn
import torch.nn.functional as N
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import re
import random
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from sklearn.metrics import roc_curve, f1_score, confusion_matrix
from tqdm import tqdm
import warnings
import gc
import psutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter in the checkpoint to a meta parameter.*")
warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*")


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
from load_datasets import *
from generic_classifier import FeatureCache
     
refusal_lst = [
        "Sorry", "sorry",
        "unfortunately", "unfortunate", "sadly",
        "explicit", "deadly", "crim", "criminal", "illegal", "dangerous", "harmful", "warning", "alarm", "caution",
        "shame", "conspiracy",
        "Subject", "contrary", "shouldn"
    ]
vocab_size = 32000

def _get_memory_info():
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

def _aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()

def test(dataset, model_path, s=16, e=31, use_cache=True, dataset_name="hidden_detect_dataset", experiment_name=None):
    """
    Test function with caching support for hidden state extraction

    Args:
        dataset: List of samples to process
        model_path: Path to the model
        s: Start layer index (default: 16)
        e: End layer index (default: 31)
        use_cache: Whether to use caching (default: True)
        dataset_name: Name for caching purposes (default: "hidden_detect_dataset")
        experiment_name: Optional experiment name for cache organization
    """
    # Initialize cache
    cache = FeatureCache("cache")
    layer_range = (s, e)
    dataset_size = len(dataset)

    # Check cache first
    if use_cache and cache.exists(dataset_name, model_path, layer_range, dataset_size, experiment_name):
        print(f"Loading cached features for {dataset_name} (size: {dataset_size}, layers: {s}-{e})...")
        cached_hidden_states, cached_labels, cached_metadata = cache.load(dataset_name, model_path, layer_range, dataset_size, experiment_name)

        # Convert cached hidden states to the format expected by the rest of the function
        # The cached format stores per-layer features, we need to convert to per-sample AUC scores
        aware_auc_all = []
        label_all = cached_labels

        # Calculate AUC for each sample from cached hidden states
        for sample_idx in range(len(cached_labels)):
            F = []
            for layer_idx in range(s, e+1):
                if layer_idx in cached_hidden_states and sample_idx < len(cached_hidden_states[layer_idx]):
                    F.append(cached_hidden_states[layer_idx][sample_idx])

            if F:
                aware_auc = np.trapz(np.array(F))
            else:
                aware_auc = None
            aware_auc_all.append(aware_auc)

        return label_all, aware_auc_all

    # If not cached, proceed with model loading and processing
    model_name = get_model_name_from_path(model_path)
    kwargs = {
    "device_map": "auto",
    "torch_dtype": torch.float16
    }
    tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    **kwargs
    )
                 
    def find_conv_mode(model_name):
        # select conversation mode based on the model name
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
        
    def adjust_query_for_images(qs):   
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        return qs

    def construct_conv_prompt(sample):        
        conv = conv_templates[find_conv_mode(model_name)].copy()  
        if (sample['img'] != None):     
            qs = adjust_query_for_images(sample['txt'])
        else:
            qs = sample['txt']
        conv.append_message(conv.roles[0], qs)  
        conv.append_message(conv.roles[1], None)       
        prompt = conv.get_prompt()
        return prompt

    def load_image(image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(image_files):
        out = []
        for image_file in image_files:
            image = load_image(image_file)
            out.append(image)
        return out
    
    def load_image_from_bytes(image_data):          
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def load_images_from_bytes(image_data_list):       
        return [load_image_from_bytes(data) for data in image_data_list]

    def prepare_imgs_tensor_both_cases(sample):
        try:
            # Case 1: Comma-separated file paths
            if isinstance(sample['img'], str):
                image_files_path = sample['img'].split(",")
                # Strip whitespace from paths
                image_files_path = [path.strip() for path in image_files_path]
                img_prompt = load_images(image_files_path)
            # Case 2: Single binary image
            elif isinstance(sample['img'], bytes):
                img_prompt = [load_image_from_bytes(sample['img'])]
            # Case 3: List of binary images
            elif isinstance(sample['img'], list):
                # Check if all elements in the list are bytes
                if all(isinstance(item, bytes) for item in sample['img']):
                    img_prompt = load_images_from_bytes(sample['img'])
                else:
                    raise ValueError("List contains non-bytes data.")
            else:
                raise ValueError("Unsupported data type in sample['img']. "
                                "Expected str, bytes, or list of bytes.")

            # Filter out None images (failed to load)
            img_prompt = [img for img in img_prompt if img is not None]

            if not img_prompt:
                print("Warning: No valid images loaded")
                return None, None

            # Compute sizes
            images_size = [img.size for img in img_prompt]

            # Process images into tensor
            images_tensor = process_images(img_prompt, image_processor, model.config)

            # Handle the case where process_images returns a list instead of tensor
            if isinstance(images_tensor, list):
                # Convert list of tensors to a single tensor
                images_tensor = torch.stack(images_tensor)

            images_tensor = images_tensor.to(model.device, dtype=torch.float16)
            return images_tensor, images_size
        except Exception as e:
            print(f"Error preparing image tensors: {e}")
            return None, None

    lm_head = model.lm_head    
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        norm = model.transformer.ln_f
    else:
        raise ValueError(f"Incorrect Model") 
    
    label_all = []
    aware_auc_all = []

    # Initialize storage for caching hidden states
    all_hidden_states = {i: [] for i in range(s, e+1)}

    refusal_token_ids = []
    for token in refusal_lst:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        refusal_token_ids.append(token_id)
    token_one_hot = torch.zeros(vocab_size)
    for token_id in refusal_token_ids:
        token_one_hot[token_id] = 1.0
    
    # Create progress bar for sample processing
    pbar = tqdm(dataset, desc="Processing samples", unit="sample")
    for sample_idx, sample in enumerate(pbar):
        try:
            # Periodic memory cleanup
            if sample_idx % 10 == 0:
                _aggressive_cleanup()
            F = []
            if sample['img'] == None:
                prompt = construct_conv_prompt(sample)
                input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                    )
                # Truncate if sequence is too long (max 4096 tokens for most models)
                max_length = getattr(model.config, 'max_position_embeddings', 4096)
                if input_ids.shape[1] > max_length:
                    print(f"Truncating sequence from {input_ids.shape[1]} to {max_length} tokens")
                    input_ids = input_ids[:, :max_length]
                input_ids = input_ids.cuda()

                with torch.no_grad():
                    outputs = model(input_ids, images=None, image_sizes=None, output_hidden_states=True)
            else:
                prompt = construct_conv_prompt(sample)
                input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                    )
                # Truncate if sequence is too long (max 4096 tokens for most models)
                max_length = getattr(model.config, 'max_position_embeddings', 4096)
                if input_ids.shape[1] > max_length:
                    print(f"Truncating sequence from {input_ids.shape[1]} to {max_length} tokens")
                    input_ids = input_ids[:, :max_length]
                input_ids = input_ids.cuda()
                images_tensor, images_size = prepare_imgs_tensor_both_cases(sample)
                if images_tensor is None:  # Skip if image loading failed
                    continue
                with torch.no_grad():
                    outputs = model(input_ids, images=images_tensor, image_sizes=images_size, output_hidden_states=True)
               
        
            for layer_idx, r in enumerate(outputs.hidden_states[1:]):
                layer_output = norm(r)
                logits = lm_head(layer_output)
                next_token_logits = logits[:, -1, :]
                reference_tokens = token_one_hot.to(next_token_logits.device)
                cos_sim = N.cosine_similarity(next_token_logits, reference_tokens)
                F.append(cos_sim.item())

                # Store hidden states for caching (only for layers in our range)
                actual_layer = layer_idx  # layer_idx starts from 0 for hidden_states[1:]
                if actual_layer >= s and actual_layer <= e:
                    all_hidden_states[actual_layer].append(cos_sim.item())

            F = F[s:e+1]
            if F:
                aware_auc = np.trapz(np.array(F))
            else:
                aware_auc = None

            label_all.append(sample['toxicity'])
            aware_auc_all.append(aware_auc)

            # Update progress bar with current stats
            processed = len(label_all)
            has_image = "📷" if sample.get('img') is not None else "📝"
            pbar.set_postfix({
                'processed': f"{processed}/{len(dataset)}",
                'type': has_image,
                'toxicity': sample['toxicity']
            })

        except Exception as e:
            print(f"Error processing sample: {e}")

    # Cache the results if caching is enabled
    if use_cache:
        metadata = {
            'dataset_size': len(dataset),
            'label_key': 'toxicity',
            'processed_samples': len(label_all),
            'layer_start': s,
            'layer_end': e
        }
        cache.save(dataset_name, model_path, layer_range, all_hidden_states, label_all,
                  metadata, dataset_size, experiment_name)

    # Final cleanup
    _aggressive_cleanup()
    final_mem = _get_memory_info()
    print(f"Feature extraction completed. Final memory usage:")
    print(f"  GPU: {final_mem['gpu_allocated']:.1f}GB allocated, {final_mem['gpu_cached']:.1f}GB cached")
    print(f"  CPU: {final_mem['cpu_used']:.1f}GB used")

    return label_all, aware_auc_all


def evaluate_AUPRC(true_labels, scores):
    # Filter out None values
    valid_pairs = [(label, score) for label, score in zip(true_labels, scores) if score is not None]
    if len(valid_pairs) == 0:
        print("Warning: No valid samples for AUPRC calculation")
        return 0.0

    valid_labels, valid_scores = zip(*valid_pairs)
    if len(set(valid_labels)) < 2:
        print("Warning: Only one class present in labels")
        return 0.0

    precision_arr, recall_arr, threshold_arr = precision_recall_curve(valid_labels, valid_scores)
    auprc = auc(recall_arr, precision_arr)
    return auprc

def evaluate_AUROC(true_labels, scores):
    # Filter out None values
    valid_pairs = [(label, score) for label, score in zip(true_labels, scores) if score is not None]
    if len(valid_pairs) == 0:
        print("Warning: No valid samples for AUROC calculation")
        return 0.0

    valid_labels, valid_scores = zip(*valid_pairs)
    if len(set(valid_labels)) < 2:
        print("Warning: Only one class present in labels")
        return 0.0

    fpr, tpr, thresholds = roc_curve(valid_labels, valid_scores)
    auroc = auc(fpr, tpr)
    return auroc

def find_optimal_threshold(train_labels, train_scores):
    """Find optimal threshold using training set"""
    # Filter out None values
    valid_pairs = [(label, score) for label, score in zip(train_labels, train_scores) if score is not None]
    if len(valid_pairs) == 0:
        print("Warning: No valid samples for threshold optimization")
        return 0.0

    valid_labels, valid_scores = zip(*valid_pairs)
    if len(set(valid_labels)) < 2:
        print("Warning: Only one class present in training labels")
        return 0.0

    # Use precision-recall curve to find optimal threshold
    precision, recall, thresholds = precision_recall_curve(valid_labels, valid_scores)

    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)

    if optimal_idx < len(thresholds):
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.0

    return optimal_threshold

def evaluate_with_threshold(true_labels, scores, threshold):
    """Evaluate accuracy using the given threshold"""
    # Filter out None values
    valid_pairs = [(label, score) for label, score in zip(true_labels, scores) if score is not None]
    if len(valid_pairs) == 0:
        print("Warning: No valid samples for accuracy calculation")
        return 0.0

    valid_labels, valid_scores = zip(*valid_pairs)
    predictions = [1 if score >= threshold else 0 for score in valid_scores]
    accuracy = accuracy_score(valid_labels, predictions)
    return accuracy

def evaluate_F1(true_labels, scores, threshold):
    """Evaluate F1 score using the given threshold"""
    # Filter out None values
    valid_pairs = [(label, score) for label, score in zip(true_labels, scores) if score is not None]
    if len(valid_pairs) == 0:
        print("Warning: No valid samples for F1 calculation")
        return 0.0

    valid_labels, valid_scores = zip(*valid_pairs)
    predictions = [1 if score >= threshold else 0 for score in valid_scores]
    f1 = f1_score(valid_labels, predictions)
    return f1

def evaluate_FPR_TPR(true_labels, scores, threshold):
    """Evaluate False Positive Rate (FPR) and True Positive Rate (TPR) using the given threshold"""
    # Filter out None values
    valid_pairs = [(label, score) for label, score in zip(true_labels, scores) if score is not None]
    if len(valid_pairs) == 0:
        print("Warning: No valid samples for FPR/TPR calculation")
        return 0.0, 0.0

    valid_labels, valid_scores = zip(*valid_pairs)
    predictions = [1 if score >= threshold else 0 for score in valid_scores]

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(valid_labels, predictions).ravel()

    # Calculate FPR and TPR
    # FPR = FP / (FP + TN) = False Positive Rate
    # TPR = TP / (TP + FN) = True Positive Rate (Sensitivity/Recall)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return fpr, tpr

def create_balanced_datasets():
    """Create balanced training and test datasets using the same configuration as balanced_jailbreak_detection.py"""
    print("Creating balanced training and test datasets...")

    # Set random seed for reproducibility
    random.seed(42)

    # Create training set
    print("\n--- Creating Training Set ---")
    train_benign = []
    train_malicious = []

    # Benign samples for training
    try:
        alpaca_samples = load_alpaca(max_samples=500)
        train_benign.extend(alpaca_samples)
        print(f"Added {len(alpaca_samples)} Alpaca samples")
    except Exception as e:
        print(f"Could not load Alpaca: {e}")

    try:
        mmvet_samples = load_mm_vet()
        mmvet_subset = mmvet_samples[:218] if len(mmvet_samples) > 218 else mmvet_samples
        train_benign.extend(mmvet_subset)
        print(f"Added {len(mmvet_subset)} MM-Vet samples")
    except Exception as e:
        print(f"Could not load MM-Vet: {e}")

    try:
        openassistant_samples = load_openassistant(max_samples=282)
        train_benign.extend(openassistant_samples)
        print(f"Added {len(openassistant_samples)} OpenAssistant samples")
    except Exception as e:
        print(f"Could not load OpenAssistant: {e}")

    # Malicious samples for training
    try:
        advbench_samples = load_advbench(max_samples=300)
        train_malicious.extend(advbench_samples)
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

        train_malicious.extend(jbv_samples)
        print(f"Added {len(jbv_samples)} JailbreakV-28K samples ({len(llm_attack_samples)} llm_transfer + {len(query_related_samples)} query_related for training)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K: {e}")

    try:
        dan_samples = load_dan_prompts(max_samples=150)
        train_malicious.extend(dan_samples)
        print(f"Added {len(dan_samples)} DAN variant samples")
    except Exception as e:
        print(f"Could not load DAN prompts: {e}")

    # Balance training set
    target_benign = 1000
    target_malicious = 1000

    if len(train_benign) > target_benign:
        train_benign = random.sample(train_benign, target_benign)
    elif len(train_benign) < target_benign:
        print(f"Warning: Only {len(train_benign)} benign samples available, target was {target_benign}")

    if len(train_malicious) > target_malicious:
        train_malicious = random.sample(train_malicious, target_malicious)
    elif len(train_malicious) < target_malicious:
        print(f"Warning: Only {len(train_malicious)} malicious samples available, target was {target_malicious}")

    print(f"Training set: {len(train_benign)} benign, {len(train_malicious)} malicious")

    # Create test set
    print("\n--- Creating Test Set ---")
    test_safe = []
    test_unsafe = []

    # Safe samples for testing
    try:
        xstest_samples = load_XSTest()
        xstest_safe = [s for s in xstest_samples if s['toxicity'] == 0]
        xstest_safe_subset = random.sample(xstest_safe, min(250, len(xstest_safe)))
        test_safe.extend(xstest_safe_subset)
        print(f"Added {len(xstest_safe_subset)} XSTest safe samples")
    except Exception as e:
        print(f"Could not load XSTest safe: {e}")

    try:
        figtxt_samples = load_FigTxt()
        figtxt_safe = [s for s in figtxt_samples if s['toxicity'] == 0]
        figtxt_safe_subset = random.sample(figtxt_safe, min(300, len(figtxt_safe)))
        test_safe.extend(figtxt_safe_subset)
        print(f"Added {len(figtxt_safe_subset)} FigTxt safe samples")
    except Exception as e:
        print(f"Could not load FigTxt safe: {e}")

    try:
        vqav2_samples = load_vqav2(max_samples=450)
        test_safe.extend(vqav2_samples)
        print(f"Added {len(vqav2_samples)} VQAv2 samples")
    except Exception as e:
        print(f"Could not load VQAv2: {e}")

    # Unsafe samples for testing
    try:
        xstest_samples = load_XSTest()
        xstest_unsafe = [s for s in xstest_samples if s['toxicity'] == 1]
        xstest_unsafe_subset = random.sample(xstest_unsafe, min(200, len(xstest_unsafe)))
        test_unsafe.extend(xstest_unsafe_subset)
        print(f"Added {len(xstest_unsafe_subset)} XSTest unsafe samples")
    except Exception as e:
        print(f"Could not load XSTest unsafe: {e}")

    try:
        figtxt_samples = load_FigTxt()
        figtxt_unsafe = [s for s in figtxt_samples if s['toxicity'] == 1]
        figtxt_unsafe_subset = random.sample(figtxt_unsafe, min(350, len(figtxt_unsafe)))
        test_unsafe.extend(figtxt_unsafe_subset)
        print(f"Added {len(figtxt_unsafe_subset)} FigTxt unsafe samples")
    except Exception as e:
        print(f"Could not load FigTxt unsafe: {e}")

    try:
        vae_samples = load_adversarial_img()
        if len(vae_samples) >= 200:
            vae_subset = random.sample(vae_samples, 200)
        test_unsafe.extend(vae_subset)
        print(f"Added {len(vae_subset)} VAE samples")
    except Exception as e:
        print(f"Could not load VAE: {e}")

    try:
        # JailbreakV-28K - 150 samples (figstep attack for testing)
        jbv_test_samples = load_JailBreakV_figstep(max_samples=150)
        test_unsafe.extend(jbv_test_samples)
        print(f"Added {len(jbv_test_samples)} JailbreakV-28K samples (figstep attack for testing)")
    except Exception as e:
        print(f"Could not load JailbreakV-28K for testing: {e}")

    # Balance test set
    target_safe = 900
    target_unsafe = 900

    if len(test_safe) > target_safe:
        test_safe = random.sample(test_safe, target_safe)
    elif len(test_safe) < target_safe:
        print(f"Warning: Only {len(test_safe)} safe samples available, target was {target_safe}")

    if len(test_unsafe) > target_unsafe:
        test_unsafe = random.sample(test_unsafe, target_unsafe)
    elif len(test_unsafe) < target_unsafe:
        print(f"Warning: Only {len(test_unsafe)} unsafe samples available, target was {target_unsafe}")

    print(f"Test set: {len(test_safe)} safe, {len(test_unsafe)} unsafe")

    # Combine and shuffle datasets
    train_dataset = train_benign + train_malicious
    test_dataset = test_safe + test_unsafe

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    print(f"\nFinal dataset sizes:")
    print(f"Training: {len(train_dataset)} samples ({len(train_benign)} benign + {len(train_malicious)} malicious)")
    print(f"Test: {len(test_dataset)} samples ({len(test_safe)} safe + {len(test_unsafe)} unsafe)")

    return train_dataset, test_dataset

if __name__ == "__main__":
    model_path = "model/llava-v1.6-vicuna-7b/"

    print("="*80)
    print("UNSUPERVISED JAILBREAK DETECTION WITH BALANCED DATASETS")
    print("="*80)

    # Create balanced datasets
    train_dataset, test_dataset = create_balanced_datasets()

    print("\n--- Processing Training Set for Threshold Optimization ---")
    train_labels, train_scores = test(train_dataset, model_path, s=16, e=29,
                                    use_cache=True, dataset_name="hidden_detect_train",
                                    experiment_name="balanced_threshold_optimization")

    print("\n--- Processing Test Set ---")
    test_labels, test_scores = test(test_dataset, model_path, s=16, e=29,
                                  use_cache=True, dataset_name="hidden_detect_test",
                                  experiment_name="balanced_evaluation")

    # Find optimal threshold using training set
    print("\n--- Finding Optimal Threshold ---")
    optimal_threshold = find_optimal_threshold(train_labels, train_scores)
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Evaluate on test set
    print("\n--- Evaluating on Test Set ---")
    test_auprc = evaluate_AUPRC(test_labels, test_scores)
    test_auroc = evaluate_AUROC(test_labels, test_scores)
    test_accuracy = evaluate_with_threshold(test_labels, test_scores, optimal_threshold)
    test_f1 = evaluate_F1(test_labels, test_scores, optimal_threshold)
    test_fpr, test_tpr = evaluate_FPR_TPR(test_labels, test_scores, optimal_threshold)

    print(f"Test AUPRC (threshold-free): {test_auprc:.4f}")
    print(f"Test AUROC (threshold-free): {test_auroc:.4f}")
    print(f"Test Accuracy (with optimal threshold {optimal_threshold:.4f}): {test_accuracy:.4f}")
    print(f"Test F1 Score (with optimal threshold {optimal_threshold:.4f}): {test_f1:.4f}")
    print(f"Test FPR (with optimal threshold {optimal_threshold:.4f}): {test_fpr:.4f}")
    print(f"Test TPR (with optimal threshold {optimal_threshold:.4f}): {test_tpr:.4f}")

    # Save results
    output_path = "results/balanced_hidden_detect_results.csv"
    try:
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Method", "Dataset", "AUPRC", "AUROC", "Accuracy", "F1", "FPR", "TPR", "Threshold", "Train_Size", "Test_Size"])
            writer.writerow([
                "Unsupervised_Cosine_Similarity",
                "Balanced_Dataset",
                f"{test_auprc:.4f}",
                f"{test_auroc:.4f}",
                f"{test_accuracy:.4f}",
                f"{test_f1:.4f}",
                f"{test_fpr:.4f}",
                f"{test_tpr:.4f}",
                f"{optimal_threshold:.4f}",
                len(train_dataset),
                len(test_dataset)
            ])
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")

    print("\n" + "="*80)
    print("UNSUPERVISED (HiddenDetect) DETECTION SUMMARY")
    print("="*80)
    print(f"Method: Cosine similarity with refusal tokens")
    print(f"Training set: {len(train_dataset)} samples (for threshold optimization)")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test FPR: {test_fpr:.4f}")
    print(f"Test TPR: {test_tpr:.4f}")
    print("="*80)
