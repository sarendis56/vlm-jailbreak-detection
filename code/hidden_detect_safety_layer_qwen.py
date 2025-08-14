import torch
import torch.nn.functional as N
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import json
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from load_datasets import *

# Refusal tokens for Qwen (including Chinese)
refusal_lst = [
    # English refusal patterns that tokenize correctly for Qwen
    "Sorry", "sorry", "apologize", "apologies",
    "cannot", "can't", "unable", "refuse",
    "illegal", "criminal", "dangerous", "explicit", "harmful", "warning",
    "unfortunately", "sadly", "regret",
    # Qwen-specific refusal patterns (Chinese)
    "抱歉", "对不起", "很遗憾", "无法", "不能", "拒绝",
    # Common refusal phrases
    "I cannot", "I can't", "I'm unable", "I'm sorry",
    "not allowed", "not permitted", "against policy"
]

def load_image_from_bytes(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def prepare_qwen25vl_messages(sample):
    """Prepare messages for Qwen2.5-VL based on sample format"""
    content = []

    # Handle different image formats
    if sample.get("img") is not None:
        img_data = sample["img"]

        # Case 1: File path(s) as string
        if isinstance(img_data, str):
            if "," in img_data:
                # Multiple comma-separated paths
                image_files = [path.strip() for path in img_data.split(",")]
                for img_path in image_files:
                    # Convert to file:// format if it's a local path
                    if not img_path.startswith(("http://", "https://", "file://")):
                        img_path = f"file://{os.path.abspath(img_path)}"
                    content.append({"type": "image", "image": img_path})
            else:
                # Single path
                if not img_data.startswith(("http://", "https://", "file://")):
                    img_data = f"file://{os.path.abspath(img_data)}"
                content.append({"type": "image", "image": img_data})

        # Case 2: Binary image data - save to temp file
        elif isinstance(img_data, bytes):
            import tempfile

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(img_data)
                tmp_path = tmp_file.name

            content.append({"type": "image", "image": f"file://{tmp_path}"})

        # Case 3: List of images (paths or bytes)
        elif isinstance(img_data, list):
            for item in img_data:
                if isinstance(item, str):
                    if not item.startswith(("http://", "https://", "file://")):
                        item = f"file://{os.path.abspath(item)}"
                    content.append({"type": "image", "image": item})
                elif isinstance(item, bytes):
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        tmp_file.write(item)
                        tmp_path = tmp_file.name
                    content.append({"type": "image", "image": f"file://{tmp_path}"})

    # Add text
    content.append({"type": "text", "text": sample["txt"]})

    # Return messages in the format expected by Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    return messages

def create_few_shot_datasets():
    """Load few-shot safe and unsafe datasets from the provided few_shot.json"""
    print("Loading few-shot datasets for safety layer detection...")

    few_shot_safe = []
    few_shot_unsafe = []
    json_file_path = "few_shot/few_shot.json"

    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found. Please extract few_shot.zip first.")
        return [], []

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        if sample.get("img") == "null":
            sample["img"] = None
        elif sample.get("img") is not None:
            # Fix image paths - replace "data/few_shot/" with "few_shot/"
            if isinstance(sample["img"], str):
                if sample["img"].startswith("data/few_shot/"):
                    sample["img"] = sample["img"].replace("data/few_shot/", "few_shot/")
                elif not sample["img"].startswith(("http://", "https://", "/")):
                    sample["img"] = f"few_shot/{sample['img']}"

        if sample.get("toxicity", 0) == 1:
            few_shot_unsafe.append(sample)
        else:
            few_shot_safe.append(sample)

    print(f"Loaded few-shot dataset: {len(few_shot_safe)} safe, {len(few_shot_unsafe)} unsafe samples")
    return few_shot_safe, few_shot_unsafe

def locate_most_safety_aware_layers(model_path):
    """Locate the most safety-aware layers using Few-shot Discrepancy Vector (FDV) method"""
    print(f"Loading Qwen2.5-VL model from {model_path}")
    
    # Try float32 first to avoid numerical instability
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda:1",
            torch_dtype=torch.float32
        )
        print("Loaded model with float32 precision")
    except Exception as e:
        print(f"Float32 failed ({e}), falling back to float16")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda:1",
            torch_dtype=torch.float16
        )
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_path)
    
    # Get tokenizer from processor for refusal token encoding
    tokenizer = processor.tokenizer
    
    # Encode refusal tokens
    refusal_token_ids = []
    for token in refusal_lst:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:  # Make sure we got valid token IDs
            refusal_token_ids.extend(token_ids)
    
    # Remove duplicates while preserving order
    seen = set()
    refusal_token_ids = [x for x in refusal_token_ids if not (x in seen or seen.add(x))]
    
    print(f"Using {len(refusal_token_ids)} refusal tokens for Qwen")
    print(f"Sample refusal token IDs: {refusal_token_ids[:10]}")  # Show first 10
    
    # Use the actual model vocab size for the one-hot tensor
    actual_vocab_size = model.lm_head.out_features
    token_one_hot = torch.zeros(actual_vocab_size)
    valid_tokens = 0
    for token_id in refusal_token_ids:
        if token_id < actual_vocab_size:  # Make sure token ID is within bounds
            token_one_hot[token_id] = 1.0
            valid_tokens += 1

    print(f"Vocab size: {actual_vocab_size}, Valid refusal tokens: {valid_tokens}")
    print(f"Refusal token one-hot sum: {token_one_hot.sum().item()}")

    if valid_tokens == 0:
        print("Error: No valid refusal tokens found!")
        return None, None
    
    # Get model components for hidden state processing
    lm_head = model.lm_head
    
    # Use the exact same normalization approach as the working hidden_detect_qwen.py
    norm = None

    # Check for Qwen2.5-VL specific structure (exactly like working version)
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "norm"):
        norm = model.model.language_model.norm
        print("Found norm at model.model.language_model.norm")
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
        print("Found norm at model.model.norm")
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        norm = model.transformer.ln_f
        print("Found norm at model.transformer.ln_f")
    elif hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        # For some models, we might need to use the last layer's norm
        if hasattr(model.model.layers[-1], "post_attention_layernorm"):
            norm = model.model.layers[-1].post_attention_layernorm
            print("Found norm at model.model.layers[-1].post_attention_layernorm")
        elif hasattr(model.model.layers[-1], "ln_2"):
            norm = model.model.layers[-1].ln_2
            print("Found norm at model.model.layers[-1].ln_2")

    if norm is None:
        # Create a dummy normalization that just returns the input (exactly like working version)
        class DummyNorm:
            def __call__(self, x):
                return x
            def forward(self, x):
                return x
        norm = DummyNorm()
        print("Using dummy normalization")
    
    # Create few-shot datasets
    few_shot_safe, few_shot_unsafe = create_few_shot_datasets()
    
    if len(few_shot_safe) == 0 or len(few_shot_unsafe) == 0:
        print("Error: Could not create few-shot datasets")
        return None, None
    
    print("Processing safe samples...")
    F_safe = []
    
    for sample in few_shot_safe:
        F = []
        
        # Prepare messages for Qwen2.5-VL
        messages = prepare_qwen25vl_messages(sample)
        
        # Process the messages using the processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)[:2]
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Truncate if sequence is too long
        max_length = getattr(model.config, 'max_position_embeddings', 32768)
        if inputs.input_ids.shape[1] > max_length:
            inputs.input_ids = inputs.input_ids[:, :max_length]
            if hasattr(inputs, 'attention_mask'):
                inputs.attention_mask = inputs.attention_mask[:, :max_length]
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Process ALL layers (exactly like working version)
            for layer_idx, r in enumerate(outputs.hidden_states[1:]):
                # Apply normalization (exactly like working version)
                if hasattr(norm, '__call__') and not isinstance(norm, type(lambda: None)):
                    layer_output = norm(r)
                elif hasattr(norm, 'forward'):
                    layer_output = norm.forward(r)
                else:
                    layer_output = r

                # Compute logits for all tokens (exactly like working version)
                logits = lm_head(layer_output)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                # Create reference tensor (exactly like working Qwen version)
                reference_tokens = torch.zeros_like(next_token_logits)
                for token_id in refusal_token_ids:
                    if token_id < reference_tokens.shape[-1]:
                        reference_tokens[:, token_id] = 1.0

                # Compute cosine similarity exactly like working version (no dim parameter!)
                cos_sim = N.cosine_similarity(next_token_logits, reference_tokens)
                F.append(cos_sim.item())
        
        F_safe.append(F)
    
    # Average over the few-shot safe dataset
    if not F_safe:
        print("Error: No safe samples processed successfully!")
        return None, None

    F_safe_arr = np.mean(F_safe, axis=0)
    F_safe = F_safe_arr.tolist()
    print(f"Safe samples processed: {len(F_safe)} layers, sample values: {F_safe[:5]}")
    
    print("Processing unsafe samples...")
    F_unsafe = []
    
    for sample in few_shot_unsafe:
        F = []
        
        # Prepare messages for Qwen2.5-VL
        messages = prepare_qwen25vl_messages(sample)
        
        # Process the messages using the processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)[:2]
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Truncate if sequence is too long
        max_length = getattr(model.config, 'max_position_embeddings', 32768)
        if inputs.input_ids.shape[1] > max_length:
            inputs.input_ids = inputs.input_ids[:, :max_length]
            if hasattr(inputs, 'attention_mask'):
                inputs.attention_mask = inputs.attention_mask[:, :max_length]
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Process ALL layers (exactly like working version)
            for layer_idx, r in enumerate(outputs.hidden_states[1:]):
                # Apply normalization (exactly like working version)
                if hasattr(norm, '__call__') and not isinstance(norm, type(lambda: None)):
                    layer_output = norm(r)
                elif hasattr(norm, 'forward'):
                    layer_output = norm.forward(r)
                else:
                    layer_output = r

                # Check for NaN in layer output before computing logits
                if torch.isnan(layer_output).any():
                    print(f"NaN detected in layer_output at layer {len(F)}, skipping...")
                    F.append(0.0)
                    continue

                # Compute logits for all tokens (exactly like working version)
                logits = lm_head(layer_output)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                # Check for NaN in logits
                if torch.isnan(next_token_logits).any():
                    print(f"NaN detected in logits at layer {len(F)}, skipping...")
                    F.append(0.0)
                    continue

                # Create reference tensor (exactly like working Qwen version)
                reference_tokens = torch.zeros_like(next_token_logits)
                for token_id in refusal_token_ids:
                    if token_id < reference_tokens.shape[-1]:
                        reference_tokens[:, token_id] = 1.0

                # Compute cosine similarity exactly like working version (no dim parameter!)
                cos_sim = N.cosine_similarity(next_token_logits, reference_tokens)
                F.append(cos_sim.item())
        
        F_unsafe.append(F)
    
    # Average over the few-shot unsafe dataset
    if not F_unsafe:
        print("Error: No unsafe samples processed successfully!")
        return None, None

    F_unsafe_arr = np.mean(F_unsafe, axis=0)
    F_unsafe = F_unsafe_arr.tolist()
    print(f"Unsafe samples processed: {len(F_unsafe)} layers, sample values: {F_unsafe[:5]}")
    
    # Refusal discrepancy vector
    FDV_arr = F_unsafe_arr - F_safe_arr
    FDV = FDV_arr.tolist()
    
    # Safety awareness exists broadly
    positive_layers = [str(i) for i, item in enumerate(FDV) if item > 0]
    print("Safety awareness broadly exists at layer: " + ", ".join(positive_layers) + ".")
    
    # Locate the most safety-aware layers by using the last layer as baseline
    most_aware_layers = [str(i) for i, item in enumerate(FDV) if item > FDV[-1]]
    print("The most safety-aware layers are: " + ", ".join(most_aware_layers) + ".")
    
    return FDV, most_aware_layers

if __name__ == "__main__":
    model_path = "./model/Qwen2.5-VL-3B-Instruct"
    
    print("="*80)
    print("QWEN2.5-VL SAFETY LAYER DETECTION")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please download the model first.")
        exit(1)
    
    result = locate_most_safety_aware_layers(model_path)

    if result is not None:
        FDV, most_aware_layers = result
        print(f"\nFull FDV (Refusal Discrepancy Vector): {FDV}")
        print(f"Most safety-aware layers: {most_aware_layers}")
        
        # Save results
        results = {
            "FDV": FDV,
            "most_aware_layers": most_aware_layers,
            "model_path": model_path
        }
        
        output_file = "results/qwen25vl_safety_layers.json"
        os.makedirs("results", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    else:
        print("Failed to compute safety layers")
