#!/usr/bin/env python3
"""
Model Download Script for VLM Jailbreak Detection Project

This script downloads all required models for the experiments.
It downloads models from HuggingFace Hub and organizes them
in the expected directory structure.

Required models:
- LLaVA-v1.6-Vicuna-7B (multimodal language model)
- FLAVA (Facebook's multimodal model)
- Additional models as needed
"""

import os
import shutil
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModel
import torch

def create_model_directory():
    """Create the model directory structure"""
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Created model directory: {model_dir}")
    return model_dir

def download_llava_v16_vicuna_7b(model_dir):
    """Download LLaVA-v1.6-Vicuna-7B model"""
    print("\n" + "="*60)
    print("DOWNLOADING LLAVA-V1.6-VICUNA-7B MODEL")
    print("="*60)
    
    model_name = "liuhaotian/llava-v1.6-vicuna-7b"
    local_dir = os.path.join(model_dir, "llava-v1.6-vicuna-7b")
    
    try:
        print(f"Downloading {model_name} to {local_dir}...")
        print("This may take a while as the model is large (~13GB)")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ Successfully downloaded LLaVA-v1.6-Vicuna-7B to {local_dir}")
        
        # Verify the download by checking key files
        key_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        missing_files = []
        
        for file in key_files:
            if not os.path.exists(os.path.join(local_dir, file)):
                # Check for alternative file names
                if file == "pytorch_model.bin":
                    # Check for safetensors or sharded models
                    safetensors_files = [f for f in os.listdir(local_dir) if f.endswith('.safetensors')]
                    bin_files = [f for f in os.listdir(local_dir) if f.startswith('pytorch_model') and f.endswith('.bin')]
                    if not safetensors_files and not bin_files:
                        missing_files.append(file)
                else:
                    missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Warning: Some expected files are missing: {missing_files}")
        else:
            print("✅ All key model files are present")
            
    except Exception as e:
        print(f"❌ Error downloading LLaVA-v1.6-Vicuna-7B: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Insufficient disk space")
        print("3. HuggingFace Hub access issues")

def download_flava_model(model_dir):
    """Download FLAVA model"""
    print("\n" + "="*60)
    print("DOWNLOADING FLAVA MODEL")
    print("="*60)
    
    model_name = "facebook/flava-full"
    local_dir = os.path.join(model_dir, "flava")
    
    try:
        print(f"Downloading {model_name} to {local_dir}...")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ Successfully downloaded FLAVA to {local_dir}")
        
        # Verify the download
        key_files = ["config.json", "pytorch_model.bin"]
        missing_files = []
        
        for file in key_files:
            if not os.path.exists(os.path.join(local_dir, file)):
                if file == "pytorch_model.bin":
                    # Check for safetensors
                    safetensors_files = [f for f in os.listdir(local_dir) if f.endswith('.safetensors')]
                    if not safetensors_files:
                        missing_files.append(file)
                else:
                    missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Warning: Some expected files are missing: {missing_files}")
        else:
            print("✅ All key model files are present")
            
    except Exception as e:
        print(f"❌ Error downloading FLAVA: {e}")

def download_clip_model(model_dir):
    """Download CLIP model (often used as a baseline)"""
    print("\n" + "="*60)
    print("DOWNLOADING CLIP MODEL")
    print("="*60)
    
    model_name = "openai/clip-vit-base-patch32"
    local_dir = os.path.join(model_dir, "clip-vit-base-patch32")
    
    try:
        print(f"Downloading {model_name} to {local_dir}...")
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ Successfully downloaded CLIP to {local_dir}")
        
    except Exception as e:
        print(f"❌ Error downloading CLIP: {e}")

def check_disk_space():
    """Check available disk space"""
    print("\n" + "="*60)
    print("CHECKING DISK SPACE")
    print("="*60)
    
    try:
        # Get disk usage statistics
        statvfs = os.statvfs('.')
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        
        print(f"Available disk space: {free_space_gb:.2f} GB")
        
        # Estimate required space
        required_space_gb = 20  # Rough estimate for all models
        
        if free_space_gb < required_space_gb:
            print(f"⚠️  Warning: You may need at least {required_space_gb} GB of free space")
            print("Consider freeing up disk space before proceeding")
            return False
        else:
            print(f"✅ Sufficient disk space available")
            return True
            
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return True  # Proceed anyway

def check_gpu_availability():
    """Check GPU availability"""
    print("\n" + "="*60)
    print("CHECKING GPU AVAILABILITY")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA is available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️  CUDA is not available. Models will run on CPU (much slower)")

def print_usage_instructions():
    """Print instructions for using the downloaded models"""
    print("\n" + "="*80)
    print("MODEL USAGE INSTRUCTIONS")
    print("="*80)
    
    print("\nThe models have been downloaded to the 'model/' directory:")
    print("├── model/")
    print("│   ├── llava-v1.6-vicuna-7b/     # Main multimodal model")
    print("│   ├── flava/                    # FLAVA multimodal model")
    print("│   └── clip-vit-base-patch32/    # CLIP baseline model")
    
    print("\nTo use the models in your code:")
    print("1. Update model paths in your scripts to point to these directories")
    print("2. The balanced_ml.py script expects LLaVA at 'model/llava-v1.6-vicuna-7b/'")
    print("3. The visualize_embeddings_pca.py script uses both LLaVA and FLAVA")
    
    print("\nModel loading examples:")
    print("# For LLaVA")
    print("model_path = 'model/llava-v1.6-vicuna-7b/'")
    print("extractor = HiddenStateExtractor(model_path)")
    print()
    print("# For FLAVA")
    print("flava_extractor = FlavaFeatureExtractor()")

def main():
    """Main function to download all models"""
    print("VLM JAILBREAK DETECTION - MODEL DOWNLOADER")
    print("="*80)
    
    # Check system requirements
    if not check_disk_space():
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    check_gpu_availability()
    
    # Create model directory
    model_dir = create_model_directory()
    
    # Download models
    download_llava_v16_vicuna_7b(model_dir)
    download_flava_model(model_dir)
    download_clip_model(model_dir)
    
    # Print usage instructions
    print_usage_instructions()
    
    print("\n" + "="*80)
    print("MODEL DOWNLOAD COMPLETE")
    print("="*80)
    print("✅ All models have been downloaded to the 'model/' directory")
    print("🚀 You can now run your experiments!")
    
    # Final verification
    total_size = 0
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    
    total_size_gb = total_size / (1024**3)
    print(f"📊 Total model size: {total_size_gb:.2f} GB")

if __name__ == "__main__":
    main()
