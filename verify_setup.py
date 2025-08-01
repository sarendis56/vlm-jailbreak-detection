#!/usr/bin/env python3
"""
Setup Verification Script for VLM Jailbreak Detection Project

This script verifies that all required datasets and models are properly downloaded
and accessible. It checks file existence, basic loading, and provides a summary
of what's available vs. what's missing.
"""

import os
import json
import sys
from pathlib import Path

def check_file_exists(filepath, description=""):
    """Check if a file exists and return status"""
    exists = os.path.exists(filepath)
    size = ""
    if exists:
        try:
            file_size = os.path.getsize(filepath)
            if file_size > 1024**3:  # GB
                size = f" ({file_size / (1024**3):.1f} GB)"
            elif file_size > 1024**2:  # MB
                size = f" ({file_size / (1024**2):.1f} MB)"
            elif file_size > 1024:  # KB
                size = f" ({file_size / 1024:.1f} KB)"
            else:
                size = f" ({file_size} bytes)"
        except:
            size = ""
    
    status = "✅" if exists else "❌"
    print(f"  {status} {filepath}{size}")
    if description and not exists:
        print(f"      {description}")
    return exists

def check_directory_exists(dirpath, description=""):
    """Check if a directory exists and count files"""
    exists = os.path.exists(dirpath)
    count = ""
    if exists and os.path.isdir(dirpath):
        try:
            file_count = len([f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))])
            count = f" ({file_count} files)"
        except:
            count = ""
    
    status = "✅" if exists else "❌"
    print(f"  {status} {dirpath}{count}")
    if description and not exists:
        print(f"      {description}")
    return exists

def verify_datasets():
    """Verify all required datasets"""
    print("="*60)
    print("VERIFYING DATASETS")
    print("="*60)
    
    datasets_status = {}
    
    # Automatic datasets (from HuggingFace)
    print("\n📦 Automatic Downloads (HuggingFace):")
    
    auto_datasets = [
        ("data/Alpaca/alpaca_samples.json", "Alpaca instruction following dataset"),
        ("data/AdvBench/advbench_samples.json", "AdvBench harmful prompts dataset"),
        ("data/DAN_Prompts/dan_prompts_samples.json", "DAN jailbreak prompts dataset"),
        ("data/OpenAssistant/openassistant_samples.json", "OpenAssistant conversational dataset"),
        ("data/VQAv2/vqav2_samples.json", "VQAv2 visual question answering dataset"),
        ("data/XSTest/xstest_samples.json", "XSTest safety evaluation dataset")
    ]
    
    auto_count = 0
    for filepath, description in auto_datasets:
        if check_file_exists(filepath, f"Run: python download_datasets.py"):
            auto_count += 1
            # Try to load and check sample count
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    print(f"      Loaded {len(data)} samples")
            except:
                print(f"      Could not read sample count")
        datasets_status[filepath] = os.path.exists(filepath)
    
    # Manual datasets
    print("\n🔧 Manual Downloads Required:")
    
    manual_datasets = [
        ("data/MM-Vet/mm-vet.json", "MM-Vet multimodal reasoning dataset"),
        ("data/FigTxt/benign_questions.csv", "FigTxt benign questions"),
        ("data/FigTxt/safebench.csv", "FigTxt safety benchmark"),
        ("data/JailBreakV-28k/mini_JailBreakV_28K.csv", "JailBreakV-28k text queries"),
        ("data/VAE/manual_harmful_instructions.csv", "VAE harmful instructions")
    ]
    
    manual_count = 0
    for filepath, description in manual_datasets:
        if check_file_exists(filepath, f"Manual download required - see RECOVERY_README.md"):
            manual_count += 1
        datasets_status[filepath] = os.path.exists(filepath)
    
    # Manual directories
    print("\n📁 Manual Directories Required:")
    
    manual_dirs = [
        ("data/JailBreakV-28k/figstep", "JailBreakV-28k figstep images"),
        ("data/JailBreakV-28k/llm_transfer_attack", "JailBreakV-28k LLM transfer attack images"),
        ("data/JailBreakV-28k/query_related", "JailBreakV-28k query related images"),
        ("data/VAE/adversarial_images", "VAE adversarial images")
    ]
    
    manual_dir_count = 0
    for dirpath, description in manual_dirs:
        if check_directory_exists(dirpath, f"Manual download required - see RECOVERY_README.md"):
            manual_dir_count += 1
        datasets_status[dirpath] = os.path.exists(dirpath)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Automatic datasets: {auto_count}/{len(auto_datasets)} ✅")
    print(f"  Manual files: {manual_count}/{len(manual_datasets)} ✅")
    print(f"  Manual directories: {manual_dir_count}/{len(manual_dirs)} ✅")
    
    total_available = auto_count + manual_count + manual_dir_count
    total_required = len(auto_datasets) + len(manual_datasets) + len(manual_dirs)
    
    return total_available, total_required, datasets_status

def verify_models():
    """Verify all required models"""
    print("\n" + "="*60)
    print("VERIFYING MODELS")
    print("="*60)
    
    models_status = {}
    
    print("\n🤖 Required Models:")
    
    models = [
        ("model/llava-v1.6-vicuna-7b", "LLaVA-v1.6-Vicuna-7B multimodal model"),
        ("model/flava", "FLAVA multimodal model"),
        ("model/clip-vit-base-patch32", "CLIP baseline model")
    ]
    
    model_count = 0
    for model_path, description in models:
        if check_directory_exists(model_path, f"Run: python download_models.py"):
            model_count += 1
            
            # Check for key model files
            key_files = ["config.json"]
            model_files = []
            
            if os.path.exists(model_path):
                for file in os.listdir(model_path):
                    if file.endswith(('.bin', '.safetensors')):
                        model_files.append(file)
                
                if model_files:
                    print(f"      Found model files: {len(model_files)} files")
                else:
                    print(f"      ⚠️  No model weight files found")
        
        models_status[model_path] = os.path.exists(model_path)
    
    print(f"\n📊 Model Summary:")
    print(f"  Available models: {model_count}/{len(models)} ✅")
    
    return model_count, len(models), models_status

def test_basic_loading():
    """Test basic loading of key components"""
    print("\n" + "="*60)
    print("TESTING BASIC LOADING")
    print("="*60)
    
    # Test dataset loading
    print("\n📚 Testing Dataset Loading:")
    try:
        sys.path.append('code')
        from load_datasets import load_alpaca, load_mm_vet
        
        # Test Alpaca loading
        try:
            alpaca_samples = load_alpaca(max_samples=5)
            if alpaca_samples:
                print(f"  ✅ Alpaca: Loaded {len(alpaca_samples)} test samples")
            else:
                print(f"  ❌ Alpaca: No samples loaded")
        except Exception as e:
            print(f"  ❌ Alpaca: Error loading - {e}")
        
        # Test MM-Vet loading
        try:
            mmvet_samples = load_mm_vet()
            if mmvet_samples:
                print(f"  ✅ MM-Vet: Loaded {len(mmvet_samples)} samples")
            else:
                print(f"  ❌ MM-Vet: No samples loaded")
        except Exception as e:
            print(f"  ❌ MM-Vet: Error loading - {e}")
            
    except ImportError as e:
        print(f"  ❌ Could not import dataset loading functions: {e}")
    
    # Test model loading (basic check)
    print("\n🤖 Testing Model Accessibility:")
    
    llava_path = "model/llava-v1.6-vicuna-7b"
    if os.path.exists(llava_path):
        config_path = os.path.join(llava_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_type = config.get('model_type', 'unknown')
                    print(f"  ✅ LLaVA config accessible (type: {model_type})")
            except Exception as e:
                print(f"  ❌ LLaVA config error: {e}")
        else:
            print(f"  ❌ LLaVA config.json not found")
    else:
        print(f"  ❌ LLaVA model directory not found")

def main():
    """Main verification function"""
    print("VLM JAILBREAK DETECTION - SETUP VERIFICATION")
    print("="*80)
    
    # Verify datasets
    dataset_available, dataset_total, dataset_status = verify_datasets()
    
    # Verify models  
    model_available, model_total, model_status = verify_models()
    
    # Test basic loading
    test_basic_loading()
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    dataset_percent = (dataset_available / dataset_total) * 100 if dataset_total > 0 else 0
    model_percent = (model_available / model_total) * 100 if model_total > 0 else 0
    
    print(f"📊 Datasets: {dataset_available}/{dataset_total} ({dataset_percent:.0f}%) available")
    print(f"🤖 Models: {model_available}/{model_total} ({model_percent:.0f}%) available")
    
    if dataset_available == dataset_total and model_available == model_total:
        print("\n🎉 SUCCESS: All datasets and models are available!")
        print("✅ You can now run your experiments:")
        print("   cd code && python balanced_ml.py")
    else:
        print("\n⚠️  INCOMPLETE SETUP:")
        if dataset_available < dataset_total:
            print("   Missing datasets - run download_datasets.py")
        if model_available < model_total:
            print("   Missing models - run download_models.py")
        
        print("\n📖 See RECOVERY_README.md for detailed instructions")
    
    # Disk space check
    try:
        statvfs = os.statvfs('.')
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        print(f"\n💾 Available disk space: {free_space_gb:.1f} GB")
    except:
        pass

if __name__ == "__main__":
    main()
