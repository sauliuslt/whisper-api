#!/usr/bin/env python3
"""Quick script to check GPU availability and PyTorch CUDA support."""

import torch
import sys

print("=" * 60)
print("GPU/CUDA CHECK")
print("=" * 60)

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")

        # Try to allocate tensor on GPU
        try:
            test_tensor = torch.zeros(100, 100).cuda(i)
            print(f"  ✓ Can allocate tensors on GPU {i}")
        except Exception as e:
            print(f"  ✗ Cannot allocate tensors on GPU {i}: {e}")

    # Test Whisper import
    print("\nTesting Whisper import...")
    try:
        import whisper
        print("✓ Whisper imported successfully")

        # Try loading a tiny model to test
        print("Testing model load on GPU...")
        model = whisper.load_model("tiny", device="cuda")
        print("✓ Successfully loaded Whisper model on GPU")

        # Test inference
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)
        result = model.transcribe(dummy_audio, fp16=True)
        print("✓ Successfully ran inference on GPU")

    except Exception as e:
        print(f"✗ Whisper test failed: {e}")
else:
    print("\n❌ CUDA is NOT available!")
    print("\nPossible solutions:")
    print("1. Check if you have an NVIDIA GPU: nvidia-smi")
    print("2. Install CUDA-enabled PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("3. Check NVIDIA drivers: nvidia-smi")
    print("4. Verify CUDA toolkit is installed")

print("\n" + "=" * 60)