#!/usr/bin/env python
"""Verify CUDA setup for CDCN project in VSCode."""

import os
import sys
import platform

print("=" * 70)
print("CUDA & GPU VERIFICATION FOR CDCN PROJECT")
print("=" * 70)

# System Info
print(f"\n[SYSTEM]")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

# CUDA Environment Variables
print(f"\n[CUDA ENVIRONMENT]")
cuda_home = os.environ.get('CUDA_HOME', 'NOT SET')
cuda_path = os.environ.get('CUDA_PATH', 'NOT SET')
cudnn_home = os.environ.get('CUDNN_HOME', 'NOT SET')
print(f"CUDA_HOME: {cuda_home}")
print(f"CUDA_PATH: {cuda_path}")
print(f"CUDNN_HOME: {cudnn_home}")

# TensorFlow
print(f"\n[TENSORFLOW]")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        # Test a simple operation on GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            c = tf.matmul(a, b)
            print(f"✓ GPU computation works! Result shape: {c.shape}")
    else:
        print("  ⚠ No GPUs detected")
except ImportError as e:
    print(f"✗ TensorFlow not installed: {e}")
except Exception as e:
    print(f"✗ Error: {e}")

# PyTorch (optional)
print(f"\n[PYTORCH (optional)]")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        # Test a simple operation
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ GPU computation works! Result shape: {z.shape}")
    else:
        print("  ⚠ CUDA not available")
except ImportError:
    print("  (PyTorch not installed - this is OK if using TensorFlow only)")
except Exception as e:
    print(f"✗ Error: {e}")

# GPU Memory
print(f"\n[GPU MEMORY]")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            total, used, free = map(int, line.split(','))
            print(f"GPU {i}:")
            print(f"  Total: {total // 1024} GB")
            print(f"  Used:  {used // 1024} GB")
            print(f"  Free:  {free // 1024} GB")
except Exception as e:
    print(f"  Could not query nvidia-smi: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
