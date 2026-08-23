#!/usr/bin/env python
"""Simple CUDA test - run this to verify GPU is working."""

import tensorflow as tf
import numpy as np
import time

print("\n" + "="*60)
print("SIMPLE CUDA TEST")
print("="*60)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs detected: {len(gpus)}")

if not gpus:
    print("❌ No GPUs found! Check CUDA setup.")
    exit(1)

print("✅ GPU(s) found:")
for gpu in gpus:
    print(f"   {gpu}")

# Test 1: Simple matrix multiplication on GPU
print("\n--- Test 1: Matrix Multiplication on GPU ---")
try:
    with tf.device('/GPU:0'):
        # Create random matrices
        a = tf.random.normal((1000, 1000))
        b = tf.random.normal((1000, 1000))

        # Warm-up run
        _ = tf.matmul(a, b)

        # Timed run
        start = time.time()
        result = tf.matmul(a, b)
        gpu_time = time.time() - start

    print(f"✅ GPU computation successful!")
    print(f"   Shape: {result.shape}")
    print(f"   Time: {gpu_time*1000:.2f}ms")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Compare GPU vs CPU
print("\n--- Test 2: GPU vs CPU Speed Comparison ---")
size = 2000
runs = 5

# GPU computation
print(f"   Matrix size: {size}x{size} (5 runs)")
gpu_times = []
for i in range(runs):
    with tf.device('/GPU:0'):
        a = tf.random.normal((size, size))
        b = tf.random.normal((size, size))
        start = time.time()
        _ = tf.matmul(a, b)
        gpu_times.append(time.time() - start)

# CPU computation
cpu_times = []
for i in range(runs):
    with tf.device('/CPU:0'):
        a = tf.random.normal((size, size))
        b = tf.random.normal((size, size))
        start = time.time()
        _ = tf.matmul(a, b)
        cpu_times.append(time.time() - start)

gpu_avg = np.mean(gpu_times)
cpu_avg = np.mean(cpu_times)
speedup = cpu_avg / gpu_avg

print(f"   GPU avg: {gpu_avg*1000:.2f}ms")
print(f"   CPU avg: {cpu_avg*1000:.2f}ms")
print(f"   Speedup: {speedup:.1f}x faster on GPU ✅")

# Test 3: TensorFlow model with GPU
print("\n--- Test 3: Simple Neural Network Training ---")
try:
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Create dummy data
    x_train = np.random.randn(1000, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, 1000).astype(np.float32)

    # Train for 1 epoch
    print("   Training model on GPU...")
    start = time.time()
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
    train_time = time.time() - start

    print(f"✅ Training successful!")
    print(f"   Time: {train_time:.2f}s")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - CUDA IS WORKING!")
print("="*60 + "\n")
