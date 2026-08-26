#!/usr/bin/env python
"""
Run actual inference on CelebA-Spoof using the trained quantized CDCN++ model.
Uses the dynamic-range INT8 quantized model (perfect accuracy, 4x compression).
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from datasets import load_dataset

OUTPUT_DIR = "c:\\Users\\anubh\\OneDrive\\Documents\\Security System\\paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUANTIZED_MODEL = "c:\\Users\\anubh\\OneDrive\\Documents\\Security System\\cdcn_quantized\\cdcnpp_dynamic_range.tflite"

COLORS = {
    'live': '#029E73',
    'spoof': '#CA0020',
}


def get_label(row):
    """Extract label from HF dataset row."""
    if row.get('labelNames') is not None:
        return 1 if 'live' in str(row['labelNames']).lower() else 0
    if row.get('labels') is not None:
        return 1 if int(row['labels']) == 0 else 0
    raw_label = row.get('label') or row.get('class label') or 0
    return 1 if str(raw_label) == '1' or 'live' in str(raw_label).lower() else 0


def run_inference_actual():
    """Load quantized model and run actual inference on real faces."""

    print("\n" + "="*70)
    print("ACTUAL INFERENCE: CDCN++ Quantized Model on CelebA-Spoof")
    print("="*70)

    # Load quantized model
    print("\n[1] Loading quantized TFLite model...")
    if not os.path.exists(QUANTIZED_MODEL):
        print(f"ERROR: Model not found at {QUANTIZED_MODEL}")
        print("Available models:")
        quant_dir = os.path.dirname(QUANTIZED_MODEL)
        for f in os.listdir(quant_dir):
            if f.endswith('.tflite'):
                print(f"  - {f}")
        return

    interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Model loaded: {os.path.basename(QUANTIZED_MODEL)}")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Outputs: {len(output_details)} ({[od['shape'] for od in output_details]})")

    # Load dataset
    print("\n[2] Loading CelebA-Spoof dataset...")
    try:
        ds = load_dataset("nguyenkhoa/celeba-spoof-for-face-antispoofing-test", split="test")
    except:
        ds = load_dataset("nguyenkhoa/celeba-spoof-for-face-antispoofing-test")

    # Collect samples
    live_samples = []
    spoof_samples = []

    print("  Scanning for samples...")
    for i in range(min(300, len(ds))):
        row = ds[i]
        label = get_label(row)
        img_pil = row.get('cropped_image') or row.get('image')

        if img_pil is None:
            continue

        img = np.array(img_pil.convert('RGB'))
        img = cv2.resize(img, (256, 256))

        if label == 1 and len(live_samples) < 5:
            live_samples.append((img, i))
        elif label == 0 and len(spoof_samples) < 5:
            spoof_samples.append((img, i))

        if len(live_samples) >= 5 and len(spoof_samples) >= 5:
            break

    print(f"  Found {len(live_samples)} live, {len(spoof_samples)} spoof samples")

    # Run inference
    print("\n[3] Running inference on real faces...")

    all_samples = live_samples + spoof_samples
    predictions = {}

    for label_type, samples in [('live', live_samples), ('spoof', spoof_samples)]:
        print(f"\n  Processing {label_type} faces:")
        for img, idx in samples:
            # Preprocess
            img_normalized = (img / 255.0).astype(np.float32)
            img_batch = np.expand_dims(img_normalized, axis=0)

            # Inference
            interpreter.set_tensor(input_details[0]['index'], img_batch)
            interpreter.invoke()

            # Get outputs
            depth_raw = interpreter.get_tensor(output_details[1]['index'])  # depth output
            cls_raw = interpreter.get_tensor(output_details[0]['index'])    # classification output

            # Store results
            predictions[idx] = {
                'type': label_type,
                'image': img,
                'depth_map': depth_raw[0, :, :, 0],  # (32, 32) depth
                'liveness_score': float(cls_raw[0, 0])
            }

            status = "LIVE" if cls_raw[0, 0] > 0.5 else "SPOOF"
            print(f"    [{idx}] Liveness: {cls_raw[0, 0]:.4f} → {status} | Depth range: [{depth_raw.min():.3f}, {depth_raw.max():.3f}]")

    # Create visualization
    print("\n[4] Creating visualization...")

    fig, axes = plt.subplots(2, 15, figsize=(20, 7))

    # Live faces: 5 samples x 3 columns (input, depth heatmap, depth 3D)
    for col, (img, idx) in enumerate(live_samples):
        pred = predictions[idx]

        # Input
        ax = axes[0, col * 3]
        ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.text(-0.35, 0.5, 'Live Face\nInput', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', ha='right', va='center',
                   color=COLORS['live'])

        # Depth heatmap
        ax = axes[0, col * 3 + 1]
        im = ax.imshow(pred['depth_map'], cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
        if col == 0:
            ax.text(-0.1, 0.5, 'Predicted\nDepth', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', ha='right', va='center',
                   color=COLORS['live'])

        # Stats
        ax = axes[0, col * 3 + 2]
        ax.axis('off')
        depth = pred['depth_map']
        stats_text = (
            f"Live Score: {pred['liveness_score']:.3f}\n"
            f"Mean depth: {depth.mean():.3f}\n"
            f"Std depth: {depth.std():.3f}\n"
            f"Max: {depth.max():.3f}"
        )
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=9, family='monospace', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['live'], alpha=0.1))

    # Spoof faces: 5 samples x 3 columns
    for col, (img, idx) in enumerate(spoof_samples):
        pred = predictions[idx]

        # Input
        ax = axes[1, col * 3]
        ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.text(-0.35, 0.5, 'Spoof Face\nInput', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', ha='right', va='center',
                   color=COLORS['spoof'])

        # Depth heatmap
        ax = axes[1, col * 3 + 1]
        im = ax.imshow(pred['depth_map'], cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
        if col == 0:
            ax.text(-0.1, 0.5, 'Predicted\nDepth', transform=ax.transAxes,
                   fontsize=10, fontweight='bold', ha='right', va='center',
                   color=COLORS['spoof'])

        # Stats
        ax = axes[1, col * 3 + 2]
        ax.axis('off')
        depth = pred['depth_map']
        stats_text = (
            f"Live Score: {pred['liveness_score']:.3f}\n"
            f"Mean depth: {depth.mean():.3f}\n"
            f"Std depth: {depth.std():.3f}\n"
            f"Max: {depth.max():.3f}"
        )
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=9, family='monospace', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['spoof'], alpha=0.1))

    fig.suptitle(
        'ACTUAL CDCN++ Inference: Real CelebA-Spoof Faces with Quantized Model Predictions',
        fontsize=13, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    plt.savefig(f"{OUTPUT_DIR}/ACTUAL_depth_maps_quantized.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/ACTUAL_depth_maps_quantized.pdf", bbox_inches='tight')
    print("  ✓ Saved: ACTUAL_depth_maps_quantized.png/pdf")

    # Summary statistics
    print("\n[5] Summary Statistics:")
    print("-" * 70)

    live_depths = [predictions[idx]['depth_map'].mean() for img, idx in live_samples]
    live_scores = [predictions[idx]['liveness_score'] for img, idx in live_samples]
    spoof_depths = [predictions[idx]['depth_map'].mean() for img, idx in spoof_samples]
    spoof_scores = [predictions[idx]['liveness_score'] for img, idx in spoof_samples]

    print(f"\nLive faces (n={len(live_samples)}):")
    print(f"  Mean depth:      {np.mean(live_depths):.4f} ± {np.std(live_depths):.4f}")
    print(f"  Liveness score:  {np.mean(live_scores):.4f} ± {np.std(live_scores):.4f}")
    print(f"  All correctly classified: {all(s > 0.5 for s in live_scores)}")

    print(f"\nSpoof faces (n={len(spoof_samples)}):")
    print(f"  Mean depth:      {np.mean(spoof_depths):.4f} ± {np.std(spoof_depths):.4f}")
    print(f"  Liveness score:  {np.mean(spoof_scores):.4f} ± {np.std(spoof_scores):.4f}")
    print(f"  All correctly classified: {all(s < 0.5 for s in spoof_scores)}")

    print(f"\nSeparation:")
    print(f"  Depth mean gap:  {np.mean(live_depths) - np.mean(spoof_depths):.4f}")
    print(f"  Score gap:       {np.mean(live_scores) - np.mean(spoof_scores):.4f}")

    print("\n" + "="*70)
    print("INFERENCE COMPLETE - ACTUAL MODEL OUTPUTS SAVED")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_inference_actual()
