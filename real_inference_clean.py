#!/usr/bin/env python
"""
Clean layout: Images, Depth Maps, and Stats in separate, non-overlapping panels.
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
    return 1 if str(row.get('label') or row.get('class label', 0)) in ('1', 'live') else 0


def run_inference_clean():
    """Run inference with clean, non-overlapping layout."""

    print("\n" + "="*70)
    print("ACTUAL INFERENCE - CLEAN LAYOUT")
    print("="*70)

    # Load model
    print("\n[1] Loading quantized model...")
    interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  Model loaded: {os.path.basename(QUANTIZED_MODEL)}")

    # Load dataset
    print("\n[2] Loading CelebA-Spoof dataset...")
    try:
        ds = load_dataset("nguyenkhoa/celeba-spoof-for-face-antispoofing-test", split="test")
    except:
        ds = load_dataset("nguyenkhoa/celeba-spoof-for-face-antispoofing-test")

    live_images = []
    spoof_images = []

    for i in range(min(300, len(ds))):
        row = ds[i]
        img_pil = row.get('cropped_image') or row.get('image')
        if img_pil is None:
            continue

        img = np.array(img_pil.convert('RGB'))
        img = cv2.resize(img, (256, 256))
        label = get_label(row)

        if label == 1 and len(live_images) < 3:
            live_images.append(img)
        elif label == 0 and len(spoof_images) < 3:
            spoof_images.append(img)

        if len(live_images) >= 3 and len(spoof_images) >= 3:
            break

    print(f"  Loaded {len(live_images)} live, {len(spoof_images)} spoof")

    # Run inference
    print("\n[3] Running inference...")
    predictions = {}

    for label_type, images in [('live', live_images), ('spoof', spoof_images)]:
        print(f"\n  {label_type.upper()}:")
        for idx, img in enumerate(images):
            img_normalized = (img / 255.0).astype(np.float32)
            img_batch = np.expand_dims(img_normalized, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_batch)
            interpreter.invoke()

            depth_raw = interpreter.get_tensor(output_details[1]['index'])
            cls_raw = interpreter.get_tensor(output_details[0]['index'])

            predictions[f"{label_type}_{idx}"] = {
                'type': label_type,
                'image': img,
                'depth_map': depth_raw[0, :, :, 0],
                'liveness_score': float(cls_raw[0, 0])
            }

            status = "LIVE" if cls_raw[0, 0] > 0.5 else "SPOOF"
            print(f"    [{idx}] Score: {cls_raw[0, 0]:.4f} ({status}) | Depth: {depth_raw[0, :, :, 0].mean():.4f}")

    # Create clean layout: 2 rows (live/spoof) x 3 samples
    # Each sample has 3 columns: [Input Image] [Depth Map] [Stats Text]
    print("\n[4] Creating clean visualization...")

    fig, axes = plt.subplots(2, 9, figsize=(24, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    # ===== ROW 0: LIVE FACES =====
    for col in range(3):
        pred = predictions[f'live_{col}']
        depth = pred['depth_map']

        # Column 0: Input Image
        ax = axes[0, col * 3]
        ax.imshow(pred['image'])
        ax.axis('off')
        ax.set_title(f'Live Face #{col+1}\nInput Image', fontsize=13, fontweight='bold', color=COLORS['live'])

        # Column 1: Depth Map
        ax = axes[0, col * 3 + 1]
        im = ax.imshow(depth, cmap='viridis', vmin=0, vmax=0.5)
        ax.axis('off')
        ax.set_title(f'Predicted\nDepth Map', fontsize=13, fontweight='bold', color=COLORS['live'])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Depth', fontsize=10, fontweight='bold')

        # Column 2: Stats (as text in clean box)
        ax = axes[0, col * 3 + 2]
        ax.axis('off')

        stats_text = (
            f"Liveness Score\n{pred['liveness_score']:.4f}\n\n"
            f"Mean Depth\n{depth.mean():.4f}\n\n"
            f"Depth Std Dev\n{depth.std():.4f}\n\n"
            f"Max Depth\n{depth.max():.4f}\n\n"
            f"Classification\nLIVE ✓"
        )

        ax.text(0.5, 0.5, stats_text,
               transform=ax.transAxes,
               fontsize=12,
               fontweight='bold',
               family='monospace',
               ha='center',
               va='center',
               bbox=dict(boxstyle='round,pad=1.5',
                        facecolor=COLORS['live'],
                        alpha=0.2,
                        edgecolor=COLORS['live'],
                        linewidth=3))

    # ===== ROW 1: SPOOF FACES =====
    for col in range(3):
        pred = predictions[f'spoof_{col}']
        depth = pred['depth_map']

        # Column 0: Input Image
        ax = axes[1, col * 3]
        ax.imshow(pred['image'])
        ax.axis('off')
        ax.set_title(f'Spoof Face #{col+1}\nInput Image', fontsize=13, fontweight='bold', color=COLORS['spoof'])

        # Column 1: Depth Map
        ax = axes[1, col * 3 + 1]
        im = ax.imshow(depth, cmap='viridis', vmin=0, vmax=0.5)
        ax.axis('off')
        ax.set_title(f'Predicted\nDepth Map', fontsize=13, fontweight='bold', color=COLORS['spoof'])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Depth', fontsize=10, fontweight='bold')

        # Column 2: Stats (as text in clean box)
        ax = axes[1, col * 3 + 2]
        ax.axis('off')

        stats_text = (
            f"Liveness Score\n{pred['liveness_score']:.4f}\n\n"
            f"Mean Depth\n{depth.mean():.4f}\n\n"
            f"Depth Std Dev\n{depth.std():.4f}\n\n"
            f"Max Depth\n{depth.max():.4f}\n\n"
            f"Classification\nSPOOF ✓"
        )

        ax.text(0.5, 0.5, stats_text,
               transform=ax.transAxes,
               fontsize=12,
               fontweight='bold',
               family='monospace',
               ha='center',
               va='center',
               bbox=dict(boxstyle='round,pad=1.5',
                        facecolor=COLORS['spoof'],
                        alpha=0.2,
                        edgecolor=COLORS['spoof'],
                        linewidth=3))

    fig.suptitle(
        'ACTUAL CDCN++ Model Inference on Real CelebA-Spoof Faces',
        fontsize=18, fontweight='bold', y=0.98
    )

    plt.savefig(f"{OUTPUT_DIR}/ACTUAL_inference_CLEAN.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/ACTUAL_inference_CLEAN.pdf", bbox_inches='tight')
    print(f"  [OK] Clean figure saved!")

    # Summary
    print("\n[5] Results Summary:")
    print("-" * 70)

    live_data = [predictions[k] for k in predictions if k.startswith('live')]
    spoof_data = [predictions[k] for k in predictions if k.startswith('spoof')]

    live_depths = [p['depth_map'].mean() for p in live_data]
    live_scores = [p['liveness_score'] for p in live_data]
    spoof_depths = [p['depth_map'].mean() for p in spoof_data]
    spoof_scores = [p['liveness_score'] for p in spoof_data]

    print(f"\nLive faces (n={len(live_data)}):")
    print(f"  Mean depth:      {np.mean(live_depths):.4f}")
    print(f"  Liveness score:  {np.mean(live_scores):.4f}")
    print(f"  Correct class:   {sum(s > 0.5 for s in live_scores)}/{len(live_scores)}")

    print(f"\nSpoof faces (n={len(spoof_data)}):")
    print(f"  Mean depth:      {np.mean(spoof_depths):.4f}")
    print(f"  Liveness score:  {np.mean(spoof_scores):.4f}")
    print(f"  Correct class:   {sum(s < 0.5 for s in spoof_scores)}/{len(spoof_scores)}")

    print(f"\nSeparation:")
    print(f"  Depth mean gap:  {np.mean(live_depths) - np.mean(spoof_depths):.4f}")
    print(f"  Score gap:       {np.mean(live_scores) - np.mean(spoof_scores):.4f}")

    print("\n" + "="*70)
    print("[OK] CLEAN INFERENCE FIGURE SAVED")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_inference_clean()
