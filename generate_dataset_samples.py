#!/usr/bin/env python
"""Load CelebA-Spoof dataset and visualize sample live vs spoof faces."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset
import os

OUTPUT_DIR = "c:\\Users\\anubh\\OneDrive\\Documents\\Security System\\paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors for the paper
COLORS = {
    'live': '#029E73',      # Green
    'spoof': '#CA0020',     # Red
}

def get_label(row):
    """Extract label from HF dataset row (1=live, 0=spoof)."""
    if row.get('labelNames') is not None:
        return 1 if 'live' in str(row['labelNames']).lower() else 0
    if row.get('labels') is not None:
        return 1 if int(row['labels']) == 0 else 0
    raw_label = row.get('label') or row.get('class label') or 0
    return 1 if str(raw_label) == '1' or 'live' in str(raw_label).lower() else 0


def create_dataset_samples():
    """Load CelebA-Spoof and display sample images."""

    print("Loading CelebA-Spoof dataset from HuggingFace...")
    try:
        ds = load_dataset("nguyenkhoa/celeba-spoof-for-face-antispoofing-test",
                         split="test", trust_remote_code=True)
        print(f"Dataset loaded: {len(ds)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting alternative approach...")
        ds = load_dataset("nguyenkhoa/celeba-spoof-for-face-antispoofing-test")
        print(f"Dataset loaded: {len(ds)} images")

    # Separate live and spoof
    print("Separating live and spoof samples...")
    live_indices = []
    spoof_indices = []

    for i in range(min(1000, len(ds))):  # Scan first 1000 for speed
        label = get_label(ds[i])
        if label == 1:
            live_indices.append(i)
        else:
            spoof_indices.append(i)
        if len(live_indices) >= 5 and len(spoof_indices) >= 5:
            break

    print(f"Found {len(live_indices)} live, {len(spoof_indices)} spoof samples")

    # Create grid: 2 rows (live, spoof) x 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))

    # TOP ROW: Live faces
    print("Loading live face samples...")
    for col in range(5):
        idx = live_indices[col]
        row = ds[idx]

        # Get image
        img_pil = row.get('cropped_image') or row.get('image')
        if img_pil is None:
            continue

        img = np.array(img_pil.convert('RGB'))

        # Display
        ax = axes[0, col]
        ax.imshow(img)
        ax.axis('off')

        if col == 0:
            ax.text(-0.15, 0.5, 'Live Faces\n(Genuine)', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', ha='right', va='center',
                   color=COLORS['live'])

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS['live'])
            spine.set_linewidth(3)
            spine.set_visible(True)

    # BOTTOM ROW: Spoof attacks
    print("Loading spoof attack samples...")
    for col in range(5):
        idx = spoof_indices[col]
        row = ds[idx]

        # Get image
        img_pil = row.get('cropped_image') or row.get('image')
        if img_pil is None:
            continue

        img = np.array(img_pil.convert('RGB'))

        # Display
        ax = axes[1, col]
        ax.imshow(img)
        ax.axis('off')

        if col == 0:
            ax.text(-0.15, 0.5, 'Spoof Attacks\n(2D Print/Screen)', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', ha='right', va='center',
                   color=COLORS['spoof'])

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS['spoof'])
            spine.set_linewidth(3)
            spine.set_visible(True)

    fig.suptitle('CelebA-Spoof Dataset: Live Faces vs Presentation Attacks',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.savefig(f"{OUTPUT_DIR}/fig0_dataset_examples.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/fig0_dataset_examples.pdf", bbox_inches='tight')
    print("[OK] Dataset samples figure saved")
    plt.close()


if __name__ == "__main__":
    print("\nGenerating CelebA-Spoof dataset sample visualization...\n")
    create_dataset_samples()
    print("\nDataset visualization complete!")
    print(f"Saved to: {OUTPUT_DIR}/fig0_dataset_examples.*")
