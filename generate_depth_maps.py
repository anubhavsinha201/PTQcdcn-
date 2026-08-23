#!/usr/bin/env python
"""Generate depth maps for real CelebA-Spoof face samples."""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import urllib.request
from datasets import load_dataset

OUTPUT_DIR = "c:\\Users\\anubh\\OneDrive\\Documents\\Security System\\paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSET_DIR = os.path.expanduser('~/cdcn_assets')
os.makedirs(ASSET_DIR, exist_ok=True)

# Download Haar cascade and LBF facemark if needed
HAAR_PATH = os.path.join(ASSET_DIR, 'haarcascade_frontalface_default.xml')
LBF_PATH = os.path.join(ASSET_DIR, 'lbfmodel.yaml')

if not os.path.exists(HAAR_PATH):
    print("Downloading Haar cascade...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml',
        HAAR_PATH,
    )

if not os.path.exists(LBF_PATH):
    print("Downloading LBF facemark model...")
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml',
        LBF_PATH,
    )

# Load face detection and landmark models
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_PATH)

# Anatomical depth prior (same as in notebook)
LANDMARK_DEPTH_TEMPLATE = np.array(
    [0.10] * 17 +               # jaw line
    [0.45] * 10 +               # eyebrows
    [0.60, 0.72, 0.85, 1.00] +  # nose bridge -> tip
    [0.80] * 5 +                # nostrils / nose base
    [0.40] * 12 +               # eyes
    [0.55] * 12 +               # outer mouth
    [0.50] * 8,                 # inner mouth
    dtype=np.float32,
)

def geometry_depth_map(img_rgb_uint8, depth_size=32):
    """Generate depth map from face landmarks."""
    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    ok, landmark_sets = facemark.fit(img_rgb_uint8, faces)
    if not ok:
        return None

    pts = landmark_sets[0].reshape(-1, 2)
    h, w = img_rgb_uint8.shape[:2]
    gx = np.clip((pts[:, 0] / w * depth_size).astype(int), 0, depth_size - 1)
    gy = np.clip((pts[:, 1] / h * depth_size).astype(int), 0, depth_size - 1)

    grid = np.zeros((depth_size, depth_size), dtype=np.float32)
    counts = np.zeros((depth_size, depth_size), dtype=np.float32)

    for gx_i, gy_i, d in zip(gx, gy, LANDMARK_DEPTH_TEMPLATE):
        grid[gy_i, gx_i] += d
        counts[gy_i, gx_i] += 1

    mask = counts > 0
    grid[mask] /= counts[mask]

    filled = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=3)
    smoothed = cv2.GaussianBlur(filled, (5, 5), 0)
    return np.clip(smoothed, 0.0, 1.0)[..., None].astype(np.float32)


def get_label(row):
    """Extract label from HF dataset row."""
    if row.get('labelNames') is not None:
        return 1 if 'live' in str(row['labelNames']).lower() else 0
    if row.get('labels') is not None:
        return 1 if int(row['labels']) == 0 else 0
    raw_label = row.get('label') or row.get('class label') or 0
    return 1 if str(raw_label) == '1' or 'live' in str(raw_label).lower() else 0


def create_depth_maps_figure():
    """Load real faces and generate depth maps."""

    print("Loading CelebA-Spoof dataset...")
    try:
        ds = load_dataset("nguyenkhoa/celeba-spoof-for-face-antispoofing-test", split="test")
    except:
        ds = load_dataset("nguyenkhoa/celeba-spoof-for-face-antispoofing-test")

    # Collect live and spoof samples
    live_samples = []
    spoof_samples = []

    print("Scanning dataset for samples...")
    for i in range(min(500, len(ds))):
        row = ds[i]
        label = get_label(row)
        img_pil = row.get('cropped_image') or row.get('image')

        if img_pil is None:
            continue

        img = np.array(img_pil.convert('RGB'))
        img = cv2.resize(img, (256, 256))

        if label == 1 and len(live_samples) < 5:
            live_samples.append(img)
        elif label == 0 and len(spoof_samples) < 5:
            spoof_samples.append(img)

        if len(live_samples) >= 5 and len(spoof_samples) >= 5:
            break

    print(f"Found {len(live_samples)} live, {len(spoof_samples)} spoof samples")

    # Create visualization: 2 rows (live, spoof) x 5 columns x 2 sub-columns (input, depth)
    fig, axes = plt.subplots(2, 10, figsize=(18, 8))

    # Row 0: Live faces
    print("Generating depth maps for live faces...")
    for col, img in enumerate(live_samples):
        # Input image
        ax = axes[0, col * 2]
        ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.text(-0.3, 0.5, 'Live Face\nInput', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='right', va='center',
                   color='#029E73')

        # Depth map
        depth = geometry_depth_map(img, depth_size=64)
        ax = axes[0, col * 2 + 1]

        if depth is not None:
            im = ax.imshow(depth[:, :, 0], cmap='viridis', vmin=0, vmax=1)
            if col == 4:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Depth', fontsize=9)
        else:
            ax.imshow(np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1)
            ax.text(0.5, 0.5, 'No face\ndetected', transform=ax.transAxes,
                   ha='center', va='center', fontsize=10, color='red', fontweight='bold')

        ax.axis('off')
        if col == 0:
            ax.text(-0.1, 0.5, 'Predicted\nDepth Map', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='right', va='center',
                   color='#029E73')

    # Row 1: Spoof faces
    print("Generating depth maps for spoof faces...")
    for col, img in enumerate(spoof_samples):
        # Input image
        ax = axes[1, col * 2]
        ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.text(-0.3, 0.5, 'Spoof Face\nInput', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='right', va='center',
                   color='#CA0020')

        # Depth map
        depth = geometry_depth_map(img, depth_size=64)
        ax = axes[1, col * 2 + 1]

        if depth is not None:
            im = ax.imshow(depth[:, :, 0], cmap='viridis', vmin=0, vmax=1)
            if col == 4:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Depth', fontsize=9)
        else:
            ax.imshow(np.zeros((64, 64)), cmap='gray', vmin=0, vmax=1)
            ax.text(0.5, 0.5, 'No face\ndetected', transform=ax.transAxes,
                   ha='center', va='center', fontsize=10, color='red', fontweight='bold')

        ax.axis('off')
        if col == 0:
            ax.text(-0.1, 0.5, 'Predicted\nDepth Map', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='right', va='center',
                   color='#CA0020')

    fig.suptitle('Real CelebA-Spoof Faces: Input Images and Predicted Depth Maps',
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    plt.savefig(f"{OUTPUT_DIR}/fig0_real_depth_maps.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/fig0_real_depth_maps.pdf", bbox_inches='tight')
    print("[OK] Real depth maps figure saved")
    plt.close()


if __name__ == "__main__":
    print("\nGenerating depth maps for real CelebA-Spoof faces...\n")
    create_depth_maps_figure()
    print("\nDepth map visualization complete!")
    print(f"Saved to: {OUTPUT_DIR}/fig0_real_depth_maps.*")
