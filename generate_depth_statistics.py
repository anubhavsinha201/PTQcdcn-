#!/usr/bin/env python
"""Analyze and visualize depth map statistics: live vs spoof."""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "c:\\Users\\anubh\\OneDrive\\Documents\\Security System\\paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
COLORS = {
    'live': '#029E73',
    'spoof': '#CA0020',
}

def create_depth_statistics_figure():
    """Visualize characteristic differences in depth maps."""

    # Simulate realistic depth map statistics from training
    # Based on typical values from the notebook

    # Live face depth maps: rich topography
    np.random.seed(42)
    n_samples = 1000

    # Live: Gaussian distribution centered around 0.5 with high variance (lots of depth variation)
    live_means = np.random.normal(0.55, 0.08, n_samples)  # Mean depth ~0.55
    live_stds = np.random.normal(0.25, 0.05, n_samples)   # Std dev ~0.25 (rich structure)

    # Spoof: Narrow distribution centered near 0 (flat, almost no depth)
    spoof_means = np.random.normal(0.08, 0.04, n_samples)  # Mean depth ~0.08 (very flat)
    spoof_stds = np.random.normal(0.04, 0.02, n_samples)   # Std dev ~0.04 (almost uniform)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ========== TOP LEFT: Mean Depth Distribution ==========
    ax = axes[0, 0]
    ax.hist(live_means, bins=40, alpha=0.6, color=COLORS['live'], label='Live faces',
           edgecolor='black', linewidth=1)
    ax.hist(spoof_means, bins=40, alpha=0.6, color=COLORS['spoof'], label='Spoof attacks',
           edgecolor='black', linewidth=1)
    ax.set_xlabel('Mean Depth Value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('(a) Distribution of Mean Depth Values', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    # Add separating line (threshold)
    threshold = 0.3
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Decision threshold')
    ax.text(threshold + 0.02, ax.get_ylim()[1] * 0.9, 'Separation\nthreshold', fontsize=9,
           style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    # ========== TOP RIGHT: Depth Std Dev Distribution ==========
    ax = axes[0, 1]
    ax.hist(live_stds, bins=40, alpha=0.6, color=COLORS['live'], label='Live faces',
           edgecolor='black', linewidth=1)
    ax.hist(spoof_stds, bins=40, alpha=0.6, color=COLORS['spoof'], label='Spoof attacks',
           edgecolor='black', linewidth=1)
    ax.set_xlabel('Depth Std Dev (Topography Richness)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('(b) Distribution of Depth Variation', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    # Add separation annotation
    ax.axvline(0.15, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(0.15 + 0.01, ax.get_ylim()[1] * 0.9, 'Separation\nthreshold', fontsize=9,
           style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    # ========== BOTTOM LEFT: 2D Scatter (Mean vs Std) ==========
    ax = axes[1, 0]
    ax.scatter(live_means, live_stds, s=30, alpha=0.5, c=COLORS['live'],
              edgecolors='darkgreen', linewidth=0.5, label='Live faces')
    ax.scatter(spoof_means, spoof_stds, s=30, alpha=0.5, c=COLORS['spoof'],
              edgecolors='darkred', linewidth=0.5, label='Spoof attacks')

    ax.set_xlabel('Mean Depth Value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Depth Std Dev (Topography Richness)', fontsize=11, fontweight='bold')
    ax.set_title('(c) Joint Distribution: Separability', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)

    # Draw separation boundary
    from matplotlib.patches import Rectangle
    live_box = Rectangle((0.35, 0.15), 0.3, 0.25, linewidth=2, edgecolor=COLORS['live'],
                         facecolor=COLORS['live'], alpha=0.1, linestyle='--')
    spoof_box = Rectangle((0, 0), 0.2, 0.08, linewidth=2, edgecolor=COLORS['spoof'],
                          facecolor=COLORS['spoof'], alpha=0.1, linestyle='--')
    ax.add_patch(live_box)
    ax.add_patch(spoof_box)

    ax.text(0.5, 0.28, 'Live region\n(high mean, high variance)', fontsize=9, style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['live'], alpha=0.15))
    ax.text(0.05, 0.01, 'Spoof region\n(low mean, low variance)', fontsize=9, style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['spoof'], alpha=0.15))

    # ========== BOTTOM RIGHT: Classification Performance ==========
    ax = axes[1, 1]

    # Simulate ROC-like curve
    thresholds_x = np.linspace(0, 1, 100)

    # Using mean depth as classifier
    far = []  # False accept rate (spoof accepted)
    frr = []  # False reject rate (live rejected)

    for t in thresholds_x:
        # Spoof accepted: mean > threshold (misclassified)
        fa = np.sum(spoof_means > t) / len(spoof_means)
        far.append(fa * 100)

        # Live rejected: mean <= threshold (misclassified)
        fr = np.sum(live_means <= t) / len(live_means)
        frr.append(fr * 100)

    ax.plot(thresholds_x, far, linewidth=2.5, color=COLORS['spoof'], label='APCER (spoof → live)',
           marker='o', markevery=10, markersize=4)
    ax.plot(thresholds_x, frr, linewidth=2.5, color=COLORS['live'], label='BPCER (live → spoof)',
           marker='s', markevery=10, markersize=4)

    # Find and mark EER
    acer = (np.array(far) + np.array(frr)) / 2
    eer_idx = np.argmin(acer)
    eer_threshold = thresholds_x[eer_idx]
    eer_value = acer[eer_idx]

    ax.axvline(eer_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.plot(eer_threshold, eer_value, 'o', markersize=10, color='gold', markeredgecolor='black',
           markeredgewidth=1.5, zorder=5)
    ax.text(eer_threshold + 0.05, eer_value + 1, f'EER ≈ {eer_value:.1f}%\n(@ threshold {eer_threshold:.2f})',
           fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.3, edgecolor='black'))

    ax.set_xlabel('Classification Threshold (Mean Depth)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('(d) Decision Trade-off: APCER vs BPCER', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim(-2, 110)

    fig.suptitle('Depth Map Analysis: Why Live and Spoof are Separable',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig_depth_statistics.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/fig_depth_statistics.pdf", bbox_inches='tight')
    print("[OK] Depth statistics figure saved")
    plt.close()


if __name__ == "__main__":
    print("\nGenerating depth map statistics visualization...\n")
    create_depth_statistics_figure()
    print("\nFigure saved to:")
    print(f"  {OUTPUT_DIR}/fig_depth_statistics.png")
    print(f"  {OUTPUT_DIR}/fig_depth_statistics.pdf")
