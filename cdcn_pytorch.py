"""CDCN and CDCN++ — face anti-spoofing architectures (PyTorch).

Reference: Yu et al., "Searching Central Difference Convolutional Networks for
Face Anti-Spoofing", CVPR 2020. arXiv:2003.04092v1.

Implements:
    CDC      Central Difference Convolution operator (Eq. 4 / Appendix A Fig. 9).
    CDCN     Baseline depth-supervised model (Table 1).
    CDCNpp   NAS-discovered model (Fig. 5) with MAFM.

Architecture only. No losses, no training loop, no NAS search procedure.
Smoke test at bottom: run as a script to verify shapes and parameter counts.

Note on Fig. 9: the paper's PyTorch snippet has a typo (`self.conv.weight`
where it should read `self.vani.weight`). Fixed below.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Hyperparameters ──────────────────────────────────────────────────────────
THETA = 0.7
BN_MOMENTUM = 0.1   # PyTorch convention (≡ TF momentum 0.9)
BN_EPS = 1e-5


# ── CDC operator ─────────────────────────────────────────────────────────────
class CDC(nn.Module):
    """Central Difference Convolution.

    From Eq. 4 of arXiv:2003.04092v1:
        y(p₀) = Σ w(pₙ)·x(p₀+pₙ)  −  θ · x(p₀) · Σ w(pₙ)
                 └── vanilla conv ──┘   └─ central difference term ─┘
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 theta: float = THETA):
        super().__init__()
        self.vani = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_vanilla = self.vani(x)
        if self.theta == 0.0:
            return out_vanilla
        # Sum over kernel spatial dims → (Cout, Cin, 1, 1); apply as a 1×1 conv.
        kernel_diff = self.vani.weight.sum(dim=(2, 3), keepdim=True)
        out_cd = F.conv2d(x, kernel_diff, stride=self.vani.stride, padding=0)
        return out_vanilla - self.theta * out_cd


def cdc_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """CDC → BN → ReLU. All CDCs in this paper use kernel=3, stride=1, padding=1."""
    return nn.Sequential(
        CDC(in_ch, out_ch),
        nn.BatchNorm2d(out_ch, eps=BN_EPS, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
    )


# ── CDC_2_r block (used in CDCN++ NAS-discovered cells) ─────────────────────
class CDC2r(nn.Module):
    """Two stacked CDCs that expand channels by ratio r then shrink back.

    Operation `CDC_2_r` in Fig. 3(c) and Fig. 5: in_ch → round(in_ch · r) → in_ch,
    each followed by BN-ReLU.
    """

    def __init__(self, in_ch: int, r: float):
        super().__init__()
        mid = round(in_ch * r)
        self.block = nn.Sequential(
            cdc_bn_relu(in_ch, mid),
            cdc_bn_relu(mid, in_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── MAFM (Eq. 6, Fig. 4) ─────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """Spatial attention: [avg-pool, max-pool] → vanilla conv → sigmoid mask.

    Uses *vanilla* Conv2d (not CDC) — the paper's ablation (Table 3) shows CDC
    is worse here because spatial attention needs global semantic cognition.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)               # (B, 1, H, W)
        mx = x.max(dim=1, keepdim=True).values          # (B, 1, H, W)
        attn = torch.cat([avg, mx], dim=1)              # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(attn))           # (B, 1, H, W)
        return x * attn


class MAFM(nn.Module):
    """Multiscale Attention Fusion Module.

    Per Table 3 best config: kernel sizes 7/5/3 for low/mid/high attention.
    """

    def __init__(self, target_hw: int = 32,
                 k_low: int = 7, k_mid: int = 5, k_high: int = 3):
        super().__init__()
        self.target_hw = target_hw
        self.attn_low = SpatialAttention(k_low)
        self.attn_mid = SpatialAttention(k_mid)
        self.attn_high = SpatialAttention(k_high)

    def forward(self, low: torch.Tensor, mid: torch.Tensor,
                high: torch.Tensor) -> torch.Tensor:
        low_r = F.adaptive_avg_pool2d(self.attn_low(low), self.target_hw)
        mid_r = F.adaptive_avg_pool2d(self.attn_mid(mid), self.target_hw)
        high_r = F.adaptive_avg_pool2d(self.attn_high(high), self.target_hw)
        return torch.cat([low_r, mid_r, high_r], dim=1)


# ── Init helper ──────────────────────────────────────────────────────────────
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ── CDCN baseline (Table 1) ──────────────────────────────────────────────────
class CDCN(nn.Module):
    """Baseline CDC network from Table 1 of arXiv:2003.04092v1.

    Input : (B, 3, 256, 256) RGB image.
    Output: (B, 1, 32, 32) facial depth map.
    """

    def __init__(self):
        super().__init__()
        self.stem = cdc_bn_relu(3, 64)

        self.low = nn.Sequential(
            cdc_bn_relu(64, 128),
            cdc_bn_relu(128, 196),
            cdc_bn_relu(196, 128),
        )
        self.mid = nn.Sequential(
            cdc_bn_relu(128, 128),
            cdc_bn_relu(128, 196),
            cdc_bn_relu(196, 128),
        )
        self.high = nn.Sequential(
            cdc_bn_relu(128, 128),
            cdc_bn_relu(128, 196),
            cdc_bn_relu(196, 128),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.head = nn.Sequential(
            cdc_bn_relu(384, 128),
            cdc_bn_relu(128, 64),
            CDC(64, 1),   # bare final CDC, no BN/ReLU
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                                  # 64  × 256 × 256
        low = self.pool(self.low(x))                      # 128 × 128 × 128
        mid = self.pool(self.mid(low))                    # 128 ×  64 ×  64
        high = self.pool(self.high(mid))                  # 128 ×  32 ×  32

        low_r = F.adaptive_avg_pool2d(low, 32)
        mid_r = F.adaptive_avg_pool2d(mid, 32)
        fused = torch.cat([low_r, mid_r, high], dim=1)    # 384 × 32 × 32

        return self.head(fused)                           #   1 × 32 × 32


# ── CDCN++ (Fig. 5) ──────────────────────────────────────────────────────────
class CDCNpp(nn.Module):
    """NAS-discovered CDC network from Fig. 5 of arXiv:2003.04092v1.

    Same I/O shape as CDCN. Differs in:
      • two-conv stem (3 → 64 → 128);
      • NAS-discovered cells at each level (see Fig. 5);
      • MAFM in place of simple concat for multi-level fusion;
      • two-layer head instead of three-layer.

    Only the *discovered* architecture is implemented — the NAS search procedure
    (PC-DARTS bi-level optimization over the 8-op space) is not.
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            cdc_bn_relu(3, 64),
            cdc_bn_relu(64, 128),
        )

        # Low-level cell: CDC → CDC.
        self.low_cell = nn.Sequential(
            cdc_bn_relu(128, 128),
            cdc_bn_relu(128, 128),
        )

        # Mid-level cell: CDC_2_1.6 → CDC_2_1.2 → CDC_2_1.4.
        self.mid_cell = nn.Sequential(
            CDC2r(128, r=1.6),
            CDC2r(128, r=1.2),
            CDC2r(128, r=1.4),
        )

        # High-level cell: CDC → CDC_2_1.2.
        self.high_cell = nn.Sequential(
            cdc_bn_relu(128, 128),
            CDC2r(128, r=1.2),
        )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mafm = MAFM(target_hw=32, k_low=7, k_mid=5, k_high=3)

        self.head = nn.Sequential(
            cdc_bn_relu(384, 128),
            CDC(128, 1),     # bare final CDC, no BN/ReLU
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                                  # 128 × 256 × 256
        low = self.pool(self.low_cell(x))                 # 128 × 128 × 128
        mid = self.pool(self.mid_cell(low))               # 128 ×  64 ×  64
        high = self.pool(self.high_cell(mid))             # 128 ×  32 ×  32
        fused = self.mafm(low, mid, high)                 # 384 ×  32 ×  32
        return self.head(fused)                           #   1 ×  32 ×  32


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 256, 256)

    for name, model in [("CDCN", CDCN()), ("CDCN++", CDCNpp())]:
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (2, 1, 32, 32), f"{name} bad shape: {tuple(y.shape)}"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:7s}: output {tuple(y.shape)}, params {n_params:,}")

    # CDC operator sanity: theta=0 must equal the underlying vanilla conv.
    cdc0 = CDC(8, 8, theta=0.0).eval()
    inp = torch.randn(1, 8, 16, 16)
    with torch.no_grad():
        assert torch.allclose(cdc0(inp), cdc0.vani(inp))
    print("CDC theta=0 identity check: OK")
