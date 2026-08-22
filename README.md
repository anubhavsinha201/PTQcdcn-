# Post-Training Quantization of CDCN++ for Face Anti-Spoofing on the Edge

Research code for the paper **"Post Training Quantization of Central Difference
Convolution Network for Facial Anti-Spoofing on Edge"** (Dr. Sishaj P. Simon,
Anubhav Sinha — National Institute of Technology, Tiruchirappalli).

The project implements the CDCN / CDCN++ face anti-spoofing architecture from
Yu et al., *"Searching Central Difference Convolutional Networks for Face
Anti-Spoofing"* (CVPR 2020, [arXiv:2003.04092](Research%20Papers/2003.04092v1.pdf)),
trains it with depth supervision on CelebA-Spoof, compresses it four different
ways with TensorFlow Lite post-training quantization (PTQ), and measures what
each compression method costs in presentation-attack-detection accuracy.

**Headline finding:** dynamic-range INT8 gives a **10.6× smaller model and 2.6×
faster CPU inference at zero accuracy loss**, while **full-integer INT8 collapses
the model** (ACER 0.05% → 21.45%, a 429× degradation). The collapse is not
gradual — it is binary. The proposed mechanism is catastrophic cancellation
inside the Central Difference Convolution: CDC subtracts two nearly-equal
quantities, and int8 activation quantization rounds both into the same bucket,
zeroing out exactly the micro-gradient signal the operator exists to capture.

---

## Results

Evaluated on a balanced 1,983-image held-out split of CelebA-Spoof
(988 live / 995 spoof), threshold calibrated per run to minimise ACER.
Source: [cdcn_quantized/full_comparison_metrics.md](cdcn_quantized/full_comparison_metrics.md).

| Variant | Params (M) | MACs (G) | Size (MB) | ACER (%) [95% CI] | EER (%) | ROC-AUC | Latency (ms) |
|---|---|---|---|---|---|---|---|
| Original (Keras, GPU) | 2.56 | 52.46 | 29.55 | 0.050 [0.000, 0.155] | 0.050 | 1.0000 | 75.5 (GPU) |
| FP32 (TFLite) | 2.56 | 52.46 | 10.88 | 0.050 [0.000, 0.155] | 0.050 | 1.0000 | 1015.5 (CPU) |
| **Dynamic-range INT8** | 2.56 | 52.46 | **2.80** | **0.050 [0.000, 0.155]** | 0.050 | 1.0000 | **386.1 (CPU)** |
| Float16 | 2.56 | 52.46 | 5.46 | 0.050 [0.000, 0.155] | 0.050 | 1.0000 | 1026.4 (CPU) |
| **Full INT8** | 2.56 | 52.46 | 2.84 | **21.450 [19.667, 23.210]** | 22.039 | 0.8629 | 272.6 (CPU) |

Operating points for the full-INT8 failure case: BPCER@APCER=1% is 62.0%
(vs. 0.1% for every other variant), APCER 12.35%, BPCER 30.55%.

**McNemar's test** (original vs. each variant, same samples):

| Comparison | Discordant (b, c) | p-value | Significant |
|---|---|---|---|
| Original vs. FP32 | (0, 0) | 1 | no |
| Original vs. Dynamic-range INT8 | (0, 0) | 1 | no |
| Original vs. Float16 | (0, 0) | 1 | no |
| Original vs. Full INT8 | (425, 0) | ~0 | **YES** |

Zero discordant pairs for the first three means those three variants make
*identical* per-image decisions to the unquantized model — the compression is
decision-equivalent, not merely similar. Full INT8 flips 425 decisions, always
in the wrong direction.

Two caveats worth stating up front:

- **ROC-AUC of 1.0000 and ACER of 0.05% are suspiciously perfect.** The
  train and validation splits both come from the single `test` split of
  `nguyenkhoa/celeba-spoof-for-face-antispoofing-test`, split 80/20 by
  `train_test_split(seed=42)`. There is no subject-disjoint guarantee: the same
  identity (and plausibly near-duplicate captures of the same spoof medium) can
  appear on both sides. These numbers characterise *quantization deltas*
  reliably — every variant sees the same split — but should not be read as
  cross-dataset generalisation.
- **CPU latency numbers are host-machine measurements**, single-threaded TFLite
  on a desktop CPU, not on a Raspberry Pi or phone. They are internally
  comparable, not an edge-device benchmark. (Note that FP32/Float16 TFLite on
  CPU is *slower* than the Keras GPU model by ~13×; the interesting comparison
  is TFLite-vs-TFLite.)

---

## Repository layout

```
.
├── CDCN_internship_project.ipynb   # ← the main artifact: model, training, eval, quantization
├── cdcn_internship.ipynb           # scratch notebook (CUDA path fix only)
├── cdcn_pytorch.py                 # standalone PyTorch reference implementation
├── model.py                        # Keras layer/model definitions extracted from the notebook
├── models.py                       # empty placeholder
│
├── cdcn_quantized/                 # quantization outputs + metrics
│   ├── cdcnpp_fp32.tflite
│   ├── cdcnpp_dynamic_range.tflite # ← the recommended deployment artifact
│   ├── cdcnpp_float16.tflite
│   ├── cdcnpp_full_int8.tflite     # kept as the documented failure case
│   ├── quantization_comparison.{csv,md}
│   ├── full_comparison_{metrics.csv,metrics.md,results.json}
│   ├── mcnemar_results.csv
│   └── fig_*.{png,pdf}
│
├── paper_figures/                  # publication figure set (PNG + PDF)
│   ├── INDEX.md                    # per-figure purpose, key finding, target section
│   ├── FIGURE_GUIDE.md             # long-form figure captions / interpretation notes
│   └── COMPLETE_SUMMARY.txt
│
├── cdcn_figures/                   # earlier training-run figures (loss, ROC, confusion matrix)
│
├── generate_*.py                   # figure generation scripts (see below)
├── real_inference_*.py             # run the quantized model on real CelebA-Spoof faces
├── verify_cuda.py, test_cuda_simple.py   # GPU environment checks
│
├── Research Papers/                # source papers (CDCN CVPR'20, FR-bypass survey)
├── crazystuff.{docx,pdf}           # paper draft
├── renooo.docx                     # paper draft (revision)
├── requirementssetup               # dependency list (see Environment)
└── .vscode/                        # CUDA env vars, run tasks, debug configs, CUDA_SETUP.md
```

---

## The model

### Central Difference Convolution (CDC)

The core operator, from Eq. 4 of the CDCN paper:

```
y(p₀) = Σ w(pₙ)·x(p₀+pₙ)  −  θ · x(p₀) · Σ w(pₙ)
         └── vanilla conv ──┘   └─ central difference ─┘
```

Implemented as a vanilla `Conv2D` minus a 1×1 convolution whose kernel is the
spatial sum of the vanilla kernel — mathematically equivalent, and it reuses one
weight tensor instead of two. `θ = 0.7` throughout (the paper's value).

Two invariants are asserted in the notebook:
- `θ = 0` must reduce exactly to the vanilla convolution.
- `θ = 1` on a constant input must produce ~0 in the interior (the two terms
  cancel), with non-zero values only at the padded border.

The PyTorch reference in [cdcn_pytorch.py](cdcn_pytorch.py) notes and fixes a typo
in the paper's own Fig. 9 snippet (`self.conv.weight` should be `self.vani.weight`).

### Architectures

Both take `(B, 256, 256, 3)` RGB and emit a `(B, 32, 32, 1)` facial depth map.

- **CDCN** (Table 1 baseline) — CDC stem → three CDC blocks (low/mid/high, each
  128→196→128) with max-pooling between → concat the three levels at 32×32 →
  3-layer CDC head.
- **CDCN++** (Fig. 5, NAS-discovered) — two-conv stem, NAS cells per level
  (`CDC_2_r` blocks with expansion ratios 1.6 / 1.2 / 1.4 at mid level),
  **MAFM** instead of plain concatenation, 2-layer head. 2.56M params, 52.46 GMACs.
  This is the model that is trained and quantized.

Only the *discovered* architecture is implemented — the PC-DARTS NAS search
procedure is not reproduced.

### MAFM (Multiscale Attention Fusion Module)

Spatial attention (`[avg-pool, max-pool] → conv → sigmoid`) applied separately to
the low/mid/high feature maps with kernel sizes **7 / 5 / 3**, then pooled to
32×32 and concatenated. The attention convolutions are deliberately **vanilla,
not CDC** — the paper's Table 3 ablation shows CDC hurts here, because spatial
attention needs global semantic context rather than local gradients.

### Auxiliary classification head (a deviation from the paper)

`CDCNpp` returns a dict `{'depth': (B,32,32,1), 'cls': (B,1)}`. The `cls` branch
is a `GlobalAveragePooling2D → Dense(1, sigmoid)` on the fused features.

This was added to fix a real failure: with depth supervision alone, liveness had
to be read out as "mean of the predicted depth map", which gives no direct
classification gradient, and the model collapsed to a constant input-independent
output. The auxiliary head supplies that gradient, and its sigmoid output — not
the depth mean — is the liveness score used in all reported metrics.

### Loss

```
L_total = L_depth + L_cls
L_depth = MSE(depth_true, depth_pred) + CDL(depth_true, depth_pred)
L_cls   = binary cross-entropy
```

`CDL` (contrastive depth loss) compares image gradients of the true and predicted
depth maps, so the network is penalised for getting the *shape* of the depth
surface wrong and not just its absolute values. Depth and classification losses
are equally weighted (1.0 / 1.0).

Note: the notebook's final `contrastive_depth_loss` uses **L1** on the gradient
difference; the earlier draft cell (and the PyTorch-side notes) use **L2**. The
L1 version is the one used in training.

---

## Data pipeline

**Dataset:** [`nguyenkhoa/celeba-spoof-for-face-antispoofing-test`](https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test)
(Hugging Face). It ships a single `test` split, which is divided 80/20 into
`ds_train_pool` / `ds_val`.

⚠️ **Label convention is inverted.** In the raw HF rows, `labels == 0` means
**live** and `labels == 1` means **spoof**. The project's internal convention is
the opposite (`1 = live`). All conversion goes through the single `get_label()`
helper — check it before writing any new evaluation code, because getting this
backwards silently inverts every metric.

**Class balancing:** CelebA-Spoof is ~70% spoof, so training draws up to
10,000 live + 10,000 spoof (auto-capped by whichever class runs out first, i.e.
live). A separate 300+300 *monitor* split, taken from the unused tail of the same
shuffled permutation, feeds `validation_data` during training — guaranteeing it
is disjoint from both the training slice and `ds_val`.

### Geometry-based depth targets

The depth-map ground truth is the part most likely to trip up a reader, so:

- **Spoof** → an all-zero 32×32 depth map (a print or screen has no 3D face).
- **Live** → a per-image depth target derived from **actual detected face
  geometry**: OpenCV Haar cascade for detection + the LBF facemark model for 68
  iBUG landmarks, combined with a fixed anatomical "protrusion toward camera"
  prior per landmark index (nose tip highest, jaw line lowest, etc.). Cached per
  dataset index on first access.

This replaced an earlier target that redrew a **random Gaussian blob on every
`__getitem__` call** — a different ground truth for the same image every epoch.
With no stable input→output mapping to learn, the network's optimal strategy was
to emit the average target and ignore the input entirely; this was the direct
cause of the "predicts live for everything" collapse in earlier runs. Determinism
of the target is the fix.

MediaPipe FaceLandmarker was the first choice (true 3D per-vertex depth) but was
abandoned: it segfaults natively in this WSL environment (GL/EGL context) and its
legacy `solutions` API conflicts with the protobuf version this TensorFlow install
requires.

Haar cascade and LBF model files are downloaded on first run into `~/cdcn_assets/`
(`cv2.data.haarcascades` is empty in this environment's `opencv-contrib-python`
install, so they are fetched from the OpenCV / GSOC2017 repos directly).

### Augmentation

Training generator only (`augment=True`); evaluation and monitoring generators
run clean. Horizontal flip + ±10° rotation + brightness/contrast jitter
(gain 0.85–1.15, bias ±0.15). **The flip is mirrored onto the depth target** —
otherwise a flipped face would be regressed against a now-wrong-sided depth map.

---

## Training

| Setting | Value | Why |
|---|---|---|
| Optimizer | AdamW, lr `1e-4`, weight decay `5e-5` | matches the CDCN paper's recipe |
| Batch size | 8 | 16 OOM'd — this GPU exposes ~5 GB to TF and a bs=16 backward pass tried to allocate 6+ GB |
| Epochs | 40 (upper bound) | EarlyStopping usually stops sooner |
| Callbacks | EarlyStopping(val_loss, patience 6, restore best), ReduceLROnPlateau(×0.5, patience 4), ModelCheckpoint | |
| Checkpoint | `cdcnpp_best.weights.h5` | **not committed to this folder** — see Known gaps |
| BatchNorm | momentum 0.9, ε 1e-5, He-normal init | |

---

## Evaluation protocol

Metrics follow **ISO/IEC 30107-3** presentation-attack-detection terminology:

- **APCER** — fraction of *spoof* samples wrongly accepted as live (security risk).
- **BPCER** — fraction of *live* samples wrongly rejected (usability cost).
- **ACER** — the mean of the two.
- **EER** — equal error rate.
- **BPCER@APCER=k%** — usability cost at a fixed security operating point; the
  metric that actually matters for deployment, and the one where full INT8 looks
  worst (62% of genuine users rejected at APCER=1%).

The decision threshold is **not** hardcoded: each evaluation run sweeps the ROC
curve and picks the point minimising ACER. 95% confidence intervals on ACER and
AUC are bootstrapped. Evaluation runs on a class-balanced subset of `ds_val`
(shuffled first — an earlier run's `select(range(2000))` grabbed a spoof-only
slice because the rows were ordered).

Inference batch size is 8, not 16: a bs=16 forward pass OOM'd after a long
training run because the GPU memory pool was fragmented by then.

---

## Quantization

Post-training quantization only. **QAT was considered and rejected**: `CDCNpp` is
a subclassed Keras model with a custom `CDC` layer, and `tfmot` targets
Functional/Sequential models with registered layer types, so QAT would require
rewriting the model *and* authoring a custom quantization config for `CDC`. PTQ
operates on the frozen inference graph regardless of how the model was built.

Four variants, all converted from the same SavedModel:

| Variant | What is quantized | Calibration |
|---|---|---|
| FP32 | nothing (fair CPU baseline) | — |
| Dynamic-range | int8 weights, float activations | none needed |
| Float16 | fp16 weights | none needed |
| Full INT8 | int8 weights **and** activations | 200 images from the **training pool only** |

The calibration set is drawn strictly from `ds_train_pool`, never from the
held-out `ds_val` used for reporting.

**Export gotcha worth remembering:** exporting via plain `tf.saved_model.save()`
on this subclassed model left a BatchNorm variable as an unresolved
`READ_VARIABLE` op, which made full-int8 conversion fail outright. Keras 3's
`Model.export()` — purpose-built for a fully-frozen inference-only SavedModel —
resolves it. That is what the notebook uses.

### Why full INT8 fails

CDC computes `vanilla_conv(x) − θ·centre_term(x)`, i.e. a difference of two
quantities of similar magnitude. Its output is the small residual. Under
int8 *activation* quantization, both operands are rounded onto the same coarse
grid; when they land in the same bucket, the residual quantizes to exactly zero
and the operator degenerates. Weight-only quantization (dynamic-range, float16)
leaves activations in float, so the subtraction stays exact — which is precisely
why those three variants are decision-identical to FP32 and full INT8 is not.

The practical takeaway for edge deployment: **use dynamic-range INT8.** It is the
smallest variant that preserves accuracy exactly (2.80 MB, 10.6× smaller than the
Keras checkpoint), and the fastest of the accuracy-preserving options.

---

## Figures

[paper_figures/INDEX.md](paper_figures/INDEX.md) and
[paper_figures/FIGURE_GUIDE.md](paper_figures/FIGURE_GUIDE.md) document each
figure's purpose, key finding, and intended paper section. Every figure exists as
both PNG (presentations) and PDF (submission).

| Figure | Generated by | Shows |
|---|---|---|
| Fig 0 — Dataset examples | `generate_dataset_samples.py` | Live vs. spoof CelebA-Spoof samples |
| Fig 0 — Real depth maps | `generate_depth_maps.py` | Real faces paired with predicted depth |
| Fig 1 — Quantization collapse | `generate_paper_figures.py`, `generate_publication_figures.py` | ACER bar chart + size/accuracy scatter (**main result**) |
| Fig 2 — Depth visualization | `generate_paper_figures.py` | Input → GT depth → predicted depth, live and spoof |
| Fig 3 — Catastrophic cancellation | `generate_paper_figures.py` | Mechanism diagram: FP32 preserves the 0.2 difference, INT8 collapses it to 0 |
| Fig 5 — Depth statistics | `generate_depth_statistics.py` | Mean/variance distributions (live ≈0.55, spoof ≈0.08), separability scatter |
| Fig X — Forest plot | `generate_forest_plot.py` | ACER point estimates with 95% bootstrap CI |
| ACTUAL_inference_* | `real_inference_{clean,large,quantized}.py` | Quantized model run on real CelebA-Spoof faces |

Colour palette is Okabe-Ito / colorblind-safe throughout
(live `#029E73` green, spoof `#CA0020` red, failure cases red).

⚠️ **Some figure scripts use synthetic stand-in data.**
`generate_dataset_samples_cached.py`, `generate_depth_maps_simplified.py`, and
`generate_depth_statistics.py` synthesise representative images/statistics when
the dataset or landmark stack is unavailable; `real_inference_fast.py` falls back
to synthetic faces. Use the non-`_simplified`/`_cached` variants
(`generate_dataset_samples.py`, `generate_depth_maps.py`,
`real_inference_quantized.py`, `real_inference_clean.py`, `real_inference_large.py`)
for anything that goes into the paper as real data.

---

## Environment

### Dependencies

From [requirementssetup](requirementssetup):

```
tensorflow>=2.21.0
keras>=3.15.0
opencv-contrib-python
datasets
scikit-learn
matplotlib
numpy
tqdm
```

Plus `scipy` (used by `generate_depth_maps_simplified.py`) and, for the PyTorch
reference implementation only, `torch`.

> Install `scikit-learn`, not `sklearn` — the latter is a deprecated PyPI
> meta-package and its install fails.

### GPU / CUDA

The first cell of the notebook prepends the pip-installed CUDA `ptxas` to `PATH`
**before** `import tensorflow`, and this is load-bearing, not cosmetic:

- The dev machine's RTX 5070 (Blackwell, compute capability 12.0a) needs a `ptxas`
  newer than 12.6.3 to JIT-compile all kernels. Without it, `model.fit()` fails
  outright on backward-conv kernels ("Autotuner could not compile any configs").
- The system CUDA 13 toolkit's `ptxas` is ABI-incompatible with the pip-installed
  CUDA 12.9 runtime libs (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`) and segfaults.
- The fix is to use the `ptxas` bundled with `nvidia-cuda-nvcc-cu12` (CUDA 12.9),
  matching the runtime libs. It must run before the first GPU op.

Verify the stack with:

```powershell
python verify_cuda.py       # env vars, driver, TF device list, cuDNN
python test_cuda_simple.py  # actual GPU matmul + timing
```

[.vscode/CUDA_SETUP.md](.vscode/CUDA_SETUP.md) documents the VS Code tasks
(`Ctrl+Shift+B`), debug configs, and GPU-monitoring commands.

---

## Reproducing

Order matters — the notebook cells are stateful and several later cells depend on
names defined earlier (`get_label`, `geometry_depth_map`, `_HAAR_PATH`,
`ds_train_pool`, `ds_val`, `model`, `history`).

1. **Verify the GPU stack** — `python verify_cuda.py`.
2. **Open `CDCN_internship_project.ipynb`** and run cells top to bottom:
   - cells 1–2: CUDA `PATH` fix, then TF import (do not reorder);
   - cells 3–12: CDC operator + invariant tests + CDCN / CDCN++ definitions;
   - cells 13–17: losses, dataset load, geometry depth targets, generator, training;
   - cells 20–23: validation metrics (APCER/BPCER/ACER, ROC) and loss curves;
   - cells 24–27: TFLite export, four-variant PTQ, comparison table + figure.
3. **Generate the paper figures** — run the `generate_*.py` scripts (they read
   `cdcn_quantized/full_comparison_results.json`, so run the quantization cells
   first) and the `real_inference_*.py` scripts for qualitative panels.

To run inference only, skip straight to
`cdcn_quantized/cdcnpp_dynamic_range.tflite` — `real_inference_clean.py` shows a
complete load-interpret-score loop, including `find_output_index()`, which
distinguishes the `cls` output (1 element) from the `depth` output (32×32) by
element count rather than by name or index, since TFLite does not preserve
output ordering.

---

## Known gaps and caveats

- **`cdcnpp_best.weights.h5` is not in this folder.** The quantization cells load
  it, so they cannot be re-run from scratch here without retraining. The `.tflite`
  files in `cdcn_quantized/` are the surviving trained artifacts.
- **`models.py` is empty**; [model.py](model.py) holds the Keras definitions but is
  a fragment lifted from the notebook — it has no imports and no `THETA` /
  `BN_MOMENTUM` / `CONV_INIT` definitions, so it will not import standalone.
  [cdcn_pytorch.py](cdcn_pytorch.py) *is* self-contained and runnable
  (`python cdcn_pytorch.py` runs a shape + parameter-count smoke test).
- **`cdcn_gpu_env/` is a broken/empty virtualenv** — it was created under WSL
  (`/usr/bin/python3.14`, POSIX `bin/` layout, no packages installed), while
  `.vscode/settings.json` and `tasks.json` point at
  `cdcn_gpu_env/Scripts/python.exe` (Windows layout), which does not exist. Use a
  system or conda interpreter, or recreate the venv on Windows.
- **`.vscode/settings.json` sets `TF_CUDA_COMPUTE_CAPABILITY=89`** (Ada), which
  does not match the RTX 5070 (12.0a) described in the notebook's CUDA comment.
- **`'''import dependancies '''.py`** is a scratch file whose entire contents are
  notes inside a docstring; the filename itself is a stray artifact.
- **Two notebooks exist.** `CDCN_internship_project.ipynb` is the real one;
  `cdcn_internship.ipynb` contains only the CUDA path-fix cell and a stored
  `SyntaxError`.
- **Latency tables differ slightly between runs** —
  `quantization_comparison.md` (e.g. FP32 1002.8 ms) and
  `full_comparison_metrics.md` (1015.5 ms) are separate measurement passes on a
  shared host; treat them as run-to-run variance, and prefer
  `full_comparison_metrics.md`, which is the more complete table.
- Absolute Windows paths are hardcoded in every `generate_*.py` and
  `real_inference_*.py` script; they need editing to run elsewhere.

---

## References

- Z. Yu et al., *Searching Central Difference Convolutional Networks for Face
  Anti-Spoofing*, CVPR 2020. [`Research Papers/2003.04092v1.pdf`](Research%20Papers/2003.04092v1.pdf)
- *Bypassing Facial Recognition Systems* — [`Research Papers/`](Research%20Papers/)
- ISO/IEC 30107-3 — presentation attack detection metrics (APCER / BPCER / ACER)
- CelebA-Spoof via Hugging Face: `nguyenkhoa/celeba-spoof-for-face-antispoofing-test`

