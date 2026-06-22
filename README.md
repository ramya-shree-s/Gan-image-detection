 🕵️ GAN Image Detection

A multi-stage forensic pipeline for detecting GAN-synthesized images and attributing them to their source generator

Frequency analysis · Patch inconsistency · Noise residuals · Ensemble classification


Overview

This project detects whether an image is a real photograph or **GAN/diffusion-synthesized**, and attempts to attribute fake images to their generator architecture (StyleGAN2, ProGAN, CycleGAN, etc.).

Instead of a single black-box CNN, it uses a **three-branch forensic pipeline** that mirrors how digital forensics experts actually examine synthetic images — by inspecting frequency artifacts, local texture coherence, and sensor noise patterns in parallel, then fusing the evidence for a final decision.

```
┌─────────────┐
│ Input Image │  Real or GAN-synthesized
└──────┬──────┘
       ▼
┌─────────────────────┐
│ STAGE 1              │
│ Feature Extraction   │  Spatial + frequency + noise features
└──────┬───────────────┘
       ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2 — Parallel Analysis                               │
├───────────────────┬───────────────────┬───────────────────┤
│ Frequency Analysis │ Patch Inconsistency│ Noise Residual    │
│ FFT spectral       │ Local texture       │ PRNU mismatch     │
│ artifacts          │ coherence           │ Sensor pattern    │
│ GAN checkerboard    │ Boundary artifacts  │ check             │
│ peaks               │                     │                   │
└───────────────────┴───────────────────┴───────────────────┘
       ▼
┌──────────────────────┐
│ STAGE 3               │
│ Classifier / Fusion   │  SVM / Random Forest / Ensemble
│ Feature vector fusion │
└──────┬────────────────┘
       ▼
┌───────────────────┐   ┌──────────────────┐
│ Real / Fake        │   │ Generator ID      │
│ Binary detection    │   │ Attribution label │
└───────────────────┘   └──────────────────┘
       ▼
┌────────────────────────────────────────────┐
│ Evaluation: AUC · Accuracy · EER ·          │
│ Cross-generator generalization              │
└────────────────────────────────────────────┘
```

## Why this approach?

| Forensic cue | What it catches |
|---|---|
| **Frequency analysis** | Transposed-convolution checkerboard patterns, periodic spectral peaks, abnormal radial energy decay that pure CNNs leave behind |
| **Patch inconsistency** | Texture/GLCM discontinuities, boundary seams, unnatural local-variance smoothness from patch-based or tiled generation |
| **Noise residual** | Missing camera PRNU fingerprint, unnatural spatial noise uniformity — real sensors leave noise real GANs don't replicate |

Fusing all three into one feature vector before classification is more robust to *unseen* generators than any single cue alone — which is also why **cross-generator generalization** is tracked as a first-class metric.

## Project Structure

```
gan_detection/
├── src/
│   ├── features/
│   │   ├── frequency_analysis.py     # FFT spectral artifacts, checkerboard detection
│   │   ├── patch_inconsistency.py    # Local texture coherence, boundary artifacts
│   │   └── noise_residual.py         # PRNU mismatch, sensor pattern check
│   ├── models/
│   │   ├── classifier.py             # SVM / RF / Ensemble + generator attribution
│   │   └── fusion.py                 # Feature vector fusion, scaling, PCA
│   ├── utils/
│   │   ├── image_loader.py           # Image loading & preprocessing
│   │   ├── evaluation.py             # AUC, Accuracy, EER, cross-generator metrics
│   │   └── visualizer.py             # FFT/noise/edge visual explanations
│   ├── pipeline.py                   # End-to-end inference pipeline (CLI)
│   └── train.py                      # Training + evaluation script (CLI)
├── configs/
│   └── config.yaml                   # All pipeline/feature/classifier hyperparameters
├── tests/
│   └── test_pipeline.py              # Unit tests for every module
├── notebooks/
│   └── demo.ipynb                    # End-to-end walkthrough on synthetic data
├── data/
│   ├── real/                         # Put real images here
│   └── fake/                         # Put GAN images here
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/gan-image-detection.git
cd gan-image-detection
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, OpenCV, scikit-learn, scikit-image, PyWavelets, SciPy.

## Quick Start

### 1. Prepare your data

```
data/
├── real/   # real photographs (.jpg, .png, ...)
└── fake/   # GAN/diffusion-generated images
```

### 2. Train

```bash
python src/train.py train \
  --real_dir data/real \
  --fake_dir data/fake \
  --config configs/config.yaml \
  --cross_validate
```

This extracts features, fits the fusion/normalization step, trains the ensemble classifier, evaluates on a held-out split, and saves everything to `saved_models/`.

### 3. Run inference on a new image

```bash
python src/pipeline.py --image path/to/image.jpg --model saved_models/
```

Sample output:

```
========================================
  Image:       path/to/image.jpg
  Decision:    Fake
  Confidence:  94.32%
  P(Real):     0.0568
  P(Fake):     0.9432
  Generator:   StyleGAN2 (78.10%)
  Time:        842.3 ms
========================================
```

A visual breakdown (FFT spectrum, noise residual, edge map, prediction scores) is saved to `results/<image>_analysis.png`.

### 4. Evaluate on a labeled test set

```bash
python src/train.py eval --test_dir data/test --model saved_models/
```

Outputs ROC curve, confusion matrix, score distribution, and (if multiple generators are present) a cross-generator generalization chart — all saved to `results/`.

### 5. Try it without any dataset

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook generates synthetic real/fake images on the fly so you can see the full pipeline run end-to-end with zero setup.

## Configuration

All hyperparameters live in [`configs/config.yaml`](configs/config.yaml) — image size, patch size/stride, PRNU sigma, wavelet levels, classifier type (`svm` / `rf` / `ensemble`), and known generator labels for attribution. Edit this file rather than the source to tune behavior.

## Evaluation Metrics

| Metric | Description |
|---|---|
| **AUC** | Area under the ROC curve |
| **Accuracy** | Overall binary classification accuracy |
| **EER** | Equal Error Rate (where false-accept rate = false-reject rate) |
| **Cross-generator generalization** | Per-generator AUC/EER/accuracy breakdown to test robustness against unseen architectures|
```
