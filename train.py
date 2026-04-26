"""
train.py — GAN Detector Training Script
========================================
Uses the SAME 8 features as analyzer.py so model is always compatible.

Run order:
  python generate_dataset.py
  python train.py
  python app.py
"""

import os
import numpy as np
import cv2
import pickle
from scipy import ndimage
from scipy.stats import kurtosis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

REAL_DIR   = 'dataset/real'
FAKE_DIR   = 'dataset/fake'
MODEL_PATH = 'models/svm_model.pkl'
IMG_SIZE   = (256, 256)

os.makedirs('models', exist_ok=True)


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — same 8 features as analyzer.py
# IMPORTANT: Keep this in sync with analyzer.py always
# ─────────────────────────────────────────────
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img  = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape

    # ── F1 & F2: Frequency domain (DFT) ──
    dft       = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)

    cy, cx = h // 2, w // 2
    center_mask = np.zeros_like(magnitude, dtype=bool)
    center_mask[cy-10:cy+10, cx-10:cx+10] = True
    peripheral = magnitude.copy()
    peripheral[center_mask] = 0

    flat     = peripheral.flatten()
    top_n    = np.sort(flat)[-50:]
    valid    = flat[flat > 0]
    mean_top = float(np.mean(top_n)) if len(top_n) > 0 else 0.0
    mean_all = float(np.mean(valid)) if len(valid) > 0 else 1.0
    if np.isnan(mean_top) or np.isnan(mean_all):
        mean_top, mean_all = 0.0, 1.0

    f1_spectral_peak = float(np.clip(mean_top / (mean_all + 1e-8) / 100, 0, 1))

    freq_magnitude = np.log1p(magnitude)
    center_energy  = float(np.sum(freq_magnitude[cy-32:cy+32, cx-32:cx+32]))
    total_energy   = float(np.sum(freq_magnitude)) + 1e-8
    f2_hf_ratio    = float(np.clip(1.0 - center_energy / total_energy, 0, 1))

    # ── F3 & F4: Noise residual (SRM filter) ──
    kernel = np.array([
        [-1, 2, -1],
        [ 2, 0,  2],
        [-1, 2, -1]
    ], dtype=np.float64) / 4.0
    residual = ndimage.convolve(gray, kernel)

    var = np.var(residual)
    f3_noise_var = float(np.clip(var / 500.0, 0, 1)) if not np.isnan(var) else 0.0

    kurt = float(kurtosis(residual.flatten()))
    f4_kurtosis = float(np.clip(abs(kurt) / 20.0, 0, 1))

    # ── F5 & F6: Patch inconsistency ──
    patch_size = 32
    patches_y  = h // patch_size
    patches_x  = w // patch_size
    patch_vars = []

    for i in range(patches_y):
        for j in range(patches_x):
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            patch_vars.append(float(np.var(residual[y1:y2, x1:x2])))

    if patch_vars:
        median_var = np.median(patch_vars)
        threshold  = median_var * 1.5
        suspicious = sum(1 for v in patch_vars if v > threshold)
        f5_patch_ratio = float(suspicious / len(patch_vars))
        f6_patch_std   = float(np.clip(np.std(patch_vars) / (np.mean(patch_vars) + 1e-8) / 5.0, 0, 1))
    else:
        f5_patch_ratio = 0.0
        f6_patch_std   = 0.0

    # ── F7: Checkerboard / autocorrelation artifact ──
    small    = cv2.resize(gray.astype(np.uint8), (64, 64)).astype(np.float64)
    small   -= small.mean()
    autocorr = np.fft.ifft2(np.abs(np.fft.fft2(small))**2).real
    autocorr = np.fft.fftshift(autocorr)
    ac_h, ac_w = autocorr.shape
    ac_cy, ac_cx = ac_h // 2, ac_w // 2
    autocorr[ac_cy-3:ac_cy+3, ac_cx-3:ac_cx+3] = 0
    ac_max  = float(np.max(np.abs(autocorr)))
    ac_mean = float(np.mean(np.abs(autocorr))) + 1e-8
    f7_checkerboard = float(np.clip(ac_max / ac_mean / 50.0, 0, 1))

    # ── F8: LBP texture uniformity ──
    def lbp_variance(img_gray):
        ig = img_gray.astype(np.float64)
        lbp = np.zeros_like(ig, dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            shifted = np.roll(np.roll(ig, dy, axis=0), dx, axis=1)
            lbp += (ig >= shifted).astype(np.uint8)
        hist, _ = np.histogram(lbp, bins=9, range=(0, 8))
        hist = hist.astype(np.float64) / (hist.sum() + 1e-8)
        return float(np.std(hist))

    f8_lbp = float(np.clip(lbp_variance(gray.astype(np.uint8)) * 10, 0, 1))

    features = [f1_spectral_peak, f2_hf_ratio, f3_noise_var,
                f4_kurtosis, f5_patch_ratio, f6_patch_std,
                f7_checkerboard, f8_lbp]

    # Replace NaN / Inf with 0
    features = [0.0 if (np.isnan(f) or np.isinf(f)) else f for f in features]
    return features


# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset():
    X, y = [], []
    supported = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    print(f"\n[1/4] Loading REAL images from '{REAL_DIR}' ...")
    real_files = [f for f in os.listdir(REAL_DIR) if f.lower().endswith(supported)]
    if not real_files:
        print(f"  ERROR: No images found. Run: python generate_dataset.py")
        return None, None

    for i, fname in enumerate(real_files):
        features = extract_features(os.path.join(REAL_DIR, fname))
        if features:
            X.append(features)
            y.append(0)  # 0 = REAL
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(real_files)} done...")
    print(f"  ✓ {sum(1 for l in y if l==0)} real images loaded")

    print(f"\n[2/4] Loading FAKE images from '{FAKE_DIR}' ...")
    fake_files = [f for f in os.listdir(FAKE_DIR) if f.lower().endswith(supported)]
    if not fake_files:
        print(f"  ERROR: No images found. Run: python generate_dataset.py")
        return None, None

    for i, fname in enumerate(fake_files):
        features = extract_features(os.path.join(FAKE_DIR, fname))
        if features:
            X.append(features)
            y.append(1)  # 1 = FAKE
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(fake_files)} done...")
    print(f"  ✓ {sum(1 for l in y if l==1)} fake images loaded")

    X = np.array(X)
    y = np.array(y)

    # Remove NaN rows
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]
    print(f"  Total after cleanup: {len(X)} samples")
    return X, y


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train():
    print("=" * 50)
    print("  GAN DETECTOR — TRAINING PIPELINE")
    print("  Features: 8 (DFT + SRM + Patches + LBP)")
    print("=" * 50)

    if not os.path.exists(REAL_DIR) or not os.path.exists(FAKE_DIR):
        print("\n ERROR: Run python generate_dataset.py first")
        return

    X, y = load_dataset()
    if X is None or len(X) < 10:
        print("\n ERROR: Not enough images. Run: python generate_dataset.py")
        return

    print(f"\n[3/4] Training SVM on {len(X)} samples ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train_sc, y_train)

    y_pred   = svm.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"\n  ✓ Training complete!")
    print(f"  Test Accuracy: {accuracy:.1f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=['REAL','FAKE'])}")

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'svm': svm, 'scaler': scaler}, f)

    print(f"[4/4] Model saved → {MODEL_PATH}")
    print("\n  Now run: python app.py")
    print("=" * 50)


if __name__ == '__main__':
    train()