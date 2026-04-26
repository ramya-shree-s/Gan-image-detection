import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import kurtosis, skew
import base64
import os
import pickle

# ─────────────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'svm_model.pkl')
_model_data = None

def load_model():
    global _model_data
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            _model_data = pickle.load(f)
        print("✓ Trained SVM model loaded")
    else:
        _model_data = None
        print("⚠ No trained model found — run train.py first")

load_model()


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — 8 features total
# ─────────────────────────────────────────────

def extract_all_features(gray):
    """
    Extract 8 discriminative features from a grayscale image.
    Returns feature vector of length 8.
    """
    gray_f = gray.astype(np.float64)
    h, w = gray.shape

    # ── Feature 1 & 2: Frequency domain (DFT) ──
    dft = np.fft.fft2(gray_f)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)

    cy, cx = h // 2, w // 2
    center_mask = np.zeros_like(magnitude, dtype=bool)
    center_mask[cy-10:cy+10, cx-10:cx+10] = True
    peripheral = magnitude.copy()
    peripheral[center_mask] = 0

    flat = peripheral.flatten()
    top_n = np.sort(flat)[-50:]
    valid = flat[flat > 0]
    mean_top = float(np.mean(top_n)) if len(top_n) > 0 else 0.0
    mean_all = float(np.mean(valid)) if len(valid) > 0 else 1.0
    if np.isnan(mean_top) or np.isnan(mean_all):
        mean_top, mean_all = 0.0, 1.0

    # F1: Spectral peak ratio — GAN has high peaks
    f1_spectral_peak = float(np.clip(mean_top / (mean_all + 1e-8) / 100, 0, 1))

    # F2: High frequency energy ratio — GAN has more high freq content
    freq_magnitude = np.log1p(magnitude)
    center_energy = float(np.sum(freq_magnitude[cy-32:cy+32, cx-32:cx+32]))
    total_energy  = float(np.sum(freq_magnitude)) + 1e-8
    f2_hf_ratio   = float(np.clip(1.0 - center_energy / total_energy, 0, 1))

    # ── Feature 3 & 4: Noise residual (SRM filter) ──
    kernel = np.array([
        [-1,  2, -1],
        [ 2,  0,  2],
        [-1,  2, -1]
    ], dtype=np.float64) / 4.0
    residual = ndimage.convolve(gray_f, kernel)

    # F3: Global noise variance — GAN synthetic noise differs from camera noise
    f3_noise_var = float(np.clip(np.var(residual) / 500.0, 0, 1))

    # F4: Noise kurtosis — real camera noise is Gaussian (kurtosis~3), GAN is not
    res_flat = residual.flatten()
    kurt = float(kurtosis(res_flat))
    f4_kurtosis = float(np.clip(abs(kurt) / 20.0, 0, 1))

    # ── Feature 5 & 6: Patch inconsistency ──
    patch_size = 32
    patches_y = h // patch_size
    patches_x = w // patch_size
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

        # F5: Suspicious patch ratio
        f5_patch_ratio = float(suspicious / len(patch_vars))

        # F6: Patch variance std — how inconsistent are patches?
        # Real images: low std (all patches similar noise)
        # GAN images: high std (patches have very different noise levels)
        f6_patch_std = float(np.clip(np.std(patch_vars) / (np.mean(patch_vars) + 1e-8) / 5.0, 0, 1))
    else:
        f5_patch_ratio = 0.0
        f6_patch_std   = 0.0

    # ── Feature 7: Checkerboard artifact score ──
    # Detect periodic grid patterns by looking at autocorrelation
    # GAN checkerboard creates regular spikes in autocorrelation
    small = cv2.resize(gray, (64, 64)).astype(np.float64)
    small -= small.mean()
    autocorr = np.fft.ifft2(np.abs(np.fft.fft2(small))**2).real
    autocorr = np.fft.fftshift(autocorr)
    ac_h, ac_w = autocorr.shape
    ac_cy, ac_cx = ac_h // 2, ac_w // 2
    # Exclude center (DC)
    autocorr[ac_cy-3:ac_cy+3, ac_cx-3:ac_cx+3] = 0
    ac_max = float(np.max(np.abs(autocorr)))
    ac_mean = float(np.mean(np.abs(autocorr))) + 1e-8
    f7_checkerboard = float(np.clip(ac_max / ac_mean / 50.0, 0, 1))

    # ── Feature 8: Local Binary Pattern uniformity ──
    # Real images have natural texture — LBP histogram is smooth
    # GAN images have artificial texture — LBP histogram is spiky
    def lbp_variance(img_gray):
        lbp = np.zeros_like(img_gray, dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            shifted = np.roll(np.roll(img_gray, dy, axis=0), dx, axis=1)
            lbp += (img_gray >= shifted).astype(np.uint8)
        hist, _ = np.histogram(lbp, bins=9, range=(0, 8))
        hist = hist.astype(np.float64) / (hist.sum() + 1e-8)
        return float(np.std(hist))

    f8_lbp = float(np.clip(lbp_variance(gray) * 10, 0, 1))

    features = [f1_spectral_peak, f2_hf_ratio, f3_noise_var,
                f4_kurtosis, f5_patch_ratio, f6_patch_std,
                f7_checkerboard, f8_lbp]

    # Replace any NaN with 0
    features = [0.0 if (np.isnan(f) or np.isinf(f)) else f for f in features]

    return features, residual


# ─────────────────────────────────────────────
# 1. FREQUENCY DOMAIN VISUALIZATION
# ─────────────────────────────────────────────
def frequency_analysis(gray):
    dft = np.fft.fft2(gray.astype(np.float64))
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    log_magnitude = np.log1p(magnitude)
    vis = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    center_mask = np.zeros_like(magnitude, dtype=bool)
    center_mask[cy-10:cy+10, cx-10:cx+10] = True
    peripheral = magnitude.copy()
    peripheral[center_mask] = 0
    flat = peripheral.flatten()
    valid = flat[flat > 0]
    top_n = np.sort(flat)[-50:]
    mean_top = float(np.mean(top_n)) if len(top_n) > 0 else 0.0
    mean_all = float(np.mean(valid)) if len(valid) > 0 else 1.0
    artifact_score = float(np.clip(mean_top / (mean_all + 1e-8) / 100, 0, 1))
    peak_detected = artifact_score > 0.35

    return vis_color, artifact_score, peak_detected


# ─────────────────────────────────────────────
# 2. NOISE RESIDUAL VISUALIZATION
# ─────────────────────────────────────────────
def noise_analysis(gray):
    gray_f = gray.astype(np.float64)
    kernel = np.array([[-1,2,-1],[2,0,2],[-1,2,-1]], dtype=np.float64) / 4.0
    residual = ndimage.convolve(gray_f, kernel)
    residual_abs = np.abs(residual)
    vis = cv2.normalize(residual_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    variance = float(np.var(residual))
    normalized_variance = float(np.clip(variance / 500.0, 0, 1))
    inconsistency = "HIGH" if normalized_variance > 0.4 else ("MEDIUM" if normalized_variance > 0.2 else "LOW")
    return vis_color, normalized_variance, inconsistency, residual


# ─────────────────────────────────────────────
# 3. PATCH ANALYSIS VISUALIZATION
# ─────────────────────────────────────────────
def patch_analysis(gray, residual, patch_size=32):
    h, w = gray.shape
    patches_y = h // patch_size
    patches_x = w // patch_size
    variances = []
    suspicious_positions = []

    for i in range(patches_y):
        for j in range(patches_x):
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            variances.append(float(np.var(residual[y1:y2, x1:x2])))

    if not variances:
        return None, 0, 0, []

    median_var = np.median(variances)
    threshold  = median_var * 1.5
    heatmap    = np.zeros((h, w), dtype=np.float32)
    suspicious_count = 0
    patch_idx = 0

    for i in range(patches_y):
        for j in range(patches_x):
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            var = variances[patch_idx]
            heatmap[y1:y2, x1:x2] = float(np.clip(var / (threshold + 1e-8), 0, 2))
            if var > threshold:
                suspicious_count += 1
                vert  = "Top"    if i < patches_y // 3 else ("Bottom" if i > 2 * patches_y // 3 else "Center")
                horiz = "left"   if j < patches_x // 3 else ("right"  if j > 2 * patches_x // 3 else "center")
                region = f"{vert}-{horiz}"
                if region not in suspicious_positions:
                    suspicious_positions.append(region)
            patch_idx += 1

    heatmap_norm  = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    total_patches = patches_y * patches_x
    return heatmap_color, suspicious_count, total_patches, suspicious_positions[:4]


# ─────────────────────────────────────────────
# 4. CLASSIFICATION
# ─────────────────────────────────────────────
def classify(features):
    feat_arr = np.array([features])

    if _model_data is not None:
        svm    = _model_data['svm']
        scaler = _model_data['scaler']
        scaled = scaler.transform(feat_arr)
        pred   = svm.predict(scaled)[0]
        proba  = svm.predict_proba(scaled)[0]
        label      = "FAKE" if pred == 1 else "REAL"
        confidence = float(np.max(proba) * 100)
        raw_score  = float(proba[1])
    else:
        # Fallback: weighted average of 8 features
        weights   = [0.15, 0.10, 0.15, 0.10, 0.15, 0.15, 0.10, 0.10]
        raw_score = float(np.clip(sum(f * w for f, w in zip(features, weights)), 0, 1))
        if raw_score > 0.5:
            label      = "FAKE"
            confidence = 50 + (raw_score - 0.5) * 100
        else:
            label      = "REAL"
            confidence = 50 + (0.5 - raw_score) * 100
        confidence = float(np.clip(confidence, 51, 99))

    return label, round(confidence, 1), round(raw_score, 3)


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def img_to_b64(img_array, upload_folder, filename):
    path = os.path.join(upload_folder, filename)
    cv2.imwrite(path, img_array)
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def analyze_image(filepath):
    upload_folder = os.path.dirname(filepath)

    img = cv2.imread(filepath)
    if img is None:
        return {'error': 'Could not read image'}

    img  = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract all 8 features + residual
    features, residual = extract_all_features(gray)

    # Visualizations
    freq_vis,  artifact_score, peak_detected = frequency_analysis(gray)
    noise_vis, noise_variance, inconsistency, _ = noise_analysis(gray)
    heatmap_vis, suspicious_count, total_patches, regions = patch_analysis(gray, residual)

    suspicious_ratio = suspicious_count / max(total_patches, 1)

    # Classification using all 8 features
    label, confidence, raw_score = classify(features)

    freq_b64    = img_to_b64(freq_vis,  upload_folder, 'freq.png')
    noise_b64   = img_to_b64(noise_vis, upload_folder, 'noise.png')
    input_b64   = img_to_b64(img,       upload_folder, 'input_resized.png')
    heatmap_b64 = img_to_b64(heatmap_vis, upload_folder, 'heatmap.png') if heatmap_vis is not None else None

    return {
        'label':      label,
        'confidence': confidence,
        'raw_score':  raw_score,
        'frequency': {
            'peak_detected': peak_detected,
            'artifact_score': round(artifact_score, 3),
            'image_b64': freq_b64
        },
        'noise': {
            'inconsistency': inconsistency,
            'srm_variance':  round(noise_variance, 3),
            'image_b64':     noise_b64
        },
        'patches': {
            'suspicious_count': suspicious_count,
            'total_patches':    total_patches,
            'regions':          regions if regions else ['None detected'],
            'heatmap_b64':      heatmap_b64
        },
        'input_b64': input_b64
    }