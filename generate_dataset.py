"""
generate_dataset.py — Dataset Generator
=========================================
REAL images  → Taken from actual built-in photo library (faces, nature, cats,
                coffee, rockets, space etc.) + augmentation to get 300 images.
                No internet needed. These are ACTUAL photographs.

FAKE images  → Same real photos with GAN artifacts injected on top
                (checkerboard, patch noise, frequency peaks, boundary lines)
                so the SVM learns the exact difference.

Run order:
  python generate_dataset.py
  python train.py
  python app.py
"""

import numpy as np
import cv2
import os
from skimage import data as skdata
from sklearn.datasets import load_sample_images

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REAL_DIR = 'dataset/real'
FAKE_DIR = 'dataset/fake'
IMG_SIZE = (256, 256)
NUM_EACH = 300

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — COLLECT ALL REAL SOURCE PHOTOS
# ─────────────────────────────────────────────
def collect_real_sources():
    sources = []

    # Color photos from skimage
    color_loaders = [
        ('astronaut',            skdata.astronaut),
        ('cat',                  skdata.cat),
        ('chelsea',              skdata.chelsea),
        ('coffee',               skdata.coffee),
        ('hubble_deep_field',    skdata.hubble_deep_field),
        ('immunohistochemistry', skdata.immunohistochemistry),
        ('retina',               skdata.retina),
        ('rocket',               skdata.rocket),
        ('colorwheel',           skdata.colorwheel),
    ]

    for name, loader in color_loaders:
        try:
            img = loader()
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            img_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            sources.append((name, img_bgr))
            print(f"  ✓ {name:30s} {img_bgr.shape}")
        except Exception as e:
            print(f"  ✗ {name:30s} {e}")

    # Grayscale photos converted to color
    gray_loaders = [
        ('moon',   skdata.moon),
        ('camera', skdata.camera),
        ('coins',  skdata.coins),
        ('clock',  skdata.clock),
        ('brick',  skdata.brick),
        ('grass',  skdata.grass),
        ('gravel', skdata.gravel),
    ]

    for name, loader in gray_loaders:
        try:
            img = loader()
            if img.dtype == bool:
                img = img.astype(np.uint8) * 255
            elif img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            sources.append((name, img_color))
            print(f"  ✓ {name:30s} {img_color.shape}")
        except Exception as e:
            print(f"  ✗ {name:30s} {e}")

    # sklearn sample photos (china temple + flower)
    try:
        sk_imgs = load_sample_images()
        names = ['china_temple', 'flower']
        for i, img in enumerate(sk_imgs.images):
            nm = names[i] if i < len(names) else f'sklearn_{i}'
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            sources.append((nm, img_bgr))
            print(f"  ✓ {nm:30s} {img_bgr.shape}")
    except Exception as e:
        print(f"  ✗ sklearn sample images: {e}")

    print(f"\n  Total source photos loaded: {len(sources)}")
    return sources


# ─────────────────────────────────────────────
# STEP 2 — AUGMENT REAL PHOTO
# ─────────────────────────────────────────────
def augment_image(img, seed):
    np.random.seed(seed)
    h, w = img.shape[:2]

    # Random crop
    crop_frac = np.random.uniform(0.5, 0.95)
    ch = int(h * crop_frac)
    cw = int(w * crop_frac)
    y0 = np.random.randint(0, max(1, h - ch))
    x0 = np.random.randint(0, max(1, w - cw))
    img = img[y0:y0+ch, x0:x0+cw]

    # Resize
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    # Random flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Random rotation
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1.0)
    img = cv2.warpAffine(img, M, IMG_SIZE, borderMode=cv2.BORDER_REFLECT)

    # Brightness & contrast
    alpha = np.random.uniform(0.7, 1.4)
    beta  = np.random.uniform(-30, 30)
    img = np.clip(img.astype(np.float64) * alpha + beta, 0, 255).astype(np.uint8)

    # Color channel variation
    for c in range(3):
        shift = np.random.uniform(-15, 15)
        img[:, :, c] = np.clip(img[:, :, c].astype(np.float64) + shift, 0, 255).astype(np.uint8)

    # Consistent Gaussian noise (real camera sensor)
    noise_level = np.random.uniform(2, 8)
    noise = np.random.normal(0, noise_level, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return img


# ─────────────────────────────────────────────
# STEP 3 — INJECT GAN ARTIFACTS
# ─────────────────────────────────────────────
def inject_gan_artifacts(real_img, seed):
    np.random.seed(seed + 10000)
    img = real_img.astype(np.float64)
    h, w = img.shape[:2]

    # Artifact 1: Checkerboard (transposed convolution)
    block    = np.random.choice([4, 8, 16])
    strength = np.random.uniform(18, 45)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if (y // block + x // block) % 2 == 0:
                img[y:y+block, x:x+block] += strength

    # Artifact 2: Inconsistent patch noise (key GAN signature)
    patch_size = 32
    for py in range(0, h, patch_size):
        for px in range(0, w, patch_size):
            ph = min(patch_size, h - py)
            pw = min(patch_size, w - px)
            patch_noise = np.random.uniform(1, 35)
            img[py:py+ph, px:px+pw] += np.random.normal(0, patch_noise, (ph, pw, 3))

    # Artifact 3: Periodic frequency peaks
    freq = np.random.choice([8, 16, 32])
    amp  = np.random.uniform(12, 30)
    for c in range(3):
        xs = np.sin(np.linspace(0, 2 * np.pi * (w // freq), w)) * amp
        ys = np.sin(np.linspace(0, 2 * np.pi * (h // freq), h)) * amp
        img[:, :, c] += np.outer(ys, xs)

    # Artifact 4: Hard boundary lines at patch borders
    bs = np.random.uniform(15, 35)
    for py in range(patch_size, h, patch_size):
        img[py-1:py+1, :] += np.random.uniform(-bs, bs, (2, w, 3))
    for px in range(patch_size, w, patch_size):
        img[:, px-1:px+1] += np.random.uniform(-bs, bs, (h, 2, 3))

    # Artifact 5: Color channel misalignment
    shift = np.random.randint(1, 4)
    img[:, :, 0] = np.roll(img[:, :, 0], shift, axis=1)
    img[:, :, 2] = np.roll(img[:, :, 2], -shift, axis=0)

    return np.clip(img, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def generate():
    print("=" * 55)
    print("  GAN DETECTOR — DATASET GENERATOR")
    print("  Using REAL photographs (no internet needed)")
    print("=" * 55)

    print("\n[1/3] Loading real source photographs ...")
    sources = collect_real_sources()

    if not sources:
        print("ERROR: No source images found.")
        return

    # Generate REAL images
    print(f"\n[2/3] Generating {NUM_EACH} REAL training images ...")
    print("      (augmented crops of actual photographs)")
    for i in range(NUM_EACH):
        src_img = sources[i % len(sources)][1]
        aug = augment_image(src_img, seed=i)
        cv2.imwrite(os.path.join(REAL_DIR, f'real_{i:04d}.png'), aug)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{NUM_EACH} done ...")
    print(f"  ✓ {NUM_EACH} real images saved to '{REAL_DIR}/'")

    # Generate FAKE images
    print(f"\n[3/3] Generating {NUM_EACH} FAKE training images ...")
    print("      (same photos + GAN artifacts injected)")
    for i in range(NUM_EACH):
        src_img = sources[i % len(sources)][1]
        aug  = augment_image(src_img, seed=i)
        fake = inject_gan_artifacts(aug, seed=i)
        cv2.imwrite(os.path.join(FAKE_DIR, f'fake_{i:04d}.png'), fake)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{NUM_EACH} done ...")
    print(f"  ✓ {NUM_EACH} fake images saved to '{FAKE_DIR}/'")

    print(f"\n  Dataset ready: {NUM_EACH} real + {NUM_EACH} fake")
    print(f"  Next step → python train.py")
    print("=" * 55)


if __name__ == '__main__':
    generate()