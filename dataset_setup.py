"""
dataset_setup.py — Auto Dataset Generator
==========================================
Creates a synthetic leaf dataset so you can train immediately
WITHOUT downloading anything.

If you already have real PlantVillage images:
  → Place them in  dataset/train/<ClassName>/
  → Re-run python train.py   (this script is skipped automatically)

Run:
    python dataset_setup.py

What it does:
  • Creates 15 class folders inside dataset/train/
  • Generates synthetic coloured leaf images (solid noise patches)
    — Good enough for a college mini-project demo
  • Real accuracy improvement requires real PlantVillage images
    (download free from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease)
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_DIR        = os.path.join("dataset", "train")
TEST_DIR         = os.path.join("dataset", "test")
IMAGES_PER_CLASS = 120      # Increase for better training (needs more time)
IMG_SIZE         = (224, 224)

# ── 15 Class definitions with colour hints for synthetic images ───────────────
CLASS_CONFIG = {
    "Tomato___Early_Blight":        {"base": (80,  130,  60),  "spot": (60,  40,  20)},
    "Tomato___Late_Blight":         {"base": (70,  120,  55),  "spot": (30,  20,  10)},
    "Tomato___Leaf_Mold":           {"base": (75,  125,  58),  "spot": (100, 80,  40)},
    "Tomato___Septoria_Leaf_Spot":  {"base": (85,  135,  65),  "spot": (200, 200, 190)},
    "Tomato___Healthy":             {"base": (60,  160,  50),  "spot": None},
    "Potato___Early_Blight":        {"base": (70,  115,  55),  "spot": (80,  50,  20)},
    "Potato___Late_Blight":         {"base": (65,  110,  50),  "spot": (40,  30,  15)},
    "Potato___Healthy":             {"base": (55,  150,  45),  "spot": None},
    "Corn___Common_Rust":           {"base": (90,  155,  50),  "spot": (180, 80,  30)},
    "Corn___Northern_Leaf_Blight":  {"base": (85,  150,  48),  "spot": (160, 140, 80)},
    "Corn___Healthy":               {"base": (50,  170,  40),  "spot": None},
    "Pepper___Bacterial_Spot":      {"base": (75,  130,  60),  "spot": (120, 60,  30)},
    "Pepper___Healthy":             {"base": (65,  160,  55),  "spot": None},
    "Apple___Apple_Scab":           {"base": (80,  140,  55),  "spot": (90,  65,  30)},
    "Apple___Healthy":              {"base": (60,  170,  50),  "spot": None},
}

CLASS_NAMES = list(CLASS_CONFIG.keys())


def make_synthetic_leaf(base_rgb, spot_rgb, img_size, has_disease):
    """
    Generate a synthetic leaf image.
    - Green ellipse on random-coloured background
    - If diseased: add dark irregular spots
    """
    bg_r = random.randint(180, 220)
    bg_g = random.randint(200, 240)
    bg_b = random.randint(170, 210)
    img  = Image.new("RGB", img_size, color=(bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # ── Draw a leaf-shaped ellipse ────────────────────────────────────────────
    pad   = random.randint(10, 30)
    r, g, b = base_rgb
    # Add noise to base colour so every image looks slightly different
    r = max(0, min(255, r + random.randint(-20, 20)))
    g = max(0, min(255, g + random.randint(-20, 20)))
    b = max(0, min(255, b + random.randint(-20, 20)))
    draw.ellipse([pad, pad, img_size[0]-pad, img_size[1]-pad], fill=(r, g, b))

    # ── Vein lines ────────────────────────────────────────────────────────────
    vein_col = (max(0,r-25), max(0,g-25), max(0,b-25))
    cx, cy   = img_size[0]//2, img_size[1]//2
    for _ in range(5):
        x1 = cx + random.randint(-20, 20)
        y1 = pad + random.randint(0, 20)
        x2 = cx + random.randint(-40, 40)
        y2 = img_size[1] - pad - random.randint(0, 20)
        draw.line([(x1, y1), (x2, y2)], fill=vein_col, width=2)

    # ── Disease spots ─────────────────────────────────────────────────────────
    if has_disease and spot_rgb is not None:
        num_spots = random.randint(5, 18)
        sr, sg, sb = spot_rgb
        for _ in range(num_spots):
            sx = random.randint(pad+10, img_size[0]-pad-10)
            sy = random.randint(pad+10, img_size[1]-pad-10)
            sw = random.randint(8, 28)
            sh = random.randint(8, 22)
            # Vary spot colour slightly
            sr2 = max(0, min(255, sr + random.randint(-15, 15)))
            sg2 = max(0, min(255, sg + random.randint(-15, 15)))
            sb2 = max(0, min(255, sb + random.randint(-15, 15)))
            draw.ellipse([sx-sw, sy-sh, sx+sw, sy+sh], fill=(sr2, sg2, sb2))

    # ── Slight blur for realism ────────────────────────────────────────────────
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # ── Add pixel noise ───────────────────────────────────────────────────────
    arr   = np.array(img, dtype=np.int16)
    noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)

    return img


def setup_dataset():
    """Create all class folders and populate with synthetic images."""

    print("\n" + "="*60)
    print("  DATASET SETUP — Generating Synthetic Leaf Images")
    print("="*60)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR,  exist_ok=True)

    total = 0
    for cls_name, cfg in CLASS_CONFIG.items():
        cls_dir  = os.path.join(TRAIN_DIR, cls_name)
        test_dir = os.path.join(TEST_DIR,  cls_name)
        os.makedirs(cls_dir,  exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        existing = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Skip if folder already has enough images (real or previously generated)
        if existing >= IMAGES_PER_CLASS:
            print(f"  ✅ {cls_name:40s} — {existing} images already present, skipping")
            total += existing
            continue

        # How many more we need to generate
        to_generate = IMAGES_PER_CLASS - existing
        has_disease = cfg["spot"] is not None

        for i in range(to_generate):
            img      = make_synthetic_leaf(cfg["base"], cfg["spot"], IMG_SIZE, has_disease)
            filename = f"synthetic_{existing+i:04d}.jpg"
            img.save(os.path.join(cls_dir, filename), "JPEG", quality=90)

        # Also generate 20% test images
        for i in range(max(1, IMAGES_PER_CLASS // 5)):
            img      = make_synthetic_leaf(cfg["base"], cfg["spot"], IMG_SIZE, has_disease)
            filename = f"test_{i:04d}.jpg"
            img.save(os.path.join(test_dir, filename), "JPEG", quality=90)

        total += IMAGES_PER_CLASS
        print(f"  ✅ {cls_name:40s} — {IMAGES_PER_CLASS} images generated")

    print(f"\n  Total images : {total}")
    print(f"  Classes      : {len(CLASS_CONFIG)}")
    print(f"  Train dir    : {TRAIN_DIR}")
    print(f"  Test dir     : {TEST_DIR}")
    print("\n  ✅ Dataset ready!  Next → python train.py\n")

    # Save class names for reference
    os.makedirs("models", exist_ok=True)
    with open(os.path.join("models", "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print(f"  Class names saved → models/class_names.json")


if __name__ == "__main__":
    setup_dataset()
