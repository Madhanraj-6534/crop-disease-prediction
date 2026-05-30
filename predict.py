"""
predict.py
==========
Load the saved CNN model and predict the disease from a leaf image.

Usage (standalone):
    python predict.py path/to/leaf_image.jpg

Also importable — used by app.py.
"""
TAMIL_MAP = {
    "Tomato — Early Blight": "தக்காளி - ஆரம்ப இலை நோய்",
    "Tomato — Late Blight": "தக்காளி - கடைசி இலை நோய்",
    "Healthy": "ஆரோக்கியமானது",
    "Moderate": "மிதமானது",
    "High": "அதிகம்"
}

import os
import sys
import json
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "crop_disease_model.h5")
CLASS_JSON  = os.path.join("models", "class_names.json")
IMG_SIZE    = (224, 224)

# ─────────────────────────────────────────────────────────────────────────────
# Disease solutions dictionary
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    # ── Tomato ──────────────────────────────────────────────────────────────
    "Tomato___Early_Blight": {
        "display_name": "Tomato — Early Blight",
        "severity":     "Moderate",
        "cause":        "Fungus: Alternaria solani",
        "symptoms":     "Dark brown concentric rings (target-board pattern) on older leaves; "
                        "yellowing around lesions; premature leaf drop.",
        "treatment": [
            "Remove and destroy infected leaves immediately.",
            "Apply copper-based fungicide (Copper Oxychloride) every 7–10 days.",
            "Use Mancozeb 75 WP @ 2.5 g/litre as a protective spray.",
            "Ensure proper plant spacing for air circulation.",
            "Avoid overhead irrigation; water at the base.",
            "Rotate crops — do not plant tomatoes in the same spot consecutively.",
        ],
        "prevention":   "Use certified disease-resistant seeds; mulch around plants.",
    },
    "Tomato___Late_Blight": {
        "display_name": "Tomato — Late Blight",
        "severity":     "High",
        "cause":        "Oomycete: Phytophthora infestans",
        "symptoms":     "Water-soaked greenish-grey lesions rapidly turning brown-black; "
                        "white mouldy growth on leaf undersides in humid conditions.",
        "treatment": [
            "Destroy severely infected plants immediately to stop spread.",
            "Apply Metalaxyl + Mancozeb (Ridomil Gold) @ 2 g/litre.",
            "Spray Chlorothalonil (Bravo) as a protective fungicide.",
            "Avoid working in the field when foliage is wet.",
            "Drain standing water near plants.",
        ],
        "prevention":   "Plant resistant varieties (e.g., Mountain Magic); avoid high humidity.",
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato — Leaf Mold",
        "severity":     "Moderate",
        "cause":        "Fungus: Passalora fulva (syn. Cladosporium fulvum)",
        "symptoms":     "Pale yellow spots on upper leaf surface; olive-green to grey "
                        "velvety mould on lower surface.",
        "treatment": [
            "Improve greenhouse ventilation to reduce humidity below 85%.",
            "Apply Chlorothalonil or Copper-based fungicide.",
            "Remove and bin infected leaves — do not compost.",
            "Avoid wetting foliage during irrigation.",
        ],
        "prevention":   "Use resistant hybrids; space plants adequately.",
    },
    "Tomato___Septoria_Leaf_Spot": {
        "display_name": "Tomato — Septoria Leaf Spot",
        "severity":     "Moderate",
        "cause":        "Fungus: Septoria lycopersici",
        "symptoms":     "Small, circular spots with dark brown border and light grey centre; "
                        "tiny black specks (pycnidia) visible in spot centres.",
        "treatment": [
            "Remove infected lower leaves at first sign.",
            "Spray Mancozeb 75 WP @ 2 g/litre every 10 days.",
            "Apply Copper Hydroxide as an alternative.",
            "Avoid overhead watering.",
            "Stake and prune plants for better airflow.",
        ],
        "prevention":   "Mulch soil surface; practice 3-year crop rotation.",
    },
    "Tomato___Healthy": {
        "display_name": "Tomato — Healthy",
        "severity":     "None",
        "cause":        "No disease detected",
        "symptoms":     "Leaves appear bright green, firm, and free of lesions.",
        "treatment": [
            "Continue regular care: balanced NPK fertilisation.",
            "Monitor weekly for early signs of pest or disease.",
            "Maintain adequate irrigation without waterlogging.",
        ],
        "prevention":   "Keep up good agricultural practices.",
    },

    # ── Potato ──────────────────────────────────────────────────────────────
    "Potato___Early_Blight": {
        "display_name": "Potato — Early Blight",
        "severity":     "Moderate",
        "cause":        "Fungus: Alternaria solani",
        "symptoms":     "Dark concentric rings on older leaves; yellowing and premature senescence.",
        "treatment": [
            "Apply Mancozeb or Chlorothalonil preventively.",
            "Remove heavily infected foliage.",
            "Ensure balanced potassium levels in soil.",
            "Avoid excess nitrogen fertilisation.",
        ],
        "prevention":   "Use certified seed potatoes; rotate with non-Solanaceae crops.",
    },
    "Potato___Late_Blight": {
        "display_name": "Potato — Late Blight",
        "severity":     "Very High",
        "cause":        "Oomycete: Phytophthora infestans",
        "symptoms":     "Rapidly expanding water-soaked lesions; white sporulation on leaf undersides; "
                        "tuber rot under severe infection.",
        "treatment": [
            "Apply Metalaxyl + Mancozeb (Ridomil) at first sign.",
            "Remove and destroy all infected plant material.",
            "Hill up soil around plants to protect tubers.",
            "Harvest early if disease is severe to save tubers.",
        ],
        "prevention":   "Plant resistant cultivars (Sarpo Mira, Defender); spray preventively in wet seasons.",
    },
    "Potato___Healthy": {
        "display_name": "Potato — Healthy",
        "severity":     "None",
        "cause":        "No disease detected",
        "symptoms":     "Normal dark-green foliage; no lesions or discolouration.",
        "treatment": [
            "Maintain regular hilling and irrigation.",
            "Scout weekly for Colorado potato beetle and aphids.",
        ],
        "prevention":   "Balanced fertilisation; avoid overhead irrigation.",
    },

    # ── Corn ────────────────────────────────────────────────────────────────
    "Corn___Common_Rust": {
        "display_name": "Corn — Common Rust",
        "severity":     "Moderate",
        "cause":        "Fungus: Puccinia sorghi",
        "symptoms":     "Brick-red to cinnamon-brown oval pustules scattered on both leaf surfaces; "
                        "may coalesce under heavy infection.",
        "treatment": [
            "Apply Propiconazole (Tilt 25 EC) @ 1 ml/litre.",
            "Use Azoxystrobin + Propiconazole (Amistar Top) for severe cases.",
            "Spray at VT stage for best results.",
        ],
        "prevention":   "Grow rust-resistant hybrids; early planting.",
    },
    "Corn___Northern_Leaf_Blight": {
        "display_name": "Corn — Northern Leaf Blight",
        "severity":     "Moderate–High",
        "cause":        "Fungus: Exserohilum turcicum",
        "symptoms":     "Long, cigar-shaped grey-green to tan lesions (5–15 cm) running parallel to "
                        "leaf veins; dark green sooty sporulation in centres.",
        "treatment": [
            "Apply Azoxystrobin @ 1 ml/litre at silking stage.",
            "Use Propiconazole or Tebuconazole as alternatives.",
            "Remove heavily infected lower leaves.",
        ],
        "prevention":   "Plant resistant hybrids; avoid dense planting.",
    },
    "Corn___Healthy": {
        "display_name": "Corn — Healthy",
        "severity":     "None",
        "cause":        "No disease detected",
        "symptoms":     "Vibrant green, turgid leaves with no lesions.",
        "treatment": [
            "Ensure adequate nitrogen top-dressing at knee-high stage.",
            "Monitor for fall armyworm and stem borers.",
        ],
        "prevention":   "Use treated certified seed; balanced fertilisation.",
    },

    # ── Pepper ──────────────────────────────────────────────────────────────
    "Pepper___Bacterial_Spot": {
        "display_name": "Pepper — Bacterial Spot",
        "severity":     "High",
        "cause":        "Bacterium: Xanthomonas campestris pv. vesicatoria",
        "symptoms":     "Small, water-soaked spots turning brown with yellow halo; "
                        "spots may coalesce; scabby lesions on fruit.",
        "treatment": [
            "Apply Copper Hydroxide (Kocide) @ 3 g/litre every 7 days.",
            "Add Mancozeb to copper spray for synergistic effect.",
            "Remove infected plant debris promptly.",
            "Avoid working in the field when leaves are wet.",
        ],
        "prevention":   "Use resistant varieties; hot-water seed treatment (52 °C / 30 min).",
    },
    "Pepper___Healthy": {
        "display_name": "Pepper — Healthy",
        "severity":     "None",
        "cause":        "No disease detected",
        "symptoms":     "Deep green, glossy leaves; no spots or lesions.",
        "treatment": [
            "Continue drip irrigation to avoid wet foliage.",
            "Apply balanced fertiliser fortnightly.",
        ],
        "prevention":   "Scout regularly; remove volunteer plants.",
    },

    # ── Apple ────────────────────────────────────────────────────────────────
    "Apple___Apple_Scab": {
        "display_name": "Apple — Apple Scab",
        "severity":     "High",
        "cause":        "Fungus: Venturia inaequalis",
        "symptoms":     "Olive-green to brown velvety lesions on leaves; scabby, corky "
                        "spots on fruit surface; premature defoliation in severe cases.",
        "treatment": [
            "Apply Captan 50 WP @ 2 g/litre at green-tip stage.",
            "Use Myclobutanil or Difenoconazole post-infection.",
            "Rake and destroy fallen leaves to break the disease cycle.",
            "Prune trees to improve light penetration and air flow.",
        ],
        "prevention":   "Plant scab-resistant cultivars (Liberty, Enterprise); dormant copper spray.",
    },
    "Apple___Healthy": {
        "display_name": "Apple — Healthy",
        "severity":     "None",
        "cause":        "No disease detected",
        "symptoms":     "Bright green, smooth leaves; no lesions.",
        "treatment": [
            "Continue dormant oil sprays for scale and mite control.",
            "Maintain annual pruning for canopy openness.",
        ],
        "prevention":   "Balanced fertilisation; monitor for fire blight in spring.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Model & class-name loading (cached for speed in Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

_model       = None
_class_names = None


def load_model_and_classes():
    """Load (and cache) model + class names from disk."""
    global _model, _class_names

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'.\n"
                "Please run: python train.py"
            )
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _class_names is None:
        if not os.path.exists(CLASS_JSON):
            raise FileNotFoundError(
                f"Class names not found at '{CLASS_JSON}'.\n"
                "Please run: python train.py"
            )
        with open(CLASS_JSON) as f:
            _class_names = json.load(f)

    return _model, _class_names


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction function
# ─────────────────────────────────────────────────────────────────────────────

def predict_disease(image_input) -> dict:
    """
    Predict the disease from a leaf image.

    Parameters
    ----------
    image_input : str | PIL.Image.Image
        Either a file path (str) or an already-opened PIL image.

    Returns
    -------
    dict with keys:
        class_name    – internal class string, e.g. "Tomato___Early_Blight"
        display_name  – human-friendly name
        confidence    – float 0-100
        top3          – list of (class_name, confidence) for top 3 predictions
        disease_info  – full DISEASE_INFO entry for the predicted class
    """
    model, class_names = load_model_and_classes()

    # ── Open / convert image ──
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    # ── Preprocess ──
    pil_img = pil_img.resize(IMG_SIZE)
    img_arr = np.array(pil_img, dtype=np.float32) / 255.0   # normalise
    img_arr = np.expand_dims(img_arr, axis=0)                # add batch dim

    # ── Inference ──
    predictions = model.predict(img_arr, verbose=0)[0]        # shape (num_classes,)

    # ── Top prediction ──
    top_idx  = int(np.argmax(predictions))
    top_conf = float(predictions[top_idx]) * 100.0
    top_cls  = class_names[top_idx]

    # ── Top-3 ──
    top3_idx = np.argsort(predictions)[::-1][:3]
    top3 = [(class_names[i], float(predictions[i]) * 100.0) for i in top3_idx]

    # ── Retrieve disease info (fallback if class not in dict) ──
    info = DISEASE_INFO.get(top_cls, {
        "display_name": top_cls.replace("___", " — ").replace("_", " "),
        "severity":     "Unknown",
        "cause":        "Unknown",
        "symptoms":     "No symptom data available.",
        "treatment":    ["Consult a local agricultural extension officer."],
        "prevention":   "Practice good crop hygiene.",
    })

    return {
        "class_name":   top_cls,
        "display_name": info["display_name"],
        "confidence":   round(top_conf, 2),
        "top3":         top3,
        "disease_info": info,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' not found.")
        sys.exit(1)

    print(f"\nAnalysing: {img_path}")
    result = predict_disease(img_path)

    print("\n" + "=" * 50)
    print(f"  Prediction  : {result['display_name']}")
    print(f"  Confidence  : {result['confidence']:.2f}%")
    print(f"  Severity    : {result['disease_info']['severity']}")
    print(f"  Cause       : {result['disease_info']['cause']}")
    print("\n  Symptoms:")
    print(f"    {result['disease_info']['symptoms']}")
    print("\n  Treatment steps:")
    for step in result["disease_info"]["treatment"]:
        print(f"    • {step}")
    print("\n  Top-3 predictions:")
    for name, conf in result["top3"]:
        print(f"    {conf:6.2f}%  {name}")
    print("=" * 50)
