# 🌾 Crop Disease Prediction — Full CNN Project

A **complete, production-ready** deep learning mini-project that detects plant leaf diseases from photos using a Convolutional Neural Network (CNN). Built with **TensorFlow/Keras** and served via a **Streamlit** web app with **Tamil + English** language support, PDF reports, and voice output.

---

## 📁 Complete Project Structure

```
crop_disease/
│
├── dataset/
│   ├── train/
│   │   ├── Tomato___Early_Blight/     ← leaf images here
│   │   ├── Tomato___Late_Blight/
│   │   ├── Tomato___Leaf_Mold/
│   │   ├── Tomato___Septoria_Leaf_Spot/
│   │   ├── Tomato___Healthy/
│   │   ├── Potato___Early_Blight/
│   │   ├── Potato___Late_Blight/
│   │   ├── Potato___Healthy/
│   │   ├── Corn___Common_Rust/
│   │   ├── Corn___Northern_Leaf_Blight/
│   │   ├── Corn___Healthy/
│   │   ├── Pepper___Bacterial_Spot/
│   │   ├── Pepper___Healthy/
│   │   ├── Apple___Apple_Scab/
│   │   └── Apple___Healthy/
│   └── test/                          ← same sub-folders, for testing
│
├── models/
│   ├── crop_disease_model.h5          ← trained model (auto-created)
│   ├── class_names.json               ← class index mapping (auto-created)
│   ├── training_history.json          ← epoch metrics (auto-created)
│   ├── training_curves.png            ← accuracy/loss plot (auto-created)
│   └── confusion_matrix.png           ← confusion matrix (auto-created)
│
├── dataset_setup.py    ← Step 1: create/verify dataset
├── train.py            ← Step 2: build & train CNN
├── predict.py          ← Step 3: load model + predict
├── app.py              ← Step 4: Streamlit web UI
├── streamlit_app.py    ← Entry point alias (imports app.py)
├── requirements.txt    ← Python dependencies
├── good.jpg            ← Sample healthy leaf
├── bad.jpg             ← Sample diseased leaf
└── README.md
```

---

## ⚡ Quick Start (Step-by-Step)

### Prerequisites
- Python 3.9 – 3.11
- VS Code (recommended) or any terminal
- 4 GB RAM minimum

---

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** If `tensorflow-cpu` fails, try:
> ```bash
> pip install tensorflow==2.12.0
> ```

---

### Step 2 — Set Up Dataset

```bash
python dataset_setup.py
```

This auto-generates **1,800 synthetic leaf images** (120 per class × 15 classes) so you can train immediately without downloading anything.

**For better accuracy** → Use real PlantVillage images:
1. Download from [Kaggle PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Extract and place images in `dataset/train/<ClassName>/`
3. Class folder names must **exactly match** those listed above
4. Re-run `python dataset_setup.py` to verify

---

### Step 3 — Train the Model

```bash
python train.py
```

**What happens:**
- Reads images from `dataset/train/`
- Applies data augmentation (rotation, flip, zoom, shift)
- Trains a 4-block CNN for up to 30 epochs
- Saves best model via `ModelCheckpoint`
- Stops early if accuracy plateaus (`EarlyStopping`)
- Prints a full `Classification Report`
- Saves `training_curves.png` and `confusion_matrix.png`

**Expected time:** ~5–15 minutes on CPU with synthetic data.

---

### Step 4 — Test from Command Line

```bash
python predict.py path/to/leaf_image.jpg
```

Example output:
```
Prediction  : Tomato — Early Blight
Confidence  : 93.47%
Severity    : Moderate
Cause       : Fungus: Alternaria solani
Treatment   :
  • Remove infected leaves immediately
  • Apply Copper Oxychloride every 7–10 days
  ...
Top-3 predictions:
  93.47%  Tomato___Early_Blight
   4.21%  Tomato___Late_Blight
   2.32%  Tomato___Healthy
```

---

### Step 5 — Launch the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

Or use:
```bash
streamlit run streamlit_app.py
```

---

## 🧠 Model Architecture

```
Input (224 × 224 × 3)
│
├── Conv2D(32,  3×3, ReLU) + L2  →  BatchNorm  →  MaxPool(2×2)   [→ 112×112]
├── Conv2D(64,  3×3, ReLU) + L2  →  BatchNorm  →  MaxPool(2×2)   [→  56× 56]
├── Conv2D(128, 3×3, ReLU) + L2  →  BatchNorm  →  MaxPool(2×2)   [→  28× 28]
├── Conv2D(256, 3×3, ReLU) + L2  →  BatchNorm  →  MaxPool(2×2)   [→  14× 14]
│
├── GlobalAveragePooling2D                                          [→ 256-dim]
├── Dense(512, ReLU) + L2  →  Dropout(0.50)
├── Dense(256, ReLU)       →  Dropout(0.30)
└── Dense(15, Softmax)                                              [output]
```

| Component | Detail |
|---|---|
| Optimizer | Adam (lr=0.001, auto-decayed) |
| Loss | Categorical Cross-Entropy |
| Regularisation | L2 (1e-4) on Conv layers |
| EarlyStopping | patience=7, restore best |
| ReduceLROnPlateau | factor=0.5, patience=3 |
| ModelCheckpoint | saves best val_accuracy |

---

## 🌿 Supported Disease Classes (15)

| Crop | Disease | Severity |
|---|---|---|
| 🍅 Tomato | Early Blight | Moderate |
| 🍅 Tomato | Late Blight | High |
| 🍅 Tomato | Leaf Mold | Moderate |
| 🍅 Tomato | Septoria Leaf Spot | Moderate |
| 🍅 Tomato | Healthy | None |
| 🥔 Potato | Early Blight | Moderate |
| 🥔 Potato | Late Blight | Very High |
| 🥔 Potato | Healthy | None |
| 🌽 Corn | Common Rust | Moderate |
| 🌽 Corn | Northern Leaf Blight | Moderate–High |
| 🌽 Corn | Healthy | None |
| 🌶️ Pepper | Bacterial Spot | High |
| 🌶️ Pepper | Healthy | None |
| 🍎 Apple | Apple Scab | High |
| 🍎 Apple | Healthy | None |

---

## 📊 Data Augmentation Pipeline

| Transform | Range |
|---|---|
| Rotation | ±20° |
| Width shift | ±15% |
| Height shift | ±15% |
| Shear | 10% |
| Zoom | ±20% |
| Horizontal flip | Yes |
| Pixel normalisation | ÷255 → [0,1] |
| Validation split | 20% (held out) |

---

## 🖥️ Web App Features

| Feature | Detail |
|---|---|
| Image upload | JPG, PNG, BMP, WEBP |
| Camera capture | Direct in-browser photo |
| Disease prediction | With confidence % |
| Confidence bar | Visual colour meter |
| Top-3 predictions | Alternative diagnoses |
| Disease info cards | Cause, symptoms, severity |
| Treatment guide | Numbered steps |
| Prevention tips | Long-term management |
| Tamil + English | Full bilingual UI |
| Voice output | gTTS inline audio + MP3 download |
| PDF report | ReportLab with image |
| WhatsApp share | One-click sharing |
| Location advice | Salem / Madurai / Chennai / Coimbatore / Trichy |
| Sidebar crops | Expandable crop list |

---

## 🔧 Customisation Guide

| What to change | Where |
|---|---|
| Max epochs | `train.py → EPOCHS` |
| Batch size | `train.py → BATCH_SIZE` |
| Learning rate | `train.py → LR` |
| Images per class | `dataset_setup.py → IMAGES_PER_CLASS` |
| Add new disease | `dataset_setup.py → CLASS_CONFIG` + `predict.py → DISEASE_INFO` |
| Model depth | `train.py → build_model()` |
| Location advice | `app.py → LOCATION_ADVICE` |

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Model not found` | Run `python train.py` first |
| `Module not found` | Run `pip install -r requirements.txt` |
| Slow training | Reduce `IMAGES_PER_CLASS` in `dataset_setup.py` |
| Low accuracy | Use real PlantVillage images from Kaggle |
| `googletrans` error | Pin version: `pip install googletrans==4.0.0-rc1` |
| Voice not working | May fail on some systems — PDF download still works |

---

## 🚀 Deployment Options

### Local
```bash
streamlit run app.py
```

### Streamlit Cloud (free)
1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file as `streamlit_app.py`
4. Deploy!

> ⚠️ Upload your trained `models/` folder to GitHub so the app has the model.

---

## 📜 License

This project is for **educational purposes** (college mini-project). The PlantVillage dataset is publicly available under academic-use terms.

---

## 👨‍💻 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.x / Keras |
| Image Processing | Pillow, NumPy |
| Web UI | Streamlit |
| Voice | gTTS (Google TTS) |
| Translation | googletrans |
| PDF | ReportLab |
| Evaluation | scikit-learn, seaborn, matplotlib |
