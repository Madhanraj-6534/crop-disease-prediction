"""
app.py  ── Crop Disease Predictor (Upgraded)
============================================
New features added:
  1. Location-based Smart Suggestions  (Salem / Madurai / Chennai …)
  2. Voice Report Download             (gTTS → MP3 download button)

Run:
    streamlit run app.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, tempfile, urllib.parse
from io        import BytesIO
from datetime  import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
from PIL        import Image
from gtts       import gTTS
from googletrans import Translator
from reportlab.platypus   import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & LOOKUP TABLES
# ══════════════════════════════════════════════════════════════════════════════
translator = Translator()


TAMIL_MAP = {
    "Tomato — Early Blight": "தக்காளி - ஆரம்ப இலை நோய்",
    "Tomato — Late Blight" : "தக்காளி - கடைசி இலை நோய்",
    "Tomato — Healthy"     : "தக்காளி - ஆரோக்கியமானது",
    "Potato — Early Blight": "உருளைக்கிழங்கு - ஆரம்ப இலை நோய்",
    "Potato — Healthy"     : "உருளைக்கிழங்கு - ஆரோக்கியமானது",
    "Corn — Healthy"       : "மக்காச்சோளம் - ஆரோக்கியமானது",
}

TAMIL_UI = {
    "confidence"     : "நம்பிக்கை அளவு",
    "solution"       : "தீர்வு",
    "result"         : "முடிவு",
    "disease"        : "நோய்",
    "cause"          : "காரணம்",
    "symptoms"       : "அறிகுறிகள்",
    "prevention"     : "தடுப்பு",
    "treatment"      : "சிகிச்சை",
    "top_predictions": "மேல் கணிப்புகள்",
}

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — Location → Smart Advice data
# ══════════════════════════════════════════════════════════════════════════════
LOCATION_ADVICE = {
    "Salem": {
        "en": [
            "🌾 Salem region: Tomato and mango crops are common — watch for Early Blight in July-Sept.",
            "💧 Irrigate in early morning to reduce leaf wetness and fungal risk.",
            "🧪 Use Mancozeb or Copper Oxychloride as a preventive spray every 10 days.",
            "🌡️ If temperature exceeds 35C, increase watering frequency.",
        ],
        "ta": [
            "🌾 சேலம் பகுதி: தக்காளி மற்றும் மாம்பழ பயிர்கள் பொதுவானவை — ஜூலை-செப்டம்பரில் ஆரம்ப நோயை கவனிக்கவும்.",
            "💧 இலை ஈரம் மற்றும் பூஞ்சை அபாயத்தை குறைக்க காலையில் நீர் பாய்ச்சவும்.",
            "🧪 தடுப்பு நடவடிக்கையாக 10 நாட்களுக்கு ஒரு முறை Mancozeb தெளிக்கவும்.",
            "🌡️ வெப்பநிலை 35C தாண்டினால், நீர்ப்பாசனம் அதிகரிக்கவும்.",
        ],
    },
    "Madurai": {
        "en": [
            "🌶️ Madurai region: Chilli and banana crops dominate — Bacterial Spot is a key threat.",
            "🌿 Remove and burn infected leaves immediately.",
            "💊 Apply Streptomycin-based spray after rain.",
            "🚿 Avoid overhead irrigation during flowering.",
        ],
        "ta": [
            "🌶️ மதுரை பகுதி: மிளகாய் மற்றும் வாழை பயிர்கள் அதிகம் — பாக்டீரியல் ஸ்பாட் முக்கிய அச்சுறுத்தல்.",
            "🌿 நோயுற்ற இலைகளை உடனே அகற்றி எரிக்கவும்.",
            "💊 மழைக்கு பிறகு Streptomycin அடிப்படையிலான தெளிப்பு பயன்படுத்தவும்.",
            "🚿 பூக்கும் தருணத்தில் மேல் நீர்ப்பாசனம் தவிர்க்கவும்.",
        ],
    },
    "Chennai": {
        "en": [
            "🌴 Chennai region: High humidity year-round increases fungal disease risk.",
            "🍃 Ensure good air circulation between plants — avoid crowding.",
            "☁️ After cloudy or rainy days, inspect leaves carefully for spots or blight.",
            "🧴 Use systemic fungicide like Propiconazole during monsoon months.",
        ],
        "ta": [
            "🌴 சென்னை பகுதி: ஆண்டு முழுவதும் அதிக ஈரப்பதம் பூஞ்சை நோய் அபாயத்தை அதிகரிக்கிறது.",
            "🍃 செடிகளுக்கிடையில் நல்ல காற்று ஓட்டம் உறுதி செய்யவும்.",
            "☁️ மேக மற்றும் மழை நாட்களுக்கு பிறகு இலைகளை கவனமாக ஆய்வு செய்யவும்.",
            "🧴 பருவமழை மாதங்களில் Propiconazole போன்ற fungicide பயன்படுத்தவும்.",
        ],
    },
    "Coimbatore": {
        "en": [
            "🌻 Coimbatore region: Cotton and turmeric are key crops — watch for leaf curl and root rot.",
            "🪱 Check soil moisture regularly; avoid waterlogging.",
            "🌬️ Spray neem oil solution (5 ml/litre) weekly as a bio-pesticide.",
            "📅 Apply potash fertilizer post-monsoon to boost immunity.",
        ],
        "ta": [
            "🌻 கோயம்புத்தூர் பகுதி: பருத்தி மற்றும் மஞ்சள் முக்கிய பயிர்கள் — இலை சுருட்டை மற்றும் வேர் அழுகலை கவனிக்கவும்.",
            "🪱 மண் ஈரப்பதத்தை தொடர்ந்து சரிபார்க்கவும்; நீர்த் தேக்கத்தை தவிர்க்கவும்.",
            "🌬️ வாரந்தோறும் வேப்ப எண்ணெய் கரைசல் தெளிக்கவும்.",
            "📅 பருவமழைக்கு பிறகு பொட்டாஷ் உரம் இடவும்.",
        ],
    },
    "Trichy": {
        "en": [
            "🌾 Trichy region: Rice and sugarcane are primary — Blast disease and stem borer are concerns.",
            "🔬 Inspect paddy weekly for brown spots or neck lesions.",
            "💦 Maintain proper drainage in paddy fields.",
            "🧪 Use Tricyclazole spray at tillering stage as prevention.",
        ],
        "ta": [
            "🌾 திருச்சி பகுதி: நெல் மற்றும் கரும்பு முதன்மை பயிர்கள் — Blast நோய் மற்றும் தண்டு துளைப்பி கவலைக்குரியவை.",
            "🔬 வாரந்தோறும் நெல்லில் பழுப்பு புள்ளிகளை சரிபார்க்கவும்.",
            "💦 நெல் வயல்களில் சரியான வடிகால் பராமரிக்கவும்.",
            "🧪 திரிசைக்ளோசோல் தெளிப்பை தடுப்புக்காக பயன்படுத்தவும்.",
        ],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Crop Disease Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.card {
    background:#fff; border:1px solid #d4e6d0; border-radius:16px;
    padding:1.5rem 1.8rem; margin-bottom:1.2rem;
    box-shadow:0 2px 12px rgba(26,71,42,0.06);
}
.result-card {
    background:linear-gradient(135deg,#f0faf0,#e6f4e8);
    border:2px solid #3a8f5c; border-radius:16px;
    padding:1.8rem; margin-bottom:1.2rem;
}
.result-card-danger {
    background:linear-gradient(135deg,#fff8f0,#fef0e0);
    border:2px solid #e07b39; border-radius:16px;
    padding:1.8rem; margin-bottom:1.2rem;
}
.loc-card {
    background:linear-gradient(135deg,#e3f2fd,#bbdefb);
    border:2px solid #1e88e5; border-radius:14px; padding:1rem 1.4rem; margin-bottom:1rem;
}
.badge {
    display:inline-block; padding:0.25rem 0.75rem; border-radius:20px;
    font-size:0.78rem; font-weight:600; letter-spacing:0.04em;
    text-transform:uppercase; margin-right:0.4rem;
}
.badge-green  { background:#d4edda; color:#155724; }
.badge-orange { background:#ffe0b2; color:#8d4004; }
.badge-red    { background:#ffd5d5; color:#8b0000; }
.conf-bar-wrap {
    background:#e8f5e9; border-radius:12px;
    height:22px; width:100%; overflow:hidden; margin-top:0.5rem;
}
.conf-bar-fill {
    height:100%; border-radius:12px; display:flex; align-items:center;
    justify-content:flex-end; padding-right:0.5rem;
    font-size:0.75rem; font-weight:600; color:white;
}
.step-item { display:flex; align-items:flex-start; gap:0.7rem; padding:0.45rem 0; }
.step-num {
    background:#1a472a; color:white; border-radius:50%;
    width:22px; height:22px; display:flex; align-items:center; justify-content:center;
    font-size:0.72rem; font-weight:700; flex-shrink:0; margin-top:2px;
}
[data-testid="stSidebar"] { background:#f4fbf4; border-right:1px solid #d4e6d0; }
.section-title {
    font-family:'DM Serif Display',serif; font-size:1.3rem;
    color:#1a472a; margin:1.2rem 0 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def severity_badge(severity: str) -> str:
    colour = {
        "None":"badge-green","Low":"badge-green",
        "Moderate":"badge-orange","High":"badge-red","Very High":"badge-red",
    }.get(severity, "badge-orange")
    return f'<span class="badge {colour}">{severity}</span>'


# ── FEATURE 3 — Build MP3 bytes in memory (no disk file needed) ───────────────
def build_voice_mp3(text: str, lang_code: str = "en") -> bytes:
    """Convert text to MP3 bytes using gTTS. Returns raw bytes."""
    tts = gTTS(text=text, lang=lang_code)
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def speak_inline(text: str, lang_code: str = "en"):
    """Play audio inline inside Streamlit."""
    st.audio(build_voice_mp3(text, lang_code), format="audio/mp3")


def generate_pdf(disease_name, confidence, cause, treatment, image, lang="English"):
    """Build a ReportLab PDF and return it as BytesIO."""
    buffer = BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    body   = []

    title = "Crop Disease Report" if lang == "English" else "பயிர் நோய் அறிக்கை"
    body.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    body.append(Spacer(1, 10))
    body.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
    body.append(Spacer(1, 10))

    if image:
        img_buf = BytesIO()
        image.save(img_buf, format="PNG")
        img_buf.seek(0)
        body.append(RLImage(img_buf, width=200, height=200))
        body.append(Spacer(1, 10))

    label_d = "Disease" if lang == "English" else "நோய்"
    body.append(Paragraph(f"<b>{label_d}:</b> {disease_name}", styles["Normal"]))
    clr = "green" if confidence > 80 else ("orange" if confidence > 50 else "red")
    body.append(Paragraph(f"<b>Confidence:</b> <font color='{clr}'>{confidence:.1f}%</font>", styles["Normal"]))
    body.append(Spacer(1, 10))

    label_c = "Cause" if lang == "English" else "காரணம்"
    body.append(Paragraph(f"<b>{label_c}:</b> {cause}", styles["Normal"]))
    body.append(Spacer(1, 10))

    label_t = "Treatment" if lang == "English" else "தீர்வு"
    body.append(Paragraph(f"<b>{label_t}:</b>", styles["Normal"]))
    for t in treatment:
        body.append(Paragraph(f"• {t}", styles["Normal"]))

    doc.build(body)
    buffer.seek(0)
    return buffer



# ── FEATURE 2 — Render location-based advice ──────────────────────────────────
def show_location_advice(location: str, lang: str):
    advice_list = LOCATION_ADVICE.get(location, {}).get(
        "ta" if lang == "Tamil" else "en", []
    )
    if not advice_list:
        return

    loc_label = "இடம் அடிப்படையிலான பரிந்துரைகள்" if lang == "Tamil" else "Smart Farming Advice"
    st.markdown(f'<p class="section-title">📍 {loc_label} — {location}</p>', unsafe_allow_html=True)
    items_html = "".join(
        f"<div style='padding:0.3rem 0'>{tip}</div>" for tip in advice_list
    )
    st.markdown(f'<div class="loc-card">{items_html}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    from predict import load_model_and_classes
    return load_model_and_classes()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("🌾 Farmer Menu")
lang = st.sidebar.selectbox("🌐 Language / மொழி", ["English", "Tamil"])

# FEATURE 2 — Location selector in sidebar
location_label = "📍 உங்கள் இடம்" if lang == "Tamil" else "📍 Your Location"
user_location  = st.sidebar.selectbox(location_label, list(LOCATION_ADVICE.keys()))

if lang == "Tamil":
    st.sidebar.info("இலை படத்தை பதிவேற்றவும் நோய் கண்டறிய")
else:
    st.sidebar.info("Upload a leaf image to detect disease")

with st.sidebar:
    if lang == "Tamil":
        st.markdown("## 🌿 பற்றி")
        st.markdown(
            "இந்த கருவி **AI (CNN)** பயன்படுத்தி பயிர் நோய்களை கண்டறிகிறது.\n\n"
            "15 வகை பயிர் நோய்களை கண்டறிய பயிற்சி பெற்றது.\n\n"
            "ஒரு தெளிவான **இலை படம்** பதிவேற்றவும்."
        )
    else:
        st.markdown("## 🌿 About")
        st.markdown(
            "This tool uses **Convolutional Neural Network (CNN)**.\n\n"
            "Trained on 15 crop disease classes.\n\n"
            "Upload a clear **leaf image**."
        )

    st.divider()
    st.markdown("### 🌾 " + ("பயிர்கள்" if lang == "Tamil" else "Supported Crops"))

    crops = {
        "🍅 Tomato" : ["Early Blight", "Late Blight"],
        "🥔 Potato" : ["Early Blight", "Late Blight"],
        "🌽 Corn"   : ["Common Rust",  "Northern Leaf Blight"],
        "🌶️ Pepper" : ["Bacterial Spot", "Healthy"],
        "🍎 Apple"  : ["Apple Scab",   "Healthy"],
    }
    for crop, diseases in crops.items():
        with st.expander(crop):
            for d in diseases:
                if lang == "Tamil":
                    try:
                        d = translator.translate(d, dest="ta").text
                    except Exception:
                        pass
                st.write(f"• {d}")

    st.divider()
    st.markdown("### 💡 " + ("குறிப்புகள்" if lang == "Tamil" else "Tips for best results"))
    tips = (
        ["தெளிவான படம் எடுக்கவும்", "ஒரே இலை மீது கவனம் செலுத்தவும்",
         "மங்கலான படங்களை தவிர்க்கவும்", "இரு பக்கங்களையும் படம் எடுக்கவும்"]
        if lang == "Tamil" else
        ["Use a clear, well-lit photo", "Focus on a single leaf",
         "Avoid blurry or dark images", "Capture both sides if unsure"]
    )
    for tip in tips:
        st.write(f"• {tip}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(90deg,#2E7D32,#66BB6A);
padding:20px;border-radius:10px;color:white;margin-bottom:1rem'>
<h2>🌾 Smart Crop Disease Detection</h2>
<p>AI-powered farming assistant for instant diagnosis</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2 : Location Smart Advice
# ─────────────────────────────────────────────────────────────────────────────
show_location_advice(user_location, lang)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOAD
# ══════════════════════════════════════════════════════════════════════════════
model_ready = os.path.exists("models/crop_disease_model.h5")

if not model_ready:
    st.warning(
        "⚠️ **Model not found.** Please train the model first:\n\n"
        "```bash\npython train.py\n```"
    )
    st.stop()

with st.spinner("Loading AI model …"):
    try:
        model, class_names = load_model()
        st.success(f"✅ Model loaded — {len(class_names)} disease classes ready.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

from predict import predict_disease, DISEASE_INFO

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE LEAF GUIDE
# ══════════════════════════════════════════════════════════════════════════════
if os.path.exists("good.jpg") and os.path.exists("bad.jpg"):
    captions = (
        ["✅ நல்ல இலை", "❌ நோயுற்ற இலை"]
        if lang == "Tamil" else
        ["✅ Healthy Leaf", "❌ Diseased Leaf"]
    )
    st.image(["good.jpg", "bad.jpg"], caption=captions, width=250)

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE INPUT
# ══════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Drag & drop or click to browse" if lang == "English" else "படத்தை பதிவேற்றவும்",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)
camera_img = st.camera_input(
    "📷 Take Leaf Photo" if lang == "English" else "📷 படம் எடுக்கவும்"
)

image = None
if camera_img is not None:
    image = Image.open(camera_img).convert("RGB")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if image is not None:

    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    image.save(tmp_path)

    with st.spinner("🔍 Analysing leaf..."):
        try:
            result = predict_disease(tmp_path)
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.stop()

    disease_name = result.get("display_name", result.get("disease", "Unknown"))
    conf         = result.get("confidence", 0.0)
    info         = result.get("disease_info", DISEASE_INFO.get(disease_name, {}))

    # Safe fallbacks so no KeyError later
    info.setdefault("cause",      "No data available")
    info.setdefault("symptoms",   "No data available")
    info.setdefault("prevention", "Practice good hygiene and crop rotation")
    info.setdefault("treatment",  ["Consult a local agronomist"])
    info.setdefault("severity",   "Moderate")

    tamil_name = TAMIL_MAP.get(disease_name, disease_name)

    # ── Image + Result columns ────────────────────────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<p class="section-title">🖼 Uploaded Image</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col_res:
        is_healthy = info["severity"] == "None"
        card_cls   = "result-card" if is_healthy else "result-card-danger"
        icon       = "✅" if is_healthy else "⚠️"

        st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)
        st.markdown(f"### {icon}  {tamil_name if lang == 'Tamil' else disease_name}")
        st.markdown(
            severity_badge(info["severity"]) +
            f' <span class="badge badge-orange">{info["cause"][:40]}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Confidence bar
        conf_col = "#3a8f5c" if conf >= 70 else ("#e07b39" if conf >= 45 else "#c0392b")
        conf_lbl = TAMIL_UI["confidence"] if lang == "Tamil" else "Confidence"
        st.markdown(f"**{conf_lbl}: {conf:.1f}%**")
        st.markdown(
            f'<div class="conf-bar-wrap">'
            f'<div class="conf-bar-fill" style="width:{min(conf,100):.1f}%;'
            f'background:linear-gradient(90deg,{conf_col},{conf_col}cc);">'
            f'{conf:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

        if conf > 80:
            st.success("🟢 High Confidence"   if lang == "English" else "🟢 அதிக நம்பிக்கை")
        elif conf > 50:
            st.warning("🟡 Medium Confidence" if lang == "English" else "🟡 நடுத்தர நம்பிக்கை")
        else:
            st.error("🔴 Low Confidence"      if lang == "English" else "🔴 குறைந்த நம்பிக்கை")

        # Inline speak button
        if st.button("🔊 Speak Full Info"):
            try:
                if lang == "Tamil":
                    c_ta = translator.translate(info["cause"], dest="ta").text
                    t_ta = " ".join(
                        translator.translate(t, dest="ta").text for t in info["treatment"]
                    )
                    speak_inline(
                        f"இந்த இலைக்கு {tamil_name} நோய் உள்ளது. காரணம் {c_ta}. தீர்வு {t_ta}",
                        lang_code="ta",
                    )
                else:
                    speak_inline(
                        f"This leaf has {disease_name}. "
                        f"Cause: {info['cause']}. "
                        f"Treatment: {' '.join(info['treatment'])}",
                        lang_code="en",
                    )
            except Exception as e:
                st.warning(f"Voice playback failed: {e}")

    # ── PDF download ──────────────────────────────────────────────────────────
    pdf_buf = generate_pdf(disease_name, conf, info["cause"], info["treatment"], image, lang)
    st.download_button(
        label    = "📄 Download Full Report" if lang == "English" else "📄 அறிக்கையை பதிவிறக்கவும்",
        data     = pdf_buf,
        file_name= "smart_crop_report.pdf",
        mime     = "application/pdf",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 3 : Voice Report Download (MP3)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="section-title">🎙️ ' +
        ("குரல் அறிக்கை" if lang == "Tamil" else "Voice Report") +
        '</p>',
        unsafe_allow_html=True,
    )
    try:
        if lang == "Tamil":
            c_ta      = translator.translate(info["cause"], dest="ta").text
            t_ta      = " ".join(
                translator.translate(t, dest="ta").text for t in info["treatment"]
            )
            voice_txt = (
                f"பயிர் நோய் அறிக்கை. "
                f"நோய்: {tamil_name}. "
                f"நம்பிக்கை அளவு: {conf:.0f} சதவீதம். "
                f"காரணம்: {c_ta}. "
                f"தீர்வு: {t_ta}."
            )
            mp3_bytes = build_voice_mp3(voice_txt, lang_code="ta")
        else:
            voice_txt = (
                f"Crop Disease Report. "
                f"Disease: {disease_name}. "
                f"Confidence: {conf:.0f} percent. "
                f"Cause: {info['cause']}. "
                f"Treatment: {' '.join(info['treatment'])}."
            )
            mp3_bytes = build_voice_mp3(voice_txt, lang_code="en")

        # Download button
        st.download_button(
            label    = "⬇️ Download Voice Report (MP3)"
                       if lang == "English" else
                       "⬇️ குரல் அறிக்கையை பதிவிறக்கவும் (MP3)",
            data     = mp3_bytes,
            file_name= "crop_voice_report.mp3",
            mime     = "audio/mp3",
        )
        # Also play inline so farmer can hear it immediately
        st.audio(mp3_bytes, format="audio/mp3")

    except Exception as e:
        st.warning(f"Voice report generation failed: {e}")

    # ── WhatsApp share ────────────────────────────────────────────────────────
    message = (
        f"🌿 Crop Disease Report\n\n"
        f"Disease: {disease_name}\nConfidence: {conf:.1f}%\n\n"
        f"Cause: {info['cause']}\n"
        f"Treatment: {', '.join(info['treatment'])}\n\n"
        f"📄 Download full report below 👇"
    )
    whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(message)}"
    st.markdown(f"[📤 Share via WhatsApp]({whatsapp_url})")

    # ── Diagnosis / Treatment / Top-3 Tabs ───────────────────────────────────
    st.divider()
    if lang == "Tamil":
        tab1, tab2, tab3 = st.tabs(["🔍 கண்டறிதல்", "💊 சிகிச்சை", "📊 கணிப்புகள்"])
    else:
        tab1, tab2, tab3 = st.tabs(["🔍 Diagnosis", "💊 Treatment", "📊 Top Predictions"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if lang == "Tamil":
                st.markdown("**🌿 காரணம்**")
                try:    st.write(translator.translate(info["cause"], dest="ta").text)
                except: st.write(info["cause"])
                st.markdown("**📄 அறிகுறிகள்**")
                try:    st.write(translator.translate(info["symptoms"], dest="ta").text)
                except: st.write(info["symptoms"])
            else:
                st.markdown("**🌿 Cause**");    st.write(info["cause"])
                st.markdown("**📄 Symptoms**"); st.write(info["symptoms"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if lang == "Tamil":
                st.markdown("**🛡️ தடுப்பு**")
                try:    st.write(translator.translate(info["prevention"], dest="ta").text)
                except: st.write(info["prevention"])
            else:
                st.markdown("**🛡️ Prevention**"); st.write(info["prevention"])
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown(
            "இந்த நோயை கட்டுப்படுத்த கீழ்க்கண்ட படிகளை பின்பற்றவும்:"
            if lang == "Tamil" else
            "Follow these treatment steps to manage the disease effectively:"
        )
        steps_html = ""
        for i, step in enumerate(info["treatment"], 1):
            disp = step
            if lang == "Tamil":
                try:    disp = translator.translate(step, dest="ta").text
                except: pass
            steps_html += (
                f'<div class="step-item">'
                f'<div class="step-num">{i}</div>'
                f'<div>{disp}</div>'
                f'</div>'
            )
        st.markdown(f'<div class="card">{steps_html}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown(
            "முன்னணி 3 கணிப்புகளின் நம்பிக்கை அளவு:"
            if lang == "Tamil" else
            "Model confidence across top 3 predictions:"
        )
        for cls_name, conf_val in result.get("top3", []):
            lbl = cls_name.replace("___", " — ").replace("_", " ")
            st.markdown(f"**{lbl}** — {conf_val:.2f}%")
            st.progress(min(conf_val / 100.0, 1.0))

else:
    # ── No image uploaded ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">🌿</div>
        <div style="font-size:1.3rem; font-family:'DM Serif Display',serif;
                    color:#1a472a; margin-bottom:0.5rem;">
            Ready to diagnose your crops
        </div>
        <div style="color:#4a7c59; font-size:0.95rem;">
            Upload a leaf image using the uploader above to begin analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DETECTABLE DISEASES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">📚 Detectable Diseases</p>', unsafe_allow_html=True)
cols = st.columns(3)
for i, (cls_key, data) in enumerate(DISEASE_INFO.items()):
    with cols[i % 3]:
        sev  = data.get("severity", "Moderate")
        icon = "✅" if sev == "None" else ("⚠️" if sev in ("Moderate", "Low") else "🔴")
        name = data.get("display_name", cls_key)
        st.markdown(
            f'<div class="card" style="padding:1rem;">'
            f'<b>{icon} {name}</b><br>'
            f'<small style="color:#666">{severity_badge(sev)}</small><br>'
            f'<small style="color:#555">{data.get("cause","")[:60]}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<center><small>🌾 Crop Disease Predictor &nbsp;|&nbsp; "
    "Built with TensorFlow + Streamlit &nbsp;|&nbsp; "
    "PlantVillage-inspired dataset</small></center>",
    unsafe_allow_html=True,
)
