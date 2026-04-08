import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Social Media Engagement Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — DARK FUTURISTIC THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

[data-testid="stSidebar"] .stMarkdown h2 {
    color: #818cf8;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.1) 50%, rgba(236,72,153,0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(99,102,241,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(168,85,247,0.06) 0%, transparent 50%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: rgba(15, 20, 35, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #818cf8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
}

/* Prediction result box */
.prediction-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.15));
    border: 2px solid rgba(99, 102, 241, 0.5);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.prediction-box::after {
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(236,72,153,0.2));
    z-index: -1;
    filter: blur(8px);
}

.prediction-score {
    font-size: 4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.prediction-tier {
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* CV features box */
.cv-box {
    background: rgba(15, 20, 35, 0.7);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}

.cv-label {
    font-size: 0.75rem;
    color: #10b981;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

.cv-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
}

/* Tips box */
.tip-box {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
}

.tip-icon {
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
}

.tip-text {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Gauge container */
.gauge-container {
    background: rgba(15, 20, 35, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 1rem;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 2rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    letter-spacing: 0.03em;
    transition: all 0.3s ease;
    cursor: pointer;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9);
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35);
}

.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99, 102, 241, 0.35) !important;
    border-radius: 16px !important;
    background: rgba(99, 102, 241, 0.05) !important;
    padding: 1rem !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
}

h1, h2, h3 {
    color: #e2e8f0 !important;
}

.stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #818cf8 !important;
}

.stTabs [data-baseweb="tab-border"] {
    background: rgba(99, 102, 241, 0.15) !important;
}

/* Info/warning boxes */
.stAlert {
    border-radius: 12px !important;
}

/* Divider */
hr {
    border-color: rgba(99, 102, 241, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = "models"
    rf = pickle.load(open(f"{base}/rf_model.pkl", "rb"))
    xgb = pickle.load(open(f"{base}/xgb_model.pkl", "rb"))
    scaler = pickle.load(open(f"{base}/scaler.pkl", "rb"))
    le = pickle.load(open(f"{base}/label_encoder.pkl", "rb"))
    features = pickle.load(open(f"{base}/feature_cols.pkl", "rb"))
    metrics = pickle.load(open(f"{base}/metrics.pkl", "rb"))
    return rf, xgb, scaler, le, features, metrics

rf_model, xgb_model, scaler, label_enc, FEATURE_COLS, MODEL_METRICS = load_models()


# ─────────────────────────────────────────────
# CV FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_cv_features(video_path):
    """Extract visual features from video thumbnail (first frame)."""
    features = {
        "brightness_score": 0.5,
        "colorfulness_score": 0.5,
        "face_count": 0,
        "female_presence_score": 0.0,
        "thumbnail": None
    }
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # Read first good frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return features, 0

        # Save thumbnail
        thumbnail = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features["thumbnail"] = thumbnail

        # ── Brightness Score ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray)[0] / 255.0
        features["brightness_score"] = round(brightness, 3)

        # ── Colorfulness Score ──
        (B, G, R) = cv2.split(frame.astype("float"))
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        std_rg, mean_rg = np.std(rg), np.mean(rg)
        std_yb, mean_yb = np.std(yb), np.mean(yb)
        colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        features["colorfulness_score"] = round(min(colorfulness / 150.0, 1.0), 3)

        # ── Face Detection ──
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_count = len(faces)
        features["face_count"] = int(face_count)

        # Female presence: estimated from face count + content analysis
        # In production this would use DeepFace gender detection
        # Here we use a heuristic: colorful + bright + faces → higher female presence probability
        if face_count > 0:
            female_score = min(
                0.3 + (features["colorfulness_score"] * 0.4) + (features["brightness_score"] * 0.3),
                1.0
            )
            features["female_presence_score"] = round(female_score, 3)

        cap.release()
        return features, round(duration, 1)

    except Exception as e:
        return features, 0


# ─────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────
def predict_engagement(input_data: dict) -> dict:
    """Run ensemble prediction and return results."""
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    # Feature engineering
    hour = input_data["hour"]
    day_name = input_data["day_name"]
    category = input_data["category"]
    followers = input_data["followers_count"]

    features = {
        "hour": hour,
        "day_num": day_map[day_name],
        "is_weekend": 1 if day_name in ["Saturday", "Sunday"] else 0,
        "is_peak_hour": 1 if 18 <= hour <= 21 else 0,
        "is_morning": 1 if 7 <= hour <= 10 else 0,
        "category_enc": label_enc.transform([category])[0] if category in label_enc.classes_ else 0,
        "is_high_eng_category": 1 if category in ["Entertainment", "Sports"] else 0,
        "followers_count": followers,
        "followers_log": np.log1p(followers),
        "duration_seconds": input_data["duration_seconds"],
        "is_short_reel": 1 if 15 <= input_data["duration_seconds"] <= 45 else 0,
        "hashtag_count": input_data["hashtag_count"],
        "hashtag_optimal": 1 if 5 <= input_data["hashtag_count"] <= 15 else 0,
        "caption_length": input_data["caption_length"],
        "has_trending_audio": input_data["has_trending_audio"],
        "female_presence_score": input_data["female_presence_score"],
        "face_count": input_data["face_count"],
        "brightness_score": input_data["brightness_score"],
        "colorfulness_score": input_data["colorfulness_score"],
        "engagement_potential": (
            np.log1p(followers) * 0.4 +
            input_data["female_presence_score"] * 0.2 +
            input_data["brightness_score"] * 0.2 +
            input_data["colorfulness_score"] * 0.2
        )
    }

    X = pd.DataFrame([features])[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    rf_pred = rf_model.predict(X_scaled)[0]
    xgb_pred = xgb_model.predict(X_scaled)[0]
    ensemble = (rf_pred + xgb_pred) / 2

    # Estimated breakdown
    total = max(ensemble, 100)
    est_likes = int(total / (1 + 3 * 0.1 + 2 * 0.08))
    est_comments = int(est_likes * 0.1)
    est_shares = int(est_likes * 0.08)

    # Tier classification
    if total < 5000:
        tier = ("Low Reach", "#ef4444", "⚠️", 20)
    elif total < 15000:
        tier = ("Growing", "#f59e0b", "📈", 45)
    elif total < 40000:
        tier = ("Good Engagement", "#10b981", "✅", 68)
    elif total < 80000:
        tier = ("High Engagement", "#6366f1", "🔥", 82)
    else:
        tier = ("Viral Potential", "#c084fc", "🚀", 95)

    return {
        "rf_pred": round(rf_pred),
        "xgb_pred": round(xgb_pred),
        "ensemble": round(ensemble),
        "est_likes": est_likes,
        "est_comments": est_comments,
        "est_shares": est_shares,
        "tier": tier[0],
        "tier_color": tier[1],
        "tier_icon": tier[2],
        "viral_score": tier[3]
    }


def get_recommendations(input_data: dict, result: dict) -> list:
    """Generate actionable recommendations based on inputs."""
    tips = []
    hour = input_data["hour"]
    day = input_data["day_name"]
    dur = input_data["duration_seconds"]
    hashtags = input_data["hashtag_count"]

    if not (18 <= hour <= 21):
        tips.append(("🕐", f"Post between 6PM–9PM for 15–20% higher reach. You chose {hour}:00."))
    if day not in ["Friday", "Saturday", "Sunday"]:
        tips.append(("📅", "Friday–Sunday reels get 18% more engagement on average."))
    if dur > 60:
        tips.append(("⏱️", "Reels under 45 seconds get 2x completion rate. Try trimming your reel."))
    if not input_data["has_trending_audio"]:
        tips.append(("🎵", "Using a trending audio track can boost reach by up to 40%."))
    if hashtags < 5 or hashtags > 15:
        tips.append(("#️⃣", f"Use 5–15 hashtags. You have {hashtags}. Sweet spot is 8–12."))
    if input_data["brightness_score"] < 0.4:
        tips.append(("💡", "Your thumbnail looks dark. Brighter visuals get 23% more clicks."))
    if input_data["colorfulness_score"] < 0.35:
        tips.append(("🎨", "Add more color contrast to your thumbnail for better thumb-stops."))
    if input_data["face_count"] == 0:
        tips.append(("👤", "Reels with human faces in the thumbnail get 35% more engagement."))

    if not tips:
        tips.append(("✨", "Your reel setup looks optimized! Focus on hook quality in the first 3 seconds."))
        tips.append(("🔄", "Consistency matters — post at least 4–5 reels per week."))

    return tips[:5]


# ─────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────
def make_gauge(score, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 36, "color": "#c084fc", "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#475569", "size": 11}, "tickcolor": "#1e293b"},
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(239,68,68,0.15)"},
                {"range": [25, 50], "color": "rgba(245,158,11,0.15)"},
                {"range": [50, 75], "color": "rgba(16,185,129,0.15)"},
                {"range": [75, 100], "color": "rgba(99,102,241,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#c084fc", "width": 4},
                "thickness": 0.85,
                "value": score
            }
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"}
    )
    return fig


def make_breakdown_chart(likes, comments, shares, score):
    categories = ["Likes ×1", "Comments ×3", "Shares ×2"]
    values = [likes * 1, comments * 3, shares * 2]
    colors = ["#6366f1", "#c084fc", "#f472b6"]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:,}" for v in values],
        textposition="outside",
        textfont={"color": "#94a3b8", "size": 12, "family": "JetBrains Mono"}
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(showgrid=False, tickfont={"color": "#64748b", "size": 12}),
        yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)", tickfont={"color": "#64748b", "size": 10}),
        bargap=0.4,
        showlegend=False
    )
    return fig


def make_model_comparison(rf_pred, xgb_pred, ensemble):
    fig = go.Figure()
    models = ["Random Forest", "XGBoost", "Ensemble"]
    values = [rf_pred, xgb_pred, ensemble]
    colors = ["rgba(99,102,241,0.7)", "rgba(168,85,247,0.7)", "rgba(236,72,153,0.9)"]

    for i, (m, v, c) in enumerate(zip(models, values, colors)):
        fig.add_trace(go.Bar(
            name=m, x=[m], y=[v],
            marker_color=c, marker_line_width=0,
            text=[f"{v:,}"], textposition="outside",
            textfont={"color": "#94a3b8", "size": 11, "family": "JetBrains Mono"}
        ))

    fig.update_layout(
        height=260, barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(showgrid=False, tickfont={"color": "#64748b", "size": 12}),
        yaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)", tickfont={"color": "#64748b", "size": 10}),
        showlegend=False
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🎬 Reel Engagement Predictor</h1>
    <p class="hero-sub">AI-powered Instagram Reels engagement forecasting · Computer Vision + Random Forest + XGBoost</p>
</div>
""", unsafe_allow_html=True)

# Model Stats Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{MODEL_METRICS['total_rows']:,}</div>
        <div class="metric-label">Training Samples</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{MODEL_METRICS['rf_r2']:.1%}</div>
        <div class="metric-label">RF Accuracy (R²)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{MODEL_METRICS['xgb_r2']:.1%}</div>
        <div class="metric-label">XGBoost Accuracy</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">20</div>
        <div class="metric-label">Features Used</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("## 📋 Post Details")
    st.markdown("---")

    followers = st.number_input(
        "Followers Count",
        min_value=100,
        max_value=10_000_000,
        value=50_000,
        step=1000,
        help="Your Instagram account followers"
    )

    day = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=4
    )

    hour = st.slider(
        "Posting Hour (24h)",
        min_value=6, max_value=23, value=19,
        help="Hour you plan to post"
    )
    st.caption(f"⏰ Selected: {'Morning' if 6<=hour<=11 else 'Afternoon' if 12<=hour<=17 else 'Evening' if 18<=hour<=21 else 'Night'} — {hour}:00")

    category = st.selectbox(
        "Content Category",
        ["Entertainment", "Sports", "Lifestyle", "Tech", "Education", "Business"],
        index=0
    )

    st.markdown("---")
    st.markdown("## 🎥 Reel Settings")

    duration = st.slider("Duration (seconds)", 7, 90, 30)
    hashtags = st.slider("Hashtag Count", 0, 30, 10)
    caption_len = st.slider("Caption Length (chars)", 0, 300, 80)
    trending_audio = st.toggle("Trending Audio 🎵", value=True)

    st.markdown("---")
    st.markdown("## 📤 Upload Reel")
    video_file = st.file_uploader(
        "Upload your reel (MP4/MOV)",
        type=["mp4", "mov", "avi"],
        help="We extract visual features from your video"
    )

# ─── CV EXTRACTION ───
cv_features = {
    "brightness_score": 0.55,
    "colorfulness_score": 0.50,
    "face_count": 1,
    "female_presence_score": 0.50,
    "thumbnail": None
}
extracted_duration = duration
video_uploaded = False

if video_file:
    with st.spinner("🔍 Analyzing video with Computer Vision..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        cv_features, extracted_duration = extract_cv_features(tmp_path)
        os.unlink(tmp_path)
        video_uploaded = True
        if extracted_duration > 0:
            duration = int(extracted_duration)

# ─── MAIN CONTENT ───
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    # CV Features Panel
    st.markdown('<div class="section-header">🔬 Computer Vision Analysis</div>', unsafe_allow_html=True)

    if video_uploaded and cv_features.get("thumbnail") is not None:
        st.image(cv_features["thumbnail"], caption="Extracted Thumbnail", use_container_width=True,
                 clamp=True)
        st.success("✅ Video analyzed successfully!")
    else:
        st.info("📹 Upload a reel video to enable Computer Vision feature extraction. Using default values for now.")

    cv1, cv2 = st.columns(2)
    with cv1:
        st.markdown(f"""<div class="cv-box">
            <div class="cv-label">Brightness</div>
            <div class="cv-value">{cv_features['brightness_score']:.2f}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="cv-box">
            <div class="cv-label">Faces Detected</div>
            <div class="cv-value">{cv_features['face_count']}</div>
        </div>""", unsafe_allow_html=True)
    with cv2:
        st.markdown(f"""<div class="cv-box">
            <div class="cv-label">Colorfulness</div>
            <div class="cv-value">{cv_features['colorfulness_score']:.2f}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="cv-box">
            <div class="cv-label">Female Presence</div>
            <div class="cv-value">{cv_features['female_presence_score']:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Manual CV override
    with st.expander("⚙️ Manually adjust CV values"):
        cv_features["brightness_score"] = st.slider("Brightness Score", 0.0, 1.0,
                                                      float(cv_features["brightness_score"]), 0.01)
        cv_features["colorfulness_score"] = st.slider("Colorfulness Score", 0.0, 1.0,
                                                        float(cv_features["colorfulness_score"]), 0.01)
        cv_features["face_count"] = st.slider("Face Count", 0, 6, int(cv_features["face_count"]))
        cv_features["female_presence_score"] = st.slider("Female Presence Score", 0.0, 1.0,
                                                           float(cv_features["female_presence_score"]), 0.01,
                                                           help="0 = no female presence, 1 = strong female presence")

    # Predict Button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict Engagement", use_container_width=True)


with right_col:
    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

    if predict_clicked or True:  # Show placeholder on load
        input_data = {
            "hour": hour,
            "day_name": day,
            "category": category,
            "followers_count": followers,
            "duration_seconds": duration,
            "hashtag_count": hashtags,
            "caption_length": caption_len,
            "has_trending_audio": int(trending_audio),
            "female_presence_score": cv_features["female_presence_score"],
            "face_count": cv_features["face_count"],
            "brightness_score": cv_features["brightness_score"],
            "colorfulness_score": cv_features["colorfulness_score"]
        }

        result = predict_engagement(input_data)

        # Main prediction box
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size:0.85rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
                Predicted Engagement Score
            </div>
            <div class="prediction-score">{result['ensemble']:,}</div>
            <div class="prediction-tier" style="color:{result['tier_color']}">
                {result['tier_icon']} {result['tier']}
            </div>
            <div style="font-size:0.8rem; color:#475569; margin-top:0.8rem;">
                RF: {result['rf_pred']:,} &nbsp;|&nbsp; XGBoost: {result['xgb_pred']:,} &nbsp;|&nbsp; Ensemble: {result['ensemble']:,}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for charts
        tab1, tab2, tab3 = st.tabs(["📈 Viral Score", "🔢 Breakdown", "🤖 Models"])

        with tab1:
            st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(result["viral_score"]), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(f"Viral probability score based on all input factors")

        with tab2:
            st.plotly_chart(
                make_breakdown_chart(result["est_likes"], result["est_comments"], result["est_shares"], result["ensemble"]),
                use_container_width=True, config={"displayModeBar": False}
            )
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Est. Likes", f"{result['est_likes']:,}")
            ec2.metric("Est. Comments", f"{result['est_comments']:,}")
            ec3.metric("Est. Shares", f"{result['est_shares']:,}")

        with tab3:
            st.plotly_chart(
                make_model_comparison(result["rf_pred"], result["xgb_pred"], result["ensemble"]),
                use_container_width=True, config={"displayModeBar": False}
            )
            st.caption("Ensemble = average of RF + XGBoost for best accuracy")


# ─── RECOMMENDATIONS ───
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">💡 Smart Recommendations</div>', unsafe_allow_html=True)

if predict_clicked or True:
    tips = get_recommendations(input_data, result)
    tip_cols = st.columns(min(len(tips), 3))
    for i, (icon, text) in enumerate(tips):
        with tip_cols[i % 3]:
            st.markdown(f"""<div class="tip-box">
                <span class="tip-icon">{icon}</span>
                <span class="tip-text">{text}</span>
            </div>""", unsafe_allow_html=True)


# ─── MODEL INSIGHTS TAB ───
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔬 Model Insights & Dataset Info"):
    mi1, mi2, mi3 = st.columns(3)
    with mi1:
        st.markdown("**Random Forest**")
        st.metric("R² Score", f"{MODEL_METRICS['rf_r2']:.1%}")
        st.metric("MAE", f"{MODEL_METRICS['rf_mae']:,.0f}")
    with mi2:
        st.markdown("**XGBoost**")
        st.metric("R² Score", f"{MODEL_METRICS['xgb_r2']:.1%}")
        st.metric("MAE", f"{MODEL_METRICS['xgb_mae']:,.0f}")
    with mi3:
        st.markdown("**Ensemble**")
        st.metric("R² Score", f"{MODEL_METRICS['ens_r2']:.1%}")
        st.metric("MAE", f"{MODEL_METRICS['ens_mae']:,.0f}")

    st.markdown("---")
    st.markdown("**Top Feature Importances**")
    fi_df = pd.DataFrame({
        "Feature": list(MODEL_METRICS["feature_importances"].keys()),
        "Importance": list(MODEL_METRICS["feature_importances"].values())
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color_discrete_sequence=["#818cf8"])
    fig_fi.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(99,102,241,0.1)",
                   tickfont={"color": "#64748b", "size": 10}),
        yaxis=dict(showgrid=False, tickfont={"color": "#94a3b8", "size": 11}),
        font={"color": "#94a3b8"}
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    **Dataset Summary**
    - Original Instagram Reels: 811 rows
    - Synthetic TikTok-style data: 3,000 rows
    - Total training rows: {MODEL_METRICS['total_rows']:,}
    - Features engineered: 20
    - Engagement formula: `Likes×1 + Comments×3 + Shares×2`
    """)

# Footer
st.markdown("""
<br>
<div style="text-align:center; color:#1e293b; font-size:0.78rem; padding:1rem 0; border-top: 1px solid rgba(99,102,241,0.1);">
    Built with Random Forest + XGBoost + OpenCV &nbsp;·&nbsp; Instagram Reel Engagement Predictor
</div>
""", unsafe_allow_html=True)
