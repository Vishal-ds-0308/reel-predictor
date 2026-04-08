import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
import tempfile

# Load model
model = joblib.load("engagement_model.pkl")

# Feature list (must match training)
features = [
    'hour','day_num','category_enc','followers_log',
    'duration_seconds','hashtag_count','caption_length',
    'has_trending_audio','female_presence_score','face_count',
    'brightness_score','colorfulness_score','is_weekend',
    'is_peak_hour','is_morning','is_high_eng_category',
    'is_short_reel','hashtag_optimal'
]

# 🎥 Extract video features
def extract_video_features(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    brightness_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(gray.mean())
    
    cap.release()
    
    avg_brightness = np.mean(brightness_values) if brightness_values else 120
    
    return duration, avg_brightness

# 🔮 Prediction
def predict_engagement(input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]

# 🧠 Label
def engagement_label(score):
    if score < 4:
        return "Low 😕"
    elif score < 6:
        return "Medium 🙂"
    else:
        return "High 🔥"

# ⏰ Best hour
def best_posting_hour(input_data):
    best_hour = 0
    best_score = -1
    
    for h in range(24):
        input_data['hour'] = h
        score = predict_engagement(input_data)
        
        if score > best_score:
            best_score = score
            best_hour = h
    
    return best_hour, best_score

# ================= UI =================

st.title("📊 AI Reel Engagement Predictor")

uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])

day = st.selectbox("Select Day", 
                   ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

hour = st.slider("Select Posting Hour", 0, 23, 18)

category = st.selectbox("Content Category", 
                        ["Entertainment","Education","Lifestyle","Fashion"])

followers = st.number_input("Followers Count", value=10000)

hashtag_count = st.slider("Number of Hashtags", 0, 30, 8)
caption_length = st.slider("Caption Length", 0, 300, 100)
trending_audio = st.selectbox("Trending Audio?", [0,1])

# Encode manually (simple)
day_map = {
    "Monday":0,"Tuesday":1,"Wednesday":2,
    "Thursday":3,"Friday":4,"Saturday":5,"Sunday":6
}

category_map = {
    "Entertainment":0,
    "Education":1,
    "Lifestyle":2,
    "Fashion":3
}

if st.button("🚀 Predict Engagement"):
    
    if uploaded_file is not None:
        duration, brightness = extract_video_features(uploaded_file)
        
        input_data = {
            'hour': hour,
            'day_num': day_map[day],
            'category_enc': category_map[category],
            'followers_log': np.log1p(followers),
            'duration_seconds': duration,
            'hashtag_count': hashtag_count,
            'caption_length': caption_length,
            'has_trending_audio': trending_audio,
            'female_presence_score': 0.5,
            'face_count': 1,
            'brightness_score': brightness,
            'colorfulness_score': 120,
            'is_weekend': 1 if day in ["Saturday","Sunday"] else 0,
            'is_peak_hour': 1 if 18 <= hour <= 22 else 0,
            'is_morning': 1 if 6 <= hour <= 11 else 0,
            'is_high_eng_category': 1,
            'is_short_reel': 1 if duration < 20 else 0,
            'hashtag_optimal': 1 if 5 <= hashtag_count <= 12 else 0
        }
        
        score = predict_engagement(input_data)
        label = engagement_label(score)
        
        best_hour, best_score = best_posting_hour(input_data.copy())
        
        st.subheader(f"📈 Predicted Engagement Score: {score:.2f}")
        st.subheader(f"🔥 Engagement Level: {label}")
        
        st.subheader(f"⏰ Best Posting Hour: {best_hour}:00")
        st.write(f"Expected Score at Best Time: {best_score:.2f}")
        
    else:
        st.warning("Please upload a video!")