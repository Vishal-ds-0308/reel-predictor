import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN ORIGINAL 8K DATASET
# ─────────────────────────────────────────────
print("📦 Loading original 8K dataset...")
df_orig = pd.read_csv("D:\Imarticus\PROJECTS\AI SOCIAL MEDIA ENGAGEMENT PREDICTOR\combined_dataset.csv")

# Keep only Instagram Reels
df_reels = df_orig[
    (df_orig["Platform"] == "Instagram") &
    (df_orig["Content_Format"] == "Reel")
].copy()
print(f"   Instagram Reels found: {len(df_reels)}")

# Build engagement score with your weights: Likes×1, Comments×3, Shares×2
df_reels["engagement_score"] = (
    df_reels["Likes"] * 1 +
    df_reels["Comments"] * 3 +
    df_reels["Shares"] * 2
)

# Rename columns to unified names
df_reels = df_reels.rename(columns={
    "Hour": "hour",
    "Day": "day_name",
    "Content_Category": "category",
    "Engagement_Rate": "engagement_rate"
})

# Add missing columns with synthetic but realistic values
df_reels["followers_count"] = np.random.randint(5000, 500000, size=len(df_reels))
df_reels["duration_seconds"] = np.random.choice([15, 30, 45, 60, 90], size=len(df_reels))
df_reels["hashtag_count"] = np.random.randint(3, 20, size=len(df_reels))
df_reels["caption_length"] = np.random.randint(10, 300, size=len(df_reels))
df_reels["has_trending_audio"] = np.random.choice([0, 1], size=len(df_reels), p=[0.4, 0.6])
df_reels["female_presence_score"] = np.round(np.random.uniform(0.0, 1.0, size=len(df_reels)), 2)
df_reels["face_count"] = np.random.randint(0, 5, size=len(df_reels))
df_reels["brightness_score"] = np.round(np.random.uniform(0.3, 1.0, size=len(df_reels)), 2)
df_reels["colorfulness_score"] = np.round(np.random.uniform(0.2, 1.0, size=len(df_reels)), 2)

df_reels["source"] = "original"
print(f"   Original reels prepared: {len(df_reels)}")


# ─────────────────────────────────────────────
# STEP 2 — BUILD SYNTHETIC TIKTOK-STYLE DATA
# ─────────────────────────────────────────────
print("\n🔧 Building synthetic TikTok/Instagram Reels data...")

N = 3000
categories = ["Entertainment", "Sports", "Lifestyle", "Tech", "Education", "Business"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Realistic follower tiers
follower_tiers = np.random.choice(
    [5000, 15000, 50000, 150000, 500000, 1000000],
    size=N,
    p=[0.25, 0.30, 0.20, 0.13, 0.08, 0.04]
)

hour_arr = np.random.choice(range(6, 24), size=N)
day_arr = np.random.choice(days, size=N)
category_arr = np.random.choice(categories, size=N, p=[0.25, 0.18, 0.20, 0.15, 0.12, 0.10])
duration_arr = np.random.choice([7, 15, 30, 45, 60, 90], size=N, p=[0.10, 0.25, 0.30, 0.15, 0.12, 0.08])
hashtag_arr = np.random.randint(3, 25, size=N)
caption_arr = np.random.randint(10, 350, size=N)
trending_arr = np.random.choice([0, 1], size=N, p=[0.35, 0.65])
female_arr = np.round(np.random.uniform(0.0, 1.0, size=N), 2)
face_arr = np.random.randint(0, 6, size=N)
brightness_arr = np.round(np.random.uniform(0.3, 1.0, size=N), 2)
colorful_arr = np.round(np.random.uniform(0.2, 1.0, size=N), 2)

# Engagement rate influenced by real factors
base_rate = np.random.uniform(0.04, 0.08, size=N)

# Peak hour bonus (6-9pm)
peak_bonus = np.where((hour_arr >= 18) & (hour_arr <= 21), 0.02, 0.0)

# Weekend bonus
weekend_bonus = np.where(np.isin(day_arr, ["Saturday", "Sunday"]), 0.015, 0.0)

# Trending audio bonus
audio_bonus = trending_arr * 0.025

# Category bonus
cat_bonus = np.where(np.isin(category_arr, ["Entertainment", "Sports"]), 0.02, 0.005)

# Duration sweet spot (15-45s)
dur_bonus = np.where((duration_arr >= 15) & (duration_arr <= 45), 0.01, 0.0)

# Female presence boost (your hypothesis)
female_bonus = female_arr * 0.02

# Follower penalty for very high counts (engagement rate drops)
follower_penalty = np.where(follower_tiers > 500000, 0.015, 0.0)

total_rate = (base_rate + peak_bonus + weekend_bonus + audio_bonus +
              cat_bonus + dur_bonus + female_bonus - follower_penalty)
total_rate = np.clip(total_rate, 0.03, 0.18)

# Calculate engagement score from rate and simulated reach
simulated_reach = follower_tiers * np.random.uniform(0.8, 2.5, size=N)
likes = (simulated_reach * total_rate * np.random.uniform(0.5, 0.7, size=N)).astype(int)
comments = (likes * np.random.uniform(0.05, 0.15, size=N)).astype(int)
shares = (likes * np.random.uniform(0.03, 0.12, size=N)).astype(int)
engagement_score = likes * 1 + comments * 3 + shares * 2

df_tiktok = pd.DataFrame({
    "hour": hour_arr,
    "day_name": day_arr,
    "category": category_arr,
    "followers_count": follower_tiers,
    "duration_seconds": duration_arr,
    "hashtag_count": hashtag_arr,
    "caption_length": caption_arr,
    "has_trending_audio": trending_arr,
    "female_presence_score": female_arr,
    "face_count": face_arr,
    "brightness_score": brightness_arr,
    "colorfulness_score": colorful_arr,
    "engagement_rate": np.round(total_rate, 4),
    "engagement_score": engagement_score,
    "source": "tiktok_synthetic"
})
print(f"   Synthetic TikTok rows built: {len(df_tiktok)}")


# ─────────────────────────────────────────────
# STEP 3 — COMBINE BOTH DATASETS
# ─────────────────────────────────────────────
print("\n🔗 Combining datasets...")

COMMON_COLS = [
    "hour", "day_name", "category", "followers_count",
    "duration_seconds", "hashtag_count", "caption_length",
    "has_trending_audio", "female_presence_score", "face_count",
    "brightness_score", "colorfulness_score",
    "engagement_rate", "engagement_score", "source"
]

df_combined = pd.concat(
    [df_reels[COMMON_COLS], df_tiktok[COMMON_COLS]],
    ignore_index=True
)
print(f"   Combined dataset shape: {df_combined.shape}")


# ─────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n⚙️  Engineering features...")

df_combined["is_weekend"] = df_combined["day_name"].isin(["Saturday", "Sunday"]).astype(int)
df_combined["is_peak_hour"] = ((df_combined["hour"] >= 18) & (df_combined["hour"] <= 21)).astype(int)
df_combined["is_morning"] = ((df_combined["hour"] >= 7) & (df_combined["hour"] <= 10)).astype(int)
df_combined["is_high_eng_category"] = df_combined["category"].isin(["Entertainment", "Sports"]).astype(int)
df_combined["is_short_reel"] = ((df_combined["duration_seconds"] >= 15) & (df_combined["duration_seconds"] <= 45)).astype(int)
df_combined["followers_log"] = np.log1p(df_combined["followers_count"])
df_combined["hashtag_optimal"] = ((df_combined["hashtag_count"] >= 5) & (df_combined["hashtag_count"] <= 15)).astype(int)
df_combined["engagement_potential"] = (
    df_combined["followers_log"] * 0.4 +
    df_combined["female_presence_score"] * 0.2 +
    df_combined["brightness_score"] * 0.2 +
    df_combined["colorfulness_score"] * 0.2
)

# Encode day name
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df_combined["day_num"] = df_combined["day_name"].map({d: i for i, d in enumerate(day_order)})

# Encode category
le = LabelEncoder()
df_combined["category_enc"] = le.fit_transform(df_combined["category"])

print(f"   Total features engineered: {df_combined.shape[1]}")


# ─────────────────────────────────────────────
# STEP 5 — TRAIN MODELS
# ─────────────────────────────────────────────
print("\n🤖 Training models...")

FEATURE_COLS = [
    "hour", "day_num", "is_weekend", "is_peak_hour", "is_morning",
    "category_enc", "is_high_eng_category",
    "followers_count", "followers_log",
    "duration_seconds", "is_short_reel",
    "hashtag_count", "hashtag_optimal",
    "caption_length", "has_trending_audio",
    "female_presence_score", "face_count",
    "brightness_score", "colorfulness_score",
    "engagement_potential"
]

TARGET = "engagement_score"

df_model = df_combined[FEATURE_COLS + [TARGET]].dropna()
X = df_model[FEATURE_COLS]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"   Random Forest  →  MAE: {rf_mae:.0f}  |  R²: {rf_r2:.3f}")

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"   XGBoost        →  MAE: {xgb_mae:.0f}  |  R²: {xgb_r2:.3f}")

# Ensemble (average of both)
ensemble_pred = (rf_pred + xgb_pred) / 2
ens_mae = mean_absolute_error(y_test, ensemble_pred)
ens_r2 = r2_score(y_test, ensemble_pred)
print(f"   Ensemble       →  MAE: {ens_mae:.0f}  |  R²: {ens_r2:.3f}")

# Feature importances
fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\n   Top 5 features:")
for feat, imp in fi.head(5).items():
    print(f"     {feat}: {imp:.3f}")


# ─────────────────────────────────────────────
# STEP 6 — SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("\n💾 Saving model artifacts...")
os.makedirs("/home/claude/reel_predictor/models", exist_ok=True)

pickle.dump(rf, open("/home/claude/reel_predictor/models/rf_model.pkl", "wb"))
pickle.dump(xgb_model, open("/home/claude/reel_predictor/models/xgb_model.pkl", "wb"))
pickle.dump(scaler, open("/home/claude/reel_predictor/models/scaler.pkl", "wb"))
pickle.dump(le, open("/home/claude/reel_predictor/models/label_encoder.pkl", "wb"))
pickle.dump(FEATURE_COLS, open("/home/claude/reel_predictor/models/feature_cols.pkl", "wb"))

# Save combined dataset
df_combined.to_csv("/home/claude/reel_predictor/models/combined_dataset.csv", index=False)

# Save metrics for display in app
metrics = {
    "rf_mae": round(rf_mae, 2), "rf_r2": round(rf_r2, 3),
    "xgb_mae": round(xgb_mae, 2), "xgb_r2": round(xgb_r2, 3),
    "ens_mae": round(ens_mae, 2), "ens_r2": round(ens_r2, 3),
    "total_rows": len(df_combined),
    "feature_importances": fi.head(10).to_dict()
}
pickle.dump(metrics, open("/home/claude/reel_predictor/models/metrics.pkl", "wb"))

print("\n✅ All artifacts saved successfully!")
print(f"\n{'='*50}")
print(f"FINAL SUMMARY")
print(f"{'='*50}")
print(f"Total training rows : {len(df_combined)}")
print(f"Original reels      : {len(df_reels)}")
print(f"Synthetic TikTok    : {len(df_tiktok)}")
print(f"Features used       : {len(FEATURE_COLS)}")
print(f"Best model (Ensemble) R²: {ens_r2:.3f}")
