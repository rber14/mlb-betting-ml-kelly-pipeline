'''
To Run: 
pip install pandas scikit-learn xgboost joblib
python3 train_model.py

You’ll end up with mlb_winprob_pipeline_v1.pkl and features_v1.json ready to load in your daily prediction workflow.
'''

#!/usr/bin/env python3
"""
train_model.py

1. Loads training_data.csv
2. Builds scaler + XGB pipeline
3. Calibrates probabilities (isotonic)
4. Saves mlb_winprob_pipeline_v1.pkl & features_v1.json
"""

import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
TRAIN_CSV     = "training_data.csv"
PIPELINE_OUT  = "mlb_winprob_pipeline_v1.pkl"
FEATURES_OUT  = "features_v1.json"

# ─── LOAD & PREPARE DATA ─────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV)

# Drop non-feature columns
drop_cols = [
    "date","time","venue",
    "home_team_id","away_team_id",
    "home_score","away_score"
]
feature_cols = [c for c in df.columns if c not in drop_cols + ["target"]]
X = df[feature_cols]
y = df["target"]

# ─── BUILD & TRAIN PIPELINE ─────────────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        use_label_encoder=False,
        eval_metric="logloss"
    ))
])
pipeline.fit(X, y)

# ─── CALIBRATE PROBABILITIES ─────────────────────────────────────────────────────
calibrator = CalibratedClassifierCV(
    base_estimator=pipeline.named_steps["model"],
    method="isotonic",
    cv="prefit"
)
calibrator.fit(X, y)
pipeline.named_steps["model"] = calibrator

# ─── PERSIST ARTIFACTS ───────────────────────────────────────────────────────────
# 1. Full pipeline
joblib.dump(pipeline, PIPELINE_OUT)
print(f"Saved pipeline ➞ {PIPELINE_OUT}")

# 2. Feature ordering
with open(FEATURES_OUT, "w") as f:
    json.dump(feature_cols, f)
print(f"Saved feature list ➞ {FEATURES_OUT}")
