#!/usr/bin/env python3
"""
calibration_log.py

After games finish, this script:
  • Loads your trained pipeline
  • Loads tomorrow's-features CSV (used for predictions)
  • Computes raw probabilities (p_pred)
  • Fetches actual game outcomes (y_true) from MLB Stats API
  • Appends all features + p_pred + y_true to calibration_log.csv
"""

import os
import sys
import json
import joblib
import pandas as pd
import requests
from datetime import datetime

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
PIPELINE_PATH = os.getenv("PIPELINE_PATH",   "mlb_winprob_pipeline_v1.pkl")
FEATURES_JSON = os.getenv("FEATURES_JSON",   "features_v1.json")
FEATURES_CSV  = os.getenv("FEATURES_CSV",    "tomorrows_games_features.csv")
LOG_CSV       = os.getenv("LOG_CSV",         "calibration_log.csv")

# ─── VALIDATE ────────────────────────────────────────────────────────────────────
if not os.path.exists(PIPELINE_PATH):
    sys.exit(f"Error: pipeline not found at {PIPELINE_PATH}")
if not os.path.exists(FEATURES_JSON):
    sys.exit(f"Error: features JSON not found at {FEATURES_JSON}")
if not os.path.exists(FEATURES_CSV):
    sys.exit(f"Error: features CSV not found at {FEATURES_CSV}")

# ─── LOAD ARTIFACTS ──────────────────────────────────────────────────────────────
pipeline = joblib.load(PIPELINE_PATH)
with open(FEATURES_JSON, "r") as f:
    feature_cols = json.load(f)

# ─── LOAD FEATURES & PREDICT ────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_CSV)
X = df[feature_cols]
df["p_pred"] = pipeline.predict_proba(X)[:, 1]

# ─── FETCH ACTUAL OUTCOMES ──────────────────────────────────────────────────────
# Assumes all rows share the same date
game_date = df["date"].iloc[0]
url       = "https://statsapi.mlb.com/api/v1/schedule"
params    = {"sportId": 1, "date": game_date, "hydrate": "linescore"}
res       = requests.get(url, params=params).json().get("dates", [])

# Build mapping: gamePk -> outcome (1 if home win, else 0)
outcomes = {}
for day in res:
    for g in day["games"]:
        if g["status"]["codedGameState"] != "F":
            continue
        pk = g["gamePk"]
        hr = g["linescore"]["teams"]["home"]["runs"]
        ar = g["linescore"]["teams"]["away"]["runs"]
        outcomes[pk] = 1 if hr > ar else 0

# Map into df (will be NaN if a game isn't found)
if "gamePk" in df.columns:
    df["y_true"] = df["gamePk"].map(outcomes)
else:
    sys.exit("Error: 'gamePk' column missing from features CSV")

# ─── APPEND TO LOG ───────────────────────────────────────────────────────────────
# Keep all cols so you have context in the log
log_df = df.copy()

# Write header only if file does not exist
if os.path.exists(LOG_CSV):
    log_df.to_csv(LOG_CSV, mode="a", index=False, header=False)
else:
    log_df.to_csv(LOG_CSV, index=False)

print(f"[{datetime.utcnow().isoformat()}] Logged {len(log_df)} rows to {LOG_CSV}")
