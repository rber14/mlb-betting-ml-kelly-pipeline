#!/usr/bin/env python3
"""
calibrate_model.py

Re-calibrates an existing mlb_winprob_pipeline by fitting an isotonic
or sigmoid calibration layer on newly logged game data.

Inputs:
  PIPELINE_PATH     – path to existing .pkl pipeline
  FEATURES_JSON     – path to feature list JSON
  LOG_CSV           – path to calibration log (features + p_pred + y_true)

Outputs:
  OUTPUT_PIPELINE   – upgraded pipeline file with new calibrator

# pip install pandas scikit-learn joblib xgboost


"""

import os
import json
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
PIPELINE_PATH   = os.getenv("PIPELINE_PATH",   "mlb_winprob_pipeline_v1.pkl")
FEATURES_JSON   = os.getenv("FEATURES_JSON",   "features_v1.json")
LOG_CSV         = os.getenv("LOG_CSV",         "calibration_log.csv")
OUTPUT_PIPELINE = os.getenv("OUTPUT_PIPELINE", "mlb_winprob_pipeline_v1_calibrated.pkl")
CALIB_METHOD    = os.getenv("CALIB_METHOD",    "isotonic")  # or "sigmoid"

# ─── LOAD ARTIFACTS & LOG ───────────────────────────────────────────────────────
pipeline    = joblib.load(PIPELINE_PATH)
feature_cols = json.load(open(FEATURES_JSON, "r"))

log_df = pd.read_csv(LOG_CSV)
# ensure p_pred and y_true exist
if "p_pred" not in log_df or "y_true" not in log_df:
    raise KeyError("calibration_log.csv must contain 'p_pred' and 'y_true' columns")

# ─── PREPARE CALIBRATION SET ─────────────────────────────────────────────────────
X_cal = log_df[feature_cols]
y_cal = log_df["y_true"]
p_cal = log_df["p_pred"]

# ─── EVALUATE CURRENT CALIBRATION ───────────────────────────────────────────────
orig_brier = brier_score_loss(y_cal, p_cal)
print(f"Original Brier score: {orig_brier:.4f}")

# ─── FIT NEW CALIBRATOR ─────────────────────────────────────────────────────────
base_model = pipeline.named_steps["model"]
calibrator = CalibratedClassifierCV(
    base_estimator=base_model,
    method=CALIB_METHOD,
    cv="prefit"
)
# fit uses X_cal & raw predicted probs
calibrator.fit(X_cal, y_cal)

# ─── REPLACE & SAVE PIPELINE ───────────────────────────────────────────────────
pipeline.named_steps["model"] = calibrator
joblib.dump(pipeline, OUTPUT_PIPELINE)
print(f"Saved recalibrated pipeline ➞ {OUTPUT_PIPELINE}")

# ─── VALIDATE NEW CALIBRATION ──────────────────────────────────────────────────
p_new = pipeline.predict_proba(X_cal)[:,1]
new_brier = brier_score_loss(y_cal, p_new)
print(f"New Brier score:      {new_brier:.4f}")
