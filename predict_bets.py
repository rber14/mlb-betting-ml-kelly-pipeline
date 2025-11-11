#!/usr/bin/env python3
'''
Proper way to run:
 Phase 1. 
 Build historical training data -> run build_training_data.py which outputs training_data.csv
 Train and Persist your model -> train_model.py which outputs mlb_winprob_pipeline_v1 and mlb_winprob_pipeline_v1

 note: You only need to rerun these two scripts when you want to expand or redo your training history or change modeling parameters.

 Phase 2.
 - Each day you'll generate tomorrow's features -> run mlb_game_features which Reads schedule + odds + weather + starter & team stats → writes tomorrows_games_features.csv
 - Produce your bet table -> run predict_bets.py which Reads features + model → writes tomorrow_bets.csv with model_p, edge, kelly, stake, etc.
 - After games conclude run log outcomes for calibration -> run calibration log

 Phase 3. Recommended 
 - After 200 plus games or accumulated sufficient rows run the calibrator which loads the mlb_winprob_pipeline_v1.pkl, features_v1.json, and calibration_log.csv
 - Then SWAP in the updated pipeline 
 Rename or move mlb_winprob_pipeline_v1_calibrated.pkl back to mlb_winprob_pipeline_v1.pkl
 Future runs of predict_bets.py will now use the recalibrated model
 Example Bash Wrapper: run_dail.sh

'''

"""
predict_bets.py

Loads tomorrow’s game features and your trained model pipeline,
then outputs a CSV with betting metrics:
  • model_p (win probability)
  • imp_p (market implied probability)
  • edge_pct
  • confidence
  • kelly_wt
  • stake_$
  • ev_$
  • risk tier

Dependencies:
  pip install pandas joblib

Environment variables:
  PIPELINE_PATH    – path to your .pkl pipeline file (default: mlb_winprob_pipeline_v1.pkl)
  FEATURES_CSV     – path to tomorrows_games_features.csv (default: tomorrows_games_features.csv)
  OUTPUT_CSV       – path for output bets CSV (default: tomorrow_bets.csv)
  BANKROLL_SIZE    – total bankroll in USD (default: 130)

Usage:
  export PIPELINE_PATH="mlb_winprob_pipeline_v1.pkl"
  export FEATURES_CSV="tomorrows_games_features.csv"
  export OUTPUT_CSV="tomorrow_bets.csv"
  export BANKROLL_SIZE=130
  python3 predict_bets.py
"""

import os
import sys
import pandas as pd
import joblib

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
PIPELINE_PATH   = os.getenv("PIPELINE_PATH", "mlb_winprob_pipeline_v1.pkl")
FEATURES_CSV    = os.getenv("FEATURES_CSV", "tomorrows_games_features.csv")
OUTPUT_CSV      = os.getenv("OUTPUT_CSV", "tomorrow_bets.csv")
BANKROLL        = float(os.getenv("BANKROLL_SIZE", 130))

# ─── UTILITIES ──────────────────────────────────────────────────────────────────
def implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (100 + odds)
    else:
        return -odds / (-odds + 100)

def kelly_weight(p: float, odds: float) -> float:
    """Compute full Kelly fraction given model P and American odds."""
    # net decimal odds b = abs(odds)/100
    b = abs(odds) / 100
    return max(0.0, (p * b - (1 - p)) / b)

def risk_tier(stake: float) -> str:
    """Bin stake into Low/Medium/High risk tiers."""
    if stake < 15:
        return "Low"
    if stake <= 30:
        return "Medium"
    return "High"

# ─── MAIN PROCESS ────────────────────────────────────────────────────────────────
def main():
    # 1. Load model pipeline
    try:
        pipeline = joblib.load(PIPELINE_PATH)
    except FileNotFoundError:
        sys.exit(f"Error: Pipeline file not found at '{PIPELINE_PATH}'")

    # 2. Load tomorrow’s features
    try:
        df = pd.read_csv(FEATURES_CSV)
    except FileNotFoundError:
        sys.exit(f"Error: Features CSV not found at '{FEATURES_CSV}'")

    # 3. Predict win probabilities
    #    Assumes pipeline expects columns in df.columns order or has .feature_names attribute
    feature_cols = getattr(pipeline, "feature_names", df.columns)
    df["model_p"] = pipeline.predict_proba(df[feature_cols])[:, 1]

    # 4. Compute implied probabilities for home & away picks
    #    In your features CSV, odds may be in 'home_odds' and 'away_odds'.
    #    Here we pivot to one row per pick: home and away.
    rows = []
    for _, row in df.iterrows():
        for side in ("home", "away"):
            odds_col = f"{side}_odds"
            team_col = f"{side}_team"
            p = row["model_p"] if side == "home" else row["model_p"]
            imp = implied_prob(row[odds_col])
            edge = p - imp
            kw = kelly_weight(p, row[odds_col])
            stake = kw * BANKROLL
            ev    = stake * edge
            rows.append({
                "date":          row["date"],
                "time":          row["time"],
                "venue":         row["venue"],
                "game":          f"{row['away_team']} @ {row['home_team']}",
                "pick":          f"{row[team_col]} ML ({int(row[odds_col]):+})",
                "odds":          row[odds_col],
                "model_p":       round(p, 3),
                "imp_p":         round(imp, 3),
                "edge_pct":      round(edge * 100, 2),
                "confidence":    round(p * 100, 1),
                "kelly_wt":      round(kw, 3),
                "stake_$":       round(stake, 2),
                "ev_$":          round(ev, 2),
                "risk":          risk_tier(stake)
            })

    out_df = pd.DataFrame(rows)

    # 5. Save output CSV
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out_df)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
