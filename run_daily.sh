#!/usr/bin/env bash
set -euo pipefail

# env vars
export ODDS_API_KEY="…"
export OPENWEATHER_API_KEY="…"
export PIPELINE_PATH="mlb_winprob_pipeline_v1.pkl"
export FEATURES_JSON="features_v1.json"
export FEATURES_CSV="tomorrows_games_features.csv"
export OUTPUT_CSV="tomorrow_bets.csv"
export LOG_CSV="calibration_log.csv"

# 1. Generate features for tomorrow’s games
python3 mlb_game_features.py

# 2. Predict bets
python3 predict_bets.py

# (Your bettors review tomorrow_bets.csv here)

# 3. At day’s end, log actual results for calibration
python3 calibration_log.py


# schedule cron 
# 0 18 * * * /path/to/run_daily.sh >> /var/log/mlb_daily.log 2>&1

# schedule calibration log
#0 5 * * * /path/to/run_daily.sh --calibrate >> /var/log/mlb_daily.log 2>&1
