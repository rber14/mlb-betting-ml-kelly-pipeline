#!/usr/bin/env python3
"""
tomorrow_features.py

A turnkey script to fetch tomorrow’s MLB game-day features:
  • Schedule & probable pitchers via MLB Stats API
  • Moneyline odds via The Odds API
  • Weather via OpenWeatherMap
  • Starter & team stats via MLB Stats API
  • Park factors & recent form

Dependencies:
  pip install requests pandas python-dateutil

Environment variables (export before running):
  ODDS_API_KEY       – your The Odds API key # https://the-odds-api.com/#get-access
  OPENWEATHER_API_KEY – your OpenWeatherMap API key

Outputs:
  tomorrows_games_features.csv
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
ODDS_API_KEY      = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY   = os.getenv("OPENWEATHER_API_KEY")
OUTPUT_CSV_PATH   = "tomorrows_games_features.csv"
ODDS_BOOKMAKER    = "draftkings"

if not ODDS_API_KEY or not WEATHER_API_KEY:
    raise RuntimeError("Set ODDS_API_KEY & OPENWEATHER_API_KEY env vars before running.")

# Tomorrow’s date string & season year
tomorrow_dt = datetime.utcnow() + timedelta(days=1)
DATE_STR    = tomorrow_dt.strftime("%Y-%m-%d")
SEASON_YEAR = tomorrow_dt.year

# All 30 MLB parks for weather lookups
VENUE_COORDS = {
    "Chase Field":                {"lat": 33.4458,  "lon": -112.0665},
    "Citizens Bank Park":         {"lat": 39.9061,  "lon": -75.1665},
    "Citi Field":                 {"lat": 40.7571,  "lon": -73.8458},
    "Comerica Park":              {"lat": 42.3390,  "lon": -83.0486},
    "Coors Field":                {"lat": 39.7559,  "lon": -104.9942},
    "Dodger Stadium":             {"lat": 34.0739,  "lon": -118.2390},
    "Fenway Park":                {"lat": 42.3467,  "lon": -71.0972},
    "Globe Life Field":           {"lat": 32.7518,  "lon": -97.0822},
    "Great American Ball Park":   {"lat": 39.0964,  "lon": -84.5060},
    "Kauffman Stadium":           {"lat": 39.0514,  "lon": -94.4803},
    "LoanDepot Park":             {"lat": 25.7781,  "lon": -80.2195},
    "Marlins Park":               {"lat": 25.7781,  "lon": -80.2195},
    "Minute Maid Park":           {"lat": 29.7573,  "lon": -95.3553},
    "Nationals Park":             {"lat": 38.8730,  "lon": -77.0074},
    "Oakland Coliseum":           {"lat": 37.7516,  "lon": -122.2005},
    "Oriole Park at Camden Yards":{"lat": 39.2839,  "lon": -76.6210},
    "Oracle Park":                {"lat": 37.7786,  "lon": -122.3893},
    "Petco Park":                 {"lat": 32.7076,  "lon": -117.1570},
    "PNC Park":                   {"lat": 40.4469,  "lon": -80.0057},
    "Progressive Field":          {"lat": 41.4953,  "lon": -81.6850},
    "Rogers Centre":              {"lat": 43.6414,  "lon": -79.3894},
    "Target Field":               {"lat": 44.9817,  "lon": -93.2777},
    "T-Mobile Park":              {"lat": 47.5915,  "lon": -122.3325},
    "Truist Park":                {"lat": 33.8908,  "lon": -84.4677},
    "Tropicana Field":            {"lat": 27.7684,  "lon": -82.6534},
    "Yankee Stadium":             {"lat": 40.8296,  "lon": -73.9262},
    "Angel Stadium":              {"lat": 33.8003,  "lon": -117.8827},
    "Guaranteed Rate Field":      {"lat": 41.8308,  "lon": -87.6339},
    "American Family Field":      {"lat": 43.0286,  "lon": -87.9712},
    "Miller Park":                {"lat": 43.0286,  "lon": -87.9712},  # alias for American Family Field
}

# Static park factors (>1 = hitter-friendly)
PARK_FACTORS = {
    "Chase Field": 0.98, "Citizens Bank Park": 1.05, "Citi Field": 1.00,
    "Comerica Park": 0.88, "Coors Field": 1.26, "Dodger Stadium": 0.91,
    "Fenway Park": 1.11, "Globe Life Field": 1.02, "Great American Ball Park": 1.13,
    "Kauffman Stadium": 0.96, "LoanDepot Park": 1.03, "Marlins Park": 1.03,
    "Minute Maid Park": 1.04, "Nationals Park": 1.01, "Oakland Coliseum": 0.90,
    "Oriole Park at Camden Yards": 0.99, "Oracle Park": 0.93, "Petco Park": 0.90,
    "PNC Park": 0.89, "Progressive Field": 0.99, "Rogers Centre": 0.94,
    "Target Field": 0.90, "T-Mobile Park": 0.85, "Truist Park": 0.95,
    "Tropicana Field": 0.97, "Yankee Stadium": 1.03, "Angel Stadium": 1.07,
    "Guaranteed Rate Field": 0.98, "American Family Field": 0.97, "Miller Park": 0.97
}

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def fetch_schedule(date_str):
    """Fetch tomorrow’s games & probable pitchers."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitchers"
    }
    data = requests.get(url, params=params).json()
    rows = []
    for day in data.get("dates", []):
        for g in day["games"]:
            pp = g.get("probablePitchers", {})
            rows.append({
                "gamePk":       g["gamePk"],
                "date":         date_str,
                "time":         datetime.fromisoformat(g["gameDate"]).strftime("%-I:%M %p"),
                "venue":        g["venue"]["name"],
                "home_team_id": g["teams"]["home"]["team"]["id"],
                "away_team_id": g["teams"]["away"]["team"]["id"],
                "home_team":    g["teams"]["home"]["team"]["name"],
                "away_team":    g["teams"]["away"]["team"]["name"],
                "home_sp_id":   pp.get("home", {}).get("id"),
                "away_sp_id":   pp.get("away", {}).get("id"),
            })
    return pd.DataFrame(rows)

def fetch_odds(date_str):
    """Get tomorrow’s moneylines from The Odds API."""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "h2h",
        "oddsFormat": "american"
    }
    data = requests.get(url, params=params).json()
    recs = []
    for game in data:
        if not game["commence_time"].startswith(date_str):
            continue
        for book in game["bookmakers"]:
            if book["key"] == ODDS_BOOKMAKER:
                outcomes = book["markets"][0]["outcomes"]
                # match home and away by name
                home_price = next(o["price"] for o in outcomes if o["name"] == game["home_team"]["name"])
                away_price = next(o["price"] for o in outcomes if o["name"] == game["away_team"]["name"])
                recs.append({
                    "home_team_id": game["home_team"]["id"],
                    "away_team_id": game["away_team"]["id"],
                    "home_odds":    home_price,
                    "away_odds":    away_price
                })
                break
    return pd.DataFrame(recs)

def fetch_sp_stats(sp_id):
    """Grab SP’s season ERA, FIP, xFIP, K/BB, and rest days."""
    if not sp_id:
        return {"era": None, "fip": None, "xfip": None, "k_bb": None, "rest": None}
    # season pitching splits
    url = f"https://statsapi.mlb.com/api/v1/people/{sp_id}/stats"
    params = {"stats": "season", "season": SEASON_YEAR, "group": "pitchingRegularSeason"}
    splits = requests.get(url, params=params).json()["stats"][0]["splits"]
    stat   = splits[0]["stat"] if splits else {}
    # last start date
    gl = requests.get(
        f"https://statsapi.mlb.com/api/v1/people/{sp_id}/gameLog",
        params={"season": SEASON_YEAR, "gameType": "R", "hydrate": "stats"}
    ).json().get("splits", [])
    last_date = gl[0]["date"] if gl else None
    rest_days = (tomorrow_dt.date() - datetime.strptime(last_date, "%Y-%m-%d").date()).days if last_date else None

    return {
        "era":   stat.get("era"),
        "fip":   stat.get("fip"),
        "xfip":  stat.get("xFIP"),
        "k_bb":  stat.get("strikeouts", 0) / max(1, stat.get("walks", 1)),
        "rest":  rest_days
    }

def fetch_all_team_stats():
    """Grab season hitting & pitching stats for every team."""
    teams = {}
    resp = requests.get("https://statsapi.mlb.com/api/v1/teams",
                        params={"season": SEASON_YEAR, "sportIds": 1}).json()["teams"]
    for t in resp:
        tid = t["id"]
        # hitting
        h = requests.get(
            f"https://statsapi.mlb.com/api/v1/teams/{tid}/stats",
            params={"season": SEASON_YEAR, "group": "hittingRegularSeason"}
        ).json()["stats"][0]["splits"][0]["stat"]
        # pitching
        p = requests.get(
            f"https://statsapi.mlb.com/api/v1/teams/{tid}/stats",
            params={"season": SEASON_YEAR, "group": "pitchingRegularSeason"}
        ).json()["stats"][0]["splits"][0]["stat"]
        teams[tid] = {
            "rpg":        p.get("runsAllowedPerGame"),
            "wRC+":       h.get("wRCPlus"),
            "babip":      h.get("battingAverageOnBallsInPlay"),
            "iso":        h.get("isoPower"),
            "drs":        h.get("defensiveRunsSaved"),
            "bp_era":     p.get("bullpenEra"),
            "bp_hl_era":  p.get("bullpenHighLeverageEra")
        }
    return teams

def compute_recent_form(days=10):
    """Compute each team’s win% & run diff over the last `days` days."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    recs = []
    for single in pd.date_range(start.date(), end.date()):
        ds = single.strftime("%Y-%m-%d")
        sched = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId":1,"date":ds,"hydrate":"linescore"}
        ).json().get("dates", [])
        for day in sched:
            for g in day["games"]:
                if g["status"]["codedGameState"] != "F":
                    continue
                hi = g["teams"]["home"]
                ai = g["teams"]["away"]
                hr = g["linescore"]["teams"]["home"]["runs"]
                ar = g["linescore"]["teams"]["away"]["runs"]
                recs.append({"team_id": hi["team"]["id"], "win": int(hr>ar), "run_diff": hr-ar})
                recs.append({"team_id": ai["team"]["id"], "win": int(ar>hr), "run_diff": ar-hr})
    df = pd.DataFrame(recs)
    agg = df.groupby("team_id").agg(last10_win_pct=("win","mean"), run_diff_last10=("run_diff","sum"))
    return agg.to_dict("index")

def fetch_weather_for_venue(venue):
    """Lookup weather at the park for first pitch."""
    coords = VENUE_COORDS.get(venue)
    if not coords:
        return {"temp_f": None, "wind_mph": None, "humidity_pct": None}
    params = {"lat": coords["lat"], "lon": coords["lon"], "appid": WEATHER_API_KEY, "units": "imperial"}
    w = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params).json()
    return {"temp_f": w["main"].get("temp"), "wind_mph": w["wind"].get("speed"), "humidity_pct": w["main"].get("humidity")}

def engineer_features():
    sched_df    = fetch_schedule(DATE_STR)
    odds_df     = fetch_odds(DATE_STR)
    team_stats  = fetch_all_team_stats()
    recent_form = compute_recent_form()

    df = sched_df.merge(odds_df, on=["home_team_id","away_team_id"], how="left")

    # Starter stats
    for side in ("home","away"):
        stats = df[f"{side}_sp_id"].apply(fetch_sp_stats).tolist()
        sp_df = pd.DataFrame(stats).add_prefix(f"{side}_sp_")
        df = pd.concat([df, sp_df], axis=1)

    # Team aggregate stats
    for side in ("home","away"):
        df[f"{side}_rpg"]        = df[f"{side}_team_id"].apply(lambda tid: team_stats.get(tid,{}).get("rpg"))
        df[f"{side}_wRC+"]       = df[f"{side}_team_id"].apply(lambda tid: team_stats.get(tid,{}).get("wRC+"))
        df[f"{side}_BABIP"]      = df[f"{side}_team_id"].apply(lambda tid: team_stats.get(tid,{}).get("babip"))
        df[f"{side}_ISO"]        = df[f"{side}_team_id"].apply(lambda tid: team_stats.get(tid,{}).get("iso"))
        df[f"{side}_DRS"]        = df[f"{side}_team_id"].apply(lambda tid: team_stats.get(tid,{}).get("drs"))
        df[f"{side}_BP_ERA"]     = df[f"{side}_team_id"].apply(lambda tid: team_stats.get(tid,{}).get("bp_era"))
        df[f"{side}_BP_HL_ERA"]  = df[f"{side}_team_id"].apply(lambda tid: team_stats.get(tid,{}).get("bp_hl_era"))

    # Recent form
    df["home_last10_win_pct"]     = df["home_team_id"].apply(lambda tid: recent_form.get(tid,{}).get("last10_win_pct"))
    df["home_run_diff_last10"]    = df["home_team_id"].apply(lambda tid: recent_form.get(tid,{}).get("run_diff_last10"))
    df["away_last10_win_pct"]     = df["away_team_id"].apply(lambda tid: recent_form.get(tid,{}).get("last10_win_pct"))
    df["away_run_diff_last10"]    = df["away_team_id"].apply(lambda tid: recent_form.get(tid,{}).get("run_diff_last10"))

    # Weather & park factor
    weather = df["venue"].apply(fetch_weather_for_venue).tolist()
    weather_df = pd.DataFrame(weather)
    df = pd.concat([df, weather_df], axis=1)
    df["park_factor"] = df["venue"].map(PARK_FACTORS)

    return df

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    features_df = engineer_features()
    features_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[{datetime.utcnow().isoformat()}] Wrote {len(features_df)} rows to {OUTPUT_CSV_PATH}")
