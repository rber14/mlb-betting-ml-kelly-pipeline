#!/usr/bin/env python3
"""
build_training_data.py

Fetches MLB data from 2018-03-01 through 2025-06-30 and produces training_data.csv
with one row per finished game:
  • Features: home/away SP stats, team stats, recent form, park factor, odds
  • Target: 1 if home team won, else 0
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
ODDS_API_KEY   = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise RuntimeError("Set ODDS_API_KEY environment variable before running")

OUTPUT_CSV     = "training_data.csv"
START_DATE     = datetime(2018, 3, 1)
END_DATE       = datetime(2025, 6, 30)
ODDS_BOOKMAKER = "draftkings"

# Static park factors (>1 = hitter-friendly)
PARK_FACTORS = {
    "Chase Field":0.98,"Citizens Bank Park":1.05,"Citi Field":1.00,
    "Comerica Park":0.88,"Coors Field":1.26,"Dodger Stadium":0.91,
    "Fenway Park":1.11,"Globe Life Field":1.02,"Great American Ball Park":1.13,
    "Kauffman Stadium":0.96,"LoanDepot Park":1.03,"Marlins Park":1.03,
    "Minute Maid Park":1.04,"Nationals Park":1.01,"Oakland Coliseum":0.90,
    "Oriole Park at Camden Yards":0.99,"Oracle Park":0.93,"Petco Park":0.90,
    "PNC Park":0.89,"Progressive Field":0.99,"Rogers Centre":0.94,
    "Target Field":0.90,"T-Mobile Park":0.85,"Truist Park":0.95,
    "Tropicana Field":0.97,"Yankee Stadium":1.03,"Angel Stadium":1.07,
    "Guaranteed Rate Field":0.98,"American Family Field":0.97,"Miller Park":0.97
}

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def daterange(start, end):
    curr = start
    while curr <= end:
        yield curr
        curr += timedelta(days=1)

def fetch_schedule(date_str):
    """Return DataFrame of finished games on date_str with probable pitchers."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId":1, "date":date_str, "hydrate":"probablePitchers,linescore"}
    data = requests.get(url, params=params).json().get("dates", [])
    rows = []
    for day in data:
        for g in day["games"]:
            if g["status"]["codedGameState"] != "F":
                continue
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]
            lines = g["linescore"]["teams"]
            pp    = g.get("probablePitchers", {})
            rows.append({
                "date":          date_str,
                "time":          datetime.fromisoformat(g["gameDate"]).strftime("%H:%M"),
                "venue":         g["venue"]["name"],
                "home_team_id":  home["id"],
                "away_team_id":  away["id"],
                "home_sp_id":    pp.get("home", {}).get("id"),
                "away_sp_id":    pp.get("away", {}).get("id"),
                "home_score":    lines["home"]["runs"],
                "away_score":    lines["away"]["runs"],
            })
    return pd.DataFrame(rows)

def fetch_odds(date_str):
    """Return DataFrame of moneyline odds for games on date_str."""
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {"apiKey":ODDS_API_KEY, "regions":"us", "markets":"h2h", "oddsFormat":"american"}
    data = requests.get(url, params=params).json()
    recs = []
    for game in data:
        if not game["commence_time"].startswith(date_str):
            continue
        for book in game["bookmakers"]:
            if book["key"] != ODDS_BOOKMAKER:
                continue
            outcomes = book["markets"][0]["outcomes"]
            home_price = next(o["price"] for o in outcomes if o["name"]==game["home_team"]["name"])
            away_price = next(o["price"] for o in outcomes if o["name"]==game["away_team"]["name"])
            recs.append({
                "home_team_id": game["home_team"]["id"],
                "away_team_id": game["away_team"]["id"],
                "home_odds":    home_price,
                "away_odds":    away_price
            })
            break
    return pd.DataFrame(recs)

def fetch_sp_stats(sp_id, game_date):
    """Return ERA, FIP, xFIP, K/BB, rest days for a starter on game_date."""
    if not sp_id:
        return {"era":None,"fip":None,"xfip":None,"k_bb":None,"rest":None}
    season = datetime.fromisoformat(game_date).year
    # season splits
    url = f"https://statsapi.mlb.com/api/v1/people/{sp_id}/stats"
    params = {"stats":"season","season":season,"group":"pitchingRegularSeason"}
    splits = requests.get(url, params=params).json()["stats"][0].get("splits",[])
    stat   = splits[0]["stat"] if splits else {}
    # last start
    gl = requests.get(
        f"https://statsapi.mlb.com/api/v1/people/{sp_id}/gameLog",
        params={"season":season,"gameType":"R","hydrate":"stats"}
    ).json().get("splits", [])
    last_date = gl[0]["date"] if gl else None
    rest_days = None
    if last_date:
        rd = datetime.fromisoformat(game_date).date() - datetime.fromisoformat(last_date).date()
        rest_days = rd.days
    return {
        "era":   stat.get("era"),
        "fip":   stat.get("fip"),
        "xfip":  stat.get("xFIP"),
        "k_bb":  stat.get("strikeouts",0)/max(1, stat.get("walks",1)),
        "rest":  rest_days
    }

def fetch_team_stats(season):
    """Return dict of team_id → season hitting/pitching stats."""
    resp = requests.get("https://statsapi.mlb.com/api/v1/teams", params={"season":season}).json()["teams"]
    teams = {}
    for t in resp:
        tid = t["id"]
        hit = requests.get(
            f"https://statsapi.mlb.com/api/v1/teams/{tid}/stats",
            params={"season":season,"group":"hittingRegularSeason"}
        ).json()["stats"][0]["splits"][0]["stat"]
        pit = requests.get(
            f"https://statsapi.mlb.com/api/v1/teams/{tid}/stats",
            params={"season":season,"group":"pitchingRegularSeason"}
        ).json()["stats"][0]["splits"][0]["stat"]
        teams[tid] = {
            "rpg":        pit.get("runsAllowedPerGame"),
            "wRC+":       hit.get("wRCPlus"),
            "babip":      hit.get("battingAverageOnBallsInPlay"),
            "iso":        hit.get("isoPower"),
            "drs":        hit.get("defensiveRunsSaved"),
            "bp_era":     pit.get("bullpenEra"),
            "bp_hl_era":  pit.get("bullpenHighLeverageEra")
        }
    return teams

def compute_recent_form(end_date, days=10):
    """Return dict of team_id → last10_win_pct & run_diff_last10 ending on end_date."""
    start = end_date - timedelta(days=days)
    recs  = []
    for d in pd.date_range(start.date(), end_date.date()):
        dsched = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId":1,"date":d.strftime("%Y-%m-%d"),"hydrate":"linescore"}
        ).json().get("dates", [])
        for day in dsched:
            for g in day["games"]:
                if g["status"]["codedGameState"]!="F":
                    continue
                hi = g["teams"]["home"]["team"]["id"]
                ai = g["teams"]["away"]["team"]["id"]
                hr = g["linescore"]["teams"]["home"]["runs"]
                ar = g["linescore"]["teams"]["away"]["runs"]
                recs.append({"team_id":hi, "win":int(hr>ar), "run_diff":hr-ar})
                recs.append({"team_id":ai, "win":int(ar>hr), "run_diff":ar-hr})
    df = pd.DataFrame(recs)
    agg = df.groupby("team_id").agg(
        last10_win_pct=("win","mean"),
        run_diff_last10=("run_diff","sum")
    )
    return agg.to_dict("index")

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = []
    for single in daterange(START_DATE, END_DATE):
        date_str = single.strftime("%Y-%m-%d")
        sched_df = fetch_schedule(date_str)
        if sched_df.empty:
            continue
        odds_df     = fetch_odds(date_str)
        season      = single.year
        team_stats  = fetch_team_stats(season)
        recent_form = compute_recent_form(single)

        df = sched_df.merge(odds_df, on=["home_team_id","away_team_id"], how="left")
        for _, row in df.iterrows():
            hs = fetch_sp_stats(row.home_sp_id, row.date)
            as_ = fetch_sp_stats(row.away_sp_id, row.date)
            ht = team_stats.get(row.home_team_id, {})
            at = team_stats.get(row.away_team_id, {})
            rf = recent_form.get(row.home_team_id, {})
            af = recent_form.get(row.away_team_id, {})

            rec = {
                "date":                      row.date,
                "time":                      row.time,
                "venue":                     row.venue,
                "home_sp_era":               hs["era"],
                "home_sp_fip":               hs["fip"],
                "home_sp_xfip":              hs["xfip"],
                "home_sp_k_bb":              hs["k_bb"],
                "home_sp_rest":              hs["rest"],
                "away_sp_era":               as_["era"],
                "away_sp_fip":               as_["fip"],
                "away_sp_xfip":              as_["xfip"],
                "away_sp_k_bb":              as_["k_bb"],
                "away_sp_rest":              as_["rest"],
                "home_rpg":                  ht.get("rpg"),
                "home_wRC+":                 ht.get("wRC+"),
                "home_BABIP":                ht.get("babip"),
                "home_ISO":                  ht.get("iso"),
                "home_DRS":                  ht.get("drs"),
                "home_BP_ERA":               ht.get("bp_era"),
                "home_BP_HL":                ht.get("bp_hl_era"),
                "away_rpg":                  at.get("rpg"),
                "away_wRC+":                 at.get("wRC+"),
                "away_BABIP":                at.get("babip"),
                "away_ISO":                  at.get("iso"),
                "away_DRS":                  at.get("drs"),
                "away_BP_ERA":               at.get("bp_era"),
                "away_BP_HL":                at.get("bp_hl_era"),
                "home_last10_win_pct":       rf.get("last10_win_pct"),
                "home_run_diff_last10":      rf.get("run_diff_last10"),
                "away_last10_win_pct":       af.get("last10_win_pct"),
                "away_run_diff_last10":      af.get("run_diff_last10"),
                "park_factor":               PARK_FACTORS.get(row.venue),
                "home_odds":                 row.home_odds,
                "away_odds":                 row.away_odds,
                "target":                    1 if row.home_score > row.away_score else 0
            }
            records.append(rec)

    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(records)} rows to {OUTPUT_CSV}")
