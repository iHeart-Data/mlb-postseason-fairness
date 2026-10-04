"""Pull every completed regular-season game, 1998-2026, from the MLB Stats API.

Used for schedule composition (division / league / interleague share by season),
schedule-adjusted team strength, and the simulator's schedule templates.

Suspended games can be listed twice under the same gamePk (once when suspended,
once when completed), so each game is kept once.

Only games with a winner are kept. That drops official ties (which don't count in
the standings) and phantom entries in older seasons (36 in 2000 alone) that show as
Final but have no winner and don't advance either team's record.

Output: data/raw/games_1998_2026.csv
"""
from pathlib import Path

import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
SEASONS = range(1998, 2027)
DONE = {"Final", "Completed Early"}
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "games_1998_2026.csv"


def pull_season(season):
    resp = requests.get(
        f"{API}/schedule", params={"sportId": 1, "season": season, "gameType": "R"}
    ).json()
    rows = []
    for day in resp["dates"]:
        for g in day["games"]:
            if g["status"]["detailedState"] not in DONE:
                continue
            home, away = g["teams"]["home"], g["teams"]["away"]
            if not (home.get("isWinner") or away.get("isWinner")):
                continue
            rows.append({
                "season": season,
                "game_pk": g["gamePk"],
                "date": g["officialDate"],
                "home_id": home["team"]["id"],
                "away_id": away["team"]["id"],
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "home_win": bool(home.get("isWinner")),
            })
    return pd.DataFrame(rows).drop_duplicates("game_pk", keep="last")


def main():
    games = []
    for season in SEASONS:
        games.append(pull_season(season))
        print(season, len(games[-1]))
    df = pd.concat(games, ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} games -> {OUT}")


if __name__ == "__main__":
    main()
