"""Pull every postseason game, 1998-2026, from the MLB Stats API.

The standings endpoint's clinch flags are unreliable for older seasons (2012 shows
8 playoff teams, not 10), so the postseason field and how far each team went come
from the games actually played.

Outputs:
  data/raw/postseason_games_1998_2026.csv   one row per game
  data/raw/postseason_teams_1998_2026.csv   one row per team-season that made it,
                                            with the furthest round reached
"""
from pathlib import Path

import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
SEASONS = range(1998, 2027)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
# F = Wild Card, D = Division Series, L = LCS, W = World Series
ROUND_ORDER = {"F": 1, "D": 2, "L": 3, "W": 4}
ROUND_NAMES = {1: "Wild Card", 2: "Division Series", 3: "LCS", 4: "World Series"}


def pull_season(season):
    resp = requests.get(
        f"{API}/schedule",
        params={"sportId": 1, "season": season, "gameType": "F,D,L,W"},
    ).json()
    rows = []
    for day in resp["dates"]:
        for g in day["games"]:
            if g["status"]["detailedState"] != "Final":
                continue
            away, home = g["teams"]["away"], g["teams"]["home"]
            rows.append({
                "season": season,
                "game_pk": g["gamePk"],
                "date": g["officialDate"],
                "round": g["gameType"],
                "series": g.get("seriesDescription"),
                "series_game": g.get("seriesGameNumber"),
                "games_in_series": g.get("gamesInSeries"),
                "home_id": home["team"]["id"],
                "away_id": away["team"]["id"],
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "winner_id": home["team"]["id"] if home.get("isWinner") else away["team"]["id"],
            })
    return rows


def team_results(games):
    """Furthest round each team reached, and whether it won the World Series."""
    long = pd.concat([
        games[["season", "round", "home_id"]].rename(columns={"home_id": "team_id"}),
        games[["season", "round", "away_id"]].rename(columns={"away_id": "team_id"}),
    ])
    long["round_num"] = long["round"].map(ROUND_ORDER)
    teams = long.groupby(["season", "team_id"], as_index=False)["round_num"].max()
    teams["furthest_round"] = teams["round_num"].map(ROUND_NAMES)

    ws = games[games["round"] == "W"]
    ws_wins = ws.groupby(["season", "winner_id"]).size().rename("w").reset_index()
    champs = ws_wins.loc[ws_wins.groupby("season")["w"].idxmax(), ["season", "winner_id"]]
    teams["won_ws"] = teams.set_index(["season", "team_id"]).index.isin(
        champs.set_index(["season", "winner_id"]).index
    )
    teams["won_pennant"] = teams["round_num"] == 4
    return teams.drop(columns="round_num")


def main():
    rows = []
    for season in SEASONS:
        rows += pull_season(season)
        print(season, "done")
    games = pd.DataFrame(rows)
    games.to_csv(RAW / "postseason_games_1998_2026.csv", index=False)
    teams = team_results(games)
    teams.to_csv(RAW / "postseason_teams_1998_2026.csv", index=False)
    print(f"{len(games)} games, {len(teams)} team-seasons")


if __name__ == "__main__":
    main()
