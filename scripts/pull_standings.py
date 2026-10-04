"""Pull final regular-season standings for every team, 1998-2026, from the MLB Stats API.

One row per team-season, with the team's league/division that season, record,
runs, how it finished, whether it made the postseason, and its record against
its own division, its own league, and the other league (interleague).

Output: data/raw/standings_1998_2026.csv
"""
from pathlib import Path

import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
SEASONS = range(1998, 2027)
LEAGUES = {103: "AL", 104: "NL"}
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "standings_1998_2026.csv"


def team_abbrevs(season):
    teams = requests.get(f"{API}/teams", params={"sportId": 1, "season": season}).json()["teams"]
    return {t["id"]: t["abbreviation"] for t in teams}


def record_vs(records, key, target_id):
    """Wins/losses against one league or division from a team's breakdown records."""
    for r in records.get(key, []):
        if r[key[:-7]]["id"] == target_id:  # "leagueRecords" -> "league"
            return r["wins"], r["losses"]
    return 0, 0


def pull_season(season):
    resp = requests.get(
        f"{API}/standings",
        params={"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"},
    ).json()
    abbrevs = team_abbrevs(season)
    rows = []
    for div in resp["records"]:
        league_id = div["league"]["id"]
        div_id = div["division"]["id"]
        div_size = len(div["teamRecords"])
        for t in div["teamRecords"]:
            recs = t["records"]
            div_w, div_l = record_vs(recs, "divisionRecords", div_id)
            lg_w, lg_l = record_vs(recs, "leagueRecords", league_id)
            other_id = 104 if league_id == 103 else 103
            inter_w, inter_l = record_vs(recs, "leagueRecords", other_id)
            rows.append({
                "season": season,
                "team_id": t["team"]["id"],
                "team": abbrevs.get(t["team"]["id"]),
                "team_name": t["team"]["name"],
                "league": LEAGUES[league_id],
                "division_id": div_id,
                "division_size": div_size,
                "wins": t["wins"],
                "losses": t["losses"],
                "runs_scored": t.get("runsScored"),
                "runs_allowed": t.get("runsAllowed"),
                "division_rank": int(t["divisionRank"]),
                "league_rank": int(t["leagueRank"]),
                "division_champ": t.get("divisionChamp", False),
                "clinched": t.get("clinched", False),
                "clinch_indicator": t.get("clinchIndicator"),
                "div_wins": div_w, "div_losses": div_l,
                "league_wins": lg_w, "league_losses": lg_l,
                "interleague_wins": inter_w, "interleague_losses": inter_l,
            })
    return rows


def main():
    rows = []
    for season in SEASONS:
        rows += pull_season(season)
        print(season, "done")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} team-seasons -> {OUT}")


if __name__ == "__main__":
    main()
