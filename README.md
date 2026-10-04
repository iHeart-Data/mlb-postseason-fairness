# Did Your Team Deserve Better in the Postseason?

Code behind the article **[Did Your Team Deserve Better in the Postseason?](MEDIUM_LINK)**, which measures how often MLB's 12-team postseason (2022–2026) gives a worse record the better outcome, on actual records and on schedule-adjusted records.

- **Did your team deserve to make the postseason?** Teams that missed the playoffs despite a top-12 record in MLB, compared with the worst team that made it.
- **Did your team deserve a better seed?** Wild Cards with a better record than the worst division winner in their league.

## How the schedule adjustment works

A team's record depends partly on who it played. `src/ratings.py` fits a Bradley–Terry model (a logistic regression on every regular-season game, with a home-field term) separately for each season. From the fitted team strengths, it computes each team's expected winning percentage against every other team on a neutral field: the record it would have had against a perfectly balanced schedule.

## Reproduce it

```bash
pip install -r requirements.txt
python scripts/pull_standings.py
python scripts/pull_postseason.py
python scripts/pull_games.py
jupyter notebook notebooks/path_analysis.ipynb
```

The pull scripts download standings, postseason results and every regular-season game for 1998–2026 from the free [MLB Stats API](https://statsapi.mlb.com) into `data/raw/`. The notebook reproduces every number in the article.

## Notes

- 2020 is excluded (60-game season, 16-team postseason).
- Win differences are in games, the way the standings measure them: ((W − L of team A) − (W − L of team B)) / 2.
- Schedule-adjusted records are estimates from one season of games; differences under one game aren't counted.

Data: MLB Stats API. Analysis and graphics by [@iHeartData](https://x.com/iHeartData).
