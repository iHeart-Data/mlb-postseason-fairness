# MLB Postseason Fairness

Code behind a series of articles on MLB's postseason format and the 2026 CBA proposals.

| Article | Notebook |
|---|---|
| [Is MLB's Postseason Rewarding the Wrong Teams? (2026)](https://medium.com/@iHeartData/is-mlbs-postseason-rewarding-the-wrong-teams-2026-a186c1721137) | `notebooks/path_analysis.ipynb` |
| [Would MLB's CBA Proposal Make the 1 Seed Worth Chasing?](https://medium.com/@iHeartData/would-mlbs-cba-proposal-make-the-1-seed-worth-chasing-a0c151d05c11) | `notebooks/drafting_analysis.ipynb` |

## Is MLB's postseason rewarding the wrong teams?

The first article measures how often MLB's 12-team postseason (2022–2026) gives a worse record the better outcome, on actual records and on schedule-adjusted records.

- **Did your team deserve to make the postseason?** Teams that missed the playoffs despite a top-12 record in MLB, compared with the worst team that made it.
- **Did your team deserve a better seed?** Wild Cards with a better record than the worst division winner in their league.

## How the schedule adjustment works

A team's record depends partly on who it played. `src/ratings.py` fits a Bradley–Terry model (a logistic regression on every regular-season game, with a home-field term) separately for each season. From the fitted team strengths, it computes each team's expected winning percentage against every other team on a neutral field: the record it would have had against a perfectly balanced schedule.

## Would MLB's CBA proposal make the 1 seed worth chasing?

MLB's October 2026 CBA proposal would let the 1 seed pick its Division Series opponent and the 3 seed pick its Wild Card opponent (from 2027). `notebooks/drafting_analysis.ipynb` simulates 260,000 seasons and measures how often each seed reaches each round under today's bracket and under drafting, with several pick rules (true talent, overall record, head-to-head, and a mix).

- `src/sim.py`: the season simulator. Each game is a weighted coin flip from the two teams' talent scores plus home-field advantage. A talent score is a team's schedule-adjusted record pulled toward .500 to remove luck (about 25% of the spread in records is luck).
- `src/bracket.py`: postseason formats, including the configurable 12-team bracket with drafting (`format_12_grid`) and the pick rules (`pick_opponents`).
- `src/evaluate.py`: season setup (talent, schedule, leagues) and the summary measures.

## Reproduce it

```bash
pip install -r requirements.txt
python scripts/pull_standings.py
python scripts/pull_postseason.py
python scripts/pull_games.py
jupyter notebook notebooks/path_analysis.ipynb
jupyter notebook notebooks/drafting_analysis.ipynb
```

The pull scripts download standings, postseason results and every regular-season game for 1998–2026 from the free [MLB Stats API](https://statsapi.mlb.com) into `data/raw/`. Each notebook reproduces every number in its article. The drafting notebook takes about 15 minutes.

## Notes

- 2020 is excluded (60-game season, 16-team postseason).
- Win differences are in games, the way the standings measure them: ((W − L of team A) − (W − L of team B)) / 2.
- Schedule-adjusted records are estimates from one season of games; differences under one game aren't counted.

Data: MLB Stats API. Analysis and graphics by [@iHeartData](https://x.com/iHeartData).
