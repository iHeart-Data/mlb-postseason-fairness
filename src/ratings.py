"""Schedule-adjusted team strength from game results.

Bradley-Terry model, one fit per season:
    P(home team wins) = sigmoid(rating_home - rating_away + home_edge)

Ratings account for who each team actually played, so a team with a hard schedule
gets credit for it. A rating is turned back into a win % with
`adjusted_pct`: the team's expected win % against every other team on a neutral
field, i.e. its record against a perfectly average schedule.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_season(games, l2=1e-3):
    """Fit ratings for one season of games (columns: home_id, away_id, home_win).

    Returns (ratings Series indexed by team_id, home_edge in log-odds).
    """
    teams = np.sort(pd.unique(games[["home_id", "away_id"]].values.ravel()))
    col = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(games), len(teams)))
    X[np.arange(len(games)), games["home_id"].map(col)] = 1
    X[np.arange(len(games)), games["away_id"].map(col)] = -1
    # Tiny L2 keeps the fit well-defined; ratings are only meaningful relative to each other.
    model = LogisticRegression(C=1 / l2, fit_intercept=True, max_iter=1000)
    model.fit(X, games["home_win"].astype(int))
    ratings = pd.Series(model.coef_[0], index=teams)
    return ratings - ratings.mean(), model.intercept_[0]


def adjusted_pct(ratings):
    """Expected win % vs. every other team on a neutral field."""
    r = ratings.values
    p = 1 / (1 + np.exp(-(r[:, None] - r[None, :])))
    np.fill_diagonal(p, np.nan)
    return pd.Series(np.nanmean(p, axis=1), index=ratings.index)


def fit_all(games):
    """Ratings for every season. Returns one row per team-season."""
    out = []
    for season, g in games.groupby("season"):
        ratings, home_edge = fit_season(g)
        out.append(pd.DataFrame({
            "season": season,
            "team_id": ratings.index,
            "rating": ratings.values,
            "adj_pct": adjusted_pct(ratings).values,
            "home_edge": home_edge,
        }))
    return pd.concat(out, ignore_index=True)
