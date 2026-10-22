"""Season simulator.

Plays a real schedule many times with known team strengths. Each game is a weighted coin
flip using the same model as `ratings.py`:

    P(home team wins) = sigmoid(strength_home - strength_away + home_edge)

All simulations run at once as arrays (one row per simulated season), so 10,000 seasons
of a 2,430-game schedule take a few seconds.
"""
import numpy as np
import pandas as pd

LUCK_VAR = 0.25 / 162     # variance of win % from coin-flip luck alone over 162 games
TALENT_SHARE = 0.75       # share of the spread in records that is real talent (2013-2026)


def pct_to_strength(pct):
    """Win % against an average team -> strength on the log-odds scale."""
    pct = np.clip(pct, 0.01, 0.99)
    return np.log(pct / (1 - pct))


def true_talent(adj_pct, talent_share=TALENT_SHARE):
    """Pull schedule-adjusted win % toward .500 so the spread of talents matches real talent.

    Scaling by sqrt(talent_share) (not talent_share) keeps the talent spread at its true
    size (SD ~.067). Scaling by talent_share would give the best guess for each team, but
    best guesses are more bunched than reality, and simulated seasons would come out too even.
    """
    return 0.5 + np.sqrt(talent_share) * (np.asarray(adj_pct) - 0.5)


class Schedule:
    """A season's games in date order, as index arrays the simulator can use directly."""

    def __init__(self, games, team_ids):
        games = games.sort_values(['date', 'game_pk']).reset_index(drop=True)
        self.team_ids = np.asarray(team_ids)
        col = {t: i for i, t in enumerate(self.team_ids)}
        self.home = games['home_id'].map(col).to_numpy()
        self.away = games['away_id'].map(col).to_numpy()
        self.dates = pd.to_datetime(games['date']).to_numpy()
        self.n_games = len(games)
        self.n_teams = len(self.team_ids)

    def games_through(self, games_per_team):
        """Number of schedule games played by the date the average team reaches `games_per_team`."""
        played = np.zeros(self.n_teams)
        for i in range(self.n_games):
            played[self.home[i]] += 1
            played[self.away[i]] += 1
            if played.mean() >= games_per_team:
                return i + 1
        return self.n_games


def simulate_games(schedule, strength, home_edge, n_sims, rng, start=0, stop=None, head_to_head=False):
    """Simulate games [start, stop) of the schedule.

    strength: (n_teams,) or (n_sims, n_teams) log-odds strengths.
    Returns (n_sims, n_teams) wins and losses over those games, plus, if head_to_head,
    an (n_sims, n_teams, n_teams) array of wins by row team over column team.
    """
    stop = schedule.n_games if stop is None else stop
    h, a = schedule.home[start:stop], schedule.away[start:stop]
    s = np.broadcast_to(strength, (n_sims, schedule.n_teams))
    p_home = 1 / (1 + np.exp(-(s[:, h] - s[:, a] + home_edge)))
    home_won = rng.random(p_home.shape) < p_home

    # Tally with matrix products (much faster than np.add.at): one-hot game -> team maps
    n_games = stop - start
    home_map = np.zeros((n_games, schedule.n_teams), dtype=np.float32)
    away_map = np.zeros((n_games, schedule.n_teams), dtype=np.float32)
    home_map[np.arange(n_games), h] = 1
    away_map[np.arange(n_games), a] = 1
    hw = home_won.astype(np.float32)
    wins = hw @ home_map + (1 - hw) @ away_map
    losses = (home_map.sum(axis=0) + away_map.sum(axis=0)) - wins
    if not head_to_head:
        return wins.astype(float), losses.astype(float)
    rows = np.arange(n_sims)[:, None]
    wins, losses = wins.astype(float), losses.astype(float)
    h2h = np.zeros((n_sims, schedule.n_teams, schedule.n_teams), dtype=np.float32)
    np.add.at(h2h, (rows, h, a), home_won)
    np.add.at(h2h, (rows, a, h), ~home_won)
    return wins, losses, h2h


def shorten_schedule(games, n_drop_home=4, n_drop_away=4, rng=None, max_tries=2000):
    """Drop games from a real schedule so every team loses exactly n_drop_home home games and
    n_drop_away road games (e.g. 162 -> 154 with 4 + 4), at most one dropped game per pair of
    teams, spread at random through the season. MLB hasn't published a 154-game schedule, so this
    is an assumption: the same opponent mix as the real schedule, slightly fewer games.
    """
    rng = rng or np.random.default_rng(154)
    games = games.reset_index(drop=True)
    teams = pd.unique(games[['home_id', 'away_id']].values.ravel())
    for _ in range(max_tries):
        home_left = dict.fromkeys(teams, n_drop_home)
        away_left = dict.fromkeys(teams, n_drop_away)
        pairs, drop = set(), []
        for i in rng.permutation(len(games)):
            h, a = games.at[i, 'home_id'], games.at[i, 'away_id']
            pair = frozenset((h, a))
            if home_left[h] and away_left[a] and pair not in pairs:
                home_left[h] -= 1
                away_left[a] -= 1
                pairs.add(pair)
                drop.append(i)
        if not any(home_left.values()) and not any(away_left.values()):
            return games.drop(index=drop).reset_index(drop=True)
    raise RuntimeError('could not shorten the schedule evenly')
