"""Shared helpers for checking formats and measuring Merit.

- season_setup: talent, schedule and league structure for one season
- seed_round_rates / real_seed_results: how far each seed got, simulated vs. real
- merit: how often the best team (by record, by talent) reaches each round
"""
import numpy as np
import pandas as pd
from scipy.stats import binom

from .ratings import fit_season, adjusted_pct
from .sim import Schedule, true_talent, pct_to_strength
from .bracket import League

ROUNDS = ['won Wild Card round', 'won Division Series', 'won pennant', 'won World Series']
ROUND_NUM = {'Wild Card': 1, 'Division Series': 2, 'LCS': 3, 'World Series': 4}


def season_setup(games, standings, season, schedule_season=None, schedule_games=None):
    """Talent from `season`, played on the schedule of `schedule_season` (default: same season).
    schedule_games: optional replacement schedule (e.g. a shortened 154-game version of it)."""
    sched_season = schedule_season or season
    sched_games = games[games['season'] == sched_season] if schedule_games is None else schedule_games
    team_ids = sorted(sched_games['home_id'].unique())
    sched = Schedule(sched_games, team_ids)
    st = standings[standings['season'] == sched_season].set_index('team_id').loc[team_ids]
    leagues = [League(np.where(st['league'] == lg)[0], st.loc[st['league'] == lg, 'division_id'].values)
               for lg in ('AL', 'NL')]
    ratings, home_edge = fit_season(games[games['season'] == season])
    adj = adjusted_pct(ratings).loc[team_ids].values
    talent = true_talent(adj)
    return dict(sched=sched, leagues=leagues, team_ids=team_ids, abbrev=st['team'].values,
                league=st['league'].values, division=st['division_id'].values,
                adj=adj, talent=talent, strength=pct_to_strength(talent), home_edge=home_edge)


def seed_round_rates(res, n_seeds, bye_seeds):
    """Share of sims where each seed reached each round, pooled over both leagues.

    bye_seeds: seeds that skip the Wild Card round (their first column is left blank).
    """
    rates = np.full((n_seeds, 4), np.nan)
    for k in range(n_seeds):
        cols = []
        for out, pennant in zip(res['leagues'], res['pennants']):
            team = out['seeds'][:, k]
            cols.append([(out['reached_ds'] == team[:, None]).any(axis=1).mean(),
                         (out['reached_lcs'] == team[:, None]).any(axis=1).mean(),
                         (pennant == team).mean(),
                         (res['champion'] == team).mean()])
        rates[k] = np.mean(cols, axis=0)
        if k + 1 in bye_seeds:
            rates[k, 0] = np.nan
    return rates


def real_seed_results(standings, post, seasons, n_wild_cards):
    """How far each real seed got. Seeds: division winners by record, then Wild Cards by record."""
    rows = []
    for season in seasons:
        st = standings[standings['season'] == season].copy()
        st['pct'] = st['wins'] / (st['wins'] + st['losses'])
        pt = post[post['season'] == season].set_index('team_id')
        st = st[st['team_id'].isin(pt.index)]
        for lg, g in st.groupby('league'):
            order = ['pct', 'league_rank']
            div_winners = g[g['division_rank'] == 1].sort_values(order, ascending=[False, True])
            wild_cards = g[g['division_rank'] != 1].sort_values(order, ascending=[False, True]).head(n_wild_cards)
            for seed, (_, t) in enumerate(pd.concat([div_winners, wild_cards]).iterrows(), start=1):
                furthest = ROUND_NUM[pt.loc[t['team_id'], 'furthest_round']]
                rows.append({'season': season, 'league': lg, 'seed': seed, 'team': t['team'],
                             'won Wild Card round': furthest >= 2, 'won Division Series': furthest >= 3,
                             'won pennant': furthest >= 4,
                             'won World Series': bool(pt.loc[t['team_id'], 'won_ws'])})
    return pd.DataFrame(rows)


def consistency_table(sim_rates, real, bye_seeds):
    """Simulated rate, expected count, 95% plausible range and the real count, per seed and round."""
    counts = real.groupby('seed')[ROUNDS].sum()
    tries = real.groupby('seed').size()
    rows = []
    for k in range(sim_rates.shape[0]):
        seed = k + 1
        for j, rnd in enumerate(ROUNDS):
            if seed in bye_seeds and j == 0:
                continue
            p, n = sim_rates[k, j], int(tries[seed])
            lo, hi = binom.ppf(0.025, n, p), binom.ppf(0.975, n, p)
            actual = int(counts.loc[seed, rnd])
            rows.append({'seed': seed, 'round': rnd, 'simulated rate': round(p, 3),
                         'expected count': round(p * n, 1), 'real tries': n,
                         'plausible range': f'{int(lo)}–{int(hi)}', 'actually happened': actual,
                         'consistent?': 'yes' if lo <= actual <= hi else 'NO'})
    return pd.DataFrame(rows)


def merit(res, leagues, wins, talent, rng):
    """How often the top team reaches each round, judged two ways:
      'best record':          best regular-season record in that simulated season
      'talent score':         highest talent score (schedule-adjusted record with luck removed;
                              the model's estimate, not something observable)
    plus how often each league's top team meet in the World Series. Ties broken randomly."""
    n_sims = wins.shape[0]
    talent = np.asarray(talent)
    noisy = wins + rng.random(wins.shape) * 1e-3

    def in_any(key, team):
        return np.any([(out[key] == team[:, None]).any(axis=1) for out in res['leagues']], axis=0)

    lenses = {
        'best record': (np.argmax(noisy, axis=1),
                        [lg.teams[np.argmax(noisy[:, lg.teams], axis=1)] for lg in leagues]),
        'talent score': (np.full(n_sims, int(np.argmax(talent))),
                                 [np.full(n_sims, lg.teams[np.argmax(talent[lg.teams])]) for lg in leagues]),
    }
    out = {}
    for label, (top, league_top) in lenses.items():
        out[(label, 'makes playoffs')] = in_any('seeds', top).mean()
        out[(label, 'reaches LCS')] = in_any('reached_lcs', top).mean()
        out[(label, 'wins pennant')] = np.any([p == top for p in res['pennants']], axis=0).mean()
        out[(label, 'wins World Series')] = (res['champion'] == top).mean()
        out[(label, "each league's top team meets in World Series")] = np.mean(
            (res['pennants'][0] == league_top[0]) & (res['pennants'][1] == league_top[1]))
    return out


def deserving_share(res, leagues, wins, talent, rng):
    """Share of each round's spots filled by the teams that earned them, per league:
      Division Series (4 spots): top 4 in the league
      LCS (2 spots):             top 2 in the league
      World Series (1 spot):     top 1 in the league
      Champion:                  top 1 in MLB
    Judged two ways: by regular-season record in that simulated season, and by talent score.
    Ties broken randomly."""
    n_sims = wins.shape[0]
    talent = np.asarray(talent)
    scores = {'best record': wins + rng.random(wins.shape) * 1e-3,
              'talent score': np.broadcast_to(talent, wins.shape)}
    out = {}
    for label, score in scores.items():
        ds, lcs, ws = [], [], []
        for lg, league_res, pennant in zip(leagues, res['leagues'], res['pennants']):
            ranked = lg.teams[np.argsort(-score[:, lg.teams], axis=1)]       # league teams, best first
            top = lambda k: ranked[:, :k]
            ds.append((league_res['reached_ds'][:, :, None] == top(4)[:, None, :]).any(axis=2).mean(axis=1))
            lcs.append((league_res['reached_lcs'][:, :, None] == top(2)[:, None, :]).any(axis=2).mean(axis=1))
            ws.append(pennant == ranked[:, 0])
        out[(label, 'Division Series spots won by top-4 teams')] = np.mean(ds)
        out[(label, 'LCS spots won by top-2 teams')] = np.mean(lcs)
        out[(label, 'World Series spots won by the top team')] = np.mean(ws)
        out[(label, 'champion is the top team in MLB')] = (res['champion'] == np.argmax(score, axis=1)).mean()
    return out


def held_back(res, leagues, wins, talent):
    """Average number of teams per season held back by the structure (relative comparisons only,
    always within a team's own league; cross-league gaps are a league-balance issue, not a format one):
      'left out':  missed the playoffs despite a better record than at least one playoff team
                   in its own league
      'Wild Card behind a worse division winner': seeded below a team in its own league with a
                   worse record. With division-winner seeding this is exactly a Wild Card with a
                   better record than a division winner; with pure record seeding it can't happen.
    Judged by regular-season record and by talent score. Strictly better; ties don't count."""
    talent = np.asarray(talent)
    scores = {'record': wins, 'talent score': np.broadcast_to(talent, wins.shape)}
    rows = np.arange(wins.shape[0])[:, None]
    out = {}
    for label, score in scores.items():
        left_out = np.zeros(wins.shape[0])
        behind = np.zeros(wins.shape[0])
        for lg, league_res in zip(leagues, res['leagues']):
            in_field = np.zeros(wins.shape, dtype=bool)
            in_field[rows, league_res['seeds']] = True
            in_league = np.zeros(wins.shape[1], dtype=bool)
            in_league[lg.teams] = True
            worst_playoff = np.take_along_axis(score, league_res['seeds'], axis=1).min(axis=1, keepdims=True)
            left_out += (in_league & ~in_field & (score > worst_playoff)).sum(axis=1)

            s = np.take_along_axis(score, league_res['seeds'], axis=1)          # in seed order
            worst_above = np.minimum.accumulate(s, axis=1)                       # worst score seeded above (incl. self)
            behind += (s[:, 1:] > worst_above[:, :-1]).sum(axis=1)
        out[(label, 'left out')] = left_out.mean()
        out[(label, 'Wild Card behind a worse division winner')] = behind.mean()
    return out
