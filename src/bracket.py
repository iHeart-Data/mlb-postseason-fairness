"""Postseason formats.

Every function works on arrays: one row per simulated season. Teams are column indices
into the season simulator's team list. A format takes the simulated standings and returns
who made the field, how they were seeded, the two pennant winners and the champion.

Series use the same game model as the regular season, with real home-field patterns:
    best-of-1:  higher seed at home
    best-of-3:  all games at the higher seed
    best-of-5:  2-2-1
    best-of-7:  2-3-2
"""
import numpy as np

HOME_PATTERNS = {
    1: [True],
    3: [True, True, True],
    5: [True, True, False, False, True],
    7: [True, True, False, False, False, True, True],
}


class League:
    """Which teams are in a league and which division each is in."""

    def __init__(self, team_index, division):
        self.teams = np.asarray(team_index)          # column indices of this league's teams
        self.division = np.asarray(division)         # division label for each of those teams
        self.div_labels = sorted(set(division))


# When run_postseason(count_games=True) is active, play_series adds each series' length
# (games actually played) here, one total per simulated postseason.
_GAME_TALLY = {'games': None}


def play_series(a, b, strength, home_edge, length, rng, a_head_start=0):
    """Higher seed `a` vs `b` (arrays of team indices, one per sim). Returns the winners.

    A best-of-n is decided the same way as "first to (n+1)/2 wins", so all games are drawn
    and the series goes to whoever wins the majority. A head start (game-in-hand) shortens
    the series by that many games and counts as wins for `a`.
    """
    n = len(a)
    need_a = (length + 1) // 2 - a_head_start
    pattern = HOME_PATTERNS[length][a_head_start:]
    games = len(pattern)
    rows = np.arange(n)
    sa = strength[rows, a][:, None]
    sb = strength[rows, b][:, None]
    edge = np.where(pattern, home_edge, -home_edge)[None, :]
    p_a = 1 / (1 + np.exp(-(sa - sb + edge)))
    a_won = rng.random((n, games)) < p_a
    if _GAME_TALLY['games'] is not None:
        need_b = (length + 1) // 2
        done = (np.cumsum(a_won, axis=1) >= need_a) | (np.cumsum(~a_won, axis=1) >= need_b)
        _GAME_TALLY['games'] += np.argmax(done, axis=1) + 1
    return np.where(a_won.sum(axis=1) >= need_a, a, b)


def seed_league(league, wins, n_wild_cards, rng):
    """Seeds for one league: division winners first (by record), then Wild Cards (by record).

    Ties are broken randomly. Returns (n_sims, n_divisions + n_wild_cards) team indices.
    """
    n_sims = wins.shape[0]
    w = wins[:, league.teams] + rng.random((n_sims, len(league.teams))) * 1e-3  # random tiebreak
    div_winner_cols = []
    is_winner = np.zeros_like(w, dtype=bool)
    for d in league.div_labels:
        cols = np.where(league.division == d)[0]
        best = cols[np.argmax(w[:, cols], axis=1)]
        is_winner[np.arange(n_sims), best] = True
        div_winner_cols.append(best)
    div_winner_cols = np.stack(div_winner_cols, axis=1)
    order = np.argsort(-np.take_along_axis(w, div_winner_cols, axis=1), axis=1)
    div_seeds = np.take_along_axis(div_winner_cols, order, axis=1)

    wc_w = np.where(is_winner, -np.inf, w)
    wc_seeds = np.argsort(-wc_w, axis=1)[:, :n_wild_cards]
    return league.teams[np.concatenate([div_seeds, wc_seeds], axis=1)]


def lowest_vs_highest(teams, seed_of):
    """Order a set of remaining teams (n_sims, 4) as two matchups: best seed vs worst, 2nd vs 3rd."""
    order = np.argsort(np.take_along_axis(seed_of, teams, axis=1), axis=1)
    t = np.take_along_axis(teams, order, axis=1)
    return (t[:, 0], t[:, 3]), (t[:, 1], t[:, 2])


def seed_lookup(seeds, n_teams):
    """(n_sims, n_teams) array of each team's seed number (1 = best); 99 if not in the field."""
    n_sims = seeds.shape[0]
    lookup = np.full((n_sims, n_teams), 99)
    lookup[np.arange(n_sims)[:, None], seeds] = np.arange(1, seeds.shape[1] + 1)
    return lookup


def higher_seed_first(x, y, seed_of):
    rows = np.arange(len(x))
    x_better = seed_of[rows, x] < seed_of[rows, y]
    return np.where(x_better, x, y), np.where(x_better, y, x)


def pick_opponents(pickers, candidates, wins, h2h, games_played, weight_record, rng, weights=None, strength=None):
    """Division winners choose Wild Card opponents in order (pickers[:, 0] first).

    Each candidate gets two ranks among the candidates still available: by record (worst
    record = most attractive) and by the picker's head-to-head win % against them (best =
    most attractive). The pick is the lowest weighted rank; ties are broken randomly.
    weights: optional {'record': w, 'h2h': w, 'talent': w} instead of weight_record; 'talent' ranks
    candidates by true strength (weakest = most attractive), i.e. perfect scouting (needs strength).
    Returns the chosen opponent for each picker, in order, plus the unpicked candidate.
    """
    n_sims = pickers.shape[0]
    rows = np.arange(n_sims)
    available = np.ones(candidates.shape, dtype=bool)
    picks = []
    for k in range(pickers.shape[1]):
        p = pickers[:, k]
        cand_pct = wins[rows[:, None], candidates] / games_played[rows[:, None], candidates]
        played = h2h[rows[:, None], p[:, None], candidates] + h2h[rows[:, None], candidates, p[:, None]]
        h2h_pct = np.where(played > 0, h2h[rows[:, None], p[:, None], candidates] / np.maximum(played, 1), 0.5)
        big = 1e6
        record_rank = np.argsort(np.argsort(np.where(available, cand_pct, big), axis=1), axis=1)
        h2h_rank = np.argsort(np.argsort(np.where(available, -h2h_pct, big), axis=1), axis=1)
        if weights is None:
            score = weight_record * record_rank + (1 - weight_record) * h2h_rank
        else:
            score = weights.get('record', 0) * record_rank + weights.get('h2h', 0) * h2h_rank
            if weights.get('talent', 0):
                cand_str = strength[rows[:, None], candidates]
                talent_rank = np.argsort(np.argsort(np.where(available, cand_str, big), axis=1), axis=1)
                score = score + weights['talent'] * talent_rank
        score = np.where(available, score + rng.random(score.shape) * 1e-3, np.inf)
        choice = np.argmin(score, axis=1)
        picks.append(candidates[rows, choice])
        available[rows, choice] = False
    leftover = candidates[rows, np.argmax(available, axis=1)]
    return picks, leftover


# ---------------------------------------------------------------------------------------
# Formats. Each returns the per-league bracket result:
#   {'seeds': (n_sims, n_seeds) team indices, 'pennant': (n_sims,) league champion}
# ---------------------------------------------------------------------------------------

def format_8_team(league, ctx):
    """1998-2011: 3 division winners + 1 Wild Card. Best-of-5 DS, best-of-7 LCS.

    #1 plays the Wild Card unless they're from the same division, then #1 plays #3.
    """
    seeds = seed_league(league, ctx['wins'], 1, ctx['rng'])
    s1, s2, s3, wc = seeds.T
    team_div = dict(zip(league.teams, league.division))
    same_div = np.vectorize(team_div.get)(s1) == np.vectorize(team_div.get)(wc)
    a1, b1 = s1, np.where(same_div, s3, wc)
    a2, b2 = s2, np.where(same_div, wc, s3)
    seed_of = seed_lookup(seeds, ctx['n_teams'])
    w1 = play_series(a1, b1, ctx['strength'], ctx['home_edge'], 5, ctx['rng'])
    w2 = play_series(*higher_seed_first(a2, b2, seed_of), ctx['strength'], ctx['home_edge'], 5, ctx['rng'])
    pennant = play_series(*higher_seed_first(w1, w2, seed_of), ctx['strength'], ctx['home_edge'], 7, ctx['rng'])
    return {'seeds': seeds, 'pennant': pennant, 'reached_ds': seeds, 'reached_lcs': np.stack([w1, w2], axis=1)}


def format_10_team(league, ctx):
    """2012-2021: 3 division winners + 2 Wild Cards. One-game Wild Card, #1 plays its winner."""
    seeds = seed_league(league, ctx['wins'], 2, ctx['rng'])
    s1, s2, s3, wc1, wc2 = seeds.T
    seed_of = seed_lookup(seeds, ctx['n_teams'])
    wc = play_series(wc1, wc2, ctx['strength'], ctx['home_edge'], 1, ctx['rng'])
    w1 = play_series(s1, wc, ctx['strength'], ctx['home_edge'], 5, ctx['rng'])
    w2 = play_series(s2, s3, ctx['strength'], ctx['home_edge'], 5, ctx['rng'])
    pennant = play_series(*higher_seed_first(w1, w2, seed_of), ctx['strength'], ctx['home_edge'], 7, ctx['rng'])
    return {'seeds': seeds, 'pennant': pennant,
            'reached_ds': np.stack([s1, s2, s3, wc], axis=1),
            'reached_lcs': np.stack([w1, w2], axis=1)}


def format_12_team(league, ctx, ds_length=5, reseed=False, game_in_hand=False, wc_length=3):
    """2022-now: 3 division winners + 3 Wild Cards. Best-of-3 Wild Card round (3v6, 4v5),
    #1 and #2 get byes.

    Options cover two proposals:
      reseed + ds_length=7: 7-game DS where #1 plays the lowest remaining seed
      game_in_hand: the #3 seed (third division winner) starts its Wild Card series up 1-0
    """
    seeds = seed_league(league, ctx['wins'], 3, ctx['rng'])
    s1, s2, s3, s4, s5, s6 = seeds.T
    seed_of = seed_lookup(seeds, ctx['n_teams'])
    st, he, rng = ctx['strength'], ctx['home_edge'], ctx['rng']
    w36 = play_series(s3, s6, st, he, wc_length, rng, a_head_start=1 if game_in_hand else 0)
    w45 = play_series(s4, s5, st, he, wc_length, rng)
    if reseed:
        (a1, b1), (a2, b2) = lowest_vs_highest(np.stack([s1, s2, w36, w45], axis=1), seed_of)
    else:
        a1, b1, a2, b2 = s1, w45, s2, w36
    w1 = play_series(a1, b1, st, he, ds_length, rng)
    w2 = play_series(a2, b2, st, he, ds_length, rng)
    pennant = play_series(*higher_seed_first(w1, w2, seed_of), st, he, 7, rng)
    return {'seeds': seeds, 'pennant': pennant,
            'ds_matchups': np.stack([a1, b1, a2, b2], axis=1),
            'reached_ds': np.stack([s1, s2, w36, w45], axis=1),
            'reached_lcs': np.stack([w1, w2], axis=1)}


def format_14_team(league, ctx, weight_record=0.5, pick=True):
    """Proposal: 3 division winners + 4 Wild Cards. #1 gets a bye; #2 then #3 pick their
    Wild Card opponents from seeds 5-7; #4 plays the one left. Best-of-3 Wild Card round.

    Division Series: #1 plays the lowest-seeded remaining team; the other two survivors play each other.
    """
    seeds = seed_league(league, ctx['wins'], 4, ctx['rng'])
    s1, s2, s3, s4 = seeds[:, 0], seeds[:, 1], seeds[:, 2], seeds[:, 3]
    seed_of = seed_lookup(seeds, ctx['n_teams'])
    st, he, rng = ctx['strength'], ctx['home_edge'], ctx['rng']
    if pick:
        (p2, p3), leftover = pick_opponents(seeds[:, 1:3], seeds[:, 4:7], ctx['wins'], ctx['h2h'],
                                            ctx['games_played'], weight_record, rng)
    else:  # fixed bracket for comparison: 2v7, 3v6, 4v5
        p2, p3, leftover = seeds[:, 6], seeds[:, 5], seeds[:, 4]
    w2 = play_series(s2, p2, st, he, 3, rng)
    w3 = play_series(s3, p3, st, he, 3, rng)
    w4 = play_series(s4, leftover, st, he, 3, rng)
    (a1, b1), (a2, b2) = lowest_vs_highest(np.stack([s1, w2, w3, w4], axis=1), seed_of)
    d1 = play_series(a1, b1, st, he, 5, rng)
    d2 = play_series(a2, b2, st, he, 5, rng)
    pennant = play_series(*higher_seed_first(d1, d2, seed_of), st, he, 7, rng)
    return {'seeds': seeds, 'pennant': pennant,
            'picks': np.stack([p2, p3, leftover], axis=1),     # #2's pick, #3's pick, #4's opponent
            'ds_matchups': np.stack([a1, b1, a2, b2], axis=1),
            'reached_ds': np.stack([s1, w2, w3, w4], axis=1),
            'reached_lcs': np.stack([d1, d2], axis=1)}



def format_12_grid(league, ctx, record_seeding=False, wc_length=3, ds_length=5, ds_rule='fixed',
                   weight_record=0.5, pick_weights=None, wc_pick=None):
    """12 teams (3 division winners + 3 Wild Cards per league), every option a switch.

    record_seeding: once the 6 teams are set, seed them purely by record (division title = a spot,
                    no seeding bump). Otherwise division winners are seeds 1-3 (as now).
    wc_length, ds_length: series lengths (Wild Card round 3v6 / 4v5, then the Division Series).
    ds_rule: 'fixed'  -> #1 plays the 4/5 winner, #2 the 3/6 winner (as now)
             'reseed' -> #1 plays the lowest remaining seed
             'pick'   -> drafting at both rounds (MLB's 2026 proposal): #3 picks its Wild Card
                         opponent from #5/#6 (#4 gets the other), then #1 picks its Division
                         Series opponent from the two Wild Card winners (#2 gets the other).
                         Picks weigh opponent record and head-to-head 50/50 (as the 14-team check).
             'pick_ds' -> only #1 picks its Division Series opponent (Wild Card round fixed)
    wc_pick: override whether #3 picks its Wild Card opponent (default: only with ds_rule='pick').
    """
    st, he, rng = ctx['strength'], ctx['home_edge'], ctx['rng']
    seeds = seed_league(league, ctx['wins'], 3, rng)
    if record_seeding:
        w = np.take_along_axis(ctx['wins'], seeds, axis=1) + rng.random(seeds.shape) * 1e-3
        seeds = np.take_along_axis(seeds, np.argsort(-w, axis=1), axis=1)
    s1, s2, s3, s4, s5, s6 = seeds.T
    seed_of = seed_lookup(seeds, ctx['n_teams'])
    pick = lambda pickers, cands: pick_opponents(pickers, cands, ctx['wins'], ctx['h2h'],
                                                 ctx['games_played'], weight_record, rng,
                                                 weights=pick_weights, strength=st)
    wc_pick = (ds_rule == 'pick') if wc_pick is None else wc_pick
    if wc_pick:
        (p3,), leftover = pick(seeds[:, 2:3], seeds[:, 4:6])
    else:
        p3, leftover = s6, s5
    w3 = play_series(s3, p3, st, he, wc_length, rng)
    w4 = play_series(s4, leftover, st, he, wc_length, rng)
    if ds_rule in ('pick', 'pick_ds'):
        (b1,), b2 = pick(s1[:, None], np.stack([w3, w4], axis=1))
        a1, a2 = s1, s2
        picks = np.stack([p3, leftover, b1, b2], axis=1)   # #3's pick, #3's alternative, #1's pick, #1's alternative
    elif ds_rule == 'reseed':
        (a1, b1), (a2, b2) = lowest_vs_highest(np.stack([s1, s2, w3, w4], axis=1), seed_of)
    else:
        a1, b1, a2, b2 = s1, w4, s2, w3
    d1 = play_series(*higher_seed_first(a1, b1, seed_of), st, he, ds_length, rng)
    d2 = play_series(*higher_seed_first(a2, b2, seed_of), st, he, ds_length, rng)
    pennant = play_series(*higher_seed_first(d1, d2, seed_of), st, he, 7, rng)
    out = {'seeds': seeds, 'pennant': pennant,
           'ds_matchups': np.stack([a1, b1, a2, b2], axis=1),
           'reached_ds': np.stack([s1, s2, w3, w4], axis=1),
           'reached_lcs': np.stack([d1, d2], axis=1)}
    if ds_rule in ('pick', 'pick_ds'):
        out['picks'] = picks
    return out


def grid_name(record_seeding, wc_length, ds_length, ds_rule):
    return (f"{'record' if record_seeding else 'division'} seeding | WC bo{wc_length} | "
            f"DS bo{ds_length} | {ds_rule}")


# Every combination of the 12-team options (24)
GRID = {grid_name(r, wc, ds, rule): dict(record_seeding=r, wc_length=wc, ds_length=ds, ds_rule=rule)
        for r in (False, True) for wc in (1, 3) for ds in (5, 7) for rule in ('fixed', 'reseed', 'pick')}


# ---------------------------------------------------------------------------------------
# Configurable bracket, for hypothetical formats
# ---------------------------------------------------------------------------------------

def seed_by_record(league, wins, n, rng):
    """Top n teams in a league by record (random tiebreak); division titles earn nothing."""
    w = wins[:, league.teams] + rng.random((wins.shape[0], len(league.teams))) * 1e-3
    return league.teams[np.argsort(-w, axis=1)[:, :n]]


def best_vs_worst(teams, seed_of):
    """Pair a set of teams (n_sims, 2k) by seed: best vs worst, 2nd vs 2nd-worst, ..."""
    order = np.argsort(np.take_along_axis(seed_of, teams, axis=1), axis=1)
    t = np.take_along_axis(teams, order, axis=1)
    k = t.shape[1]
    return [(t[:, i], t[:, k - 1 - i]) for i in range(k // 2)]


def format_generic(league, ctx, per_league, seeding='record', lengths=(3, 5, 7), fixed_bracket=False):
    """Any field from 1 to 8 teams per league.

    seeding: 'record' (top teams by record, seeded by record), 'division' (3 division winners
             first, then Wild Cards), or 'division field' (division winners + Wild Cards make the
             field, as now, but the field is then seeded purely by record).
    Rounds per league: a first round that cuts the field to 4 (top 8 - per_league seeds get byes),
    then a 4-team round (the Division Series) and a 2-team round (the LCS). Every round is
    reseeded (best remaining seed vs worst) unless fixed_bracket=True (8 teams only: 1v8, 4v5,
    2v7, 3v6, winners meet in that order). lengths = series length for each of the three rounds.
    """
    st, he, rng = ctx['strength'], ctx['home_edge'], ctx['rng']
    n = per_league
    if seeding == 'record':
        seeds = seed_by_record(league, ctx['wins'], n, rng)
    else:
        seeds = seed_league(league, ctx['wins'], n - 3, rng)
        if seeding == 'division field':
            w = np.take_along_axis(ctx['wins'], seeds, axis=1) + rng.random(seeds.shape) * 1e-3
            seeds = np.take_along_axis(seeds, np.argsort(-w, axis=1), axis=1)
    seed_of = seed_lookup(seeds, ctx['n_teams'])
    series = lambda a, b, length: play_series(*higher_seed_first(a, b, seed_of), st, he, length, rng)

    if n > 4:
        byes = 8 - n
        if fixed_bracket and n == 8:
            pos = [(0, 7), (3, 4), (1, 6), (2, 5)]
            first = [series(seeds[:, i], seeds[:, j], lengths[0]) for i, j in pos]
        else:
            first = [series(a, b, lengths[0]) for a, b in best_vs_worst(seeds[:, byes:], seed_of)]
        ds = np.concatenate([seeds[:, :byes], np.stack(first, axis=1)], axis=1)
    else:
        first, ds = None, seeds

    m = ds.shape[1]
    if m == 4:
        if fixed_bracket and n == 8:
            lcs = np.stack([series(first[0], first[1], lengths[1]), series(first[2], first[3], lengths[1])], axis=1)
        else:
            lcs = np.stack([series(a, b, lengths[1]) for a, b in best_vs_worst(ds, seed_of)], axis=1)
    elif m == 3:
        order = np.argsort(np.take_along_axis(seed_of, ds, axis=1), axis=1)
        t = np.take_along_axis(ds, order, axis=1)
        lcs = np.stack([t[:, 0], series(t[:, 1], t[:, 2], lengths[1])], axis=1)
    else:
        lcs = ds
    pennant = series(lcs[:, 0], lcs[:, 1], lengths[2]) if lcs.shape[1] == 2 else lcs[:, 0]
    return {'seeds': seeds, 'pennant': pennant, 'reached_ds': ds, 'reached_lcs': lcs}


def format_nba(league, ctx):
    """NBA-style: top 10 by record. Play-in (single games): 7v8 (winner = 7 seed), 9v10 (loser out),
    then the 7v8 loser hosts the 9v10 winner for the 8 seed. Then 8 teams, fixed bracket
    (1v8, 4v5, 2v7, 3v6), every round best-of-7."""
    st, he, rng = ctx['strength'], ctx['home_edge'], ctx['rng']
    seeds = seed_by_record(league, ctx['wins'], 10, rng)
    seed_of = seed_lookup(seeds, ctx['n_teams'])
    s7, s8, s9, s10 = seeds[:, 6], seeds[:, 7], seeds[:, 8], seeds[:, 9]
    win78 = play_series(s7, s8, st, he, 1, rng)
    lose78 = np.where(win78 == s7, s8, s7)
    win910 = play_series(s9, s10, st, he, 1, rng)
    eighth = play_series(lose78, win910, st, he, 1, rng)
    final8 = np.concatenate([seeds[:, :6], win78[:, None], eighth[:, None]], axis=1)
    seed8 = seed_lookup(final8, ctx['n_teams'])        # bracket position = seed from here on
    series = lambda a, b: play_series(*higher_seed_first(a, b, seed8), st, he, 7, rng)
    first = [series(final8[:, i], final8[:, j]) for i, j in [(0, 7), (3, 4), (1, 6), (2, 5)]]
    lcs = np.stack([series(first[0], first[1]), series(first[2], first[3])], axis=1)
    pennant = series(lcs[:, 0], lcs[:, 1])
    return {'seeds': seeds, 'pennant': pennant, 'reached_ds': np.stack(first, axis=1), 'reached_lcs': lcs}


RECORD_SWEEP = {f'record seeding, {2 * n} teams': n for n in range(3, 9)}   # 6 to 16 teams
# Division winners still guaranteed a spot, but once the field is set it is seeded by record only
DIVISION_FIELD_SWEEP = {f'division field, seeded by record, {2 * n} teams': n for n in range(4, 9)}  # 8 to 16

FORMATS = {
    '8 teams (1998-2011)': format_8_team,
    '10 teams (2012-21)': format_10_team,
    '12 teams (current)': format_12_team,
    '12 teams + 7-game DS, reseeded': lambda lg, ctx: format_12_team(lg, ctx, ds_length=7, reseed=True),
    '12 teams + game-in-hand': lambda lg, ctx: format_12_team(lg, ctx, game_in_hand=True),
    '14 teams, pick your opponent': format_14_team,
    # Pick-rule variants (how division winners choose): only record, or only head-to-head
    '14 teams, pick by record only': lambda lg, ctx: format_14_team(lg, ctx, weight_record=1.0),
    '14 teams, pick by head-to-head only': lambda lg, ctx: format_14_team(lg, ctx, weight_record=0.0),
    # Same 14-team field and reseeded DS, but no picking: 2v7, 3v6, 4v5 (isolates what the pick is worth)
    '14 teams, fixed bracket': lambda lg, ctx: format_14_team(lg, ctx, pick=False),
    # Hypotheticals (anchors for the 0-10 scores)
    '12 teams, every round best-of-7': lambda lg, ctx: format_12_team(lg, ctx, ds_length=7, wc_length=7),
    'NBA-style (20 teams, play-in)': format_nba,
}
# Pure record seeding, reseeded every round, at every field size from 6 to 16 teams.
# 'record seeding, 12 teams' = current field size seeded by record and reseeded after the Wild Card
# round; 'record seeding, 16 teams' = 8 per league, reseeded before both rounds (Merit-cap candidate).
for _name, _n in RECORD_SWEEP.items():
    FORMATS[_name] = (lambda n: lambda lg, ctx: format_generic(lg, ctx, n, 'record'))(_n)
# Division winners + Wild Cards make the field (as now), then seeded by record and reseeded every round
for _name, _n in DIVISION_FIELD_SWEEP.items():
    FORMATS[_name] = (lambda n: lambda lg, ctx: format_generic(lg, ctx, n, 'division field'))(_n)
for _name, _opts in GRID.items():
    FORMATS[_name] = (lambda o: lambda lg, ctx: format_12_grid(lg, ctx, **o))(_opts)
WS_LENGTH = {}  # World Series length overrides by format (default best-of-7)


def run_postseason(fmt, leagues, wins, losses, h2h, strength, home_edge, rng, count_games=False):
    """Run one format for both leagues and the World Series. Returns arrays per sim.

    count_games: also return 'postseason_games', the total games played in each simulated postseason.
    """
    n_sims, n_teams = wins.shape
    _GAME_TALLY['games'] = np.zeros(n_sims) if count_games else None
    ctx = {'wins': wins, 'games_played': wins + losses, 'h2h': h2h, 'n_teams': n_teams,
           'strength': np.broadcast_to(strength, (n_sims, n_teams)), 'home_edge': home_edge, 'rng': rng}
    al, nl = (FORMATS[fmt](lg, ctx) for lg in leagues)
    # World Series home field goes to the better regular-season record
    rows = np.arange(n_sims)
    al_better = wins[rows, al['pennant']] + rng.random(n_sims) * 1e-3 > wins[rows, nl['pennant']]
    a = np.where(al_better, al['pennant'], nl['pennant'])
    b = np.where(al_better, nl['pennant'], al['pennant'])
    champion = play_series(a, b, ctx['strength'], home_edge, WS_LENGTH.get(fmt, 7), rng)
    out = {'seeds': (al['seeds'], nl['seeds']), 'pennants': (al['pennant'], nl['pennant']),
           'champion': champion, 'leagues': (al, nl)}
    if count_games:
        out['postseason_games'] = _GAME_TALLY['games']
        _GAME_TALLY['games'] = None
    return out
