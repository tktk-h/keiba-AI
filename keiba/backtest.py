"""Backtest the win model: split data by time, bet, measure ROI (回収率).

race_id encodes the date (YYYY + ...), so sorting by race_id is chronological.
We train on older races and evaluate on newer ones to avoid look-ahead bias.
ROI (回収率) = returned / spent; 1.0 means break-even, >1.0 means profit.
"""
import pandas as pd
from keiba.dataset import FEATURE_COLUMNS, LABEL_COLUMN
from keiba.model import train_model, model_win_probabilities


def time_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split into (train, test) by race chronology. Newest races -> test.

    Whole races stay together (never split a race across train/test)."""
    race_ids = sorted(df["race_id"].unique())
    n_test = max(1, int(round(len(race_ids) * test_fraction)))
    test_ids = set(race_ids[-n_test:])
    test = df[df["race_id"].isin(test_ids)]
    train = df[~df["race_id"].isin(test_ids)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def simulate_win_bets(df: pd.DataFrame, picks: dict, stake: float = 100) -> dict:
    """Bet `stake` on one picked horse per race; tally payout.

    picks: {race_id: horse_name}. A win pays stake * win_odds.
    Returns spent / returned / roi / bets / hits.
    """
    spent = 0.0
    returned = 0.0
    bets = 0
    hits = 0
    for race_id, horse_name in picks.items():
        race_rows = df[df["race_id"] == race_id]
        row = race_rows[race_rows["name"] == horse_name]
        bets += 1
        spent += stake
        if row.empty:
            continue
        r = row.iloc[0]
        if int(r["won"]) == 1:
            hits += 1
            returned += stake * float(r["win_odds"])
    roi = (returned / spent) if spent else 0.0
    return {"bets": bets, "hits": hits, "spent": spent,
            "returned": returned, "roi": roi}


def model_picks(model, test: pd.DataFrame, min_ev: float = 1.0) -> dict:
    """For each test race, pick the horse with the highest expected value
    (prob * odds), but only if its EV clears `min_ev` (skip -EV races)."""
    picks = {}
    for race_id, race in test.groupby("race_id"):
        probs = model_win_probabilities(model, race)
        best_name, best_ev = None, -1.0
        for _, row in race.iterrows():
            p = probs.get(row["name"], 0.0)
            odds = row["win_odds"]
            ev = p * odds
            if ev > best_ev:
                best_name, best_ev = row["name"], ev
        if best_name is not None and best_ev >= min_ev:
            picks[race_id] = best_name
    return picks


def favorite_picks(test: pd.DataFrame) -> dict:
    """Baseline: always bet the lowest-odds (most popular) horse per race."""
    picks = {}
    for race_id, race in test.groupby("race_id"):
        row = race.loc[race["win_odds"].idxmin()]
        picks[race_id] = row["name"]
    return picks


def run_backtest(df: pd.DataFrame, test_fraction: float = 0.2,
                 min_ev: float = 1.0, stake: float = 100) -> dict:
    """Train on older races, evaluate model vs favorite baseline on newer."""
    train, test = time_split(df, test_fraction)
    model = train_model(train)

    model_result = simulate_win_bets(test, model_picks(model, test, min_ev), stake)
    fav_result = simulate_win_bets(test, favorite_picks(test), stake)
    return {
        "train_races": train["race_id"].nunique(),
        "test_races": test["race_id"].nunique(),
        "model": model_result,
        "favorite": fav_result,
    }
