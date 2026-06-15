"""Decisive edge test: drop ALL market-derived features (odds, popularity,
and their past-run aggregates) and check whether a 'fundamentals only' model
has any predictive power orthogonal to the market price.
"""
import numpy as np
import pandas as pd
from keiba.backtest import time_split
from keiba.model import train_model
from keiba.dataset import FEATURE_COLUMNS, LABEL_COLUMN

MARKET = ["win_odds", "log_odds", "odds_rank_pct", "popularity",
          "prev_avg_popularity", "prev_avg_log_odds"]
NOODDS = [c for c in FEATURE_COLUMNS if c not in MARKET]

df = pd.read_csv("data_combined_v3.csv")
train, test = time_split(df, 0.4)
print("NOODDS features:", NOODDS, "\n")

model = train_model(train, feature_columns=NOODDS)
test = test.copy()
test["won"] = test[LABEL_COLUMN].astype(int)
test["p_raw"] = model.predict_proba(test[NOODDS])[:, 1]
test["mkt_raw"] = 1.0 / test["win_odds"]
test["p"] = test.groupby("race_id")["p_raw"].transform(lambda s: s / s.sum())
test["mkt"] = test.groupby("race_id")["mkt_raw"].transform(lambda s: s / s.sum())
test["payout"] = test["win_odds"] * test["won"]

print(f"test races={test['race_id'].nunique()}  horses={len(test)}  "
      f"base win rate={test['won'].mean():.3f}")
print(f"Brier  no-odds model={((test['p']-test['won'])**2).mean():.4f}  "
      f"market={((test['mkt']-test['won'])**2).mean():.4f}\n")

# --- Does model/market DISAGREEMENT predict winners? ---
# Bucket by (model_p - market_p). If the model adds orthogonal info, then where
# the model says "higher than market", actual win rate should exceed market's.
test["diff"] = test["p"] - test["mkt"]
test["db"] = pd.qcut(test["diff"], 5, labels=["mkt>>mdl", "mkt>mdl", "~equal",
                                              "mdl>mkt", "mdl>>mkt"])
g = test.groupby("db", observed=True).agg(
    n=("won", "size"), mkt=("mkt", "mean"), model=("p", "mean"),
    actual=("won", "mean"), blind_roi=("payout", "mean"))
print("--- Where model disagrees with market, who is right? ---")
print(f"  {'bucket':>9} {'n':>6} {'mkt%':>6} {'model%':>7} {'actual%':>8} {'blindROI':>9}")
for b, r in g.iterrows():
    edge = r.actual - r.mkt   # >0 means market underpriced these (model right)
    print(f"  {b:>9} {int(r.n):>6} {r.mkt*100:>5.1f} {r.model*100:>6.1f} "
          f"{r.actual*100:>7.1f} {r.blind_roi*100:>7.1f}%   "
          f"(actual-mkt={edge*100:+.1f}pt)")
print()

# --- EV sweep using the fundamentals-only model probability ---
test["ev"] = test["p"] * test["win_odds"]
print("--- EV-threshold sweep (no-odds model EV >= t, flat bet) ---")
print(f"  {'min_ev':>7} {'bets':>6} {'ROI':>8} {'hit%':>6}")
for t in [1.0, 1.2, 1.5, 2.0, 3.0]:
    sel = test[test["ev"] >= t]
    if len(sel) == 0:
        print(f"  {t:>7} {0:>6} {'n/a':>8}")
        continue
    print(f"  {t:>7} {len(sel):>6} {sel['payout'].sum()/len(sel)*100:>7.1f}% "
          f"{sel['won'].mean()*100:>5.1f}")
