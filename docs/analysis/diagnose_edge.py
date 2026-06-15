"""Diagnostic: is the win model better-calibrated than the market, and is
there any odds range where we have an edge? Time-split (train old / test new),
no code changes to the pipeline.
"""
import numpy as np
import pandas as pd
from keiba.backtest import time_split
from keiba.model import train_model
from keiba.dataset import FEATURE_COLUMNS, LABEL_COLUMN

df = pd.read_csv("data_combined_v3.csv")
train, test = time_split(df, 0.4)
model = train_model(train)

test = test.copy()
test["p_raw"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
test["won"] = test[LABEL_COLUMN].astype(int)
test["mkt_raw"] = 1.0 / test["win_odds"]

# Normalize both model and market probabilities within each race (sum to 1).
def _norm(col):
    return test.groupby("race_id")[col].transform(lambda s: s / s.sum())
test["p"] = _norm("p_raw")
test["mkt"] = _norm("mkt_raw")
test["payout"] = test["win_odds"] * test["won"]   # flat-stake return if bet

print(f"test races={test['race_id'].nunique()}  horses={len(test)}  "
      f"base win rate={test['won'].mean():.3f}\n")

# ---- A) Calibration: model vs market (deciles of predicted prob) ----
def calibration(prob_col, label):
    t = test.copy()
    t["bin"] = pd.qcut(t[prob_col], 10, labels=False, duplicates="drop")
    g = t.groupby("bin").agg(n=("won", "size"),
                             pred=(prob_col, "mean"),
                             actual=("won", "mean"))
    print(f"--- Calibration: {label} (pred vs actual win rate) ---")
    for _, r in g.iterrows():
        flag = "over" if r.pred > r.actual else "under"
        print(f"  pred={r.pred:5.3f}  actual={r.actual:5.3f}  n={int(r.n):4d}  ({flag})")
    # Brier score
    brier = ((test[prob_col] - test["won"]) ** 2).mean()
    print(f"  Brier={brier:.4f}\n")

calibration("p", "MODEL")
calibration("mkt", "MARKET (1/odds, normalized)")

# ---- B) Edge by odds bucket ----
bins = [0, 2, 3, 5, 7, 10, 20, 50, 1e9]
test["ob"] = pd.cut(test["win_odds"], bins)
g = test.groupby("ob", observed=True).agg(
    n=("won", "size"),
    win=("won", "mean"),
    mkt=("mkt", "mean"),
    model=("p", "mean"),
    blind_roi=("payout", "mean"),
)
print("--- Edge by odds bucket (blind_roi = ROI if you bet every horse) ---")
print(f"  {'odds':>12} {'n':>6} {'win%':>6} {'mkt%':>6} {'model%':>7} {'blindROI':>9}")
for ob, r in g.iterrows():
    print(f"  {str(ob):>12} {int(r.n):>6} {r.win*100:>5.1f} {r.mkt*100:>5.1f} "
          f"{r.model*100:>6.1f} {r.blind_roi*100:>8.1f}%")
print()

# ---- C) EV-threshold sweep (model prob * odds), flat bet on every +EV horse ----
test["ev"] = test["p"] * test["win_odds"]
print("--- EV-threshold sweep (bet every horse with model EV >= t) ---")
print(f"  {'min_ev':>7} {'bets':>6} {'ROI':>8} {'hit%':>6}")
for t in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
    sel = test[test["ev"] >= t]
    if len(sel) == 0:
        print(f"  {t:>7} {0:>6} {'n/a':>8}")
        continue
    roi = sel["payout"].sum() / len(sel)
    hit = sel["won"].mean()
    print(f"  {t:>7} {len(sel):>6} {roi*100:>7.1f}% {hit*100:>5.1f}")
