"""ウォークフォワード検証: LightGBM vs ロジスティックの再現性(単勝)。

全券種のオッズ/払戻は2026の377レースしか無いが、単勝なら v3 月次データ
(win_odds と won を保持)で多期間に検証できる。各テスト月について、それ以前の
全月で学習し、その月の単勝 +EV 買い目(確率フロアつき)の回収率を測る。前進。

『88.9%が単一窓の過学習でないか』を、モデル差の符号の安定性で確認するのが狙い。

Usage: python walk_forward.py
"""
import glob
import re
import pandas as pd
from keiba.model import train_model, train_lgbm_model, model_win_probabilities
from keiba.expected_value import ev as ev_of

FLOORS = [0.0, 0.05, 0.10]
MIN_TRAIN_MONTHS = 12


def month_of(path):
    m = re.search(r"(\d{4})_(\d{2})_v3", path)
    return f"{m.group(1)}-{m.group(2)}"


def sim_win(df, model, floor, stake=100):
    """Bet every horse with prob>=floor and +EV on its win odds. Pooled."""
    spent = ret = bets = hits = 0
    for _, race in df.groupby("race_id"):
        probs = model_win_probabilities(model, race)
        for _, row in race.iterrows():
            p = probs.get(row["name"], 0.0)
            o = row["win_odds"]
            if not o or o <= 0 or p < floor or ev_of(p, o) < 0:
                continue
            bets += 1
            spent += stake
            if int(row["won"]) == 1:
                hits += 1
                ret += stake * float(o)
    return {"bets": bets, "hits": hits, "spent": spent, "returned": ret,
            "roi": (ret / spent if spent else 0.0)}


def main():
    files = sorted(glob.glob("data_months/*_v3.csv"))
    months = [month_of(f) for f in files]
    frames = {month_of(f): pd.read_csv(f) for f in files}
    print(f"  月数: {len(files)}  テスト窓: {len(files) - MIN_TRAIN_MONTHS}\n")

    pooled = {f: {"log": {"spent": 0.0, "ret": 0.0, "bets": 0},
                  "lgb": {"spent": 0.0, "ret": 0.0, "bets": 0}} for f in FLOORS}
    wins = {f: 0 for f in FLOORS}   # months where LGBM ROI > logistic
    n_windows = 0

    for i in range(MIN_TRAIN_MONTHS, len(files)):
        train = pd.concat([frames[m] for m in months[:i]], ignore_index=True)
        test = frames[months[i]]
        mlog = train_model(train)
        mlgb = train_lgbm_model(train)
        n_windows += 1
        line = [months[i]]
        for f in FLOORS:
            rl = sim_win(test, mlog, f)
            rg = sim_win(test, mlgb, f)
            for key, r in (("log", rl), ("lgb", rg)):
                pooled[f][key]["spent"] += r["spent"]
                pooled[f][key]["ret"] += r["returned"]
                pooled[f][key]["bets"] += r["bets"]
            if rg["roi"] > rl["roi"]:
                wins[f] += 1
            if f == 0.10:
                line.append(f"log {rl['roi']:.0%} / lgb {rg['roi']:.0%}")
        print("  " + "  ".join(line))

    print(f"\n=== プール回収率(全テスト窓合算) ===")
    print("floor".ljust(8) + "ロジスティック".rjust(18) + "LightGBM".rjust(18) +
          "LGBM勝ち月".rjust(14))
    for f in FLOORS:
        pl, pg = pooled[f]["log"], pooled[f]["lgb"]
        rl = pl["ret"] / pl["spent"] if pl["spent"] else 0
        rg = pg["ret"] / pg["spent"] if pg["spent"] else 0
        print(f"{f:>5.0%}   "
              f"{rl:>9.1%}({pl['bets']:>5})   {rg:>9.1%}({pg['bets']:>5})   "
              f"{wins[f]:>2}/{n_windows}")


if __name__ == "__main__":
    main()
