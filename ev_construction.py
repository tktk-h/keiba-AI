"""『買い方の工夫』で回収率が上がるかの実測(キャッシュ377レース)。

各レースで、モデルが +EV と判定した買い目(単勝/複勝/ワイド/馬連, 確率フロア5%)を
候補に、次の買い方を比較する:
  A. all-flat   : +EV を全部100円ずつ
  B. max-EV     : 期待値が最大の1点だけ100円
  C. max-odds   : 合成オッズ最大=最も高オッズの+EV 1点だけ100円
  D. dutch合成  : +EV候補に合計100円を 1/オッズ で配分(均等払い戻し)
期待値の線形性により、どれも回収率は控除率の壁(~80%)に収束するはず。

Usage: python ev_construction.py
"""
import pandas as pd
from keiba.dataset import build_dataset
from keiba.model import train_model
from sim_bets import load_train, collect_weekend
from sim_ev import load_odds, candidate_bets

DATES = ["20260502", "20260614"]
MIN_PROB = 0.05


def realized(combo_type, combo, payouts_race, stake, odds):
    """配当があれば stake*odds/... 実際の払戻はオッズ=配当/100 と一致するので
    payout テーブルに当該組があれば当たり。"""
    combos = payouts_race.get(combo_type, {})
    if combo in combos:
        return stake * combos[combo] / 100.0
    return 0.0


def main():
    model = train_model(load_train())
    raw, payouts = collect_weekend(DATES)
    feat = build_dataset(raw)
    odds_all = load_odds(DATES, [str(r) for r in feat["race_id"].unique()])

    tot = {k: {"spent": 0.0, "ret": 0.0} for k in ["A", "B", "C", "D"]}
    example = None

    for rid, race in feat.groupby("race_id"):
        rid = str(rid)
        raw_race = [r for r in raw if str(r["race_id"]) == rid]
        odds = odds_all.get(rid)
        if not odds:
            continue
        cands = [(t, c, p, o) for (t, c, p, o) in
                 candidate_bets(race, raw_race, odds, model)
                 if p >= MIN_PROB and p * o - 1 >= 0]
        if not cands:
            continue
        pay = payouts.get(rid, {})

        # A: all flat 100 each
        for t, c, p, o in cands:
            tot["A"]["spent"] += 100
            tot["A"]["ret"] += realized(t, c, pay, 100, o)
        # B: single max-EV
        t, c, p, o = max(cands, key=lambda x: x[2] * x[3] - 1)
        tot["B"]["spent"] += 100
        tot["B"]["ret"] += realized(t, c, pay, 100, o)
        # C: single max-odds (合成オッズ最大)
        t, c, p, o = max(cands, key=lambda x: x[3])
        tot["C"]["spent"] += 100
        tot["C"]["ret"] += realized(t, c, pay, 100, o)
        # D: dutch — 100円を 1/odds で配分
        inv = sum(1.0 / o for _, _, _, o in cands)
        for t, c, p, o in cands:
            stake = 100 * (1.0 / o) / inv
            tot["D"]["spent"] += stake
            tot["D"]["ret"] += realized(t, c, pay, stake, o)

        if example is None and 2 <= len(cands) <= 4:
            synth = 1.0 / sum(1.0 / o for _, _, _, o in cands)
            example = (rid, cands, synth)

    print(f"\n=== 買い方別 回収率 (377レース・確率フロア{MIN_PROB:.0%}) ===")
    names = {"A": "全+EVフラット", "B": "max-EV 1点",
             "C": "max-odds(合成最大) 1点", "D": "dutch合成配分"}
    for k in ["A", "B", "C", "D"]:
        s, r = tot[k]["spent"], tot[k]["ret"]
        print(f"  {names[k]:<22} 回収率 {r/s:6.1%}  (投資{s:.0f}円)")

    if example:
        rid, cands, synth = example
        print(f"\n=== 例: race_id={rid} の +EV 構成 ===")
        for t, c, p, o in cands:
            nums = "-".join(map(str, sorted(c)))
            print(f"  [{t:<8}] {nums:<7} オッズ{o:>7.1f} 推定確率{p:5.1%} 期待値{p*o-1:+.2f}")
        print(f"  → この構成の合成オッズ(1/Σ(1/オッズ)) = {synth:.2f}")
        print("  ※ 合成オッズを上げるほど的中率は下がる。期待値(=回収率)は不変。")


if __name__ == "__main__":
    main()
