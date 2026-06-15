import sys
from keiba.report import build_race_report


def main(race_id: str, enrich: bool = False):
    print(f"レース {race_id} を取得・計算中...")
    if enrich:
        print("  各馬の過去成績も取得します(時間がかかります)...")
    rep = build_race_report(race_id, enrich=enrich)
    m = rep["meta"]
    print(f"  {m['name']} / {m['surface']}{m['distance']}m / {m['field_size']}頭")

    print("\n=== ① 予想(モデル vs 市場・妙味つき) ===")
    print("  予想=モデル勝率 / 市場=オッズが示す勝率 / 妙味=モデルが人気より高評価")
    for i, r in enumerate(rep["predictions"], 1):
        pp = f"{r['place_prob']:.1%}" if r["place_prob"] is not None else "—"
        print(f"{i:>2}. {r['name']:<11} 予想{r['win_prob']:5.1%} 市場{r['market_prob']:5.1%} "
              f"{r['value']:<5}複勝{pp:>6} 確信{r['level']}")

    print("\n=== ② 買い目のEV(参考・優位性は未確認) ===")
    print("※ 250レースのバックテストでは控除率を越える優位性は確認されていません"
          "(回収率~75%)。EVは目安です。")
    if not rep["bets"]:
        print("推奨できる買い目はありません(確率・確信度が基準を満たす候補なし)。")
        return
    if not rep["any_positive"]:
        print("※ +EVの買い目はありません。今レースは見送り推奨(以下は参考)。")
    for i, b in enumerate(rep["bets"], 1):
        mark = "＋" if b["ev"] >= 0 else "－"
        print(f"{i}. [{b['type']}] {b['sel']:<11} オッズ{b['odds']:>6} "
              f"推定{b['prob']:5.1%} 期待値{b['ev']:+.2f}{mark} 確信{b['level']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    enrich = "--enrich" in sys.argv
    if not args:
        print("使い方: python main.py <race_id> [--enrich]")
        sys.exit(1)
    main(args[0], enrich=enrich)
