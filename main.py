import sys
from keiba.report import build_race_report


def _fmt_score(s: int) -> str:
    return f"+{s}" if s > 0 else (str(s) if s < 0 else "0")


def main(race_id: str, enrich: bool = False):
    print(f"レース {race_id} を取得・計算中...")
    if enrich:
        print("  各馬の過去成績も取得します(時間がかかります)...")
    rep = build_race_report(race_id, enrich=enrich)
    m = rep["meta"]
    print(f"  {m['name']} / {m['surface']}{m['distance']}m / {m['field_size']}頭")

    if not m.get("has_odds", True):
        print("\n※ オッズ未発表のため予想はまだ出せません(前売り確定後＝概ね前日以降)。")
        print("  出走馬一覧:")
        for r in rep["predictions"]:
            print(f"    {r['name']}")
        return

    print("\n=== AI予想印(理由つき) ===")
    by_mark = {p["mark"]: p for p in rep["predictions"] if p.get("mark")}
    for s in ["◎", "○", "▲", "△", "注", "穴"]:
        p = by_mark.get(s)
        if p:
            print(f"  {s} {p['name']:<11} 予想{p['win_prob']:.0%}/市場{p['market_prob']:.0%}"
                  f"/評価{_fmt_score(p['score'])} — {p['mark_reason']}")

    print("\n=== ① 予想ダッシュボード(資料・枠順) ===")
    print("  予想=モデル勝率 / 市場=オッズが示す勝率")
    print("  評価=市場との乖離 -5〜+5(+は過小評価/妙味・-は過大評価)")
    print(f"  {'馬番':>2} {'馬名':<11} {'騎手':<7} {'オッズ':>6} {'人気':>2} "
          f"{'予想':>5} {'市場':>5} {'評価':>3} {'複勝':>4} 確信")
    for r in sorted(rep["predictions"], key=lambda x: (x.get("number") or 99)):
        pp = f"{r['place_prob']:.0%}" if r["place_prob"] is not None else "—"
        od = f"{r['win_odds']:.1f}" if r.get("win_odds") else "—"
        pop = r.get("popularity") or "—"
        print(f"  {r.get('number') or '?':>2} {r['name']:<11} {(r.get('jockey') or '')[:6]:<7} "
              f"{od:>6} {pop:>2} {r['win_prob']:>5.1%} {r['market_prob']:>5.1%} "
              f"{_fmt_score(r['score']):>3} {pp:>4} {r['level']}")

    print("\n=== ② 乖離している馬の根拠(オッズに無い要因) ===")
    devs = [r for r in rep["predictions"] if r.get("reasons")]
    if not devs:
        print("  市場と概ね一致。際立った乖離馬はありません。")
    for r in devs:
        factors = " ・ ".join(f"{lab}{arrow}" for lab, arrow in r["reasons"])
        print(f"  {r['name']}(評価{_fmt_score(r['score'])} "
              f"予想{r['win_prob']:.0%}): {factors}")

    print("\n=== ③ 参考: 買い目のEV ===")
    print("  ※ 市場効率につき控除率を越える優位性は確認されていません(参考値)。")
    if not rep["bets"]:
        print("  該当なし。")
    else:
        for i, b in enumerate(rep["bets"], 1):
            mark = "＋" if b["ev"] >= 0 else "－"
            print(f"  {i}. [{b['type']}] {b['sel']:<11} オッズ{b['odds']:>6} "
                  f"推定{b['prob']:5.1%} EV{b['ev']:+.2f}{mark}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    enrich = "--enrich" in sys.argv
    if not args:
        print("使い方: python main.py <race_id> [--enrich]")
        sys.exit(1)
    main(args[0], enrich=enrich)
