import os
import sys
from keiba.scraper import scrape_race
from keiba.features import build_features
from keiba.predictor import win_probabilities
from keiba.recommend import predict_ranking, recommend_all
from keiba.odds_page import fetch_odds
from keiba.model import load_model, DEFAULT_MODEL_PATH


def main(race_id: str, enrich: bool = False):
    print(f"レース {race_id} を取得中...")
    if enrich:
        print("  各馬の過去成績・血統も取得します(時間がかかります)...")
    race = scrape_race(race_id, enrich=enrich)
    print(f"  {race.name} / {race.surface}{race.distance}m / {len(race.horses)}頭")

    # オッズを先に取得。出馬表に単勝オッズが無い場合(締め切り後など)は
    # 確定オッズ(=締め切り直前の価格)で各馬を補完してからモデルにかける。
    try:
        odds = fetch_odds(race_id)
    except Exception as exc:  # noqa: BLE001
        print(f"  (オッズ取得に失敗: {exc})")
        odds = {"win": {}, "place": {}, "quinella": {}, "wide": {}}
    filled = 0
    for h in race.horses:
        if h.win_odds is None and odds["win"].get(h.number) is not None:
            h.win_odds = odds["win"][h.number]
            filled += 1
    # 人気は単勝オッズの順位(出馬表に無い=締め切り後なら確定オッズ順で補完)
    runners = [h for h in race.horses if h.win_odds is not None]
    if runners and all(h.popularity is None for h in race.horses):
        for rank, h in enumerate(sorted(runners, key=lambda x: x.win_odds), 1):
            h.popularity = rank
    if filled:
        print(f"  確定オッズで単勝 {filled}頭ぶんを補完しました(締め切り直前を再現)。")

    df = build_features(race)
    model = None
    if os.path.exists(DEFAULT_MODEL_PATH):
        model = load_model(DEFAULT_MODEL_PATH)
        print("  学習済みモデルで予測します。")
    else:
        print("  モデル未学習のためオッズベースで予測します(train.pyで学習可)。")
    win_probs = win_probabilities(df, model=model)

    # ① 統計ベース予想(全馬)
    print("\n=== ① 予想(統計ベース・確信度つき) ===")
    for i, r in enumerate(predict_ranking(race, win_probs), 1):
        pp = f"{r['place_prob']:.1%}" if r["place_prob"] is not None else "—"
        print(f"{i:>2}. {r['name']:<12} 勝率{r['win_prob']:5.1%} "
              f"複勝{pp:>6} 確信度{r['level']}({r['confidence']:.2f})")

    # ② EV提案(単勝/複勝/ワイド/馬連)— オッズは上で取得済み
    bets, any_positive = recommend_all(race, win_probs, odds, top_n=8)
    print("\n=== ② EV提案(期待値の高い買い方) ===")
    if not bets:
        if not any(odds.get(k) for k in ("win", "place", "quinella", "wide")):
            print("(オッズ未確定のため計算できません。レース直前のIDを指定してください)")
        else:
            print("推奨できる買い目はありません(確率・確信度が基準を満たす候補なし)。")
        return
    if not any_positive:
        print("※ +EVの買い目はありません。今レースは見送り推奨(以下は参考)。")
    for i, b in enumerate(bets, 1):
        mark = "＋" if b["ev"] >= 0 else "－"
        print(f"{i}. [{b['type']}] {b['sel']:<12} オッズ{b['odds']:>6} "
              f"推定{b['prob']:5.1%} 期待値{b['ev']:+.2f}{mark} 確信度{b['level']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    enrich = "--enrich" in sys.argv
    if not args:
        print("使い方: python main.py <race_id> [--enrich]")
        sys.exit(1)
    main(args[0], enrich=enrich)
