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

    # ② EV提案(単勝/複勝/ワイド/馬連)
    try:
        odds = fetch_odds(race_id)
    except Exception as exc:  # noqa: BLE001
        print(f"\n(オッズ取得に失敗: {exc} — 単勝のみで提案します)")
        odds = {"place": {}, "quinella": {}, "wide": {}}
    bets, any_positive = recommend_all(race, win_probs, odds, top_n=8)
    print("\n=== ② EV提案(期待値の高い買い方) ===")
    if not bets:
        print("(オッズ未確定のため計算できません。レース直前のIDを指定してください)")
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
