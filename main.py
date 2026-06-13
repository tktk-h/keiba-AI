import sys
from keiba.scraper import scrape_race
from keiba.features import build_features
from keiba.predictor import win_probabilities
from keiba.recommend import recommend_bets


def main(race_id: str):
    print(f"レース {race_id} を取得中...")
    race = scrape_race(race_id)
    df = build_features(race)
    win_probs = win_probabilities(df)
    recs = recommend_bets(df, win_probs, top_n=5)
    print("\n=== 期待値の高い買い方 TOP5 ===")
    for i, b in enumerate(recs, 1):
        print(f"{i}. {b['bet_type']} {b['name']} "
              f"オッズ{b['odds']} 推定確率{b['prob']:.1%} 期待値{b['ev']:+.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python main.py <race_id>")
        sys.exit(1)
    main(sys.argv[1])
