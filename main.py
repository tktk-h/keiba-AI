import sys
from keiba.scraper import scrape_race
from keiba.features import build_features
from keiba.predictor import win_probabilities
from keiba.recommend import recommend_bets


def main(race_id: str, enrich: bool = False):
    print(f"レース {race_id} を取得中...")
    if enrich:
        print("  各馬の過去成績・血統も取得します(時間がかかります)...")
    race = scrape_race(race_id, enrich=enrich)
    print(f"  {race.name} / {race.surface}{race.distance}m / {len(race.horses)}頭")
    df = build_features(race)
    win_probs = win_probabilities(df)
    recs = recommend_bets(df, win_probs, top_n=5)
    print("\n=== 期待値の高い買い方 TOP5 ===")
    if not recs:
        print("(オッズ未確定のため計算できません。レース直前のIDを指定してください)")
        return
    for i, b in enumerate(recs, 1):
        print(f"{i}. {b['bet_type']} {b['name']} "
              f"オッズ{b['odds']} 推定確率{b['prob']:.1%} 期待値{b['ev']:+.2f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    enrich = "--enrich" in sys.argv
    if not args:
        print("使い方: python main.py <race_id> [--enrich]")
        sys.exit(1)
    main(args[0], enrich=enrich)
