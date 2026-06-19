"""CLI と Web が共有する、1レース分のレポート組み立て。

assemble_report は純粋(テスト対象)。build_race_report はネットワーク全工程
(出馬表→確定オッズ補完→特徴量→モデル→オッズ→組み立て)。
"""
import os
from keiba.scraper import scrape_race
from keiba.features import build_features
from keiba.predictor import win_probabilities
from keiba.odds_page import fetch_odds
from keiba.model import load_model, DEFAULT_MODEL_PATH
from keiba.recommend import predict_ranking, recommend_all


def assemble_report(race, win_probs: dict, odds: dict,
                    model=None, features=None) -> dict:
    """予想ランキングと買い目を、表示しやすい dict にまとめる。

    model と features(build_features の DataFrame)が与えられれば、乖離馬に
    非オッズ根拠 reasons を付ける。
    """
    predictions = predict_ranking(race, win_probs)
    if model is not None and features is not None:
        from keiba.explain import attach_reasons
        attach_reasons(predictions, model, features)
    bets, any_positive = recommend_all(race, win_probs, odds)
    return {
        "meta": {
            "race_id": race.race_id,
            "name": race.name,
            "surface": race.surface,
            "distance": race.distance,
            "field_size": len(race.horses),
        },
        "predictions": predictions,
        "bets": bets,
        "any_positive": any_positive,
    }


def _backfill_market(race, odds) -> None:
    """締め切り後などで出馬表に単勝/人気が無い場合、確定オッズで補完する。"""
    for h in race.horses:
        if h.win_odds is None and odds["win"].get(h.number) is not None:
            h.win_odds = odds["win"][h.number]
    runners = [h for h in race.horses if h.win_odds is not None]
    if runners and all(h.popularity is None for h in race.horses):
        for rank, h in enumerate(sorted(runners, key=lambda x: x.win_odds), 1):
            h.popularity = rank


def build_race_report(race_id: str, enrich: bool = True) -> dict:
    """1レースを取得・予測して assemble_report の dict を返す(ネットワーク)。"""
    race = scrape_race(race_id, enrich=enrich)
    try:
        odds = fetch_odds(race_id)
    except Exception:  # noqa: BLE001
        odds = {"win": {}, "place": {}, "quinella": {}, "wide": {}}
    _backfill_market(race, odds)
    df = build_features(race)
    model = (load_model(DEFAULT_MODEL_PATH)
             if os.path.exists(DEFAULT_MODEL_PATH) else None)
    win_probs = win_probabilities(df, model=model)
    return assemble_report(race, win_probs, odds, model=model, features=df)
