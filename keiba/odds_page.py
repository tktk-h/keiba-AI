"""netkeibaオッズAPIから複勝・ワイド・馬連オッズを取得・整形する。

fetch_odds はネットワーク(手動利用、テスト対象外)。parse_odds と
_extract_* は純粋関数でテスト対象。netkeibaの実レスポンス構造は
api_get_jra_odds.html?type=N で切り替わる(確認済み):
  type=1: data.odds["1"]=単勝 {馬番: [odds, "0.0", 人気]}、
          data.odds["2"]=複勝 {馬番: [下限, 上限, 人気]}
  type=4: data.odds["4"]=馬連 {"0102": [odds, "0.0", 人気]}
  type=5: data.odds["5"]=ワイド {"0102": [下限, 上限, 人気]}
馬番キーはゼロ詰め("01")、連系は4桁("0102"=1番-2番)。オッズ値は
"2,612.8" のようにカンマを含むことがあり、取消馬は "-2.0"。
"""
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}
API = "https://race.netkeiba.com/api/api_get_jra_odds.html"


def _pair_key(s: str):
    a, b = s.split("_")
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def parse_odds(sample: dict) -> dict:
    """整形済みJSON -> {'win':{int:float}, 'place':{int:(lo,hi)},
    'quinella':{(a,b):float}, 'wide':{(a,b):float}}。

    sample の値は数値文字列(カンマ除去済み)を前提とする(_extract_* が整形)。
    """
    win = {int(k): float(v) for k, v in sample.get("win", {}).items()}
    place = {int(k): (float(v[0]), float(v[1]))
             for k, v in sample.get("place", {}).items()}
    quinella = {_pair_key(k): float(v) for k, v in sample.get("quinella", {}).items()}
    wide = {_pair_key(k): float(v) for k, v in sample.get("wide", {}).items()}
    return {"win": win, "place": place, "quinella": quinella, "wide": wide}


def _clean(s) -> str:
    """'2,612.8' -> '2612.8'。"""
    return str(s).replace(",", "")


def _extract_win(raw: dict) -> dict:
    """type=1 生レスポンス -> {'馬番': 'オッズ'}(単勝、block "1")。取消馬は除外。"""
    block = raw.get("data", {}).get("odds", {}).get("1", {})
    out = {}
    for num, vals in block.items():
        if not isinstance(vals, list) or not vals:
            continue
        v = _clean(vals[0])
        try:
            if float(v) < 0:
                continue
        except ValueError:
            continue
        out[num] = v
    return out


def _extract_place(raw: dict) -> dict:
    """type=1 生レスポンス -> {'馬番': ['下限','上限']}。取消馬(負値)は除外。"""
    block = raw.get("data", {}).get("odds", {}).get("2", {})
    out = {}
    for num, vals in block.items():
        if not isinstance(vals, list) or len(vals) < 2:
            continue
        lo, hi = _clean(vals[0]), _clean(vals[1])
        try:
            if float(lo) < 0:
                continue
        except ValueError:
            continue
        out[num] = [lo, hi]
    return out


def _extract_pairs(raw: dict, block_key: str) -> dict:
    """type=4/5 生レスポンス -> {'a_b': 'odds'}(下限採用)。取消・無効は除外。"""
    block = raw.get("data", {}).get("odds", {}).get(block_key, {})
    out = {}
    for combo, vals in block.items():
        if len(combo) != 4:
            continue
        a, b = int(combo[:2]), int(combo[2:])
        val = _clean(vals[0] if isinstance(vals, list) else vals)
        try:
            if float(val) < 0:
                continue
        except ValueError:
            continue
        out[f"{a}_{b}"] = val
    return out


def _get(race_id: str, type_: int) -> dict:
    resp = requests.get(API, params={"race_id": race_id, "type": type_,
                                     "action": "update"}, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_odds(race_id: str) -> dict:
    """実APIから単勝・複勝(type1)/馬連(type4)/ワイド(type5)を取得し整形する。"""
    raw1 = _get(race_id, 1)   # 単勝(block "1") と 複勝(block "2") を含む
    sample = {
        "win": _extract_win(raw1),
        "place": _extract_place(raw1),
        "quinella": _extract_pairs(_get(race_id, 4), "4"),
        "wide": _extract_pairs(_get(race_id, 5), "5"),
    }
    return parse_odds(sample)
