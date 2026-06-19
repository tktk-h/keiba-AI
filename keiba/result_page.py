"""Parse a netkeiba race RESULT page into labeled rows for model training.

URL: https://db.netkeiba.com/race/<race_id>/
table.race_table_01 column indices (verified against a real page):
  0=着順 3=馬名(+/horse/<id>) 4=性齢 5=斤量 7=タイム 15=上り
  16=単勝オッズ 17=人気 18=馬体重
Each returned dict carries features plus the training label `won` (1 if 着順==1).
"""
import re
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"}


def fetch_html(url: str) -> str:
    # Network function - manual use only, not exercised by tests.
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = "EUC-JP"
    return resp.text


def _to_float(text):
    try:
        return float((text or "").strip())
    except ValueError:
        return None


def _to_int(text):
    m = re.search(r"\d+", text or "")
    return int(m.group()) if m else None


def _time_to_seconds(text):
    """'2:24.3' -> 144.3 ; '57.3' -> 57.3 ; '' -> None."""
    text = (text or "").strip()
    if not text:
        return None
    if ":" in text:
        minutes, rest = text.split(":", 1)
        try:
            return int(minutes) * 60 + float(rest)
        except ValueError:
            return None
    return _to_float(text)


def _sex_age(text):
    text = (text or "").strip()
    if not text:
        return "", None
    return text[0], _to_int(text[1:])


def _parse_race_meta(soup):
    """Extract surface/distance/track_condition from the race-data block,
    e.g. "11 R 第91回東京優駿(GI) 芝右2400m / 天候 : 晴 / 芝 : 良 / 発走 : 15:40"."""
    el = soup.select_one("dl.racedata") or soup.select_one(".data_intro")
    text = el.get_text(" ", strip=True) if el else ""
    surface = distance = track_condition = None
    m = re.search(r"(芝|ダ|障)(左|右|直)?(\d+)m", text)
    if m:
        surface = m.group(1)
        distance = int(m.group(3))
    m = re.search(r"(芝|ダート|障)\s*[:：]\s*(\S+?)\s*/", text)
    if m:
        track_condition = m.group(2)
    return surface, distance, track_condition


def parse_result_page(html: str, race_id: str):
    soup = BeautifulSoup(html, "html.parser")
    surface, distance, track_condition = _parse_race_meta(soup)
    table = soup.select_one("table.race_table_01")
    if table is None:
        return []
    rows = []
    for tr in table.select("tr")[1:]:  # skip header
        tds = tr.select("td")
        if len(tds) < 19:
            continue
        finish = _to_int(tds[0].get_text(strip=True))
        if finish is None:  # 中止/除外 rows have non-numeric 着順
            continue
        name_a = tds[3].select_one("a")
        horse_id = None
        if name_a and name_a.get("href"):
            m = re.search(r"/horse/(\w+)", name_a["href"])
            if m:
                horse_id = m.group(1)
        sex, age = _sex_age(tds[4].get_text(strip=True))
        body = re.match(r"(\d+)", tds[18].get_text(strip=True))
        rows.append({
            "race_id": race_id,
            "horse_id": horse_id,
            "number": _to_int(tds[2].get_text(strip=True)),  # 馬番
            "name": name_a.get_text(strip=True) if name_a else tds[3].get_text(strip=True),
            "finish": finish,
            "won": 1 if finish == 1 else 0,
            "sex": sex,
            "age": age,
            "weight_carried": _to_float(tds[5].get_text(strip=True)),
            "time": _time_to_seconds(tds[7].get_text(strip=True)),
            "last_3f": _to_float(tds[15].get_text(strip=True)),
            "win_odds": _to_float(tds[16].get_text(strip=True)),
            "popularity": _to_int(tds[17].get_text(strip=True)),
            "body_weight": int(body.group(1)) if body else None,
            "surface": surface,
            "distance": distance,
            "track_condition": track_condition,
        })
    return rows


def fetch_result_rows(race_id: str):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    return parse_result_page(fetch_html(url), race_id)
