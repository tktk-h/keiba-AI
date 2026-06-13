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

HEADERS = {"User-Agent": "Mozilla/5.0 (keiba-research)"}


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


def _sex_age(text):
    text = (text or "").strip()
    if not text:
        return "", None
    return text[0], _to_int(text[1:])


def parse_result_page(html: str, race_id: str):
    soup = BeautifulSoup(html, "html.parser")
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
            "name": name_a.get_text(strip=True) if name_a else tds[3].get_text(strip=True),
            "finish": finish,
            "won": 1 if finish == 1 else 0,
            "sex": sex,
            "age": age,
            "weight_carried": _to_float(tds[5].get_text(strip=True)),
            "last_3f": _to_float(tds[15].get_text(strip=True)),
            "win_odds": _to_float(tds[16].get_text(strip=True)),
            "popularity": _to_int(tds[17].get_text(strip=True)),
            "body_weight": int(body.group(1)) if body else None,
        })
    return rows


def fetch_result_rows(race_id: str):
    url = f"https://db.netkeiba.com/race/{race_id}/"
    return parse_result_page(fetch_html(url), race_id)
