"""Parse a horse's race-results page and pedigree page on db.netkeiba.com.

- Results: https://db.netkeiba.com/horse/result/<horse_id>/
  table.db_h_race_results, column indices (verified against a real page):
    0=日付 1=開催 4=レース名 9=単勝オッズ 10=人気 11=着順 12=騎手 13=斤量
    14=距離(芝2400) 16=馬場 18=タイム(2:23.0) 27=上り(上がり3F)
- Pedigree: https://db.netkeiba.com/horse/ped/<horse_id>/
  table.blood_table; td[rowspan=16] = 父/母, td[rowspan=8][2] = 母父.
"""
import re
import requests
from bs4 import BeautifulSoup
from keiba.models import PastRun

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
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else None


def _time_to_seconds(text):
    """'2:23.0' -> 143.0 ; '57.3' -> 57.3 ; '' -> None."""
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


def _course_from_kaisai(text):
    """'5東京8' -> '東京' (strip leading/trailing digits)."""
    return re.sub(r"\d+", "", text or "").strip()


def _surface_distance(text):
    """'芝2400' -> ('芝', 2400)."""
    m = re.match(r"(芝|ダ|障)\s*(\d+)", (text or "").strip())
    if m:
        return m.group(1), int(m.group(2))
    return "", None


def _race_class(race_name):
    """Extract a grade tag like 'G1'/'G2'/'G3' from a race name, else ''."""
    m = re.search(r"\((G[I1-3]+|L|OP|J\.?G[I1-3]+)\)", race_name or "")
    return m.group(1) if m else ""


def parse_results(html: str, limit: int = 5):
    """Return up to `limit` most-recent PastRun records (newest first)."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.db_h_race_results")
    if table is None:
        return []
    runs = []
    for row in table.select("tr")[1:]:  # skip header
        tds = row.select("td")
        if len(tds) < 28:
            continue
        text = [td.get_text(strip=True) for td in tds]
        surface, distance = _surface_distance(text[14])
        runs.append(PastRun(
            date=text[0],
            finish=_to_int(text[11]),
            course=_course_from_kaisai(text[1]),
            distance=distance,
            surface=surface,
            track_condition=text[16],
            time=_time_to_seconds(text[18]),
            last_3f=_to_float(text[27]),
            popularity=_to_int(text[10]),
            weight_carried=_to_float(text[13]),
            jockey=text[12],
            race_class=_race_class(text[4]),
            win_odds=_to_float(text[9]),
        ))
        if len(runs) >= limit:
            break
    return runs


def _ped_name(td):
    a = td.select_one("a")
    txt = a.get_text(strip=True) if a else td.get_text(strip=True)
    # Strip trailing birth-year/coat-color text, e.g. 'レイデオロ2014 鹿毛'.
    return re.sub(r"\d{4}.*$", "", txt).strip()


def parse_pedigree(html: str):
    """Return (sire, dam, broodmare_sire) from a blood_table page."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.blood_table")
    if table is None:
        return None, None, None
    r16 = [td for td in table.select("td") if td.get("rowspan") == "16"]
    r8 = [td for td in table.select("td") if td.get("rowspan") == "8"]
    sire = _ped_name(r16[0]) if len(r16) > 0 else None
    dam = _ped_name(r16[1]) if len(r16) > 1 else None
    bms = _ped_name(r8[2]) if len(r8) > 2 else None
    return sire, dam, bms


def fetch_results(horse_id: str, limit: int = 5):
    url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
    return parse_results(fetch_html(url), limit=limit)


def fetch_pedigree(horse_id: str):
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    return parse_pedigree(fetch_html(url))
