import re
import time
import requests
from bs4 import BeautifulSoup
from keiba.models import Race, Horse
from keiba.horse_page import fetch_results

HEADERS = {"User-Agent": "Mozilla/5.0 (keiba-research)"}


def fetch_html(url: str) -> str:
    # Network function - used manually only, not exercised by tests.
    # race.netkeiba.com (出馬表/オッズ) は UTF-8。EUC-JP を強制すると日本語が
    # 文字化けし、芝/ダや距離のパースも失敗する(db.netkeiba.com の result/horse
    # ページは EUC-JP のままで、そちらは各モジュールで個別に指定している)。
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = "UTF-8"
    return resp.text


def _to_float(text):
    """Parse odds/weight strings; netkeiba shows '---.-' before a race."""
    text = (text or "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(text):
    text = (text or "").strip()
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def _parse_race_meta(soup):
    """Extract race name and course info from the shutuba page header.

    RaceData01 looks like: '15:40発走 / 芝2400m (左 C) / 天候:晴 / 馬場:良'
    """
    name_el = soup.select_one(".RaceName")
    name = name_el.get_text(strip=True) if name_el else ""

    surface, distance, turn, weather, condition = "", 0, "", "", ""
    data_el = soup.select_one(".RaceData01")
    if data_el:
        text = data_el.get_text(" ", strip=True).replace("\xa0", " ")
        m = re.search(r"(芝|ダ|障)\s*(\d+)m", text)
        if m:
            surface = m.group(1)
            distance = int(m.group(2))
        m = re.search(r"\((左|右|直)", text)
        if m:
            turn = m.group(1)
        m = re.search(r"天候\s*[:：]\s*(\S+)", text)
        if m:
            weather = m.group(1)
        m = re.search(r"馬場\s*[:：]\s*(\S+)", text)
        if m:
            condition = m.group(1)
    return name, surface, distance, turn, weather, condition


def _parse_body_weight(text):
    """'524(-8)' -> (524, -8). Returns (None, None) if unavailable."""
    m = re.match(r"(\d+)\(([+-]?\d+)\)", (text or "").strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _parse_sex_age(text):
    """'牡3' -> ('牡', 3)."""
    text = (text or "").strip()
    if not text:
        return "", 0
    sex = text[0]
    age = _to_int(text[1:]) or 0
    return sex, age


def parse_race(html: str, race_id: str) -> Race:
    """Parse a netkeiba shutuba (出馬表) page into a Race.

    Verified against the real page structure of
    https://race.netkeiba.com/race/shutuba.html?race_id=...
    (table.Shutuba_Table > tr.HorseList). Cell index per row:
      0=枠 1=馬番 3=馬名 4=性齢 5=斤量 6=騎手 7=厩舎 8=馬体重 9=単勝オッズ 10=人気
    Note: 単勝オッズ/人気 are '---.-'/'**' until shortly before the race;
    those parse to None. Past-成績・血統・調教 are not on this page and stay
    empty here (collected separately from each horse's db.netkeiba page).
    """
    soup = BeautifulSoup(html, "html.parser")
    name, surface, distance, turn, weather, condition = _parse_race_meta(soup)

    horses = []
    for row in soup.select("table.Shutuba_Table tr.HorseList"):
        tds = row.select("td")
        if len(tds) < 11:
            continue
        name_a = tds[3].select_one("a")
        horse_name = name_a.get_text(strip=True) if name_a else tds[3].get_text(strip=True)
        horse_id = None
        if name_a and name_a.get("href"):
            m = re.search(r"/horse/(\w+)", name_a["href"])
            if m:
                horse_id = m.group(1)
        sex, age = _parse_sex_age(tds[4].get_text(strip=True))
        body_weight, body_weight_diff = _parse_body_weight(tds[8].get_text(strip=True))
        horses.append(Horse(
            name=horse_name,
            sex=sex,
            age=age,
            weight_carried=_to_float(tds[5].get_text(strip=True)),
            jockey=tds[6].get_text(strip=True),
            post=_to_int(tds[0].get_text(strip=True)),
            number=_to_int(tds[1].get_text(strip=True)),
            win_odds=_to_float(tds[9].get_text(strip=True)),
            popularity=_to_int(tds[10].get_text(strip=True)),
            body_weight=body_weight,
            body_weight_diff=body_weight_diff,
            running_style=None,
            sire=None,
            dam=None,
            broodmare_sire=None,
            training_time=None,
            training_course=None,
            training_eval=None,
            horse_id=horse_id,
            past_runs=[],
        ))
    return Race(race_id=race_id, name=name, date="", course="", distance=distance,
                surface=surface, turn=turn, track_condition=condition,
                weather=weather, horses=horses)


def enrich_horses(race: Race, limit: int = 5, delay: float = 0.5) -> Race:
    """Fill each horse's past_runs from db.netkeiba horse pages.

    Network-bound (manual use only, not exercised by tests). A polite `delay`
    between horse pages avoids hammering the site. Pedigree is intentionally
    NOT fetched: the model uses no pedigree features, so skipping it halves the
    per-horse requests (≈2x faster load).
    """
    for horse in race.horses:
        if not horse.horse_id:
            continue
        try:
            horse.past_runs = fetch_results(horse.horse_id, limit=limit)
        except Exception as exc:  # noqa: BLE001 - keep going on a single failure
            print(f"  ! {horse.name} の詳細取得に失敗: {exc}")
        time.sleep(delay)
    return race


def scrape_race(race_id: str, enrich: bool = False) -> Race:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    race = parse_race(fetch_html(url), race_id)
    if enrich:
        enrich_horses(race)
    return race
