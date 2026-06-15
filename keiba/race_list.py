"""Discover race IDs automatically from netkeiba.

- Calendar page lists which dates have racing:
    https://race.netkeiba.com/top/calendar.html?year=YYYY&month=M
- Per-date list page links every race that day:
    https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD

`collect_race_ids` ties them together (network, manual use only).
"""
import re
import time
from datetime import date as _date
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (keiba-research)"}


def fetch_html(url: str) -> str:
    # Network function - manual use only, not exercised by tests.
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = "EUC-JP"
    return resp.text


def parse_kaisai_dates(html: str):
    """Return sorted unique 'YYYYMMDD' racing dates found on a calendar page."""
    return sorted(set(re.findall(r"kaisai_date=(\d{8})", html)))


def parse_race_ids(html: str):
    """Return sorted unique 12-digit race IDs found on a race-list page."""
    return sorted(set(re.findall(r"race_id=(\d{12})", html)))


def fetch_kaisai_dates(year: int, month: int):
    url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
    return parse_kaisai_dates(fetch_html(url))


def fetch_race_ids(kaisai_date: str):
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={kaisai_date}"
    return parse_race_ids(fetch_html(url))


def collect_race_ids(year: int, month: int, delay: float = 1.0):
    """All race IDs for every racing date in a given year/month (network)."""
    race_ids = []
    for d in fetch_kaisai_dates(year, month):
        try:
            race_ids.extend(fetch_race_ids(d))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {d} のレース一覧取得に失敗: {exc}")
        time.sleep(delay)
    return sorted(set(race_ids))


VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
          "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def race_card_from_id(race_id: str) -> dict:
    """race_id から会場・R番号を導出。race_id=YYYY+場(2)+開催(2)+日(2)+R(2)。"""
    return {"race_id": race_id,
            "venue": VENUES.get(race_id[4:6], race_id[4:6]),
            "number": int(race_id[10:12]),
            "name": ""}


def parse_race_cards(html: str) -> list:
    """一覧ページHTMLから {race_id, venue, number, name} のリストを返す。"""
    return [race_card_from_id(rid) for rid in parse_race_ids(html)]


def fetch_race_cards(kaisai_date: str) -> list:
    """指定日(YYYYMMDD)の開催レースカード一覧(ネットワーク)。"""
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={kaisai_date}"
    return parse_race_cards(fetch_html(url))


def today_cards() -> list:
    """今日(ローカル日付)の開催レースカード一覧(ネットワーク)。"""
    return fetch_race_cards(_date.today().strftime("%Y%m%d"))
