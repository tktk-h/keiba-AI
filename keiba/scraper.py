import requests
from bs4 import BeautifulSoup
from keiba.models import Race, Horse

HEADERS = {"User-Agent": "Mozilla/5.0 (keiba-research)"}


def fetch_html(url: str) -> str:
    # Network function - used manually only, not exercised by tests.
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    return resp.text


def parse_race(html: str, race_id: str) -> Race:
    soup = BeautifulSoup(html, "html.parser")
    # NOTE: selectors below match our test fixture
    # (tests/fixtures/race_sample.html). The real netkeiba page structure
    # differs and MUST be verified/adjusted against an actual
    # https://race.netkeiba.com/race/shutuba.html?race_id=... page before
    # live use. Cell order expected per row:
    # [number, name, sexage, weight_carried, jockey, win_odds, popularity]
    horses = []
    for row in soup.select("table.shutuba tr.horse"):
        cells = [c.get_text(strip=True) for c in row.select("td")]
        if len(cells) < 6:
            continue
        sexage = cells[2]
        horses.append(Horse(
            name=cells[1], sex=sexage[0], age=int(sexage[1:]),
            weight_carried=float(cells[3]), jockey=cells[4],
            post=int(cells[0]), number=int(cells[0]),
            win_odds=float(cells[5]) if cells[5] else None,
            popularity=int(cells[6]) if len(cells) > 6 and cells[6] else None,
            body_weight=None, body_weight_diff=None, running_style=None,
            sire=None, dam=None, broodmare_sire=None,
            training_time=None, training_course=None, training_eval=None,
            past_runs=[],
        ))
    return Race(race_id=race_id, name="", date="", course="", distance=0,
                surface="", turn="", track_condition="", weather="", horses=horses)


def scrape_race(race_id: str) -> Race:
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    return parse_race(fetch_html(url), race_id)
