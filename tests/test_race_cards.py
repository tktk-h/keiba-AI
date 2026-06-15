from keiba.race_list import race_card_from_id, parse_race_cards


def test_race_card_from_id_derives_venue_and_number():
    c = race_card_from_id("202605030411")   # 05=東京, 末尾11=11R
    assert c["race_id"] == "202605030411"
    assert c["venue"] == "東京"
    assert c["number"] == 11
    assert c["name"] == ""                    # v1は名称未取得(空)


def test_parse_race_cards_from_list_html():
    html = ('<a href="/race/shutuba.html?race_id=202605030411">x</a>'
            '<a href="/race/shutuba.html?race_id=202609030401">y</a>')
    cards = parse_race_cards(html)
    ids = {c["race_id"] for c in cards}
    assert ids == {"202605030411", "202609030401"}
    venues = {c["venue"] for c in cards}
    assert venues == {"東京", "阪神"}           # 05=東京, 09=阪神
