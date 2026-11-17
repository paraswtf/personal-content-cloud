"""The library area's seed (ios/UITests/LibraryScreens.swift, ios/UnitTests/LibraryDecodingTests.swift).

Eight drawn photos named library lantern NN.jpg, taken in January and February 2026: the first three are food, the
fourth a favourite, the sixth a possible mis-click (left out of the library until "Include mis-clicks"), and the
first, fifth and seventh are tagged "Library sample". The tests filter by that tag, so what other areas add to the
library does not change what they find. Nothing else in the library is touched.
"""
from __future__ import annotations

from datetime import datetime, timezone

PHOTOS = 8
PREFIX = "library lantern "
TAG = "Library sample"
TAGGED = (1, 5, 7)


def library(kit) -> dict:
    made = []
    for n in range(1, PHOTOS + 1):
        month = 1 if n <= 4 else 2
        taken = datetime(2026, month, 1 + 6 * ((n - 1) % 4), 9 + n, 15, tzinfo=timezone.utc)
        img = kit.scene(7000 + n, [])
        item = kit.make_item(kit.conn, data=kit.jpeg(img), width=kit.W, height=kit.H,
                             filename=f"{PREFIX}{n:02d}.jpg", taken_at=taken.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                             camera="CI Camera", category="food" if n <= 3 else "misc",
                             favorite=1 if n == 4 else 0, misclick=1 if n == 6 else 0)
        kit.previews(item, img)
        made.append(item["uid"])
    return {"photos": len(made)}


def api(kit) -> dict:
    by_name = {}
    for key in ("2026-01", "2026-02"):
        page = kit.client.call("GET", f"/api/timeline/bucket?key={key}&misclicks=include&limit=500")
        for card in page["items"]:
            if card["filename"].startswith(PREFIX):
                by_name[card["filename"]] = card["uid"]
    if len(by_name) != PHOTOS:
        raise SystemExit(f"library seed: found {len(by_name)} of its {PHOTOS} photos")
    for n in TAGGED:
        kit.client.call("POST", f"/api/items/{by_name[f'{PREFIX}{n:02d}.jpg']}/user-tags", {"name": TAG})
    return {"photos": len(by_name), "tagged": len(TAGGED), "tag": TAG}
