"""The albums area's seed (ios/UITests/AlbumsScreens.swift): its own photos, albums and share links, so its tests
change only what is theirs and leave the core's "Weekend trip", "Hills" and their share as they are.

library: four drawn photos from February 2026, ALBUMS_1.JPG to ALBUMS_4.JPG (older than the core's photos).
api: the album "Albums test: Beach" with the first three, the empty album "Albums test: Shelf" (where the add test
puts them), a share of Beach titled "Albums test link" (the edit test changes it) and a share of the fourth photo
titled "Albums test revoke" (the revoke test takes it down).
"""
from __future__ import annotations

from urllib.parse import quote

PREFIX = "ALBUMS_"


def library(kit) -> dict:
    made = []
    for k in range(4):
        img = kit.scene(5100 + k, [])
        item = kit.make_item(kit.conn, data=kit.jpeg(img), width=kit.W, height=kit.H, filename=f"{PREFIX}{k + 1}.JPG",
                             taken_at=f"2026-02-{10 + k:02d}T09:30:00.000Z", camera="CI Camera", category="misc")
        kit.previews(item, img)
        made.append(item["uid"])
    return {"photos": len(made)}


def _my_items(client) -> list[str]:
    """The uids of this seed's photos, by file name, in name order."""
    found: dict[str, str] = {}
    cursor = None
    for _ in range(20):
        path = "/api/items?limit=200" + (f"&cursor={quote(cursor, safe='')}" if cursor else "")
        page = client.call("GET", path)
        for it in page.get("items", []):
            if str(it.get("filename", "")).startswith(PREFIX):
                found[it["filename"]] = it["uid"]
        cursor = page.get("next_cursor")
        if not cursor:
            break
    return [found[name] for name in sorted(found)]


def api(kit) -> dict:
    c = kit.client
    uids = _my_items(c)
    if len(uids) < 4:
        raise SystemExit(f"the albums seed found {len(uids)} of its 4 photos")
    beach = c.call("POST", "/api/albums", {"name": "Albums test: Beach", "description": "Made by the albums seed",
                                           "item_uids": uids[:3]})
    shelf = c.call("POST", "/api/albums", {"name": "Albums test: Shelf"})
    link = c.call("POST", "/api/shares", {"kind": "album", "album_id": beach["id"], "title": "Albums test link"})
    gone = c.call("POST", "/api/shares", {"kind": "items", "item_uids": uids[3:4], "title": "Albums test revoke",
                                          "expires_days": 7})
    return {"albums": [beach["id"], shelf["id"]], "shares": [link["id"], gone["id"]]}
