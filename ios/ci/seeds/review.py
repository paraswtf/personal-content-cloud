"""The review area's seed (ios/UITests/ReviewScreens.swift, ios/UnitTests/ReviewDecodingTests.swift).

Two named people of this area, "Review Person" and "Review Other". Review Person has two faces the owner named
and a guess of every kind the Review screen tells apart: a tattoo (traits alone), trained recognition, clothes,
a person box by clothes, the position, and a face the app had grouped with one the owner named. Review Other has
one clothes guess. Besides them two mis-clicks, two forwards from messaging apps and a group of three near
duplicates. Everything is drawn and dated in January and February 2026, older than the core's photos. The screen
tests change only these rows.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

PERSON = "Review Person"
OTHER = "Review Other"


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def library(kit) -> dict:
    from PIL import ImageDraw, ImageEnhance, ImageFilter

    conn = kit.conn
    W, H = kit.W, kit.H
    start = datetime(2026, 2, 20, 11, 0, tzinfo=timezone.utc)
    person = kit.make_person(conn, PERSON)
    other = kit.make_person(conn, OTHER)
    skin, hair = (214, 168, 128), (52, 36, 24)
    face_box = (0.42, 0.30, 0.13, 0.19)

    def photo(name: str, when: datetime, *, seed: int, faces=(), img=None, **extra):
        if img is None:
            img = kit.scene(seed, [(b, skin, hair) for b in faces])
        item = kit.make_item(conn, data=kit.jpeg(img), width=W, height=H, filename=f"{name}.JPG",
                             taken_at=_iso(when), camera="CI Camera", category="personal", **extra)
        kit.previews(item, img)
        return item, img

    def face(item, img, who=None, **extra) -> int:
        face_id = kit.make_face(conn, int(item["id"]), who, box=face_box, size_px=int(face_box[2] * W),
                                assigned_by="user" if who else "auto", **extra)
        kit.crop(img, face_box).save(kit.storage.face_crop_path(face_id), "WEBP", quality=85)
        return face_id

    def guessed(who: int, score: float, evidence: dict) -> dict:
        return {"guess_person_id": who, "guess_score": score, "guess_evidence": json.dumps(evidence)}

    # the two faces the owner named: the guesses below refer to them
    ref1, img1 = photo("REVIEW_01", start, seed=7001, faces=[face_box])
    ref1_face = face(ref1, img1, person)
    ref2, img2 = photo("REVIEW_02", start + timedelta(minutes=30), seed=7002, faces=[face_box])
    ref2_face = face(ref2, img2, person)

    guesses = [
        ("REVIEW_03", 0.95, {"basis": "trait",
                             "traits": [{"kind": "tattoo", "part": "forearm_l", "ref_item_uid": ref1["uid"],
                                         "ref_taken_at": _iso(start), "days_apart": 20, "ref_before": True,
                                         "support": 1, "strong": True}],
                             "against": [{"kind": "nails", "part": "hand_r", "why": "bare"}]}),
        ("REVIEW_04", 0.92, {"basis": "face", "face": 0.61, "also": "clothes"}),
        ("REVIEW_05", 0.85, {"minutes_apart": 25, "scene": 0.86, "clothes": 0.91, "face": 0.42,
                             "ref_face_id": ref1_face, "ref_item_uid": ref1["uid"]}),
        ("REVIEW_06", 0.77, {"basis": "place", "place": {"ref_face_id": ref2_face, "ref_item_uid": ref2["uid"],
                                                         "seconds_apart": 8, "ref_before": True, "others": 1}}),
        ("REVIEW_07", 0.62, {"basis": "group", "face": 0.62}),
    ]
    for n, (name, score, evidence) in enumerate(guesses, start=3):
        item, img = photo(name, start - timedelta(days=n, minutes=7 * n), seed=7000 + n, faces=[face_box])
        face(item, img, None, **guessed(person, score, evidence))

    # a person box guessed by clothes
    body_box = (0.30, 0.20, 0.38, 0.72)
    item, img = photo("REVIEW_08", start - timedelta(days=8), seed=7008, faces=[face_box])
    body_id = kit.make_body(conn, int(item["id"]), None, box=body_box,
                            **guessed(person, 0.80, {"minutes_apart": 12, "scene": 0.9, "clothes": 0.88}))
    x0, y0 = int(body_box[0] * W), int(body_box[1] * H)
    x1, y1 = int((body_box[0] + body_box[2]) * W), int((body_box[1] + body_box[3]) * H)
    body_crop = kit.storage.body_crop_path(body_id)
    body_crop.parent.mkdir(parents=True, exist_ok=True)
    img.crop((x0, y0, x1, y1)).save(body_crop, "WEBP", quality=85)

    # Review Other's guess
    item, img = photo("REVIEW_09", start - timedelta(days=9), seed=7009, faces=[face_box])
    face(item, img, None, **guessed(other, 0.70, {"minutes_apart": 40, "scene": 0.8, "clothes": 0.83}))

    # mis-clicks: one blurred and dark, one taken in a pocket
    blurred = kit.scene(7101, []).filter(ImageFilter.GaussianBlur(9))
    photo("REVIEW_MISCLICK_1", datetime(2026, 2, 3, 9, 12, tzinfo=timezone.utc), seed=7101, img=blurred,
          misclick=1, misclick_reasons=json.dumps({"blurry": 0.21, "dark": 0.12}))
    pocket = ImageEnhance.Brightness(kit.scene(7102, [])).enhance(0.18)
    photo("REVIEW_MISCLICK_2", datetime(2026, 2, 4, 21, 40, tzinfo=timezone.utc), seed=7102, img=pocket,
          misclick=1, misclick_reasons=json.dumps({"pocket": 1}))

    # forwards from messaging apps: a greeting with text on it, twice
    def greeting(seed: int, text: str):
        img = kit.scene(seed, [])
        d = ImageDraw.Draw(img)
        d.rectangle((60, 60, W - 60, 200), fill=(250, 240, 210))
        d.text((90, 110), text, fill=(120, 40, 40))
        return img

    photo("REVIEW_FORWARD_1", datetime(2026, 1, 28, 8, 0, tzinfo=timezone.utc), seed=7201,
          img=greeting(7201, "Good morning"), forward=1, received_via="whatsapp", forward_checked=1,
          forward_reasons=json.dumps({"meme_p": 0.81, "text": True, "small": True, "stripped": True,
                                      "name": "whatsapp"}))
    photo("REVIEW_FORWARD_2", datetime(2026, 1, 27, 8, 0, tzinfo=timezone.utc), seed=7202,
          img=greeting(7202, "Have a good week"), forward=1, received_via="telegram", forward_checked=1,
          forward_reasons=json.dumps({"meme_p": 0.74, "text": True, "jpeg_quality": 75, "name": "telegram"}))

    # three near-identical copies of one picture: the first is the best
    dup = kit.scene(7301, [])
    best, _ = photo("REVIEW_DUP_1", datetime(2026, 1, 20, 16, 0, tzinfo=timezone.utc), seed=7301, img=dup)
    copies = []
    for k, scale in ((2, 2), (3, 3)):
        smaller = dup.resize((W // scale, H // scale)).resize((W, H))
        copy, _ = photo(f"REVIEW_DUP_{k}", datetime(2026, 1, 20, 16, k, tzinfo=timezone.utc), seed=7300 + k,
                        img=smaller)
        copies.append(copy)
    with kit.tx(conn):
        conn.execute("UPDATE items SET dup_group_id = ?, dup_role = 'best' WHERE id = ?", (best["id"], best["id"]))
        for copy in copies:
            conn.execute("UPDATE items SET dup_group_id = ?, dup_role = 'copy' WHERE id = ?", (best["id"], copy["id"]))

    return {"people": [PERSON, OTHER], "guesses": len(guesses) + 2, "misclicks": 2, "forwards": 2,
            "duplicates": 1 + len(copies)}
