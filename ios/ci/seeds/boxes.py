"""The viewer's box editing area: its own people and photos for ios/UITests/BoxesScreens.swift.

Five drawn photos taken from 30 to 26 April 2026 (older than the core's, so they sit after them in the library),
each for one test, so the tests never touch each other's boxes or the core's:

- 30 April: a face nobody named (the test names it "Boxes Alpha").
- 29 April: a face the system guessed as "Boxes Beta" (the test confirms the guess).
- 28 April: a face of "Boxes Alpha" the owner tagged (the test moves the box; the info panel test reads it).
- 27 April: nothing on it (the test draws a person box).
- 26 April: a dog nobody named (the pet editor test), next to the pet "Boxes Rex".

Everything is made up: the people are named after the area and the photos are drawn.
"""
from PIL import ImageDraw

FACE = (0.40, 0.30, 0.14, 0.20)
SKIN, HAIR = (214, 176, 146), (72, 52, 40)


def library(kit) -> dict:
    conn = kit.conn
    W, H = kit.W, kit.H
    alpha = kit.make_person(conn, "Boxes Alpha")
    beta = kit.make_person(conn, "Boxes Beta")
    rex = kit.make_person(conn, "Boxes Rex")
    with kit.tx(conn):
        conn.execute("UPDATE persons SET kind = 'pet', species = 'dog' WHERE id = ?", (rex,))

    def photo(n: int, day: int, faces: list):
        img = kit.scene(4100 + n, [(box, SKIN, HAIR) for box in faces])
        return img, day

    def store(n: int, day: int, img):
        item = kit.make_item(conn, data=kit.jpeg(img), width=W, height=H, filename=f"IMG_{4950 + n}.JPG",
                             taken_at=f"2026-04-{day:02d}T10:00:00.000Z", camera="CI Camera", category="personal")
        kit.previews(item, img)
        return item

    def face(item, img, box, person=None, **extra) -> int:
        face_id = kit.make_face(conn, int(item["id"]), person, box=box, size_px=int(box[2] * W), **extra)
        kit.crop(img, box).save(kit.storage.face_crop_path(face_id), "WEBP", quality=85)
        return face_id

    # 30 April: a face nobody named
    img, day = photo(0, 30, [FACE])
    unnamed = face(store(0, day, img), img, FACE, None, assigned_by="auto")

    # 29 April: a face guessed as Boxes Beta, waiting for an answer
    img, day = photo(1, 29, [FACE])
    # A face-model guess has no time window, so the server's startup check of old guesses keeps it.
    guessed = face(store(1, day, img), img, FACE, None, assigned_by="guess", guess_person_id=beta,
                   guess_score=0.82, guess_evidence='{"basis": "face", "face": 0.82}')

    # 28 April: Boxes Alpha's face, tagged by the owner
    img, day = photo(2, 28, [FACE])
    tagged = face(store(2, day, img), img, FACE, alpha, assigned_by="user")

    # 27 April: an empty landscape to draw on
    img, day = photo(3, 27, [])
    store(3, day, img)

    # 26 April: a dog nobody named
    img, day = photo(4, 26, [])
    d = ImageDraw.Draw(img)
    dog = (0.30, 0.45, 0.30, 0.30)
    x0, y0, x1, y1 = dog[0] * W, dog[1] * H, (dog[0] + dog[2]) * W, (dog[1] + dog[3]) * H
    d.ellipse((x0, y0 + 0.25 * (y1 - y0), x1, y1), fill=(150, 104, 60))  # body
    d.ellipse((x0 + 0.6 * (x1 - x0), y0, x1 + 0.1 * (x1 - x0), y0 + 0.55 * (y1 - y0)), fill=(160, 112, 66))  # head
    item = store(4, day, img)
    with kit.tx(conn):
        cur = conn.execute(
            "INSERT INTO animals(item_id, person_id, assigned_by, x, y, w, h, det_score, species, source, created_at) "
            "VALUES (?, NULL, 'auto', ?, ?, ?, ?, 0.9, 'dog', 'detector', ?)",
            (int(item["id"]), *dog, "2026-04-26T10:00:00.000Z"))
        animal = int(cur.lastrowid)
    path = kit.storage.animal_crop_path(animal)
    path.parent.mkdir(parents=True, exist_ok=True)
    kit.crop(img, dog).save(path, "WEBP", quality=85)

    return {"people": [alpha, beta, rex], "faces": [unnamed, guessed, tagged], "animals": [animal]}
