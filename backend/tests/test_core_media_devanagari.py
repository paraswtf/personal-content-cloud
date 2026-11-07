"""Page renders of office and plain-text documents draw Devanagari (Hindi, Marathi) with a font that
has the glyphs, in runs next to Latin text, and shrink a line that would overflow the page."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.media import documents as D
from tests.helpers import devanagari_font_available

HINDI = "राहुल कुमार शर्मा"
MIXED = "जन्म तिथि / DOB: 15/08/1988"


def test_text_runs_split_by_script():
    assert D.text_runs("Rahul") == [("Rahul", False)]
    assert D.text_runs(HINDI) == [(HINDI, True)]
    assert D.text_runs(MIXED) == [("जन्म तिथि ", True), ("/ DOB: 15/08/1988", False)]
    assert D.text_runs("Name: राहुल (Rahul)") == [("Name: ", False), ("राहुल ", True), ("(Rahul)", False)]
    assert D.text_runs("") == []


def _ink(img: Image.Image) -> int:
    return int((np.asarray(img.convert("L")) < 128).sum())


@pytest.mark.skipif(not devanagari_font_available(), reason="Noto Sans Devanagari is not downloaded")
def test_devanagari_lines_use_a_devanagari_font_and_have_ink():
    img = Image.new("RGB", (900, 120), "white")
    draw = ImageDraw.Draw(img)
    family = D._draw_line(draw, 10, 10, HINDI, D._font(40), 40, (0, 0, 0), 880)
    assert "Devanagari" in family or "FreeSerif" in family
    assert _ink(img) > 500
    # the same line through the Latin font alone draws missing-glyph boxes: a different picture
    plain = Image.new("RGB", (900, 120), "white")
    ImageDraw.Draw(plain).text((10, 10), HINDI, fill=(0, 0, 0), font=D._font(40))
    assert not np.array_equal(np.asarray(img), np.asarray(plain))


@pytest.mark.skipif(not devanagari_font_available(), reason="Noto Sans Devanagari is not downloaded")
def test_mixed_line_keeps_the_latin_part_readable_and_fits_the_width():
    img = Image.new("RGB", (600, 80), "white")
    draw = ImageDraw.Draw(img)
    D._draw_line(draw, 10, 10, MIXED, D._font(36), 36, (0, 0, 0), 300)  # narrower than the text: shrinks
    arr = np.asarray(img.convert("L")) < 128
    cols = np.where(arr.any(axis=0))[0]
    assert cols.size and cols.max() <= 10 + 300 + 2  # nothing drawn past the allowed width
    assert _ink(img) > 300


def test_latin_lines_are_drawn_as_before():
    img = Image.new("RGB", (400, 60), "white")
    draw = ImageDraw.Draw(img)
    assert D._draw_line(draw, 5, 5, "Plain English line 123", D._font(24), 24, (0, 0, 0), 390) == ""
    assert _ink(img) > 100


@pytest.mark.skipif(not devanagari_font_available(), reason="Noto Sans Devanagari is not downloaded")
def test_text_page_image_renders_marathi_text():
    text = "भाडे करार\n\nहा भाडे करार दिनांक 12 मार्च 2026 रोजी करण्यात आला आहे.\nMonthly rent Rs 25,000."
    page = D.text_page_image(text, 800)
    assert page.size[1] == 800 and _ink(page) > 300  # four short lines at the page's small font size
