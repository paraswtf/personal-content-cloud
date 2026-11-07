"""Regenerates the synthetic document fixtures in this folder (run with the backend venv python).

boarding_pass.png  a boarding pass with printed text and a PDF417 barcode holding an IATA BCBP string
event_ticket.png   an event ticket with a QR code holding a JSON payload
pan_card.png       a PAN-card-like render (no barcode) for the OCR test
All names and numbers are made up; check digits are valid where the format has them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zxingcpp
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BCBP = "M1SHARMA/RAHUL KUMAR  EABC123 DELBOMAI 0865 123Y012A0045 100"
TICKET = '{"booking": "TKT8X2K91", "seat": "K-23", "gate": "5"}'


def font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def code_image(text: str, fmt, scale: int) -> Image.Image:
    bc = zxingcpp.create_barcode(text, fmt)
    return Image.fromarray(np.array(zxingcpp.write_barcode_to_image(bc, scale=scale))).convert("RGB")


def page(lines: list[str], size=(900, 560), fs=26) -> tuple[Image.Image, int]:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 24
    for ln in lines:
        d.text((32, y), ln, fill="black", font=font(fs))
        y += int(fs * 1.5)
    return img, y


def main() -> None:
    img, y = page(["BOARDING PASS", "Air India", "Name: SHARMA/RAHUL KUMAR MR", "From DEL To BOM",
                   "Flight AI 865   Seat 12A   Gate 23"], size=(900, 520))
    img.paste(code_image(BCBP, zxingcpp.BarcodeFormat.PDF417, 2), (32, y + 10))
    img.save(HERE / "boarding_pass.png", optimize=True)

    img, y = page(["Coldplay Music of the Spheres", "Venue: DY Patil Stadium", "Date: 19 Jan 2025"],
                  size=(700, 520))
    img.paste(code_image(TICKET, zxingcpp.BarcodeFormat.QRCode, 5), (32, y + 10))
    img.save(HERE / "event_ticket.png", optimize=True)

    img, _y = page(["INCOME TAX DEPARTMENT     GOVT. OF INDIA", "Permanent Account Number Card", "ABCPS1234K",
                    "Name", "RAHUL SHARMA", "Father's Name", "SURESH SHARMA", "Date of Birth", "15/08/1988"],
                   size=(1000, 640), fs=30)
    img.save(HERE / "pan_card.png", optimize=True)


if __name__ == "__main__":
    main()
