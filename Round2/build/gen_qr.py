"""Write qr.png for the Round 2 cover, using the project's own QR encoder.

No new dependency: deck/build/qrgen.py encodes and self-decodes the payload,
so a bad render fails here rather than on a judge's phone.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deck" / "build"))

import qrgen  # noqa: E402
from PIL import Image  # noqa: E402

URL = "https://surface-vision.github.io"
OUT = Path(__file__).resolve().parent / "qr.png"
QUIET = 4      # modules of quiet zone, per spec
SCALE = 26     # px per module (~960 px, crisp in print)

qr = qrgen.encode(URL, level="Q")
assert qrgen.self_decode(qr) == URL, "QR did not round-trip"

n = qr.size + 2 * QUIET
img = Image.new("L", (n, n), 255)
px = img.load()
for y, row in enumerate(qr.modules):
    for x, cell in enumerate(row):
        if cell:
            px[x + QUIET, y + QUIET] = 0
img.resize((n * SCALE, n * SCALE), Image.NEAREST).save(OUT)
print(f"wrote {OUT} ({qr.version}, {qr.size}x{qr.size} modules, level Q, self-decode OK)")
