"""Render the built .pptx to PNG previews so the layout can be inspected.

There is no PowerPoint or LibreOffice on this host, so this draws the slide
from the shapes python-pptx reports, using the same font metrics the build
script uses for its overflow checks. It approximates PowerPoint's layout
closely enough to catch collisions, clipping and bad line breaks.

Run:
    .venv/bin/python deck/build/preview.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN

ROOT = Path("/Users/prathmeshwalimbe/Downloads/JSW-PS1")
PPTX_PATH = ROOT / "deck" / "JSW_Surface_Defect_Detection.pptx"
OUT_DIR = ROOT / "deck" / "build" / "preview"
SCALE = 110  # pixels per inch
EMU = 914400.0

ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ARIAL_ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"

_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font_for(size_pt: float, bold: bool, italic: bool) -> ImageFont.FreeTypeFont:
    path = ARIAL_BOLD if bold else (ARIAL_ITALIC if italic else ARIAL)
    px = max(1, int(round(size_pt / 72.0 * SCALE)))
    key = (path, px)
    if key not in _CACHE:
        _CACHE[key] = ImageFont.truetype(path, px)
    return _CACHE[key]


def rgb(colour, default=(0x33, 0x41, 0x4F)):
    try:
        value = colour.rgb
        return (value[0], value[1], value[2])
    except Exception:
        return default


def draw_textframe(draw: ImageDraw.ImageDraw, shape) -> None:
    frame = shape.text_frame
    x0 = shape.left / EMU * SCALE
    y0 = shape.top / EMU * SCALE
    width = shape.width / EMU * SCALE
    cursor = y0
    for paragraph in frame.paragraphs:
        tokens: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]] = []
        max_px = 1
        for piece in paragraph.runs:
            size = piece.font.size.pt if piece.font.size else 11.0
            bold = bool(piece.font.bold)
            italic = bool(piece.font.italic)
            face = font_for(size, bold, italic)
            colour = rgb(piece.font.color)
            max_px = max(max_px, int(round(size / 72.0 * SCALE)))
            words = piece.text.split(" ")
            for i, word in enumerate(words):
                if word == "" and i != len(words) - 1:
                    continue
                tokens.append((word, face, colour))
        if not tokens:
            cursor += max_px * 1.22
            continue

        lines: list[list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]]] = [[]]
        line_w = 0.0
        space = tokens[0][1].getlength(" ")
        for word, face, colour in tokens:
            word_w = face.getlength(word)
            gap = space if lines[-1] else 0.0
            if line_w + gap + word_w > width and lines[-1]:
                lines.append([])
                line_w = 0.0
                gap = 0.0
            lines[-1].append((word, face, colour))
            line_w += gap + word_w
        for line in lines:
            total = sum(face.getlength(word) for word, face, _ in line)
            total += tokens[0][1].getlength(" ") * (len(line) - 1)
            if paragraph.alignment == PP_ALIGN.RIGHT:
                pen = x0 + width - total
            elif paragraph.alignment == PP_ALIGN.CENTER:
                pen = x0 + (width - total) / 2
            else:
                pen = x0
            for word, face, colour in line:
                draw.text((pen, cursor), word, font=face, fill=colour)
                pen += face.getlength(word) + tokens[0][1].getlength(" ")
            cursor += max_px * 1.22
        if paragraph.space_after is not None:
            cursor += paragraph.space_after.pt / 72.0 * SCALE


def render(slide, index: int) -> Path:
    width = int(13.333 * SCALE)
    height = int(7.5 * SCALE)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for shape in slide.shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            image = Image.open(shape.image.blob and shape._element and None) if False else None
            blob = shape.image.blob
            import io

            image = Image.open(io.BytesIO(blob)).convert("RGB")
            box = (
                int(round(shape.left / EMU * SCALE)),
                int(round(shape.top / EMU * SCALE)),
                int(round(shape.width / EMU * SCALE)),
                int(round(shape.height / EMU * SCALE)),
            )
            canvas.paste(image.resize((max(box[2], 1), max(box[3], 1)), Image.LANCZOS),
                         (box[0], box[1]))
        elif shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
            x0 = shape.left / EMU * SCALE
            y0 = shape.top / EMU * SCALE
            x1 = x0 + shape.width / EMU * SCALE
            y1 = y0 + shape.height / EMU * SCALE
            fill = rgb(shape.fill.fore_color, (0xF3, 0xF5, 0xF7))
            try:
                outline = rgb(shape.line.color, None)
            except Exception:
                outline = None
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline)
        if shape.has_text_frame and shape.text_frame.text.strip():
            draw_textframe(draw, shape)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"slide_{index}.png"
    canvas.save(path)
    return path


def main() -> None:
    prs = Presentation(str(PPTX_PATH))
    for i, slide in enumerate(prs.slides, start=1):
        print(render(slide, i))


if __name__ == "__main__":
    main()
