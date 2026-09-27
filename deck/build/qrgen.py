"""A self-contained QR encoder, written here so the deck needs no new dependency.

Nothing in the project's requirements provides QR generation (no qrcode, no segno,
no pyqrcode -- checked), and adding a package for one image on one slide is not
worth it. This implements the subset of ISO/IEC 18004 the deck actually needs:
byte mode, versions 1-6, error-correction levels L/M/Q/H.

It is checked three ways:
  1. self_decode() reverses every step -- format bits, mask, module placement,
     block de-interleaving -- and asserts the payload comes back byte-identical.
  2. Reed-Solomon syndromes of every received block are recomputed and asserted
     zero, which is what a scanner's decoder checks first.
  3. deck/build/verify_qr.swift decodes the rendered PNG with Apple's Vision and
     Core Image readers -- the same detectors behind the iOS and macOS camera.

Run:
    .venv/bin/python deck/build/qrgen.py          # self-test, prints the matrix
"""

from __future__ import annotations

from dataclasses import dataclass

# --- GF(256), primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 = 0x11D ---------
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _rs_generator(degree: int) -> list[int]:
    """Product of (x - alpha^i) for i in 0..degree-1, as coefficients high->low."""
    poly = [1]
    for i in range(degree):
        nxt = [0] * (len(poly) + 1)
        root = _EXP[i]
        for j, coeff in enumerate(poly):
            nxt[j] ^= coeff
            nxt[j + 1] ^= _mul(coeff, root)
        poly = nxt
    return poly


def _rs_remainder(data: list[int], degree: int) -> list[int]:
    gen = _rs_generator(degree)
    rem = [0] * degree
    for byte in data:
        factor = byte ^ rem[0]
        rem = rem[1:] + [0]
        for i, coeff in enumerate(gen[1:]):
            rem[i] ^= _mul(coeff, factor)
    return rem


def _rs_syndromes(block: list[int], degree: int) -> list[int]:
    """Evaluate the received block at alpha^0..alpha^(degree-1)."""
    out = []
    for i in range(degree):
        root = _EXP[i]
        acc = 0
        for coeff in block:
            acc = _mul(acc, root) ^ coeff
        out.append(acc)
    return out


# --- block structure: version -> level -> (ec_per_block, [(blocks, data_cw)]) -
# Totals are asserted against TOTAL_CODEWORDS at import time.
TOTAL_CODEWORDS = {1: 26, 2: 44, 3: 70, 4: 100, 5: 134, 6: 172}

BLOCKS: dict[int, dict[str, tuple[int, list[tuple[int, int]]]]] = {
    1: {"L": (7, [(1, 19)]), "M": (10, [(1, 16)]), "Q": (13, [(1, 13)]), "H": (17, [(1, 9)])},
    2: {"L": (10, [(1, 34)]), "M": (16, [(1, 28)]), "Q": (22, [(1, 22)]), "H": (28, [(1, 16)])},
    3: {"L": (15, [(1, 55)]), "M": (26, [(1, 44)]), "Q": (18, [(2, 17)]), "H": (22, [(2, 13)])},
    4: {"L": (20, [(1, 80)]), "M": (18, [(2, 32)]), "Q": (26, [(2, 24)]), "H": (16, [(4, 9)])},
    5: {"L": (26, [(1, 108)]), "M": (24, [(2, 43)]),
        "Q": (18, [(2, 15), (2, 16)]), "H": (22, [(2, 11), (2, 12)])},
    6: {"L": (18, [(2, 68)]), "M": (16, [(4, 27)]), "Q": (24, [(4, 19)]), "H": (28, [(4, 15)])},
}

for _v, _levels in BLOCKS.items():
    for _lvl, (_ec, _groups) in _levels.items():
        _nblocks = sum(n for n, _ in _groups)
        _total = sum(n * k for n, k in _groups) + _nblocks * _ec
        assert _total == TOTAL_CODEWORDS[_v], (_v, _lvl, _total)

FORMAT_BITS = {"L": 1, "M": 0, "Q": 3, "H": 2}
# Remainder bits appended after the codeword stream (ISO/IEC 18004 table 1).
REMAINDER_BITS = {1: 0, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7}
# Single alignment-pattern centre for versions 2-6.
ALIGN_CENTRE = {2: 18, 3: 22, 4: 26, 5: 30, 6: 34}


def data_capacity_bytes(version: int, level: str) -> int:
    """Byte-mode payload capacity: data codewords minus the 12-bit header."""
    ec, groups = BLOCKS[version][level]
    data_cw = sum(n * k for n, k in groups)
    return (data_cw * 8 - 4 - 8) // 8


# --- encoding ---------------------------------------------------------------
def _bitstream(payload: bytes, version: int, level: str) -> list[int]:
    ec, groups = BLOCKS[version][level]
    data_cw = sum(n * k for n, k in groups)
    bits: list[int] = []

    def put(value: int, width: int) -> None:
        for shift in range(width - 1, -1, -1):
            bits.append((value >> shift) & 1)

    put(0b0100, 4)          # byte mode
    put(len(payload), 8)    # count indicator, 8 bits for versions 1-9
    for byte in payload:
        put(byte, 8)

    capacity = data_cw * 8
    if len(bits) > capacity:
        raise ValueError("payload does not fit")
    put(0, min(4, capacity - len(bits)))            # terminator
    while len(bits) % 8:
        bits.append(0)
    pad = [0xEC, 0x11]
    i = 0
    while len(bits) < capacity:
        put(pad[i % 2], 8)
        i += 1
    return bits


def _codewords(payload: bytes, version: int, level: str) -> list[int]:
    """Data + EC codewords, interleaved as the standard requires."""
    bits = _bitstream(payload, version, level)
    stream = [int("".join(str(b) for b in bits[i:i + 8]), 2) for i in range(0, len(bits), 8)]

    ec, groups = BLOCKS[version][level]
    data_blocks: list[list[int]] = []
    ec_blocks: list[list[int]] = []
    pos = 0
    for count, size in groups:
        for _ in range(count):
            block = stream[pos:pos + size]
            pos += size
            data_blocks.append(block)
            ec_blocks.append(_rs_remainder(block, ec))
    assert pos == len(stream)

    out: list[int] = []
    for i in range(max(len(b) for b in data_blocks)):
        for block in data_blocks:
            if i < len(block):
                out.append(block[i])
    for i in range(ec):
        for block in ec_blocks:
            out.append(block[i])
    assert len(out) == TOTAL_CODEWORDS[version]
    return out


# --- matrix -----------------------------------------------------------------
@dataclass
class QR:
    version: int
    level: str
    mask: int
    modules: list[list[bool]]      # True = dark
    function: list[list[bool]]     # True = function module (not data)
    payload: bytes

    @property
    def size(self) -> int:
        return len(self.modules)


def _blank(size: int) -> list[list[bool]]:
    return [[False] * size for _ in range(size)]


def _draw_function_patterns(mod, fun, version: int) -> None:
    size = len(mod)

    def set_fn(col: int, row: int, dark: bool) -> None:
        mod[row][col] = dark
        fun[row][col] = True

    # finder patterns with separators
    for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < size and 0 <= y < size:
                    dist = max(abs(dx), abs(dy))
                    set_fn(x, y, dist != 2 and dist != 4)

    # timing patterns
    for i in range(size):
        if not fun[6][i]:
            set_fn(i, 6, i % 2 == 0)
        if not fun[i][6]:
            set_fn(6, i, i % 2 == 0)

    # single alignment pattern (versions 2-6)
    if version >= 2:
        centre = ALIGN_CENTRE[version]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                set_fn(centre + dx, centre + dy, max(abs(dx), abs(dy)) != 1)

    # format-information areas, reserved now and written later
    for i in range(9):
        if not fun[8][i]:
            set_fn(i, 8, False)
        if not fun[i][8]:
            set_fn(8, i, False)
    for i in range(8):
        set_fn(size - 1 - i, 8, False)
        set_fn(8, size - 1 - i, False)
    set_fn(8, size - 8, True)      # dark module


def _format_bits(level: str, mask: int) -> int:
    data = (FORMAT_BITS[level] << 3) | mask
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    return ((data << 10) | rem) ^ 0x5412


def _draw_format(mod, fun, level: str, mask: int) -> None:
    size = len(mod)
    bits = _format_bits(level, mask)

    def bit(i: int) -> bool:
        return bool((bits >> i) & 1)

    def set_fn(col: int, row: int, dark: bool) -> None:
        mod[row][col] = dark
        fun[row][col] = True

    for i in range(6):
        set_fn(8, i, bit(i))
    set_fn(8, 7, bit(6))
    set_fn(8, 8, bit(7))
    set_fn(7, 8, bit(8))
    for i in range(9, 15):
        set_fn(14 - i, 8, bit(i))
    for i in range(8):
        set_fn(size - 1 - i, 8, bit(i))
    for i in range(8, 15):
        set_fn(8, size - 15 + i, bit(i))
    set_fn(8, size - 8, True)


def _data_positions(size: int, fun) -> list[tuple[int, int]]:
    """Module coordinates in the order the codeword bits are written."""
    order: list[tuple[int, int]] = []
    for right in range(size - 1, 0, -2):
        # column 6 is the vertical timing pattern, so every pair to its left
        # shifts one column further left
        col_right = right - 1 if right <= 6 else right
        for vert in range(size):
            for j in range(2):
                col = col_right - j
                upward = ((col_right + 1) & 2) == 0
                row = (size - 1 - vert) if upward else vert
                if not fun[row][col]:
                    order.append((col, row))
    return order


def _mask_bit(mask: int, col: int, row: int) -> bool:
    if mask == 0:
        return (col + row) % 2 == 0
    if mask == 1:
        return row % 2 == 0
    if mask == 2:
        return col % 3 == 0
    if mask == 3:
        return (col + row) % 3 == 0
    if mask == 4:
        return (row // 2 + col // 3) % 2 == 0
    if mask == 5:
        return (col * row) % 2 + (col * row) % 3 == 0
    if mask == 6:
        return ((col * row) % 2 + (col * row) % 3) % 2 == 0
    if mask == 7:
        return ((col + row) % 2 + (col * row) % 3) % 2 == 0
    raise ValueError(mask)


def _penalty(mod) -> int:
    size = len(mod)
    score = 0

    # rule 1: runs of five or more, plus rule 3 on the same scan
    finder = [True, False, True, True, True, False, True]
    for is_row in (True, False):
        for a in range(size):
            line = [mod[a][b] if is_row else mod[b][a] for b in range(size)]
            run_colour, run_len = line[0], 1
            for value in line[1:]:
                if value == run_colour:
                    run_len += 1
                else:
                    if run_len >= 5:
                        score += 3 + (run_len - 5)
                    run_colour, run_len = value, 1
            if run_len >= 5:
                score += 3 + (run_len - 5)
            padded = [False] * 4 + line + [False] * 4
            for i in range(len(padded) - 10):
                window = padded[i:i + 11]
                if window[:7] == finder and not any(window[7:]):
                    score += 40
                if window[4:] == finder and not any(window[:4]):
                    score += 40

    # rule 2: 2x2 blocks of one colour
    for row in range(size - 1):
        for col in range(size - 1):
            value = mod[row][col]
            if (mod[row][col + 1] == value and mod[row + 1][col] == value
                    and mod[row + 1][col + 1] == value):
                score += 3

    # rule 4: deviation from an even split of dark and light
    dark = sum(sum(1 for v in row if v) for row in mod)
    total = size * size
    score += 10 * (abs(dark * 20 - total * 10) // total)
    return score


def encode(text: str, level: str = "Q", version: int | None = None) -> QR:
    payload = text.encode("iso-8859-1")
    if version is None:
        for candidate in sorted(BLOCKS):
            if len(payload) <= data_capacity_bytes(candidate, level):
                version = candidate
                break
        if version is None:
            raise ValueError(f"{len(payload)} bytes will not fit versions 1-6 at level {level}")
    if len(payload) > data_capacity_bytes(version, level):
        raise ValueError("payload does not fit that version")

    size = 17 + 4 * version
    codewords = _codewords(payload, version, level)
    bits: list[int] = []
    for word in codewords:
        for shift in range(7, -1, -1):
            bits.append((word >> shift) & 1)
    bits += [0] * REMAINDER_BITS[version]

    base_mod, base_fun = _blank(size), _blank(size)
    _draw_function_patterns(base_mod, base_fun, version)
    order = _data_positions(size, base_fun)
    assert len(order) == len(bits), (len(order), len(bits))
    for (col, row), bit in zip(order, bits):
        base_mod[row][col] = bool(bit)

    best = None
    for mask in range(8):
        mod = [row[:] for row in base_mod]
        fun = [row[:] for row in base_fun]
        for row in range(size):
            for col in range(size):
                if not fun[row][col] and _mask_bit(mask, col, row):
                    mod[row][col] = not mod[row][col]
        _draw_format(mod, fun, level, mask)
        score = _penalty(mod)
        if best is None or score < best[0]:
            best = (score, mask, mod, fun)

    _, mask, mod, fun = best
    return QR(version=version, level=level, mask=mask, modules=mod,
              function=fun, payload=payload)


# --- verification -----------------------------------------------------------
def self_decode(qr: QR) -> str:
    """Reverse every step of encode() and return the recovered payload.

    Also asserts the Reed-Solomon syndromes of every block are zero, which is
    the first thing a real decoder checks.
    """
    size = qr.size
    mod = qr.modules

    # recover the format information from copy 1, exhaustively, and cross-check copy 2
    read1 = 0
    coords1 = [(8, i) for i in range(6)] + [(8, 7), (8, 8), (7, 8)] + \
              [(14 - i, 8) for i in range(9, 15)]
    for i, (col, row) in enumerate(coords1):
        if mod[row][col]:
            read1 |= 1 << i
    read2 = 0
    coords2 = [(size - 1 - i, 8) for i in range(8)] + \
              [(8, size - 15 + i) for i in range(8, 15)]
    for i, (col, row) in enumerate(coords2):
        if mod[row][col]:
            read2 |= 1 << i
    if read1 != read2:
        raise AssertionError("the two format-information copies disagree")
    found = [(lvl, m) for lvl in FORMAT_BITS for m in range(8)
             if _format_bits(lvl, m) == read1]
    if len(found) != 1:
        raise AssertionError(f"format information does not decode: {found}")
    level, mask = found[0]
    if (level, mask) != (qr.level, qr.mask):
        raise AssertionError("format information does not match the encoder")

    # rebuild the function map from the version alone, unmask, read the bits back
    ref_mod, fun = _blank(size), _blank(size)
    _draw_function_patterns(ref_mod, fun, qr.version)
    _draw_format(ref_mod, fun, level, mask)
    order = _data_positions(size, fun)
    bits: list[int] = []
    for col, row in order:
        value = mod[row][col]
        if _mask_bit(mask, col, row):
            value = not value
        bits.append(1 if value else 0)
    bits = bits[:len(bits) - REMAINDER_BITS[qr.version]]
    stream = [int("".join(str(b) for b in bits[i:i + 8]), 2) for i in range(0, len(bits), 8)]
    if len(stream) != TOTAL_CODEWORDS[qr.version]:
        raise AssertionError("wrong codeword count read back")

    # de-interleave into blocks, then check each block's syndromes
    ec, groups = BLOCKS[qr.version][qr.level]
    sizes = [k for count, k in groups for _ in range(count)]
    nblocks = len(sizes)
    data_blocks: list[list[int]] = [[] for _ in range(nblocks)]
    pos = 0
    for i in range(max(sizes)):
        for b in range(nblocks):
            if i < sizes[b]:
                data_blocks[b].append(stream[pos])
                pos += 1
    ec_blocks: list[list[int]] = [[] for _ in range(nblocks)]
    for i in range(ec):
        for b in range(nblocks):
            ec_blocks[b].append(stream[pos])
            pos += 1
    assert pos == len(stream)
    for b in range(nblocks):
        syndromes = _rs_syndromes(data_blocks[b] + ec_blocks[b], ec)
        if any(syndromes):
            raise AssertionError(f"block {b} has non-zero Reed-Solomon syndromes")

    flat: list[int] = []
    for block in data_blocks:
        flat += block
    payload_bits = []
    for word in flat:
        for shift in range(7, -1, -1):
            payload_bits.append((word >> shift) & 1)
    mode = int("".join(str(b) for b in payload_bits[0:4]), 2)
    if mode != 0b0100:
        raise AssertionError(f"mode indicator is {mode:04b}, expected 0100")
    count = int("".join(str(b) for b in payload_bits[4:12]), 2)
    out = bytearray()
    for i in range(count):
        chunk = payload_bits[12 + i * 8: 20 + i * 8]
        out.append(int("".join(str(b) for b in chunk), 2))
    return bytes(out).decode("iso-8859-1")


# --- rendering --------------------------------------------------------------
def render_png(qr: QR, path, module_px: int = 24, quiet: int = 4,
               dark: tuple[int, int, int] = (0x14, 0x1C, 0x24),
               light: tuple[int, int, int] = (0xFF, 0xFF, 0xFF)):
    """Write the code as a PNG with an integer number of pixels per module."""
    from PIL import Image

    size = qr.size
    side = (size + 2 * quiet) * module_px
    image = Image.new("RGB", (side, side), light)
    pixels = image.load()
    for row in range(size):
        for col in range(size):
            if not qr.modules[row][col]:
                continue
            x0 = (col + quiet) * module_px
            y0 = (row + quiet) * module_px
            for y in range(y0, y0 + module_px):
                for x in range(x0, x0 + module_px):
                    pixels[x, y] = dark
    image.save(path)
    return path


def decode_png(data: bytes | str, expected_version: int | None = None) -> str:
    """Read a rendered QR back out of PNG bytes (or a path) and decode it.

    Used by deck/build/verify_deck.py against the image actually embedded in the
    .pptx, so the check covers the copy a judge's phone will point at rather
    than the copy the encoder happened to produce.
    """
    import io

    from PIL import Image

    source = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    image = Image.open(source).convert("L")
    width, height = image.size
    if width != height:
        raise AssertionError(f"QR image is not square: {width}x{height}")
    pixels = image.load()

    # find the quiet zone: the first dark pixel in from each edge
    def first_dark_row() -> int:
        for y in range(height):
            if any(pixels[x, y] < 128 for x in range(width)):
                return y
        raise AssertionError("no dark pixels in the QR image")

    top = first_dark_row()
    left = next(x for x in range(width) if any(pixels[x, y] < 128 for y in range(height)))
    bottom = height - 1 - next(
        d for d in range(height) if any(pixels[x, height - 1 - d] < 128 for x in range(width)))
    right = width - 1 - next(
        d for d in range(width) if any(pixels[width - 1 - d, y] < 128 for y in range(height)))
    if abs(top - left) > 1:
        raise AssertionError(f"quiet zone is not symmetric: top {top}, left {left}")

    # the finder pattern is 7 modules wide, so the first dark run gives the pitch
    run_len = 0
    y_probe = top + 1
    while pixels[left + run_len, y_probe] < 128:
        run_len += 1
    module = run_len / 7.0
    size = int(round((right - left + 1) / module))
    if size not in {17 + 4 * v for v in BLOCKS}:
        raise AssertionError(f"implausible module count {size}")

    modules = [[False] * size for _ in range(size)]
    for row in range(size):
        for col in range(size):
            x = int(left + (col + 0.5) * module)
            y = int(top + (row + 0.5) * module)
            modules[row][col] = pixels[x, y] < 128

    version = (size - 17) // 4
    if expected_version is not None and version != expected_version:
        raise AssertionError(f"expected version {expected_version}, read {version}")

    # the level and mask live in the format information; read them, then let
    # self_decode() re-derive everything else and check the syndromes
    read = 0
    coords = [(8, i) for i in range(6)] + [(8, 7), (8, 8), (7, 8)] + \
             [(14 - i, 8) for i in range(9, 15)]
    for i, (col, row) in enumerate(coords):
        if modules[row][col]:
            read |= 1 << i
    found = [(lvl, m) for lvl in FORMAT_BITS for m in range(8)
             if _format_bits(lvl, m) == read]
    if len(found) != 1:
        raise AssertionError("format information in the image does not decode")
    level, mask = found[0]
    return self_decode(QR(version=version, level=level, mask=mask, modules=modules,
                          function=_blank(size), payload=b""))


def to_text(qr: QR, dark: str = "##", light: str = "  ") -> str:
    return "\n".join("".join(dark if cell else light for cell in row) for row in qr.modules)


# --- self-test --------------------------------------------------------------
def main() -> None:
    url = "https://surface-vision.github.io"
    for level in ("L", "M", "Q", "H"):
        qr = encode(url, level=level)
        back = self_decode(qr)
        assert back == url, (level, back)
        print(f"level {level}: version {qr.version}, {qr.size}x{qr.size} modules, "
              f"mask {qr.mask}, payload {len(qr.payload)} B, capacity "
              f"{data_capacity_bytes(qr.version, level)} B, self-decode OK")

    # a couple of unrelated payloads, to show the encoder is not URL-specific
    for probe in ("A", "0123456789", "x" * 40, "https://example.com/a/b?c=d&e=f"):
        for level in ("M", "Q"):
            qr = encode(probe, level=level)
            assert self_decode(qr) == probe, (probe, level)
    print("extra payload round-trips OK (4 payloads x 2 levels)")

    qr = encode(url, level="Q")
    print()
    print(to_text(qr))


if __name__ == "__main__":
    main()
