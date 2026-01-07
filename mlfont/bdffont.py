# Copyright (c) Katsuya Iida.  All Rights Reserved.
# See LICENSE in the project root for license information.

r"""
Loading BDF font. See
https://adobe-type-tools.github.io/font-tech-notes/pdfs/5005.BDF_Spec.pdf
"""

from typing import TextIO
from dataclasses import dataclass, field
import numpy as np

__all__ = ["BDFFont", "BDFChar", "load"]

KEYWORDS = {
    # s - accepts a string arg
    # n - accepts number args
    # g - global
    # c - glyph
    "STARTFONT": "sgc",
    "COMMENT": "gc",
    "CONTENTVERSION": "1gc",
    "FONT": "sgc",
    "SIZE": "3gc",
    "FONTBOUNDINGBOX": "4gc",
    "METRICSSET": "1gc",
    "STARTPROPERTIES": "1gc",
    "ENDPROPERTIES": "gc",
    "CHARS": "1gc",
    "STARTCHAR": "sgc",
    "ENCODING": "1gc",
    "BBX": "4gc",
    "BITMAP": "gc",
    "ENDCHAR": "gc",
    "ENDFONT": "gc",
    "SWIDTH": "2gc",
    "DWIDTH": "2gc",
    "SWIDTH1": "2gc",
    "DWIDTH1": "2gc",
    "VVECTOR": "2gc",
}


class State:
    START = 0
    FONT = 1
    PROPERTIES = 2
    CHARS = 3
    CHAR = 4
    BITMAP = 5
    END = 6


@dataclass
class BDFChar:
    name: str = ""
    encoding: int | tuple[int, int] = 0
    bbx: tuple[int, int, int, int] | None = None
    swidth: tuple[int, int] | None = None
    dwidth: tuple[int, int] | None = None
    swidth1: tuple[int, int] | None = None
    dwidth1: tuple[int, int] | None = None
    bitmap: int | bytes = 0

    def __repr__(self) -> str:
        return f"BDFChar(name='{self.name}')"


@dataclass
class BDFFont:
    version: str = ""
    comment: str = ""
    contentversion: int | None = None
    font: str = ""
    size: tuple[int, int, int] = (0, 0, 0)
    fontboundingbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    metricsset: int | None = 0
    swidth: tuple[int, int] = (0, 0)
    dwidth: tuple[int, int] = (0, 0)
    swidth1: tuple[int, int] = (0, 0)
    dwidth1: tuple[int, int] = (0, 0)
    vvector: tuple[int, int] | None = None
    properties: dict[str, str | int] = field(default_factory=dict)
    chars: list[BDFChar] = field(default_factory=list)
    bitmap: bytes = b""

    def find(
        self, *, name: str | None = None, encoding: int | tuple[int, int] | None = None
    ) -> BDFChar | None:
        if name is not None:
            for char in self.chars:
                if char.name == name:
                    return char
            return None
        elif encoding is not None:
            for char in self.chars:
                if char.encoding == encoding:
                    return char
            return None
        else:
            raise ValueError("Either one of name or encoding must be given")

    def tobytes(self, char: BDFChar) -> bytes:
        bitmap_or_index = char.bitmap
        if isinstance(bitmap_or_index, bytes):
            return bitmap_or_index
        assert isinstance(bitmap_or_index, int)
        if char.bbx is None:
            raise ValueError("char.bbx is not available.")
        bitmap_start = bitmap_or_index
        bitmap_width = char.bbx[0]
        bitmap_height = char.bbx[1]
        bitmap_rowbytes = (bitmap_width + 7) // 8
        bitmap_end = bitmap_start + bitmap_rowbytes * bitmap_height
        return self.bitmap[bitmap_start:bitmap_end]

    def numpy(self, char: BDFChar) -> np.ndarray:
        if char.bbx is None:
            raise ValueError("char.bbx is not available.")
        bitmap_width = char.bbx[0]
        bitmap_height = char.bbx[1]
        bitmap_rowbytes = (bitmap_width + 7) // 8
        bitmap = self.tobytes(char)
        x = np.frombuffer(bitmap, dtype=np.uint8)
        mask = 1 << (7 - np.arange(8))
        x = ((x[:, None] & mask[None, :]) != 0).astype(np.uint8)
        x = x.reshape([bitmap_height, bitmap_rowbytes * 8])
        x = x[:, :bitmap_width] * 255
        return x

    def __repr__(self) -> str:
        return f"BDFFont(font='{self.font}')"


def parse_tuple(args: str, n: int | None) -> tuple[int, ...] | int:
    parts = tuple(int(x) for x in args.strip().split())
    assert n is None or len(parts) == n
    assert len(parts) > 0
    return parts if len(parts) > 1 else parts[0]


class BDFFontReader:
    def __init__(self):
        self.state = State.START

    def read(self, fp: TextIO) -> BDFFont:  # noqa PLR0915
        version = None
        font_data = {}
        properties_data = {}
        comments = []
        chars_data = []
        char_data = {}
        bitmaps_data = []
        bitmap_data = []
        bitmap_index = 0
        bitmap_height = -1
        bitmap_rowbytes = -1
        for line in fp:
            if self.state not in [State.PROPERTIES, State.BITMAP]:
                kw, _, args = line.strip().partition(" ")
                kw = kw.upper()
                kwtype = KEYWORDS[kw]
                if kw == "COMMENT":
                    # Argument of COMMENT is specially handled
                    pass
                elif kw == "ENCODING":
                    args = parse_tuple(args, None)
                    assert isinstance(args, int) or len(args) == 2
                elif "s" in kwtype:
                    args = args.strip()
                    if args.startswith('"'):
                        assert args.endswith('"')
                        args = args[1:-1]
                elif "1" in kwtype:
                    args = parse_tuple(args, 1)
                elif "2" in kwtype:
                    args = parse_tuple(args, 2)
                elif "3" in kwtype:
                    args = parse_tuple(args, 3)
                elif "4" in kwtype:
                    args = parse_tuple(args, 4)
                else:
                    assert not args.strip(), line
                    args = None
                if self.state == State.START:
                    version = args
                elif self.state == State.FONT:
                    if kw == "COMMENT":
                        comments.append(args)
                    else:
                        assert kw not in font_data, line
                        font_data[kw.lower()] = args
                elif self.state == State.PROPERTIES:
                    assert kw not in properties_data, line
                    properties_data[kw.lower()] = args
                elif self.state == State.CHARS:
                    if kw == "STARTCHAR":
                        char_data = {}
                        char_data["name"] = args
                    elif kw == "COMMENT":
                        # 'font-dec-misc-1.0.3/deccurs.bdf' has
                        # comments before STARTCHAR.
                        pass
                    else:
                        assert kw == "ENDFONT", line
                elif self.state == State.CHAR:
                    assert kw not in char_data, line
                    char_data[kw.lower()] = args
                else:
                    assert args is None, line
                if kw == "STARTFONT":
                    assert self.state == State.START
                    self.state = State.FONT
                elif kw == "STARTPROPERTIES":
                    assert self.state == State.FONT
                    self.state = State.PROPERTIES
                elif kw == "CHARS":
                    assert self.state == State.FONT
                    self.state = State.CHARS
                elif kw == "STARTCHAR":
                    assert self.state == State.CHARS
                    self.state = State.CHAR
                elif kw == "BITMAP":
                    assert self.state == State.CHAR
                    self.state = State.BITMAP
                    bbx = char_data["bbx"]
                    bitmap_rowbytes = (bbx[0] + 7) // 8
                    bitmap_height = bbx[1]
                elif kw == "ENDFONT":
                    assert self.state == State.CHARS
                    self.state = State.END
            elif self.state == State.PROPERTIES:
                kw, _, args = line.strip().partition(" ")
                if kw == "ENDPROPERTIES":
                    assert not args.strip()
                    self.state = State.FONT
                elif kw == "COMMENT":
                    # 'font-adobe-100dpi-1.0.3/ncenBI14.bdf' has
                    # comments in properties section
                    pass
                else:
                    if args.startswith('"'):
                        assert args.endswith('"')
                        args = args[1:-1]
                    else:
                        args = [int(x) for x in args.split()]
                        assert args
                        if len(args) == 1:
                            args = args[0]
                    properties_data[kw.lower()] = args
            elif self.state == State.BITMAP:
                kw = line.strip()
                if kw == "ENDCHAR":
                    char_data["bitmap"] = bitmap_index
                    char = BDFChar(**char_data)
                    chars_data.append(char)
                    assert len(bitmap_data) == bitmap_height
                    bitmap = b"".join(bitmap_data)
                    bitmap_data = []
                    bitmaps_data.append(bitmap)
                    bitmap_index += len(bitmap)
                    self.state = State.CHARS
                else:
                    # assert len(kw) == bitmap_rowbytes * 2, f"Invalid bitmap data {kw}"
                    if len(kw) > bitmap_rowbytes * 2:
                        # 'font-bitstream-100dpi-1.0.3/charR18.bdf' has
                        # extra data. Truncate it.
                        kw = kw[: bitmap_rowbytes * 2]
                    else:
                        # 'font-schumacher-misc-1.1.2/clB9x15.bdf' has
                        # need more data. Append to it.
                        kw = kw + "0" * (bitmap_rowbytes * 2 - len(kw))
                    x = [int(kw[i : i + 2], 16) for i in range(0, len(kw), 2)]
                    x = bytes(x)
                    bitmap_data.append(x)
            else:
                assert False
        assert self.state == State.END
        if "startproperties":
            assert len(properties_data) == font_data["startproperties"], properties_data
            del font_data["startproperties"]
            font_data["properties"] = properties_data
        else:
            font_data["startproperties"] = {}
        assert len(chars_data) == font_data["chars"], "Incorrect number of chars"
        del font_data["chars"]
        font_data["version"] = version
        font_data["chars"] = chars_data
        font_data["bitmap"] = b"".join(bitmaps_data)
        font_data["comment"] = "\n".join(comments)

        font = BDFFont(**font_data)
        return font


def _dump_property_value(v: str | int | tuple[int, ...] | None) -> str:
    if isinstance(v, tuple):
        return " ".join(str(x) for x in v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return '"' + v + '"'
    assert False


class BDFFontWriter:
    def write(self, font: BDFFont, fp: TextIO) -> None:
        size = _dump_property_value(font.size)
        fontboundingbox = _dump_property_value(font.fontboundingbox)
        fp.write(f"STARTFONT {font.version}\n")
        for t in font.comment.split("\n"):
            fp.write(f"COMMENT {t}\n")
        fp.write(f"FONT {font.font}\n")
        fp.write(f"SIZE {size}\n")
        fp.write(f"FONTBOUNDINGBOX {fontboundingbox}\n")
        fp.write(f"STARTPROPERTIES {len(font.properties)}\n")
        for k, v in font.properties.items():
            k = k.upper()
            v = _dump_property_value(v)
            fp.write(f"{k} {v}\n")
        fp.write("ENDPROPERTIES\n")
        fp.write(f"CHARS {len(font.chars)}\n")
        for char in font.chars:
            swidth = _dump_property_value(char.swidth)
            dwidth = _dump_property_value(char.dwidth)
            bbx = _dump_property_value(char.bbx)
            fp.write(f"STARTCHAR {char.name}\n")
            fp.write(f"ENCODING {char.encoding}\n")
            fp.write(f"SWIDTH {swidth}\n")
            fp.write(f"DWIDTH {dwidth}\n")
            fp.write(f"BBX {bbx}\n")
            fp.write("BITMAP\n")
            if char.bbx is None:
                raise ValueError("char.bbx is not available.")
            bitmap = font.tobytes(char)
            bitmap_width = char.bbx[0]
            bitmap_rowbytes = (bitmap_width + 7) // 8
            for i in range(0, len(bitmap), bitmap_rowbytes):
                for b in bitmap[i : i + bitmap_rowbytes]:
                    fp.write("%02x" % b)
                fp.write("\n")
            fp.write("ENDCHAR\n")
        fp.write("ENDFONT\n")


def load(file: str) -> BDFFont:
    reader = BDFFontReader()
    with open(file) as fp:
        font = reader.read(fp)
    return font


def save(font: BDFFont, file: str) -> None:
    writer = BDFFontWriter()
    with open(file, "w") as fp:
        writer.write(font, fp)


def get_extent(glyphs: list[tuple[BDFFont, BDFChar]]) -> tuple[int, int, int, int]:
    ax, ay, bx, by, cx, cy = 0, 0, 0, 0, 0, 0
    for _, fch in glyphs:
        assert fch.bbx is not None
        assert fch.dwidth is not None
        dx = cx + fch.bbx[2]
        dy = cy - fch.bbx[1] - fch.bbx[3]
        ax = min(ax, dx)
        ay = min(ay, dy)
        dx += fch.bbx[0]
        dy += fch.bbx[1]
        bx = max(bx, dx)
        by = max(by, dy)
        cx += fch.dwidth[0]
        cy += fch.dwidth[1]
    return ax, ay, bx, by


def get_glyphs(font: BDFFont, text: str) -> list[tuple[BDFFont, BDFChar]]:
    res = []
    for ch in text:
        fch = font.find(encoding=ord(ch))
        if fch:
            res.append((font, fch))
    return res


def draw_glyphs(glyphs: list[tuple[BDFFont, BDFChar]]) -> np.ndarray:
    ax, ay, bx, by = get_extent(glyphs)
    X = np.zeros([by - ay, bx - ax], dtype=np.uint8)
    cx, cy = -ax, -ay
    for font, fch in glyphs:
        assert fch.bbx is not None
        assert fch.dwidth is not None
        dx = cx + fch.bbx[2]
        dy = cy - fch.bbx[1] - fch.bbx[3]
        g = font.numpy(fch)
        h, w = g.shape
        X[dy : dy + h, dx : dx + w] |= g
        cx += fch.dwidth[0]
        cy += fch.dwidth[1]
    return X


def draw_text(font: BDFFont, text: str) -> np.ndarray:
    glyphs = get_glyphs(font, text)
    return draw_glyphs(glyphs)
