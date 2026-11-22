"""Reading the CSV tables users bring (layouts, DNI points, map libraries, scenario lists, weather).

Spreadsheets saved on Windows are often not UTF-8: Excel's "CSV" uses the ANSI code page (GBK / GB18030 on
Chinese systems, cp1252 on Western ones), "CSV UTF-8" adds a byte-order mark, and headers come with spaces
or capitals ("X ", "DNI"). ``read_user_csv`` accepts all of these and gives the same frame on every platform.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "gb18030", "latin-1")  # latin-1 decodes any byte string: last resort


def decode_bytes(data: bytes) -> tuple[str, str]:
    """(text, encoding) of a user file: UTF-8 (with or without BOM), else GB18030, else Latin-1."""
    for enc in ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodes every byte string")  # pragma: no cover


def read_user_csv(src, lower_header: bool = True, **kwargs) -> pd.DataFrame:
    """``pd.read_csv`` for a path or an uploaded file object, independent of the file's encoding.

    Header names are stripped (and lower-cased unless ``lower_header`` is False) so that "X , Y ,DNI"
    from a spreadsheet matches the documented columns x, y, dni."""
    if hasattr(src, "getvalue"):
        data = src.getvalue()
    elif hasattr(src, "read"):
        pos = src.tell() if hasattr(src, "tell") else None
        data = src.read()
        if pos is not None:
            src.seek(pos)
    else:
        data = Path(src).read_bytes()
    text = data if isinstance(data, str) else decode_bytes(data)[0]
    df = pd.read_csv(io.StringIO(text), **kwargs)
    names = [str(c).strip() for c in df.columns]
    df.columns = [c.lower() for c in names] if lower_header else names
    return df


_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name) -> str:
    """A file or folder name that is valid on Windows as well as Linux.

    Replaces the characters Windows forbids (< > : " / \\ | ? * and control characters), drops trailing
    dots and spaces, and prefixes reserved device names (CON, NUL, COM1, ...). Names that are already
    valid (ASCII or Chinese) are returned unchanged."""
    import re

    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).rstrip(" .") or "_"
    if s.split(".")[0].upper() in _RESERVED:
        s = "_" + s
    return s


def export_name(layout_name) -> str:
    """Folder of a layout's exported values: ``+`` (swap rounds) spelled out, then made a safe file name."""
    return safe_filename(str(layout_name).replace("+", "_plus_"))
