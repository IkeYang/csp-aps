"""Windows: spreadsheets saved in the ANSI code page, CRLF line endings, consoles that cannot print Unicode."""

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

from cspaps.field import read_layout
from cspaps.tables import decode_bytes, read_user_csv


def test_user_tables_in_gbk_bom_and_padded_headers(tmp_path):
    rows = "id,X , Y ,备注\n1,-20.5,92.25,北区\n2,20,108,北区\n"
    variants = {"gbk.csv": rows.encode("gbk"), "bom.csv": rows.encode("utf-8-sig"), "plain.csv": rows.encode("utf-8"),
                "crlf.csv": rows.replace("\n", "\r\n").encode("gbk")}
    for name, data in variants.items():
        p = tmp_path / name
        p.write_bytes(data)
        df = read_user_csv(p)
        assert list(df.columns) == ["id", "x", "y", "备注"], name
        assert df["备注"].tolist() == ["北区", "北区"], name
        xyz, ids = read_layout(p, 5.0)
        assert xyz.tolist() == [[-20.5, 92.25, 5.0], [20.0, 108.0, 5.0]] and ids.tolist() == [1, 2], name
    assert decode_bytes(b"\xff\xfe x")[1] == "latin-1"  # neither UTF-8 nor GB18030: never fails


def test_uploaded_file_objects_keep_their_position():
    class Upload(io.BytesIO):
        name = "dni.csv"

    up = Upload("x,y,DNI\n0,0,800\n".encode("gbk"))
    df = read_user_csv(up, dtype=str)
    assert list(df.columns) == ["x", "y", "dni"] and up.tell() == 0


def test_scenario_table_in_gbk_with_chinese_notes(small_case, tmp_path):
    from cspaps.scenarios import load_table

    p = tmp_path / "scen.csv"
    p.write_bytes(("scenario_id,elevation_deg,azimuth_deg,hours,dni_w_m2,map_id,说明\n"
                   "S01,40,180,100,800,uniform,正午\nS02,20,120,50,600,uniform,上午\n").encode("gbk"))
    sset = load_table(small_case, p)
    assert len(sset) == 2 and sset.ids == ["S01", "S02"]


def test_scenario_fingerprint_does_not_depend_on_the_line_ending(small_case, monkeypatch):
    from cspaps.scenarios import build_scenarios

    small_case.scenarios.n_scenarios = 3
    sset = build_scenarios(small_case)
    unix = sset.fingerprint()
    monkeypatch.setattr(os, "linesep", "\r\n")  # what pandas uses for to_csv on Windows
    assert sset.fingerprint() == unix


def test_cli_output_never_fails_on_a_narrow_console(monkeypatch):
    from cspaps.cli import _safe_console

    raw = io.BytesIO()
    narrow = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", narrow)
    with pytest.raises(UnicodeEncodeError):
        print("工况 λ = 0.3", file=narrow)
        narrow.flush()
    _safe_console()
    print("工况 λ = 0.3，峰值 475 kW/m²", file=sys.stdout)
    sys.stdout.flush()
    assert b"kW/m" in raw.getvalue()


def test_pools_in_background_threads_are_threads():
    import threading
    from multiprocessing.pool import ThreadPool

    from cspaps.parallel import pool

    seen = {}

    def job():
        p = pool(2)
        seen["thread_pool"] = isinstance(p, ThreadPool)
        seen["result"] = p.map(np.square, [1, 2, 3])
        p.close()
        p.join()

    t = threading.Thread(target=job)
    t.start()
    t.join()
    assert seen == {"thread_pool": True, "result": [1, 4, 9]}


def test_folder_names_are_valid_on_windows(small_case, tmp_path):
    from cspaps.tables import export_name, safe_filename
    from cspaps.workflows import out_dir

    assert safe_filename("布局 A") == "布局 A" and safe_filename("search_1") == "search_1"
    assert safe_filename('a:b/c\\d?"e*<f>|') == "a_b_c_d__e__f__"
    assert safe_filename("CON") == "_CON" and safe_filename("nul.csv") == "_nul.csv"
    assert safe_filename("trailing. ") == "trailing"
    assert export_name("search_1+swap2") == "search_1_plus_swap2"
    small_case.name = "site: test?"
    out = out_dir(tmp_path, small_case, "run")
    assert out.parent.name == "site_ test_" and out.exists()
