"""Bundled resources: the Aksai demo case ships inside the package so that an installed wheel works."""

from __future__ import annotations

from pathlib import Path

DATA = Path(__file__).resolve().with_name("data")


def demo_case_path() -> Path:
    """The Aksai demo case: the repository copy in a source checkout, else the packaged copy."""
    repo = Path(__file__).resolve().parents[2] / "cases" / "aksai_demo.yaml"
    return repo if repo.exists() else DATA / "aksai_demo.yaml"
