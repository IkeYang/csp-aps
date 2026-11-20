"""Process pools that work from scripts, the CLI and the GUI.

Start method:
* single-threaded process (CLI, plain scripts, tests) on Linux: ``fork`` -- the
  children do not re-import ``__main__``, so scripts without an
  ``if __name__ == "__main__":`` guard are safe;
* multi-threaded host (the Streamlit GUI runs scripts in worker threads, where
  forking can deadlock): ``forkserver`` (or ``spawn``), which re-imports the
  main module -- Streamlit's entry point is guarded;
* other platforms: ``spawn``. Scripts that use parallel runs on Windows/macOS
  need the usual ``if __name__ == "__main__":`` guard.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading


def n_workers(setting: int) -> int:
    """-1 = all CPUs, otherwise at least 1."""
    return (os.cpu_count() or 1) if setting == -1 else max(1, int(setting))


def start_method() -> str:
    methods = mp.get_all_start_methods()
    if "fork" in methods and threading.active_count() == 1:
        return "fork"
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file and os.path.exists(main_file):
        return "forkserver" if "forkserver" in methods else "spawn"
    return "fork" if "fork" in methods else "spawn"


def pool(workers: int):
    return mp.get_context(start_method()).Pool(workers)
