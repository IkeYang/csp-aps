"""Pools for scripts and background application jobs.

Background GUI jobs use threads so hot-reloadable module objects never cross
a pickle boundary. Main-thread scripts use processes:
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
from multiprocessing.pool import ThreadPool
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


def _picklable(obj) -> bool:
    import pickle

    try:
        pickle.dumps(obj)
        return True
    except Exception:  # noqa: BLE001 - PicklingError, AttributeError, TypeError ...
        return False


def pool(workers: int, probe=None):
    """A pool of ``workers``.

    Main thread: processes. Background thread (GUI jobs): threads by default, so that objects of
    hot-reloaded modules never cross a pickle boundary; with ``probe`` (the function and a typical
    argument of the work) processes are used when the probe pickles, so a long GUI job can use every
    core. ``CSPAPS_POOL=thread`` / ``process`` forces one kind."""
    forced = os.environ.get("CSPAPS_POOL", "").strip().lower()
    if forced == "thread":
        return ThreadPool(workers)
    if threading.current_thread() is not threading.main_thread():
        if forced != "process" and (probe is None or not _picklable(probe)):
            return ThreadPool(workers)
    return mp.get_context(start_method()).Pool(workers)
