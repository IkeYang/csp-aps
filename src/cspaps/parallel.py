"""Pools for scripts and background application jobs.

Background GUI jobs use threads so hot-reloadable module objects never cross
a pickle boundary. Main-thread scripts use processes:
* single-threaded process (CLI, plain scripts, tests) on Linux: ``fork`` -- the
  children do not re-import ``__main__``, so scripts without an
  ``if __name__ == "__main__":`` guard are safe;
* multi-threaded host (the Streamlit GUI runs scripts in worker threads, where
  forking can deadlock): ``forkserver`` (or ``spawn``), which re-imports the
  main module. Under Streamlit ``__main__`` is the GUI script itself; ``gui_app``
  marks it with a ``__main__`` module spec so that the workers leave it alone
  instead of re-running the whole GUI (Streamlit's "missing ScriptRunContext"
  warnings);
* other platforms: ``spawn``. Scripts that use parallel runs on Windows/macOS
  need the usual ``if __name__ == "__main__":`` guard;
* GPU backend switched on (``CSPAPS_GPU=1``): ``forkserver`` on Linux too, because CUDA cannot be used in
  a child forked after the parent initialised it. Scripts then need the ``__main__`` guard as well.
  Each worker gets its own device, round-robin over ``CSPAPS_GPU_DEVICES`` (``backend.worker_init``).
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
    from .backend import enabled as gpu_enabled

    methods = mp.get_all_start_methods()
    if gpu_enabled() and "forkserver" in methods:
        return "forkserver"  # every worker opens its own CUDA context
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
    ctx = mp.get_context(start_method())
    from .backend import enabled as gpu_enabled

    if gpu_enabled():  # every worker its own GPU, round-robin, and its share of that GPU's memory
        from .backend import worker_init

        return ctx.Pool(workers, initializer=worker_init, initargs=(ctx.Value("i", 0), workers))
    return ctx.Pool(workers)
