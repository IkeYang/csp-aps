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
* Workers take the parent's ``CSPAPS_*`` settings as they are when the pool is created (a fork server keeps
  the environment it was started with), and each caps its BLAS / OpenMP threads at its share of the cores
  (``worker_threads``): W processes with machine-sized thread pools would oversubscribe the CPU.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import os
import sys
import threading


def cpu_count() -> int:
    """CPUs this process may use: its CPU affinity (taskset, cluster schedulers) and, in a container, the cgroup
    CPU quota -- not the machine's core count, which can be far larger than what a job is given."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # Windows / macOS
        n = os.cpu_count() or 1
    try:  # cgroup v2 quota, e.g. "400000 100000" = 4 CPUs
        quota, period = open("/sys/fs/cgroup/cpu.max", encoding="ascii").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def n_workers(setting: int) -> int:
    """-1 = all usable CPUs (``cpu_count``), otherwise at least 1."""
    return cpu_count() if setting == -1 else max(1, int(setting))


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
    # the parent's CSPAPS_* settings as they are now: a fork server keeps the environment it was started with
    env = {k: v for k, v in os.environ.items() if k.startswith("CSPAPS_") and k not in _PER_WORKER}
    return ctx.Pool(workers, initializer=_worker_init,
                    initargs=(env, ctx.Value("i", 0), workers, worker_threads(workers)))


_PER_WORKER = ("CSPAPS_GPU_DEVICE", "CSPAPS_GPU_SHARE")  # assigned to each worker by backend.worker_init
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def worker_threads(workers: int) -> int:
    """BLAS / OpenMP threads per pool worker: ``CSPAPS_WORKER_THREADS``, else its share of the cores.

    Every process otherwise starts a BLAS pool as large as the machine, so W workers run W x cores threads; on
    4 cores 4 workers became 2.6x faster with one thread each (same single-process speed)."""
    raw = os.environ.get("CSPAPS_WORKER_THREADS", "").strip()
    if raw:
        return max(1, int(raw))
    return max(1, cpu_count() // max(1, int(workers)))


def limit_threads(n: int) -> None:
    """Cap the BLAS / OpenMP pools of this process at ``n`` threads (loaded libraries and later ones)."""
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except Exception:  # noqa: BLE001 - threadpoolctl missing: the variables cover libraries loaded later
        pass


def _worker_init(env: dict, counter, n_workers: int, threads: int = 1) -> None:
    """Pool-worker initializer: the parent's CSPAPS_* settings (stale ones removed), this worker's share of the
    cores for BLAS / OpenMP, and with the GPU backend on the next device round-robin and its share of it."""
    for key in [k for k in os.environ if k.startswith("CSPAPS_") and k not in env]:
        del os.environ[key]
    os.environ.update(env)
    limit_threads(threads)
    from .backend import enabled, worker_init

    if enabled():
        worker_init(counter, n_workers)
