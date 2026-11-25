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


def reserved_cpus(total: int | None = None) -> int:
    """CPUs left to the system and the user's other programs (the desktop, the browser running the GUI):
    ``CSPAPS_RESERVE_CPUS``, else about 15 % of the usable CPUs (at least one when there are two or more)."""
    raw = os.environ.get("CSPAPS_RESERVE_CPUS", "").strip()
    total = cpu_count() if total is None else int(total)
    if raw:
        return min(max(0, int(raw)), max(0, total - 1))
    return 0 if total <= 1 else max(1, round(0.15 * total))


def available_cpus() -> int:
    """CPUs the calculations may use: the usable CPUs minus the reserve (at least one)."""
    total = cpu_count()
    return max(1, total - reserved_cpus(total))


def n_workers(setting: int) -> int:
    """-1 = all CPUs the calculations may use (``available_cpus``: a reserve stays free), otherwise at least 1."""
    return available_cpus() if setting == -1 else max(1, int(setting))


def memory_bytes() -> tuple[int, int] | None:
    """(available, total) physical memory in bytes, None when unknown (Linux /proc/meminfo, Windows API)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys), int(st.ullTotalPhys)
        info = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) * 1024
        return info["MemAvailable"], info["MemTotal"]
    except Exception:  # noqa: BLE001 - macOS and others: no guard
        return None


def memory_cap(workers: int) -> int:
    """At most as many worker processes as the free memory carries: each is assumed to need
    ``CSPAPS_WORKER_MEM_GB`` (default 1.5; large fields need more), and max(2 GB, 15 % of RAM) stays free, so
    the computer does not start swapping and freeze."""
    mem = memory_bytes()
    if mem is None:
        return workers
    avail, total = mem
    try:
        per = float(os.environ.get("CSPAPS_WORKER_MEM_GB", "1.5")) * 2 ** 30
    except ValueError:
        per = 1.5 * 2 ** 30
    reserve = max(2 * 2 ** 30, 0.15 * total)
    return max(1, min(int(workers), int((avail - reserve) // max(per, 1.0))))


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
    workers = memory_cap(workers)  # leave memory to the system: no swapping
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
    return max(1, available_cpus() // max(1, int(workers)))


def limit_threads(n: int) -> None:
    """Cap the BLAS / OpenMP pools of this process at ``n`` threads (loaded libraries and later ones)."""
    for var in _THREAD_VARS:
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except Exception:  # noqa: BLE001 - threadpoolctl missing: the variables cover libraries loaded later
        pass


def lower_priority() -> None:
    """Run this (worker) process below normal priority, so the desktop and the GUI stay responsive while every
    worker is busy: nice + ``CSPAPS_WORKER_NICE`` (default 10; 0 = off) on Linux / macOS, "below normal" on
    Windows."""
    try:
        nice = int(os.environ.get("CSPAPS_WORKER_NICE", "10"))
    except ValueError:
        nice = 10
    if nice <= 0:
        return
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(nice)
    except Exception:  # noqa: BLE001 - priority is a courtesy, never a failure
        pass


def _worker_init(env: dict, counter, n_workers: int, threads: int = 1) -> None:
    """Pool-worker initializer: the parent's CSPAPS_* settings (stale ones removed), this worker's share of the
    cores for BLAS / OpenMP, and with the GPU backend on the next device round-robin and its share of it."""
    for key in [k for k in os.environ if k.startswith("CSPAPS_") and k not in env]:
        del os.environ[key]
    os.environ.update(env)
    limit_threads(threads)
    lower_priority()
    from .backend import enabled, worker_init

    if enabled():
        worker_init(counter, n_workers)
