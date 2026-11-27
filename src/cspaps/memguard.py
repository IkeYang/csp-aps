"""Machine-wide coordination of large allocations, and the out-of-memory killer's choice.

Every process checks the free memory before it builds a large model (the optimality certificate's model is
about 3.5 GB per 1,000 mirrors with 21 candidate aims). Processes that start together -- the workers of one pool,
or several runs started side by side, one per GPU -- all see the same free memory before any of them has
allocated, and each concludes there is room: four 42 GB certificate models (with 12 GB aim problems) took a
251 GB computer down and the out-of-memory killer ended the desktop session.

A reservation removes the race. Under a machine-wide lock a process registers what it is about to allocate in a
ledger (one small file per reservation in the temp directory). The next process counts that as taken until it
shows up in its owner's resident memory, so nothing is counted twice. A process that does not fit waits while
another process holds memory it will release, and gives up (``MemoryShortage``) when nobody can free any -- every
other holder is waiting as well, or nobody else holds any -- or after ``CSPAPS_MEMORY_WAIT_S`` seconds
(default 3600). Without a usable ledger (read-only temp directory) the check falls back to the free memory alone.

``prefer_oom_kill`` raises this process's ``oom_score_adj`` (Linux): when memory runs out anyway, the kernel ends
a calculation instead of the desktop, the browser or the session bus.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

GB = 2 ** 30
POLL_S = 1.0


class MemoryShortage(MemoryError):
    """Not enough memory for a reservation, and no other process will free any."""

    def __init__(self, message: str, need_bytes: int, room_bytes: int):
        super().__init__(message)
        self.need_bytes = int(need_bytes)
        self.room_bytes = int(room_bytes)


def reserve_floor(total: int) -> int:
    """Memory always left to the system and the user's other programs: max(2 GB, 15 %)."""
    return int(max(2 * GB, 0.15 * total))


def prefer_oom_kill(score: int) -> None:
    """Linux: make the out-of-memory killer pick this process before the desktop (``oom_score_adj`` is raised
    only; lowering it needs privileges). 1000 = first; pool workers use 1000, the CLI and the GUI 500."""
    path = "/proc/self/oom_score_adj"
    try:
        with open(path, encoding="ascii") as f:
            current = int(f.read().strip())
        if score > current:
            with open(path, "w", encoding="ascii") as f:
                f.write(str(int(score)))
    except (OSError, ValueError):
        pass


def ledger_dir() -> Path:
    d = Path(os.environ.get("CSPAPS_MEMORY_LEDGER") or Path(tempfile.gettempdir()) / "cspaps-memory")
    if not d.is_dir():
        d.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(d, 0o1777)  # shared by every user of the computer (sticky: each removes only its own files)
    return d


@contextlib.contextmanager
def _locked(d: Path):
    fd = os.open(d / ".lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    os.lseek(fd, 0, 0)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.02)
            try:
                yield
            finally:
                os.lseek(fd, 0, 0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _start_token(pid: int) -> str | None:
    """Start time of a process (Linux), so a reused process id is not taken for the owner of an old entry."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _alive(pid: int, token: str | None) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return token is None or _start_token(pid) in (None, token)


def _rss(pid: int) -> int | None:
    """Resident memory of a process in bytes (Linux), None when unknown."""
    try:
        with open(f"/proc/{pid}/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _entries(d: Path) -> list[dict]:
    out = []
    for p in d.glob("*.json"):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
            e["_path"] = p
            pid = int(e["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not _alive(pid, e.get("start")):
            with contextlib.suppress(OSError):
                p.unlink()
            continue
        out.append(e)
    return out


def _pending(entries: list[dict]) -> int:
    """What one process has reserved but not allocated yet: its reservations minus its resident growth since the
    first of them (all of it when its resident memory is unknown)."""
    reserved = sum(int(e["bytes"]) for e in entries)
    rss = _rss(int(entries[0]["pid"]))
    base = min(int(e.get("rss0") or 0) for e in entries)
    if rss is None or base <= 0:
        return reserved
    return max(0, reserved - max(0, rss - base))


def _waiting(d: Path, pid: int) -> bool:
    return (d / f"{pid}.waiting").exists()


class Reservation:
    """``need_bytes`` of the computer's memory held for a large allocation until ``release`` (or the end of a
    ``with`` block). Raises ``MemoryShortage`` when it cannot be had; see the module docstring."""

    def __init__(self, need_bytes: int, what: str = "model", wait_s: float | None = None):
        self.need = int(max(0, need_bytes))
        self.what = what
        self.path: Path | None = None
        self.waited_s = 0.0
        from . import parallel

        mem = parallel.memory_bytes()
        if mem is None or self.need == 0:
            return
        try:
            d = ledger_dir()
            with _locked(d):
                pass
        except OSError:  # no usable ledger: the free memory alone, as before
            avail, total = mem
            room = avail - reserve_floor(total)
            if self.need > room:
                raise MemoryShortage(self._message(room, 0, 0), self.need, room) from None
            return
        if wait_s is None:
            try:
                wait_s = float(os.environ.get("CSPAPS_MEMORY_WAIT_S", "3600"))
            except ValueError:
                wait_s = 3600.0
        self._acquire(d, wait_s)

    def _message(self, room: int, pending: int, holders: int) -> str:
        held = (f" and {pending / GB:.1f} GB reserved by {holders} other running calculation(s)"
                if holders else "")
        return (f"the {self.what} needs about {self.need / GB:.1f} GB, more than the {max(room, 0) / GB:.1f} GB free "
                f"after the system's reserve{held}")

    def _acquire(self, d: Path, wait_s: float) -> None:
        from . import parallel

        pid = os.getpid()
        marker = d / f"{pid}.waiting"
        t0 = time.monotonic()
        said = False
        try:
            while True:
                with _locked(d):
                    mem = parallel.memory_bytes()
                    if mem is None:
                        return
                    avail, total = mem
                    owners: dict[int, list[dict]] = {}
                    for e in _entries(d):
                        if int(e["pid"]) != pid:
                            owners.setdefault(int(e["pid"]), []).append(e)
                    pending = sum(_pending(v) for v in owners.values())
                    room = avail - reserve_floor(total) - pending
                    if self.need <= room:
                        path = d / f"{pid}-{uuid.uuid4().hex}.json"
                        tmp = path.with_suffix(".tmp")
                        tmp.write_text(json.dumps({"pid": pid, "start": _start_token(pid), "bytes": self.need,
                                                   "rss0": _rss(pid) or 0, "what": self.what, "t": time.time()}),
                                       encoding="utf-8")
                        os.replace(tmp, path)
                        self.path = path
                        return
                    movers = [p for p in owners if not _waiting(d, p)]
                    if not movers or time.monotonic() - t0 > wait_s:
                        raise MemoryShortage(self._message(room, pending, len(owners)), self.need, room)
                    marker.write_text("", encoding="ascii")
                if not said:
                    print(f"cspaps: waiting for memory -- {self._message(room, pending, len(owners))}",
                          file=sys.stderr, flush=True)
                    said = True
                time.sleep(POLL_S)
        finally:
            self.waited_s = time.monotonic() - t0
            with contextlib.suppress(OSError):
                marker.unlink()

    def release(self) -> None:
        path, self.path = self.path, None
        if path is not None:
            with contextlib.suppress(OSError):
                path.unlink()

    def __enter__(self) -> "Reservation":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def reserve(need_bytes: int, what: str = "model", wait_s: float | None = None) -> Reservation:
    """Hold memory for a large allocation; ``with reserve(n, "..."):`` releases it at the end of the block."""
    return Reservation(need_bytes, what, wait_s)
