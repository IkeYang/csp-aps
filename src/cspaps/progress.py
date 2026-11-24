"""Progress bars and remaining-time estimates for long runs.

Long tasks report progress as text, e.g. ``"APS scenarios B: 40/223"``, ``"review 2/5: ..."``, ``"generation
3/16: ..."`` (some low-level loops call ``progress(done, total)`` instead). ``ProgressTracker`` is a drop-in
progress callback for both forms: it keeps the messages and turns the latest ``done/total`` counter into the
fraction of the current stage and a remaining-time estimate from the stage's own pace.

A stage is the text before the counter (``"APS scenarios B:"``, ``"review"``). Its work started when the message
before its first counter arrived (the first counter is reported only after the first item finished), so the
pace is (time since that moment) / items done, and the remaining time is pace x items left. A run made of
several stages shows the current one; an optional ``expected_total_s`` (e.g. the advisor's estimate for the
chosen depth) adds an overall estimate.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

_COUNTER = re.compile(r"(\d+)\s*/\s*(\d+)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class ProgressState:
    message: str
    stage: str
    done: int | None
    total: int | None
    fraction: float | None  # of the current stage, 0..1 (None: no counter yet)
    elapsed_s: float  # since the run started
    stage_elapsed_s: float  # since the current stage's work started
    remaining_s: float | None  # of the current stage (None: not enough information yet)
    overall_remaining_s: float | None  # from ``expected_total_s``, when given


def parse_counter(message: str) -> tuple[int, int] | None:
    """The first ``done/total`` counter of a message (total > 0), else None."""
    m = _COUNTER.search(message)
    if not m:
        return None
    done, total = int(m.group(1)), int(m.group(2))
    return (min(done, total), total) if total > 0 else None


def stage_of(message: str) -> str:
    """The stage of a message: the text before its first counter (numbers masked), or the whole message.

    ``"APS scenarios B: 40/223"`` -> ``"APS scenarios B:"``; ``"review 2/5: centre aiming"`` -> ``"review"``;
    ``"APS scenarios, all layouts: 40/669 (B 13/223)"`` -> ``"APS scenarios, all layouts:"``."""
    m = _COUNTER.search(message)
    head = message[:m.start()] if m else message
    return _NUMBER.sub("#", head).strip()


class ProgressTracker:
    """Progress callback (``tracker("msg")`` or ``tracker(done, total)``) with bar fraction and remaining time."""

    def __init__(self, expected_total_s: float | None = None, clock=time.monotonic, on_message=None):
        self._clock = clock
        self.started = clock()
        self.expected_total_s = expected_total_s
        self.messages: list[str] = []
        self._on_message = on_message
        self._lock = threading.Lock()
        self._stage = ""
        self._stage_start = self.started
        self._last_message_t = self.started
        self._done: int | None = None
        self._total: int | None = None
        self._done_t = self.started

    def __call__(self, msg, total=None) -> None:
        if total is not None and not isinstance(msg, str):  # progress(done, total)
            msg = f"{int(msg)}/{int(total)}"
        msg = str(msg)
        now = self._clock()
        with self._lock:
            counter = parse_counter(msg)
            stage = stage_of(msg)
            if stage != self._stage or counter is None:
                # a new stage starts its work now (or, for a counter message, at the previous message)
                self._stage = stage
                self._stage_start = self._last_message_t if counter is not None else now
                self._done = self._total = None
            if counter is not None:
                self._done, self._total = counter
                self._done_t = now
            self._last_message_t = now
            self.messages.append(msg)
        if self._on_message is not None:
            self._on_message(msg)

    # list-like access for code that read ``job.messages`` before
    def append(self, msg) -> None:
        self(msg)

    def state(self) -> ProgressState:
        now = self._clock()
        with self._lock:
            msg = self.messages[-1] if self.messages else ""
            done, total = self._done, self._total
            stage_elapsed = now - self._stage_start
            fraction = remaining = None
            if done is not None and total:
                fraction = done / total
                if done > 0:
                    pace = (self._done_t - self._stage_start) / done  # seconds per item
                    remaining = max(0.0, pace * (total - done) - (now - self._done_t))
                    if done >= total:
                        remaining = 0.0
            elapsed = now - self.started
            overall = None
            if self.expected_total_s:
                overall = max(0.0, self.expected_total_s - elapsed)
            return ProgressState(msg, self._stage, done, total, fraction, elapsed, stage_elapsed, remaining, overall)


def fmt_seconds_zh(s: float | None) -> str:
    if s is None:
        return "—"
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.0f} 秒"
    if s < 3600:
        m, sec = divmod(int(round(s)), 60)
        return f"{m} 分 {sec:02d} 秒"
    h, rest = divmod(int(round(s)), 3600)
    return f"{h} 小时 {rest // 60:02d} 分"


def fmt_seconds(s: float | None) -> str:
    if s is None:
        return "?"
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        m, sec = divmod(int(round(s)), 60)
        return f"{m}m{sec:02d}s"
    h, rest = divmod(int(round(s)), 3600)
    return f"{h}h{rest // 60:02d}m"


def describe_zh(state: ProgressState) -> str:
    """One line for the GUI: elapsed, stage progress and remaining time."""
    parts = [f"已用 {fmt_seconds_zh(state.elapsed_s)}"]
    if state.fraction is not None:
        parts.append(f"本阶段 {state.done}/{state.total}（{100 * state.fraction:.0f}%）")
        if state.remaining_s is not None:
            parts.append(f"本阶段预计还需 {fmt_seconds_zh(state.remaining_s)}")
    if state.overall_remaining_s is not None:
        parts.append(f"整体预计还需约 {fmt_seconds_zh(state.overall_remaining_s)}")
    return " · ".join(parts)


def cli_line(state: ProgressState) -> str:
    """The message with stage percent and remaining time, for the command line."""
    if state.fraction is None:
        return f"  {state.message}  [{fmt_seconds(state.elapsed_s)}]"
    eta = f", ~{fmt_seconds(state.remaining_s)} left" if state.remaining_s is not None else ""
    return f"  {state.message}  [{100 * state.fraction:3.0f}%, {fmt_seconds(state.elapsed_s)} elapsed{eta}]"
