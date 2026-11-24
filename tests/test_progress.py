"""Progress bar fraction and remaining-time estimate of long runs (GUI and command line)."""

import pytest

from cspaps.progress import ProgressTracker, cli_line, describe_zh, fmt_seconds_zh, parse_counter, stage_of


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_counter_and_stage_parsing():
    assert parse_counter("APS scenarios B: 40/223") == (40, 223)
    assert parse_counter("review 2/5: centre aiming on 30 states") == (2, 5)
    assert parse_counter("certificate: lone-heliostat potential map") is None
    assert parse_counter("weird 3/0") is None
    assert stage_of("APS scenarios B: 40/223") == stage_of("APS scenarios B: 41/223") == "APS scenarios B:"
    assert stage_of("APS scenarios, all layouts: 4/9 (B 2/3)") == stage_of("APS scenarios, all layouts: 5/9 (A 3/3)")
    assert stage_of("review 2/5: centre aiming") == stage_of("review 3/5: constrained APS on 12 states") == "review"


def test_remaining_time_from_the_stage_pace():
    clock = Clock()
    tr = ProgressTracker(clock=clock)
    clock.now = 4.0
    tr("formal ranking: APS on 100 scenarios")  # the stage's work starts here
    clock.now = 14.0
    tr("APS scenarios B: 1/100")
    clock.now = 24.0
    tr("APS scenarios B: 2/100")
    s = tr.state()
    assert s.fraction == pytest.approx(0.02)
    assert s.remaining_s == pytest.approx(98 * 10.0)  # 10 s per scenario since the stage started
    clock.now = 30.0  # 6 s into the next scenario: the estimate counts down
    assert tr.state().remaining_s == pytest.approx(98 * 10.0 - 6.0)
    assert tr.state().elapsed_s == pytest.approx(30.0)


def test_new_stage_restarts_the_estimate():
    clock = Clock()
    tr = ProgressTracker(clock=clock)
    clock.now = 10.0
    tr("APS scenarios A: 10/10")
    assert tr.state().remaining_s == 0.0
    clock.now = 12.0
    tr("control variate A: centre-aim energy of the member hours")
    s = tr.state()
    assert s.fraction is None and s.remaining_s is None and s.stage_elapsed_s == 0.0
    clock.now = 14.0
    tr("APS scenarios B: 1/10")
    assert tr.state().remaining_s == pytest.approx(9 * 2.0)  # paced from the previous message (t = 12)


def test_two_argument_callbacks_and_overall_estimate():
    clock = Clock()
    tr = ProgressTracker(expected_total_s=100.0, clock=clock)
    clock.now = 5.0
    tr(5, 20)  # simulate.annual / realtime style: progress(done, total)
    s = tr.state()
    assert (s.done, s.total) == (5, 20) and s.remaining_s == pytest.approx(15.0)
    assert s.overall_remaining_s == pytest.approx(95.0)
    assert "本阶段 5/20（25%）" in describe_zh(s) and "整体预计还需约" in describe_zh(s)
    assert "25%" in cli_line(s) and "left" in cli_line(s)
    assert tr.messages == ["5/20"]


def test_durations_in_chinese():
    assert fmt_seconds_zh(42) == "42 秒"
    assert fmt_seconds_zh(125) == "2 分 05 秒"
    assert fmt_seconds_zh(3 * 3600 + 20 * 60) == "3 小时 20 分"


def test_cli_progress_lines(capsys):
    from cspaps import cli

    cli._TRACKER = None
    cli._progress("APS scenarios B: 1/4")
    cli._progress("APS scenarios B: 2/4")
    err = capsys.readouterr().err.splitlines()
    assert err[0].startswith("  APS scenarios B: 1/4  [ 25%") and "50%" in err[1]


def test_gui_job_tracks_progress(monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.setenv("CSPAPS_GUI_SYNC", "1")
    from cspaps import gui_app  # importing the script does not draw the GUI (main() is guarded)

    def work(case, results, progress=None):
        for k in range(1, 4):
            progress(f"APS scenarios B: {k}/3")
        return {"summary": {}}

    job = gui_app.Job(work, None, None, progress=True, expected_s=60.0)
    assert job.result == {"summary": {}} and job.messages[-1] == "APS scenarios B: 3/3"
    s = job.tracker.state()
    assert s.fraction == 1.0 and s.remaining_s == 0.0 and job.tracker.expected_total_s == 60.0
