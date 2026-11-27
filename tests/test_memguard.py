"""Machine-wide memory reservations (``memguard``), the worker-pool cap and the out-of-memory priority.

Field report: with 4 parallel processes the student's 251 GB computer ran out of memory -- four Python processes of
about 52 GB each (full-field aim problem + optimality-certificate model) -- and the out-of-memory killer ended the
desktop session. Every process had checked the free memory before any of them allocated."""

import gc
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pytest

from cspaps import memguard, parallel

GB = 2 ** 30
linux_fork = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork start method and /proc (Linux)")


@pytest.fixture
def fake_memory(monkeypatch, tmp_path):
    """10 GB available of 20 GB: 3 GB stay free (15 %), 7 GB can be reserved. Ledger in a private directory."""
    monkeypatch.setattr(parallel, "memory_bytes", lambda: (10 * GB, 20 * GB))
    monkeypatch.setenv("CSPAPS_MEMORY_LEDGER", str(tmp_path / "ledger"))
    monkeypatch.setattr(memguard, "POLL_S", 0.05)
    return tmp_path


def _hold(args):
    """Child: reserve 3 GB, keep it 0.4 s, report the interval (or the failure)."""
    need, out, i = args
    try:
        with memguard.reserve(need, "test"):
            t0 = time.monotonic()
            time.sleep(0.4)
            t1 = time.monotonic()
        out.put((i, t0, t1, None))
    except memguard.MemoryShortage as exc:
        out.put((i, None, None, str(exc)))


@linux_fork
def test_parallel_processes_do_not_take_the_same_free_memory(fake_memory):
    ctx = mp.get_context("fork")  # the children inherit the fake memory
    out = ctx.Queue()
    procs = [ctx.Process(target=_hold, args=((3 * GB, out, i),)) for i in range(4)]
    for p in procs:
        p.start()
    res = [out.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    assert all(err is None for *_, err in res)  # nobody failed: the others waited for memory
    spans = sorted((t0, t1) for _, t0, t1, _ in res)
    # at most two 3 GB reservations at a time in 7 GB
    for t0, _ in spans:
        assert sum(1 for a, b in spans if a <= t0 < b) <= 2
    assert not list((fake_memory / "ledger").glob("*.json"))  # every reservation released


def test_a_reservation_nobody_can_free_fails_at_once(fake_memory):
    t0 = time.monotonic()
    with pytest.raises(memguard.MemoryShortage) as err:
        memguard.reserve(8 * GB, "certificate")
    assert time.monotonic() - t0 < 5
    assert err.value.need_bytes == 8 * GB and err.value.room_bytes == 7 * GB
    assert "more than the 7.0 GB free after the system's reserve" in str(err.value)
    with memguard.reserve(7 * GB, "certificate") as r:  # exactly the room: fine
        assert r.path is not None and r.path.exists()
    assert not r.path


def _nested(args):
    """Child: hold 3 GB, then ask for 5 GB more. ``go`` orders the second requests."""
    name, ready, go, out = args
    try:
        with memguard.reserve(3 * GB, "problem"):
            ready.set()
            go.wait(30)
            try:
                with memguard.reserve(5 * GB, "certificate"):
                    out.put((name, "ok"))
            except memguard.MemoryShortage:
                out.put((name, "skipped"))
    except Exception as exc:  # noqa: BLE001
        out.put((name, repr(exc)))


@linux_fork
def test_waiting_processes_do_not_deadlock(fake_memory):
    """A and B hold 3 GB each and both ask for 5 GB (4 GB room each). A waits for B; B sees that the only other
    holder is waiting, so nobody will free memory: it skips (its caller drops the certificate) and releases its
    3 GB, after which A gets its 5 GB."""
    ctx = mp.get_context("fork")
    out = ctx.Queue()
    ra, rb, ga, gb = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Event()
    a = ctx.Process(target=_nested, args=(("A", ra, ga, out),))
    b = ctx.Process(target=_nested, args=(("B", rb, gb, out),))
    a.start()
    b.start()
    assert ra.wait(30) and rb.wait(30)
    ga.set()  # A asks first and starts waiting
    time.sleep(0.5)
    gb.set()
    res = dict(out.get(timeout=60) for _ in range(2))
    a.join(30)
    b.join(30)
    assert res == {"A": "ok", "B": "skipped"}


@linux_fork
def test_reservations_of_ended_processes_are_ignored(fake_memory):
    ctx = mp.get_context("fork")
    p = ctx.Process(target=time.sleep, args=(0,))
    p.start()
    p.join()
    d = memguard.ledger_dir()
    stale = d / f"{p.pid}-dead.json"
    stale.write_text(f'{{"pid": {p.pid}, "start": null, "bytes": {6 * GB}, "rss0": 0}}', encoding="utf-8")
    with memguard.reserve(5 * GB, "test"):  # 6 GB of a dead process would leave only 1 GB
        pass
    assert not stale.exists()


def test_certificate_model_memory_is_reserved_while_the_model_lives(fake_memory, small_case, monkeypatch):
    from cspaps.aps import build_problem
    from cspaps.certify import CertificateTooLarge, _build_model, model_bytes
    from cspaps.field import build_field
    from cspaps.receiver import build_grid
    from cspaps.simulate import sun_state
    from cspaps.uniformity import region_of_grid

    fld = build_field(small_case)
    _, el, az, dni = sun_state(small_case, "2025-06-21 11:00")
    prob = build_problem(small_case, fld, el, az, dni, grid=build_grid(small_case.receiver))
    h = region_of_grid(small_case, prob.grid, small_case.uniformity)
    ledger = fake_memory / "ledger"

    def held(what):
        return [f for f in ledger.glob("*.json") if what in f.read_text(encoding="utf-8")]

    assert len(held("aim problem")) == 1  # the problem's own reservation, held while it lives
    model = _build_model(prob, h)
    cert = held("optimality certificate")
    assert len(cert) == 1 and str(model_bytes(prob)) in cert[0].read_text(encoding="utf-8")
    del model
    gc.collect()
    assert not held("optimality certificate") and len(held("aim problem")) == 1  # released with the model
    need = model_bytes(prob)
    monkeypatch.setattr(parallel, "memory_bytes", lambda: (2 * GB + need // 2, 8 * GB))  # too little room
    with pytest.raises(CertificateTooLarge) as err:
        _build_model(prob, h)
    assert err.value.need_bytes == need and "the optimality certificate needs about" in str(err.value)


def test_pool_size_follows_the_field_size(monkeypatch):
    from cspaps.aps import problem_memory_bytes

    monkeypatch.setattr(parallel, "memory_bytes", lambda: (240 * GB, 251 * GB))
    monkeypatch.delenv("CSPAPS_WORKER_MEM_GB", raising=False)
    per = problem_memory_bytes(11_960, 21)
    assert 14 * GB < per < 18 * GB  # measured peaks scale to ~14 GB for the Aksai field
    assert parallel.memory_cap(16, per) == int((240 - 0.15 * 251) * GB // per)
    assert parallel.memory_cap(16) == 16  # small work: 1.5 GB each, all 16 fit
    monkeypatch.setenv("CSPAPS_WORKER_MEM_GB", "100")  # the user's estimate wins
    assert parallel.memory_cap(16, per) == 2


@pytest.mark.skipif(not os.path.exists("/proc/self/oom_score_adj"), reason="Linux out-of-memory score")
def test_calculations_go_before_the_desktop_when_memory_runs_out():
    import subprocess

    code = ("from cspaps.memguard import prefer_oom_kill as p; p(700); p(100); "
            "print(open('/proc/self/oom_score_adj').read().strip())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert int(out) >= 700  # raised, never lowered again


def test_pool_workers_ask_to_be_killed_first(monkeypatch):
    if not os.path.exists("/proc/self/oom_score_adj"):
        pytest.skip("Linux out-of-memory score")
    monkeypatch.setenv("CSPAPS_POOL", "process")
    p = parallel.pool(2)
    try:
        scores = p.map(_oom_score, range(2))
    finally:
        p.close()
        p.join()
    assert all(s == 1000 for s in scores)


def _oom_score(_):
    return int(open("/proc/self/oom_score_adj", encoding="ascii").read())


def test_aim_problem_memory_is_reserved_while_the_problem_lives(fake_memory, small_case, monkeypatch):
    from cspaps.aps import build_problem, candidate_count, problem_memory_bytes
    from cspaps.field import build_field
    from cspaps.simulate import sun_state

    fld = build_field(small_case)
    _, el, az, dni = sun_state(small_case, "2025-06-21 11:00")
    ledger = fake_memory / "ledger"
    prob = build_problem(small_case, fld, el, az, dni)
    need = problem_memory_bytes(fld.n, candidate_count(small_case.aps))
    entries = list(ledger.glob("*.json"))
    assert len(entries) == 1 and f'"bytes": {need}' in entries[0].read_text(encoding="utf-8")
    del prob
    gc.collect()
    assert not list(ledger.glob("*.json"))
    monkeypatch.setattr(parallel, "memory_bytes", lambda: (2 * GB + need // 2, 8 * GB))
    with pytest.raises(memguard.MemoryShortage) as err:
        build_problem(small_case, fld, el, az, dni)
    assert "the aim problem needs about" in str(err.value)
    assert not list(ledger.glob("*.json"))  # nothing left behind by the failed build


def _certify_job(args):
    """Child: the single-instant pipeline -- aim problem, greedy plan, relaxation rounding with certificate."""
    case, out, i = args
    from cspaps.aps import build_problem, solve
    from cspaps.certify import improve_and_certify
    from cspaps.field import build_field
    from cspaps.simulate import sun_state

    try:
        fld = build_field(case)
        _, el, az, dni = sun_state(case, "2025-06-21 11:00")
        prob = build_problem(case, fld, el, az, dni)
        res = solve(prob, "greedy", case.aps)
        _, cert = improve_and_certify(prob, res, case.aps, rounds=1, max_iter=30, time_limit_s=5.0)
        out.put((i, cert["status"]))
    except Exception as exc:  # noqa: BLE001
        out.put((i, f"{type(exc).__name__}: {exc}"))


@linux_fork
@pytest.mark.parametrize("n_jobs", [3, 6])
def test_many_parallel_runs_queue_for_their_certificates(fake_memory, small_case, monkeypatch, n_jobs):
    """Any number of simultaneous runs (workers, GPUs, separate programs) through the real pipeline, with memory
    for every aim problem and about one certificate model: no deadlock, every run finishes with its certificate
    and nothing stays reserved. (That the reservations never overcommit is checked above with fixed holds: here
    the fake free memory does not shrink when the children allocate.)"""
    from cspaps.aps import build_problem, candidate_count, problem_memory_bytes
    from cspaps.certify import model_bytes
    from cspaps.field import build_field
    from cspaps.simulate import sun_state

    fld = build_field(small_case)
    _, el, az, dni = sun_state(small_case, "2025-06-21 11:00")
    model = model_bytes(build_problem(small_case, fld, el, az, dni))
    gc.collect()
    problem = problem_memory_bytes(fld.n, candidate_count(small_case.aps))
    total = 10 * GB
    avail = 2 * GB + n_jobs * problem + int(1.5 * model)  # the 2 GB floor + every problem + one model
    monkeypatch.setattr(parallel, "memory_bytes", lambda: (avail, total))
    ctx = mp.get_context("fork")
    out = ctx.Queue()
    procs = [ctx.Process(target=_certify_job, args=((small_case, out, i),)) for i in range(n_jobs)]
    for p in procs:
        p.start()
    res = dict(out.get(timeout=600) for _ in procs)
    for p in procs:
        p.join(60)
    assert all(v == "certified" for v in res.values()), res
    assert not list((fake_memory / "ledger").glob("*.json"))
