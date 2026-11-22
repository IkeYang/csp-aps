import multiprocessing as mp
import subprocess
import sys
import threading
import textwrap

from .conftest import ROOT


def test_parallel_pool_in_background_thread_does_not_pickle(tmp_path):
    from multiprocessing.pool import ThreadPool

    from cspaps.parallel import pool

    results = []

    def run():
        workers = pool(2)
        try:
            results.extend(workers.map(lambda value: value * value, [2, 3]))
            results.append(isinstance(workers, ThreadPool))
        finally:
            workers.close()
            workers.join()

    task = threading.Thread(target=run)
    task.start()
    task.join(timeout=10)
    assert not task.is_alive()
    assert results == [4, 9, True]


def test_parallel_run_from_script(tmp_path):
    """Fork runs top-level scripts directly. Spawn requires a guarded application entry."""
    script = tmp_path / "parallel_annual.py"
    body = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        print("TOP-LEVEL", flush=True)
        from tests.conftest import small_case_dict
        from cspaps import case_from_dict, sun
        from cspaps.field import build_field
        from cspaps.simulate import annual
        case = case_from_dict(small_case_dict())
        fld = build_field(case)
        bins = sun.sun_bins(sun.hourly_samples(case), 4)
        r = annual(case, fld, samples=bins, with_flux=False, workers=2)
        print("ENERGY", r.energy_mwh["intercept"], flush=True)
    """)
    if "fork" not in mp.get_all_start_methods():
        body = 'if __name__ == "__main__":\n' + textwrap.indent(body, "    ")
    script.write_text(body)
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("TOP-LEVEL") == 1
    assert "ENERGY" in out.stdout


def test_parallel_and_serial_annual_agree(small_case):
    from cspaps import sun
    from cspaps.field import build_field
    from cspaps.simulate import annual

    fld = build_field(small_case)
    bins = sun.sun_bins(sun.hourly_samples(small_case), 5)
    a = annual(small_case, fld, samples=bins, with_flux=False, workers=1)
    b = annual(small_case, fld, samples=bins, with_flux=False, workers=2)
    assert abs(a.energy_mwh["intercept"] - b.energy_mwh["intercept"]) < 1e-6 * a.energy_mwh["intercept"]
