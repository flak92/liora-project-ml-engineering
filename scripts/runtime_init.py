"""Runtime setup that has to happen before NumPy, XGBoost or any BLAS is imported.

Import this FIRST — above numpy, above pipeline, above everything numeric. The thread limits are
read by those libraries once, at their own import time, so a line placed after them is a comment.

Why it matters, measured rather than assumed: XGBoost is pinned to one thread by
config/xgb.json (XGBOOST_N_JOBS), but OpenBLAS and OpenMP each default to one thread per core.
Three worker processes therefore opened up to twelve threads on four cores and spent the
difference on context switching — the load average sat at 8.65 while three jobs ran. Capping the
pools took the same work from 12.33 s to 8.15 s, a 1.51x speedup with identical arithmetic.

    import runtime_init                 # noqa: F401 — must precede numeric imports
    runtime_init.apply()

Scratch isolation is here too. Runners used to share xgb/tools/.search_scratch, which is harmless
while each worker holds a different ticker and fatal the moment two do not — in a four-worker
benchmark on one ticker, two processes died on the shared parquet. A per-run, per-ticker, per-pid
directory removes the class of failure rather than the instance.
"""
import os
from pathlib import Path

# One thread per process, everywhere. Set before the libraries read their environment.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

ROOT = Path(__file__).resolve().parents[1]


def apply():
    """Cap every numeric thread pool. Idempotent, and never overrides an explicit setting."""
    for v in THREAD_VARS:
        os.environ.setdefault(v, "1")
    os.environ.setdefault("LIORA_EPOCH", "sealed")
    return {v: os.environ[v] for v in THREAD_VARS}


def scratch_dir(run_id, ticker):
    """A directory no other worker writes to: scratch/<run_id>/<ticker>/<pid>/."""
    d = ROOT / "xgb" / "tools" / ".search_scratch" / str(run_id) / str(ticker) / str(os.getpid())
    d.mkdir(parents=True, exist_ok=True)
    return d


def thread_report():
    """What the pools actually ended up as — for the run manifest, not for trust."""
    try:
        import threadpoolctl
        return [{"api": p["user_api"], "impl": p.get("internal_api"), "threads": p["num_threads"]}
                for p in threadpoolctl.threadpool_info()]
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]


def env_report():
    import platform
    import sys
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "threads": {v: os.environ.get(v) for v in THREAD_VARS},
            "pools": thread_report()}


apply()
