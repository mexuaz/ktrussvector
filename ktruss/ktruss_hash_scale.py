#!/usr/bin/env python3
"""
CPU scaling harness for ktruss_hash.py (segmented multi_arange formulation).

Emits one CSV row per configuration. The thread-limit variables have to be set
before numpy is imported, and graphio imports numpy at module scope, so the
core count is pulled straight off argv here -- ahead of every other import.

ktruss_hash.py exposes two independent approaches; both are driven from here:

  support   one full support pass over the whole graph (triangulation only),
            the phase whose segments are cut on vertex boundaries and so is
            the part expected to scale.
  peel      the complete k-truss, which re-runs the support pass every round
            and interleaves it with the serial compact / prune_low_degree.

Usage:
    ktruss_hash_scale.py <graph> --cores N --mode support|peel
                         [--budget N] [--threads cores|1]
"""

import os
import sys

# ---------------------------------------------------------------------------
# Thread limits. Must precede the numpy import that graphio performs, and must
# precede graphio's own setdefault(..., "1") for these same names.
# ---------------------------------------------------------------------------


def _argv_value(flag, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


_CORES = int(_argv_value("--cores", "1"))
_POLICY = _argv_value("--threads", "cores")
_THREADS = str(_CORES) if _POLICY == "cores" else "1"

os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = _THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _THREADS
# Slurm exports OMP_PROC_BIND; numpy imported under it pins the whole process
# to one core, which would flatten every curve below. graphio pops it too, but
# do it here so the setting is not order dependent.
for _v in ("OMP_PROC_BIND", "OMP_PLACES"):
    os.environ.pop(_v, None)

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import shared_memory  # noqa: E402
from timeit import default_timer as timer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ktruss_hash as kh  # noqa: E402
from graphio import load_mtx, offsets  # noqa: E402

import numpy as np  # noqa: E402

FIELDS = ("dataset", "mode", "cores", "threads", "budget", "time_sec",
          "segments", "rounds", "triangles", "k_max", "k_avg", "vertices",
          "edges", "cpus_available", "omp_num_threads")


def run_support(path, cores, budget):
    """Time a single support pass over the full graph with `cores` workers."""
    S, E, D, nV = load_mtx(path)
    nE = int(S.size)
    O = offsets(D)
    val, mxx, key = kh.build_keys(S, E, D, O, nV)

    shm_objs, spec = {}, {}

    def share(name, arr):
        shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 8))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[...] = arr
        shm_objs[name] = shm
        spec[name] = (shm.name, arr.dtype.str, arr.shape)
        return view

    g = {
        "cS": share("cS", S), "cE": share("cE", E),
        "cO": share("cO", O), "cD": share("cD", D),
        "val": share("val", val), "mxx": share("mxx", mxx),
        "key": share("key", key), "O0": share("O0", O),
        "alive": share("alive", np.ones(nE, dtype=np.uint8)),
        "sup": share("sup", np.zeros(nE, dtype=np.uint32)),
    }

    bounds = kh.segment_bounds(O, D, S, E, nV, nE, budget, cores * 4)
    pool = mp.get_context("fork").Pool(cores, initializer=kh._worker_init,
                                       initargs=(spec,))
    try:
        # Warm up: fault in the shared pages and fork the workers, so the
        # timed pass measures the expansion and not first-touch page faults.
        g["sup"][...] = 0
        pool.map(kh._support_segment, bounds, chunksize=1)

        g["sup"][...] = 0
        t = timer()
        pool.map(kh._support_segment, bounds, chunksize=1)
        dt = timer() - t
        tris = int(g["sup"].sum()) // 6
    finally:
        pool.close()
        pool.join()
        for shm in shm_objs.values():
            shm.close()
            shm.unlink()

    return {"time_sec": round(dt, 4), "segments": len(bounds), "rounds": 1,
            "triangles": tris, "k_max": "", "k_avg": "",
            "vertices": nV, "edges": nE}


def run_peel(path, cores, budget):
    r = kh.ktruss_hash(path, procs=cores, budget=budget)
    return {"time_sec": r["peeling_time_sec"], "segments": "",
            "rounds": r["rounds"], "triangles": r["triangle_count"],
            "k_max": r["K_max"], "k_avg": r["K_avg"],
            "vertices": r["vertex_count"], "edges": r["edge_count"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--mode", choices=("support", "peel"), default="support")
    ap.add_argument("--budget", type=int, default=1 << 22)
    ap.add_argument("--threads", choices=("cores", "1"), default="cores")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    row = {"dataset": os.path.basename(args.graph), "mode": args.mode,
           "cores": args.cores, "threads": _THREADS, "budget": args.budget,
           "cpus_available": len(os.sched_getaffinity(0)),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "")}
    try:
        fn = run_support if args.mode == "support" else run_peel
        row.update(fn(args.graph, args.cores, args.budget))
    except Exception:
        traceback.print_exc()
        row.update({k: "" for k in FIELDS if k not in row})
        row["time_sec"] = "FAIL"

    line = ",".join(str(row.get(f, "")) for f in FIELDS)
    print(line, flush=True)
    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a") as fh:
            if new:
                fh.write(",".join(FIELDS) + "\n")
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
