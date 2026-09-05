#!/usr/bin/env python3
"""
CPU scaling harness for ktruss_cantor.py (Cantor pairing + Cuckoo hashing).

Companion to ktruss_hash_scale.py. Emits one CSV row per configuration.

ktruss_cantor.run() already times its phases separately -- cuckoo table build,
triangulation, peel -- so a single run per configuration yields every curve at
once, unlike ktruss_hash.py which needed a separate mode per phase.

The peel variants selected by --variant:

  peel      chunk=0 -> peel(), single threshold per round. Workers split the
            *triangle list* (step = nT // procs), so every core is usable.
  multik    chunk=C -> peel_multik(), a batch of thresholds per round. Workers
            are handed run-axis *rows*, so nw = min(procs, R) and
            R = RUN_TRIANGLE_CAP // nT. On triangle-dense graphs the default
            cap pins R far below the core count.
  multik-uncapped
            same, with RUN_TRIANGLE_CAP raised so R = max(chunk, procs) and
            every core gets a row. Isolates the formulation's own scaling from
            the cap. Costs R x nT work per round, so only tractable on graphs
            with a modest triangle count.

Thread limits are set before numpy is imported, and before graphio's own
setdefault(..., "1") for the same names, so they actually take effect.

Usage:
    ktruss_cantor_scale.py <graph> --cores N
                           [--variant peel|multik|multik-uncapped]
                           [--chunk C] [--load F] [--threads cores|1]
"""

import os
import sys


def _argv_value(flag, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


_CORES = int(_argv_value("--cores", "1"))
_VARIANT = _argv_value("--variant", "peel")
_POLICY = _argv_value("--threads", "cores")
_THREADS = str(_CORES) if _POLICY == "cores" else "1"

os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = _THREADS
os.environ["NUMEXPR_NUM_THREADS"] = _THREADS
# Slurm exports OMP_PROC_BIND; numpy imported under it pins the process to one
# core, which would flatten every curve.
for _v in ("OMP_PROC_BIND", "OMP_PLACES"):
    os.environ.pop(_v, None)

# Read at ktruss_cantor import time, so it has to be set here.
if _VARIANT == "multik-uncapped":
    os.environ["RUN_TRIANGLE_CAP"] = _argv_value("--cap", str(10 ** 12))

import argparse  # noqa: E402
import traceback  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ktruss_cantor as kc  # noqa: E402

FIELDS = ("dataset", "variant", "cores", "threads", "chunk", "load",
          "run_cap", "max_R", "table_build_sec", "triangle_sec", "peel_sec",
          "total_sec", "rounds", "table_passes", "triangles", "k_max",
          "k_avg", "vertices", "undirected_edges", "cpus_available",
          "omp_num_threads")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--variant", default="peel",
                    choices=("peel", "multik", "multik-uncapped"))
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--load", type=float, default=0.45)
    ap.add_argument("--threads", choices=("cores", "1"), default="cores")
    ap.add_argument("--cap", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    chunk = 0 if args.variant == "peel" else args.chunk

    row = {"dataset": os.path.basename(args.graph), "variant": args.variant,
           "cores": args.cores, "threads": _THREADS, "chunk": chunk,
           "load": args.load, "run_cap": kc.RUN_TRIANGLE_CAP,
           "cpus_available": len(os.sched_getaffinity(0)),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "")}
    try:
        r, _ = kc.run(args.graph, load=args.load, procs=args.cores,
                      chunk=chunk)
        nT = max(int(r["triangle_count"]), 1)
        # Widest run axis the first outer chunk can open, hence the most
        # workers peel_multik can occupy. 1 for the single-threshold peel,
        # whose workers split the triangle list instead.
        if chunk == 0:
            max_R = 1
        else:
            max_R = max(1, min(max(chunk, args.cores),
                               kc.RUN_TRIANGLE_CAP // nT))
        row.update({
            "max_R": max_R,
            "table_build_sec": r["table_build_time_sec"],
            "triangle_sec": r["triangle_time_sec"],
            "peel_sec": r["peeling_time_sec"],
            "total_sec": r["total_time_sec"],
            "rounds": r["rounds"], "table_passes": r["table_passes"],
            "triangles": r["triangle_count"], "k_max": r["K_max"],
            "k_avg": r["K_avg"], "vertices": r["vertex_count"],
            "undirected_edges": r["undirected_edges"],
        })
    except Exception:
        traceback.print_exc()
        for f in FIELDS:
            row.setdefault(f, "")
        row["total_sec"] = "FAIL"

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
