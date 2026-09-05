#!/usr/bin/env python3
"""NumPy k-truss: Cantor/Cuckoo triangulation with the dissertation's peel.

Triangulation is ktruss_cantor.py's, unchanged and imported: orient by degree,
Cantor-pair each edge, hold the codes in a Cuckoo table, and close every wedge
with two vectorised probes.

The peel is the difference. ktruss_cantor.peel() keeps the full triangle list
for the whole run and recomputes support each round as
`alive[a] & alive[b] & alive[c]`, so every round costs the ORIGINAL triangle
count no matter how few triangles survive. This file follows Algorithm 7
(`ps-peeling`) of Chapter 4: carry the triangle-edge incidence array T, count
it, peel, and *shrink* T by the triangles the peeled edges destroyed, so round
cost tracks the triangles still alive.

Algorithm 7, and how this follows it:

    k = 1
    while T.size() > 0:
        T_u, T_c = unique(T)              -> run boundaries of a sorted T
        while (T_c == k).any():           -> see (1)
            T_k = T_u[T_c == k]           -> see (1)
            T_h = T_u[T_c > k]
            K[T_k] = k
            K[T_h] = K[T_h] + 1           -> see (2)
            T = update(T, G0, G1, G2, T_k)-> _compact()
        k++

Three deviations, all deliberate:

(1) `T_c == k` is replaced by `T_c <= k`. Removing a batch of edges can drop
    another edge's count by more than one, so a count can step from k+2 to
    k-1 without ever equalling k. Under `== k` such an edge matches no branch
    ever again -- not T_k, not T_h -- so it is never assigned a truss value and
    never leaves T, and `while T.size() > 0` does not terminate. Measured on
    the 53 local test graphs, the literal form breaks 9 of them: 3 strand
    edges outright, 6 finish with wrong truss values. `<= k` is what
    ktruss_cantor.py and ktruss_hash.py both use, as does the C++
    implementation in the gpu-nucleus repository.

(2) `K[T_h] = K[T_h] + 1` is INERT -- every edge in T_h is later assigned by
    `K[T_k] = k`, which overwrites it -- but it costs a scatter over every
    surviving edge, every round. It is off by default and restored with
    --alg7-increment, under which the result is bit-identical.

(3) Counts come from run boundaries of a sorted T, not `unique(T)` or a
    bincount. Sorting the incidence array once (and compaction preserves
    sortedness, since it only deletes) makes the count of every edge a run
    length, which is O(|T|) with NO term in the edge count m. It also makes
    each edge's incidences contiguous, which is what lets the parallel path
    below partition without a reduction.

Edges in no triangle never enter T. They are given truss value 0 up front,
which agrees with ktruss_cantor.py peeling them in its k=0 round.

Parallel peeling, and why the obvious version is worse than serial. The first
version of this file, and ktruss_cantor.peel(), both split the triangle list
across workers and had each worker accumulate into its OWN m-sized support
array, which the parent then summed. That reduction is O(nw * m) per round and
is pure overhead: it grows with the worker count while the useful work per
worker shrinks. Job 20012126 measured the result -- peel time linear in nw,
4.76s -> 14.9 -> 25.2 -> 48.2 on amazon-2008 at 1/16/32/96 cores, and the same
shape for ktruss_cantor.peel(). At 96 workers on soc-Slashdot0811 the
reduction alone moves 155 GB.

This version has no reduction at all. Because T is sorted, an edge's
incidences are contiguous, so a contiguous slice of T owns a contiguous RANGE
of edges. Each worker is given one such slice and is then simply the serial
algorithm restricted to its own edges: it counts its runs, decides its own
T_k, and writes its own K entries and its own slice of T. Every write is
disjoint. The only shared state is `alive_tri`, which all workers write, and
only ever write False -- a triangle dies once and stays dead, so concurrent
writes cannot disagree and no lock is needed. The parent aggregates nothing
but scalars.

Even so, more workers are not always better: a round's work is O(|T|), which
shrinks, while the dispatch cost is O(nw) and does not. A peel that runs many
short rounds (soc-Slashdot0811: 431) cannot pay for process dispatch at all.
_worker_count() therefore sizes the pool from the work actually available and
returns 1 when a round cannot pay for parallelism, so raising --cores can
leave the peel alone but never slows it down. `workers` in the CSV reports
what was actually used.

Hence TWO layouts, chosen by that same worker count:

  _peel_dense()  1 worker. Per-triangle layout, support by bincount,
                 compaction filters G0/G1/G2. Touches 3nT values a round and
                 needs no sort.
  peel_arr()     >1 worker. Sorted incidence layout, the one described above.
                 Touches 6nT a round and pays an argsort up front, in exchange
                 for partitioning with no reduction.

The sorted layout is the slower of the two when it runs alone -- measured at
0.53x on amazon-2008 and 0.67x on soc-Slashdot0811 -- so it is used only when
its partitioning is actually going to be used. Both produce identical K.

Usage:
    ktruss_cantor_scale_arr.py <graph.mtx> [--cores N] [--load F]
                               [--threads cores|1] [--alg7-increment]
                               [--check] [--json] [--csv F]
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


# Thread limits have to be set before numpy is imported, and before graphio's
# own setdefault(..., "1") for the same names -- same preamble as
# ktruss_cantor_scale.py, for the same reason.
_CORES = int(_argv_value("--cores", "1"))
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

import argparse           # noqa: E402
import json               # noqa: E402
import multiprocessing as mp  # noqa: E402
import traceback          # noqa: E402
from multiprocessing import shared_memory  # noqa: E402
from timeit import default_timer as timer  # noqa: E402

import numpy as np        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ktruss_cantor as kc  # noqa: E402

# Incidences a worker must be handed before a round can pay for its dispatch.
# A round costs O(|T|) and |T| shrinks; dispatch costs O(nw) and does not.
# Overridable so a scaling run can separate "the peel has no work to share"
# from "the threshold was set wrong".
MIN_INCIDENCES_PER_WORKER = int(
    os.environ.get("MIN_INCIDENCES_PER_WORKER", 1 << 21))


def _worker_count(n_incidences, procs):
    """Workers this peel can actually keep busy. 1 means run it serially."""
    if procs <= 1:
        return 1
    return max(1, min(int(procs),
                      int(n_incidences) // MIN_INCIDENCES_PER_WORKER))


# --------------------------------------------------------------------------
# Flat-array helpers
# --------------------------------------------------------------------------

def _runs(T):
    """(start, value, length) of every run in a sorted array.

    On the sorted incidence array this is exactly Algorithm 7's
    `T_u, T_c = unique(T)`, at O(|T|) and with no array of size m.
    """
    if T.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e
    b = np.flatnonzero(np.concatenate(([True], T[1:] != T[:-1])))
    return b, T[b], np.diff(np.concatenate((b, [T.size])))


def _missing(a, b):
    """Values of sorted `a` that do not occur in sorted `b`.

    Used for the alive edges that have fallen out of T entirely: their count
    is 0, so they are peeled at the current level rather than stranded.
    """
    if a.size == 0 or b.size == 0:
        return a
    i = np.searchsorted(b, a)
    np.minimum(i, b.size - 1, out=i)
    return a[b[i] != a]


# --------------------------------------------------------------------------
# One segment of the peel. Serial and parallel run the SAME two functions;
# the parallel path just has more than one segment and runs them in workers.
#
# meta[j] = (t_start, t_len, e_start, e_len): where segment j's incidences and
# its alive-edge list live. The starts are fixed, the lengths shrink.
# --------------------------------------------------------------------------

def _step(Tv, Tt, Ev, meta, alive_tri, K, j, k, increment):
    """One inner iteration of Algorithm 7 over segment j.

    Marks the triangles killed here in the shared alive_tri but does NOT
    compact: a triangle can span segments, so compaction has to wait until
    every segment has finished marking.
    """
    ts, tn, es, en = (int(meta[j, 0]), int(meta[j, 1]),
                      int(meta[j, 2]), int(meta[j, 3]))
    b, u, c = _runs(Tv[ts:ts + tn])

    peel = c <= k                          # T_c <= k, see docstring (1)
    Tk = u[peel]
    drop = _missing(Ev[es:es + en], u)     # alive here, no incidence left
    if Tk.size == 0 and drop.size == 0:
        return 0

    K[Tk] = k
    if drop.size:
        K[drop] = k
    surv = u[~peel]                        # T_h
    if increment and surv.size:
        K[surv] += 1                       # Algorithm 7 line 8; inert
    Ev[es:es + surv.size] = surv
    meta[j, 3] = surv.size

    if Tk.size:
        # Expand the per-run peel mask to per-incidence and kill those
        # triangles. Only ever writes False, so it is safe unsynchronised.
        alive_tri[Tt[ts:ts + tn][np.repeat(peel, c)]] = False
    return int(Tk.size + drop.size)


def _compact(Tv, Tt, meta, alive_tri, j):
    """update(T, G0, G1, G2, T_k): drop the dead triangles' incidences.

    In place within the segment, so the next round touches only survivors.
    Deletion preserves sortedness, which the next _step relies on.
    """
    ts, tn = int(meta[j, 0]), int(meta[j, 1])
    if tn == 0:
        return
    tt = Tt[ts:ts + tn]
    keep = alive_tri[tt]
    n = int(np.count_nonzero(keep))
    if n != tn:
        # boolean indexing copies before the write, so the overlap is safe
        Tv[ts:ts + n] = Tv[ts:ts + tn][keep]
        Tt[ts:ts + n] = tt[keep]
        meta[j, 1] = n


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

_W = {}


def _init_worker(spec):
    """Attach to the parent's segment buffers. Every array is shared."""
    names, shapes, nw = spec
    _W["shm"] = []
    for key in ("Tv", "Tt", "Ev", "meta", "alive_tri", "K"):
        shm = shared_memory.SharedMemory(name=names[key])
        _W["shm"].append(shm)
        dtype = np.bool_ if key == "alive_tri" else np.int64
        _W[key] = np.ndarray(shapes[key], dtype=dtype, buffer=shm.buf)
    _W["nw"] = nw


def _w_step(args):
    j, k, increment = args
    return _step(_W["Tv"], _W["Tt"], _W["Ev"], _W["meta"],
                 _W["alive_tri"], _W["K"], j, k, increment)


def _w_compact(j):
    _compact(_W["Tv"], _W["Tt"], _W["meta"], _W["alive_tri"], j)


# --------------------------------------------------------------------------
# Peeling (Algorithm 7)
# --------------------------------------------------------------------------

def _peel_dense(G, m, increment):
    """Algorithm 7 over the per-triangle layout. The SERIAL path.

    Support is a bincount over the incidences and compaction filters G0/G1/G2
    directly, so a round touches 3nT values. The sorted layout below touches
    6nT -- the incidence array plus the triangle companion it needs to know
    what to delete -- and pays an argsort up front, which measured 0.53x on
    amazon-2008 and 0.67x on soc-Slashdot0811 against this one when it ran
    alone (job 20012913 vs 20012126). That cost buys the disjoint partitioning
    the parallel path needs, and buys nothing when there is one worker, so one
    worker runs this instead.

    O(m) per round in the bincount, which is why it loses once |T| is large
    enough to be worth splitting -- exactly where _worker_count() switches.
    """
    K = np.zeros(m, dtype=np.int64)
    g0 = np.ascontiguousarray(G[:, 0]).astype(np.int64)
    g1 = np.ascontiguousarray(G[:, 1]).astype(np.int64)
    g2 = np.ascontiguousarray(G[:, 2]).astype(np.int64)
    T = np.concatenate((g0, g1, g2))

    # Edges in no triangle never enter T: truss value 0, already in K.
    alive = np.zeros(m, dtype=bool)
    alive[T] = True

    k, rounds = 1, 0
    while T.size:
        Tc = np.bincount(T, minlength=m)
        while True:
            rounds += 1
            tk = alive & (Tc <= k)          # T_c <= k, see docstring (1)
            if not tk.any():
                break
            if increment:
                K[alive & ~tk] += 1         # Algorithm 7 line 8; inert
            K[tk] = k
            alive &= ~tk
            keep = alive[g0] & alive[g1] & alive[g2]
            g0, g1, g2 = g0[keep], g1[keep], g2[keep]
            T = np.concatenate((g0, g1, g2))
            Tc = np.bincount(T, minlength=m)
        k += 1
    return K, rounds


def peel_arr(G, m, procs=1, increment=False):
    """Trussness per undirected edge. Returns (K, rounds, workers)."""
    K = np.zeros(m, dtype=np.int64)
    nT = int(G.shape[0])
    if nT == 0:
        return K, 0, 1

    # One worker: the sorted layout's only advantage is that it partitions,
    # so skip it and its argsort entirely.
    if _worker_count(3 * nT, procs) == 1:
        K, rounds = _peel_dense(G, m, increment)
        return K, rounds, 1

    # Sorted incidence array plus the triangle each incidence belongs to.
    Tv = np.concatenate((G[:, 0], G[:, 1], G[:, 2])).astype(np.int64)
    Tt = np.tile(np.arange(nT, dtype=np.int64), 3)
    order = np.argsort(Tv, kind="stable")
    Tv, Tt = Tv[order], Tt[order]

    b, u, c = _runs(Tv)
    alive_tri = np.ones(nT, dtype=bool)
    Ev = u.copy()                          # alive edges; edges absent keep K=0
    nw = _worker_count(Tv.size, procs)

    # Cut the run axis so every segment gets about the same number of
    # incidences. Cutting on runs (never inside one) is what keeps each
    # edge wholly owned by one segment. Coincident cuts collapse, so nw can
    # come out lower than asked for on a graph with few distinct edges.
    cum = np.cumsum(c)
    targets = (np.arange(1, nw) * (cum[-1] / nw)).astype(np.int64)
    cuts = np.unique(np.concatenate(([0], np.searchsorted(cum, targets),
                                     [u.size])))
    nw = int(cuts.size - 1)

    meta = np.zeros((nw, 4), dtype=np.int64)
    for j in range(nw):
        r0, r1 = int(cuts[j]), int(cuts[j + 1])
        t0 = int(b[r0])
        t1 = int(b[r1]) if r1 < u.size else int(Tv.size)
        meta[j] = (t0, t1 - t0, r0, r1 - r0)

    pool = shms = None
    try:
        if nw > 1:
            arrays = {"Tv": Tv, "Tt": Tt, "Ev": Ev, "meta": meta,
                      "alive_tri": alive_tri, "K": K}
            shms, names, shapes = {}, {}, {}
            for key, a in arrays.items():
                shm = shared_memory.SharedMemory(create=True,
                                                 size=max(a.nbytes, 8))
                shms[key] = shm
                names[key], shapes[key] = shm.name, a.shape
                view = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
                view[...] = a
                arrays[key] = view
            Tv, Tt, Ev = arrays["Tv"], arrays["Tt"], arrays["Ev"]
            meta, alive_tri, K = (arrays["meta"], arrays["alive_tri"],
                                  arrays["K"])
            pool = mp.get_context("fork").Pool(
                nw, initializer=_init_worker,
                initargs=((names, shapes, nw),))

        k, rounds = 1, 0
        while int(meta[:, 3].sum()):
            while True:
                rounds += 1
                if pool is None:
                    n = _step(Tv, Tt, Ev, meta, alive_tri, K, 0, k, increment)
                else:
                    n = sum(pool.map(_w_step,
                                     [(j, k, increment) for j in range(nw)],
                                     chunksize=1))
                if n == 0:
                    break
                if pool is None:
                    _compact(Tv, Tt, meta, alive_tri, 0)
                else:
                    pool.map(_w_compact, range(nw), chunksize=1)
            k += 1

        if pool is not None:
            K = np.array(K)                # copy out before the segment goes
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if shms:
            for shm in shms.values():
                shm.close()
                shm.unlink()

    return K, rounds, nw


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(path, load=0.45, procs=1, increment=False):
    """Same phase breakdown and result keys as ktruss_cantor.run()."""
    t = timer()
    S, E, D, nV = kc.load_mtx(path)
    t_load = timer() - t

    t = timer()
    s, e, d, o = kc.orient(S, E, D, nV)
    m = int(s.size)
    codes = kc.cantor(s, e)
    table = kc.Cuckoo(codes, np.arange(m, dtype=np.int64), load=load)
    t_build = timer() - t

    t = timer()
    G = kc.triangles(s, e, d, o, table, m, procs)
    t_tri = timer() - t

    t = timer()
    K, rounds, workers = peel_arr(G, m, procs, increment)
    t_peel = timer() - t

    return {
        "file": os.path.basename(path),
        "vertex_count": nV,
        "edge_count": int(S.size),
        "undirected_edges": m,
        "triangle_count": int(G.shape[0]),
        "K_max": int(K.max()) if K.size else 0,
        "K_avg": round(float(K.mean()), 2) if K.size else 0.0,
        "load_time_sec": round(t_load, 4),
        "table_build_time_sec": round(t_build, 4),
        "triangle_time_sec": round(t_tri, 4),
        "peeling_time_sec": round(t_peel, 4),
        "total_time_sec": round(t_build + t_tri + t_peel, 4),
        "rounds": rounds,
        "workers": workers,
        "alg7_increment": int(bool(increment)),
        "table_passes": table.passes,
        "table_bytes": table.bytes(),
        "load_factor": load,
        "procs": procs,
    }, K


FIELDS = ("dataset", "variant", "cores", "threads", "workers", "load",
          "table_build_sec", "triangle_sec", "peel_sec", "total_sec",
          "rounds", "table_passes", "triangles", "k_max", "k_avg",
          "vertices", "undirected_edges", "cpus_available",
          "omp_num_threads", "alg7_increment")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--load", type=float, default=0.45,
                    help="cuckoo table load factor")
    ap.add_argument("--threads", choices=("cores", "1"), default="cores")
    ap.add_argument("--alg7-increment", action="store_true",
                    help="restore Algorithm 7's inert K[T_h] += 1")
    ap.add_argument("--check", action="store_true",
                    help="assert K equals ktruss_cantor.peel() elementwise")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    row = {"dataset": os.path.basename(args.graph), "variant": "arr",
           "cores": args.cores, "threads": _THREADS, "load": args.load,
           "cpus_available": len(os.sched_getaffinity(0)),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "")}
    res = None
    try:
        res, K = run(args.graph, load=args.load, procs=args.cores,
                     increment=args.alg7_increment)
        row.update({
            "workers": res["workers"],
            "alg7_increment": res["alg7_increment"],
            "table_build_sec": res["table_build_time_sec"],
            "triangle_sec": res["triangle_time_sec"],
            "peel_sec": res["peeling_time_sec"],
            "total_sec": res["total_time_sec"],
            "rounds": res["rounds"], "table_passes": res["table_passes"],
            "triangles": res["triangle_count"], "k_max": res["K_max"],
            "k_avg": res["K_avg"], "vertices": res["vertex_count"],
            "undirected_edges": res["undirected_edges"],
        })
        if args.check:
            # Rebuild the same triangles and peel them with the reference
            # implementation, so the comparison isolates the peel.
            S, E, D, nV = kc.load_mtx(args.graph)
            s, e, d, o = kc.orient(S, E, D, nV)
            m = int(s.size)
            table = kc.Cuckoo(kc.cantor(s, e),
                              np.arange(m, dtype=np.int64), load=args.load)
            G = kc.triangles(s, e, d, o, table, m, 1)
            Kref, _ = kc.peel(G, m, 1)
            if not np.array_equal(K, Kref):
                bad = np.flatnonzero(K != Kref)
                raise SystemExit(
                    f"trussness mismatch on {row['dataset']}: "
                    f"{bad.size} of {m} edges differ, first at edge {bad[0]} "
                    f"(arr={K[bad[0]]}, cantor={Kref[bad[0]]})")
            res["checked_against"] = "ktruss_cantor.peel"
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        for f in FIELDS:
            row.setdefault(f, "")
        row["total_sec"] = "FAIL"

    if args.json:
        print(json.dumps(res if res else row, indent=4))
    else:
        print(",".join(str(row.get(f, "")) for f in FIELDS), flush=True)

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a") as fh:
            if new:
                fh.write(",".join(FIELDS) + "\n")
            fh.write(",".join(str(row.get(f, "")) for f in FIELDS) + "\n")


if __name__ == "__main__":
    main()
