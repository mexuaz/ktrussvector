#!/usr/bin/env python3
"""
NumPy k-truss, segmented multi_arange formulation.

Reference for the `seg` wedge kernel of the C++ implementation in the
gpu-nucleus repository. Every
directed edge (s,d) gets the key mxx[s] + d, where mxx is the prefix sum of
(max_neighbour(v) + 1); because the edge list is sorted by (src,dst) the key
array is sorted too, so a wedge s -> d -> x is closed with a vectorized
np.searchsorted instead of a per-element hash probe.

Parallelism: segments are cut on *vertex* boundaries. All wedges expanded from
a source-vertex range [v0,v1) close into edges whose source also lies in
[v0,v1), and the original edge list is grouped by source, so each segment
accumulates into a disjoint contiguous slice of the support array. Workers
therefore write into shared memory with no locks and no reduction.

The CSR construction is shared with the k-core peels via graphio.py.

Usage:
    ktruss_hash.py <graph.mtx> [--procs N] [--budget N] [--scaling 1,2,4,...]
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from multiprocessing import shared_memory
from timeit import default_timer as timer

# graphio clears OMP_PROC_BIND before it pulls in numpy, so import it first.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphio import ETYPE, VTYPE, load_mtx, multi_arange, offsets  # noqa: E402

import numpy as np  # noqa: E402

KTYPE = np.uint64


def build_keys(S, E, D, O, nV):
    """val[v] = largest neighbour of v; keys of v live in [mxx[v], mxx[v]+val[v]]."""
    val = np.zeros(nV, dtype=VTYPE)
    nz = D > 0
    val[nz] = E[(O[:-1][nz] + D[nz] - 1).astype(np.int64)]

    span = np.zeros(nV, dtype=KTYPE)
    span[nz] = val[nz].astype(KTYPE) + 1
    mxx = np.zeros(nV + 1, dtype=KTYPE)
    np.cumsum(span, out=mxx[1:])

    key = mxx[S.astype(np.int64)] + E.astype(KTYPE)
    return val, mxx, key


# --------------------------------------------------------------------------
# Shared memory plumbing
# --------------------------------------------------------------------------

_G = {}


def _worker_init(spec):
    for name, (shm_name, dt, shape) in spec.items():
        shm = shared_memory.SharedMemory(name=shm_name)
        _G["_shm_" + name] = shm
        _G[name] = np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf)


def _support_segment(task):
    return support_segment(task, _G)


def support_segment(task, g):
    """Expand every wedge rooted in the source-vertex range [v0,v1)."""
    v0, v1 = task
    cO, cD, cS, cE = g["cO"], g["cD"], g["cS"], g["cE"]
    val, mxx, key = g["val"], g["mxx"], g["key"]
    alive, sup, O0 = g["alive"], g["sup"], g["O0"]

    lo, hi = int(cO[v0]), int(cO[v1])
    if hi <= lo:
        return 0

    dst = cE[lo:hi]
    cnt = cD[dst.astype(np.int64)]

    # m = multi_arange(nei[dst], deg[dst]) ; st = np.repeat(src, deg[dst])
    m = multi_arange(cO[dst.astype(np.int64)], cnt)
    if m.size == 0:
        return 0
    st = np.repeat(cS[lo:hi], cnt)
    dn = cE[m]

    # col = val[st] >= dn : past val[st] no edge (st,dn) can exist.
    col = dn <= val[st.astype(np.int64)]
    st = st[col]
    dn = dn[col]
    if st.size == 0:
        return 0

    q = mxx[st.astype(np.int64)] + dn.astype(KTYPE)
    idx = np.searchsorted(key, q)
    ok = idx < key.size
    idx = idx[ok]
    ok = key[idx] == q[ok]
    idx = idx[ok]
    idx = idx[alive[idx] != 0]
    if idx.size == 0:
        return 0

    o0, o1 = int(O0[v0]), int(O0[v1])
    sup[o0:o1] += np.bincount(idx - o0, minlength=o1 - o0).astype(sup.dtype)
    return int(idx.size)


def segment_bounds(cO, cD, cS, cE, nV, n, budget, min_segments):
    """Vertex aligned cuts so that each segment expands ~<= budget wedges."""
    if n == 0:
        return [(0, nV)]

    w = cD[cE[:n].astype(np.int64)].astype(np.int64)
    cw = np.cumsum(w)
    total = int(cw[-1])
    if total == 0:
        return [(0, nV)]

    nseg = max(min_segments, int(math.ceil(total / budget)), 1)
    if nseg <= 1:
        return [(0, nV)]

    targets = np.arange(1, nseg, dtype=np.float64) * (total / nseg)
    pos = np.clip(np.searchsorted(cw, targets), 0, n - 1)
    cuts = cS[pos].astype(np.int64) + 1
    b = np.unique(np.concatenate(([0], cuts, [nV])))
    return list(zip(b[:-1].tolist(), b[1:].tolist()))


def prune_low_degree(k, nV, cS, cE, cID, cD, alive, K):
    """k-core style pre-peel: a support-k edge needs both endpoints at degree >= k+1."""
    if k <= 0:
        return 0

    removed = 0
    deg = cD.astype(np.int64)
    live = np.ones(cS.size, dtype=bool)
    while True:
        low = (deg > 0) & (deg <= k)
        if not low.any():
            break
        kill = (low[cS.astype(np.int64)] | low[cE.astype(np.int64)]) & live
        if not kill.any():
            break
        ids = cID[kill]
        alive[ids] = 0
        K[ids] = k
        removed += int(ids.size)
        live &= ~kill
        deg = np.bincount(cS[live], minlength=nV).astype(np.int64)

    return removed


def ktruss_hash(path, procs=1, budget=1 << 22, verbose=False):
    S, E, D, nV = load_mtx(path)
    nE = int(S.size)
    O = offsets(D)
    val, mxx, key = build_keys(S, E, D, O, nV)

    # Shared state; cS/cE/cID keep the full capacity, cO/cD describe the residual.
    shm_objs = {}
    spec = {}

    def share(name, arr):
        shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 8))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[...] = arr
        shm_objs[name] = shm
        spec[name] = (shm.name, arr.dtype.str, arr.shape)
        return view

    g = {}
    g["cS"] = share("cS", S)
    g["cE"] = share("cE", E)
    g["cO"] = share("cO", O)
    g["cD"] = share("cD", D)
    g["val"] = share("val", val)
    g["mxx"] = share("mxx", mxx)
    g["key"] = share("key", key)
    g["O0"] = share("O0", O)
    g["alive"] = share("alive", np.ones(nE, dtype=np.uint8))
    g["sup"] = share("sup", np.zeros(nE, dtype=np.uint32))

    pool = None
    if procs > 1:
        pool = mp.get_context("fork").Pool(procs, initializer=_worker_init,
                                           initargs=(spec,))

    cS = S.copy()
    cE = E.copy()
    cID = np.arange(nE, dtype=np.uint32)
    cD = D.copy()

    K = np.zeros(nE, dtype=np.int64)
    alive = g["alive"]
    sup = g["sup"]

    alive_count = nE
    n = nE
    k = 0
    triangles = 0
    first_round = True
    rounds = 0

    def publish():
        g["cS"][:n] = cS
        g["cE"][:n] = cE
        g["cO"][...] = offsets(cD)
        g["cD"][...] = cD

    def compact():
        nonlocal cS, cE, cID, cD, n
        keep = alive[cID.astype(np.int64)] != 0
        cS = cS[keep]
        cE = cE[keep]
        cID = cID[keep]
        cD = np.bincount(cS, minlength=nV).astype(VTYPE)
        n = int(cS.size)
        publish()

    t0 = timer()
    publish()

    while alive_count:
        pruned = prune_low_degree(k, nV, cS, cE, cID, cD, alive, K)
        if pruned:
            alive_count -= pruned
            compact()
            if not alive_count:
                break

        sup[...] = 0
        bounds = segment_bounds(g["cO"], cD, cS, cE, nV, n, budget, procs * 4)
        if pool is None:
            for b in bounds:
                support_segment(b, g)
        else:
            pool.map(_support_segment, bounds, chunksize=1)

        rounds += 1
        if first_round:
            triangles = int(sup.sum()) // 6
            first_round = False

        R = np.flatnonzero((alive != 0) & (sup <= k))
        if R.size == 0:
            k += 1
            continue

        K[R] = k
        alive[R] = 0
        alive_count -= int(R.size)
        compact()

    elapsed = timer() - t0

    if pool is not None:
        pool.close()
        pool.join()

    # One value per undirected edge, as the C++ implementations report.
    Ku = K[S < E]
    result = {
        "file": os.path.basename(path),
        "vertex_count": nV,
        "edge_count": nE,
        "triangle_count": triangles,
        "K_max": int(Ku.max()) if Ku.size else 0,
        "K_count": int(Ku.size),
        "K_avg": round(float(Ku.mean()), 2) if Ku.size else 0.0,
        "peeling_time_sec": round(elapsed, 4),
        "rounds": rounds,
        "procs": procs,
        "cpus_available": len(os.sched_getaffinity(0)),
    }

    for shm in shm_objs.values():
        shm.close()
        shm.unlink()

    if verbose:
        print(json.dumps(result, indent=2), file=sys.stderr)
    return result


def scaling(path, proc_list, budget):
    """Time a single support pass over the full graph for each worker count."""
    S, E, D, nV = load_mtx(path)
    nE = int(S.size)
    O = offsets(D)
    val, mxx, key = build_keys(S, E, D, O, nV)

    rows = []
    base = None
    for procs in proc_list:
        shm_objs = {}
        spec = {}

        def share(name, arr):
            shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 8))
            view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            view[...] = arr
            shm_objs[name] = shm
            spec[name] = (shm.name, arr.dtype.str, arr.shape)
            return view

        g = {
            "cS": share("cS", S),
            "cE": share("cE", E),
            "cO": share("cO", O),
            "cD": share("cD", D),
            "val": share("val", val),
            "mxx": share("mxx", mxx),
            "key": share("key", key),
            "O0": share("O0", O),
            "alive": share("alive", np.ones(nE, dtype=np.uint8)),
            "sup": share("sup", np.zeros(nE, dtype=np.uint32)),
        }

        bounds = segment_bounds(O, D, S, E, nV, nE, budget, procs * 4)
        pool = mp.get_context("fork").Pool(procs, initializer=_worker_init,
                                           initargs=(spec,))

        # Warm up: fault in the shared pages and the worker processes.
        g["sup"][...] = 0
        pool.map(_support_segment, bounds, chunksize=1)

        g["sup"][...] = 0
        t = timer()
        pool.map(_support_segment, bounds, chunksize=1)
        dt = timer() - t

        tris = int(g["sup"].sum()) // 6
        pool.close()
        pool.join()
        for shm in shm_objs.values():
            shm.close()
            shm.unlink()

        if base is None:
            base = dt
        rows.append((procs, len(bounds), dt, base / dt, tris))

    print(f"{'procs':>6} {'segments':>9} {'time_sec':>10} {'speedup':>8} {'triangles':>12}")
    for procs, nseg, dt, sp, tris in rows:
        print(f"{procs:6d} {nseg:9d} {dt:10.3f} {sp:8.2f}x {tris:12d}")
    return rows
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", help="graph in .mtx edge list format")
    ap.add_argument("--procs", type=int, default=1, help="worker processes")
    ap.add_argument("--budget", type=int, default=1 << 22,
                    help="max wedges materialized per segment")
    ap.add_argument("--scaling", default=None,
                    help="comma separated worker counts; times one support pass each")
    args = ap.parse_args()

    if args.scaling:
        scaling(args.graph, [int(x) for x in args.scaling.split(",")], args.budget)
        return

    print(json.dumps(ktruss_hash(args.graph, args.procs, args.budget), indent=4))


if __name__ == "__main__":
    main()
