#!/usr/bin/env python3
"""NumPy k-truss via Cantor pairing and Cuckoo hashing.

The formulation the dissertation's k-truss chapter describes: encode each edge
as a single integer with a Cantor pairing of its endpoints, hold those codes in
a Cuckoo hash table, and let triangle enumeration and edge peeling both become
operations over flat arrays. The point is not a speed record -- GPU truss
decomposition is a mature field -- but that the same flat-array formulation
extracts multi-core and SIMD parallelism from NumPy alone, with no GPU and no
C++.

Why these two pieces:

  * **Cantor pairing** turns an edge into one integer, so an edge set becomes a
    flat integer array and edge identity becomes integer equality. Applied to
    the ordered pair (min, max) it is injective on undirected edges, so no
    canonicalisation table is needed.
  * **Cuckoo hashing** answers "is (u,w) an edge?" in O(1) worst case with two
    probes, and both probes vectorise: one gather per table, no bucket walk and
    no branching. ktruss/ktruss_hash.py answers the same question with a
    searchsorted over a sorted key array, which is O(log m) and needs the keys
    kept in order; the Cuckoo table trades that for flat O(1) lookup.

Both give identical trussness -- --check asserts it against the sorted-key
implementation when it is importable.

Usage:
    ktruss_cantor.py <graph.mtx> [--check] [--load N] [--json] [--csv-out F]
"""

import argparse
import inspect
import json
import os
import sys
from timeit import default_timer as timer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphio import load_mtx, offsets  # noqa: E402

import multiprocessing as mp  # noqa: E402
from multiprocessing import shared_memory  # noqa: E402
import numpy as np  # noqa: E402

# chunk * |triangles| ceiling for the multi-k peel's batched state. This also
# bounds the multi-k peel's parallelism: workers are handed run-axis rows, so
# nw = min(procs, R) and R = cap // |triangles|. Overridable so a scaling run
# can separate "the formulation does not scale" from "the cap pinned R below
# the core count" -- default is unchanged.
RUN_TRIANGLE_CAP = int(os.environ.get("RUN_TRIANGLE_CAP", 200_000_000))

# _tri_chunk creates a shared segment in a worker and hands the NAME back for
# the parent to attach. With the default resource tracking that is a race: the
# worker's resource_tracker unlinks the segment when the worker exits, and it
# can win against the parent's attach -- measured at 4 of 32 handoffs lost with
# 32 workers, which is what made triangles() raise FileNotFoundError
# intermittently for procs > 1. track=False (Python 3.13+) takes the tracker
# out of it and makes the parent solely responsible for unlinking, which
# triangles() now does in a finally.
_SHM_UNTRACKED = (
    {"track": False}
    if "track" in inspect.signature(
        shared_memory.SharedMemory.__init__).parameters
    else {}
)

# Odd multipliers for the two independent hash functions. Odd keeps the low
# bits from collapsing, and the two differ so a pair colliding in one table is
# unlikely to collide in the other -- which is what makes two probes enough.
_H0 = np.int64(0x9E3779B97F4A7C15 & ((1 << 62) - 1))
_H1 = np.int64(0xC2B2AE3D27D4EB4F & ((1 << 62) - 1))


def cantor(u, v):
    """Injective code for the undirected edge {u, v}.

    pi(a,b) = (a+b)(a+b+1)/2 + b on the ORDERED pair (min, max), so {u,v} and
    {v,u} land on the same code without a separate canonicalisation step.
    """
    a = np.minimum(u, v).astype(np.int64)
    b = np.maximum(u, v).astype(np.int64)
    s = a + b
    return (s * (s + 1)) // 2 + b


class Cuckoo:
    """Two-table Cuckoo hash over int64 keys, built and probed in batches.

    Insertion alternates tables: every key still homeless is scattered into the
    current table, the winners of each slot are those that read their own key
    back, and the losers retry in the other table on the next pass. That is a
    vectorised variant of cuckoo's evict-and-reinsert -- the loser is retried
    rather than displacing the resident -- and it terminates for the load
    factors used here because each pass places a constant fraction.
    """

    EMPTY = np.int64(-1)

    def __init__(self, keys, vals, load=0.45, max_pass=200):
        keys = keys.astype(np.int64)
        vals = vals.astype(np.int64)
        # On the rare failure to converge, grow and retry: cuckoo insertion can
        # cycle, and a bigger table is the standard escape.
        for attempt in range(4):
            if self._build(keys, vals, load / (2 ** attempt), max_pass):
                self.load_used = load / (2 ** attempt)
                return
        raise RuntimeError("cuckoo build did not converge after 4 rehashes")

    def _build(self, keys, vals, load, max_pass):
        n = max(int(keys.size / load), 8)
        self.m = np.int64(n | 1)                 # odd size, better spread
        self.key = [np.full(self.m, self.EMPTY, dtype=np.int64) for _ in range(2)]
        self.val = [np.zeros(self.m, dtype=np.int64) for _ in range(2)]

        k, v = keys, vals
        t, self.passes = 0, 0
        for _ in range(max_pass):
            if k.size == 0:
                return True
            self.passes += 1
            pos = self._slot(t, k)

            # Real cuckoo eviction, vectorised. Read the residents BEFORE
            # scattering, then re-insert whoever got displaced. Writing only
            # into empty slots instead would deadlock -- a key whose two slots
            # are both permanently occupied could never be placed -- and
            # scattering without capturing the residents would lose them
            # silently, which is worse: every later probe for such a key misses
            # and triangles quietly go uncounted.
            prev_k = self.key[t][pos].copy()
            prev_v = self.val[t][pos].copy()
            self.key[t][pos] = k                 # duplicates: last writer wins
            self.val[t][pos] = v
            won = self.key[t][pos] == k

            lost = ~won                          # lost a same-slot clash
            eject = won & (prev_k != self.EMPTY) & (prev_k != k)
            k = np.concatenate((k[lost], prev_k[eject]))
            v = np.concatenate((v[lost], prev_v[eject]))
            t ^= 1
        return k.size == 0

    def _slot(self, t, k):
        h = _H0 if t == 0 else _H1
        # keys are already well spread by the pairing, so a multiply-shift is
        # enough; the mask keeps the product non-negative in int64
        return ((k * h) & np.int64((1 << 62) - 1)) % self.m

    def lookup(self, q):
        """Vectorised probe: (found, value) for every query, two gathers."""
        q = q.astype(np.int64)
        p0 = self._slot(0, q)
        hit0 = self.key[0][p0] == q
        p1 = self._slot(1, q)
        hit1 = self.key[1][p1] == q
        val = np.where(hit0, self.val[0][p0], self.val[1][p1])
        return hit0 | hit1, val

    def bytes(self):
        return sum(a.nbytes for a in self.key) + sum(a.nbytes for a in self.val)


def orient(S, E, D, nV):
    """Keep only u < v, so each undirected edge appears once."""
    keep = S < E
    s, e = S[keep].astype(np.int64), E[keep].astype(np.int64)
    order = np.lexsort((e, s))
    s, e = s[order], e[order]
    d = np.bincount(s, minlength=nV).astype(np.int64)
    return s, e, d, offsets(d)[:nV].astype(np.int64)


def _tri_slice(s, e, d, o, table, lo, hi, budget=1 << 22):
    """Triangles from oriented edges [lo, hi), in wedge-bounded blocks.

    The wedge expansion is the memory peak of the whole program, so it is cut
    into blocks whose expanded length stays under `budget` rather than
    materialised in one array.
    """
    out = []
    i = lo
    while i < hi:
        j, acc = i, 0
        while j < hi and (acc == 0 or acc + int(d[e[j]]) <= budget):
            acc += int(d[e[j]])
            j += 1
        cnt = d[e[i:j]]
        total = int(cnt.sum())
        if total:
            base = np.arange(i, j, dtype=np.int64)
            grp = np.repeat(base, cnt)
            excl = np.cumsum(cnt) - cnt
            pos = o[e[i:j]][grp - i] + (np.arange(total, dtype=np.int64)
                                        - excl[grp - i])
            found, idx_uw = table.lookup(cantor(s[grp], e[pos]))
            if found.any():
                out.append(np.stack((grp[found], pos[found], idx_uw[found]), 1))
        i = j
    if not out:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(out) if len(out) > 1 else out[0]


_TW = {}


def _init_tri_worker(state):
    _TW.update(state)


def _tri_chunk(bounds):
    """Worker side: enumerate a slice and hand back a shared-memory handle.

    The result is published through shared memory and only its NAME is
    returned. Pickling the triangles themselves would dominate: soc-digg's 62.7M
    triples are 1.5 GB, and that cost would fall on every worker.
    """
    lo, hi = bounds
    T = _tri_slice(_TW["s"], _TW["e"], _TW["d"], _TW["o"], _TW["table"], lo, hi)
    if T.shape[0] == 0:
        return None, 0
    shm = shared_memory.SharedMemory(create=True, size=T.nbytes,
                                     **_SHM_UNTRACKED)
    np.ndarray(T.shape, dtype=np.int64, buffer=shm.buf)[...] = T
    name = shm.name
    shm.close()                                  # parent unlinks after reading
    return name, T.shape[0]


def triangles(s, e, d, o, table, m, procs=1):
    """Every triangle once, as a triple of undirected edge ids.

    The graph is oriented (u<v), so walking w over adj(v) from the edge (u,v)
    and probing for (u,w) finds each triangle exactly once: u < v < w. No
    deduplication pass is needed -- and the edges partition cleanly, so this is
    the phase that parallelises best.
    """
    if procs <= 1 or e.size < (1 << 14):
        return _tri_slice(s, e, d, o, table, 0, e.size)

    # split on equal wedge counts, not equal edge counts: degree skew would
    # otherwise leave one worker with most of the expansion
    wedges = d[e].astype(np.int64)
    cw = np.cumsum(wedges)
    targets = (np.arange(1, procs) * (cw[-1] / procs)).astype(np.int64)
    cuts = np.unique(np.concatenate(([0], np.searchsorted(cw, targets),
                                     [e.size])))
    bounds = [(int(cuts[i]), int(cuts[i + 1])) for i in range(cuts.size - 1)]

    state = {"s": s, "e": e, "d": d, "o": o, "table": table}
    with mp.get_context("fork").Pool(procs, initializer=_init_tri_worker,
                                     initargs=(state,)) as pool:
        handles = pool.map(_tri_chunk, bounds)

    # The parent owns every segment's lifetime (see _SHM_UNTRACKED), so unlink
    # in a finally: an exception part way through this loop would otherwise
    # leave the remaining segments behind in /dev/shm.
    parts = []
    opened = []
    try:
        for name, n in handles:
            if not name:
                continue
            shm = shared_memory.SharedMemory(name=name, **_SHM_UNTRACKED)
            opened.append(shm)
            parts.append(np.ndarray((n, 3), dtype=np.int64,
                                    buffer=shm.buf).copy())
    finally:
        for shm in opened:
            shm.close()
            shm.unlink()
    if not parts:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def _support_chunk(args):
    """Accumulate one slice of the triangle list into this worker's own row.

    The partial sum is written into shared memory rather than returned: an
    m-sized array per worker per round pickled back through the pool costs more
    than the arithmetic it carries (432 rounds x 4 workers x 3.75 MB on
    soc-Slashdot0811 made the 4-worker run 4x SLOWER than serial).
    """
    slot, lo, hi = args
    T, alive, m = _W["T"], _W["alive"], _W["m"]
    out = _W["out"][slot]
    out[...] = 0
    a, b, c = T[lo:hi, 0], T[lo:hi, 1], T[lo:hi, 2]
    w = (alive[a] & alive[b] & alive[c]).astype(np.int64)
    np.add.at(out, a, w)
    np.add.at(out, b, w)
    np.add.at(out, c, w)


_W = {}


def _init_worker(spec):
    """Attach the worker to the parent's shared triangle list and alive mask.

    `alive` has to be SHARED, not passed at pool creation: it changes every
    round, and a copy captured at fork time would leave workers computing
    support against a stale mask.
    """
    nT, m, procs, tname, aname, oname = spec
    _W["shm_T"] = shared_memory.SharedMemory(name=tname)
    _W["shm_a"] = shared_memory.SharedMemory(name=aname)
    _W["shm_o"] = shared_memory.SharedMemory(name=oname)
    _W["T"] = np.ndarray((nT, 3), dtype=np.int64, buffer=_W["shm_T"].buf)
    _W["alive"] = np.ndarray((m,), dtype=np.bool_, buffer=_W["shm_a"].buf)
    _W["out"] = np.ndarray((procs, m), dtype=np.int64, buffer=_W["shm_o"].buf)
    _W["m"] = m


def peel(T, m, procs=1):
    """Trussness per undirected edge, support recomputed each round.

    With procs > 1 the per-round support pass is split across workers over the
    triangle list; the alive mask lives in shared memory so every worker sees
    the current round's state.
    """
    K = np.zeros(m, dtype=np.int64)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    nT = T.shape[0]
    rounds = 0

    parallel = procs > 1 and nT > (1 << 16)
    shm_T = shm_a = shm_o = pool = None
    if parallel:
        shm_T = shared_memory.SharedMemory(create=True, size=max(T.nbytes, 8))
        Ts = np.ndarray(T.shape, dtype=np.int64, buffer=shm_T.buf)
        Ts[...] = T
        shm_a = shared_memory.SharedMemory(create=True, size=max(m, 8))
        alive = np.ndarray((m,), dtype=np.bool_, buffer=shm_a.buf)
        alive[...] = True
        shm_o = shared_memory.SharedMemory(create=True,
                                           size=max(procs * m * 8, 8))
        out = np.ndarray((procs, m), dtype=np.int64, buffer=shm_o.buf)
        step = -(-nT // procs)
        bounds = [(i, s0, min(s0 + step, nT))
                  for i, s0 in enumerate(range(0, nT, step))]
        pool = mp.get_context("fork").Pool(
            procs, initializer=_init_worker,
            initargs=((nT, m, procs, shm_T.name, shm_a.name, shm_o.name),))
    else:
        alive = np.ones(m, dtype=bool)

    try:
        live, k = m, 0
        while live:
            rounds += 1
            if parallel:
                pool.map(_support_chunk, bounds)
                sup = out[:len(bounds)].sum(axis=0)
            else:
                w = (alive[a] & alive[b] & alive[c]).astype(np.int64)
                sup = np.zeros(m, dtype=np.int64)
                np.add.at(sup, a, w)
                np.add.at(sup, b, w)
                np.add.at(sup, c, w)
            B = alive & (sup <= k)
            n = int(B.sum())
            if n == 0:
                k += 1
                continue
            K[B] = k
            alive &= ~B
            live -= n
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        for shm in (shm_T, shm_a, shm_o):
            if shm is not None:
                shm.close()
                shm.unlink()
    return K, rounds


def _mk_init(spec):
    nT, m, R, tname, aname, sname = spec
    _MK["shm_T"] = shared_memory.SharedMemory(name=tname)
    _MK["shm_a"] = shared_memory.SharedMemory(name=aname)
    _MK["shm_s"] = shared_memory.SharedMemory(name=sname)
    _MK["T"] = np.ndarray((nT, 3), dtype=np.int64, buffer=_MK["shm_T"].buf)
    _MK["alive"] = np.ndarray((R, m), dtype=np.bool_, buffer=_MK["shm_a"].buf)
    _MK["sup"] = np.ndarray((R, m), dtype=np.int64, buffer=_MK["shm_s"].buf)


_MK = {}


def _mk_rows(bounds):
    """Support for run rows [r0, r1) over the whole triangle list.

    The RUN axis is split, not the triangle list: each worker then owns disjoint
    rows of `sup` and writes them in place, so there is no cross-worker
    reduction and nothing has to be pickled back. Splitting triangles instead
    would need a (procs, R, m) buffer and a summation -- 7 GB at 32 workers on
    amazon-2008.
    """
    r0, r1 = bounds
    T, alive, sup = _MK["T"], _MK["alive"], _MK["sup"]
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    for r in range(r0, r1):
        av = alive[r]
        w = (av[a] & av[b] & av[c]).astype(np.int64)
        out = sup[r]
        out[...] = 0
        np.add.at(out, a, w)
        np.add.at(out, b, w)
        np.add.at(out, c, w)


def peel_multik(T, m, chunk=8, procs=1):
    """Multi-$k$ peel: a chunk of thresholds shares one pass per round."""
    K = np.zeros(m, dtype=np.int64)
    nT = T.shape[0]
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    aliveg = np.ones(m, dtype=bool)
    base_k, width, rounds = 0, max(chunk, procs), 0

    shm_T = None
    if procs > 1:
        shm_T = shared_memory.SharedMemory(create=True, size=max(T.nbytes, 8))
        np.ndarray(T.shape, dtype=np.int64, buffer=shm_T.buf)[...] = T
    try:
        while aliveg.any():
            R = max(1, min(width, RUN_TRIANGLE_CAP // max(nT, 1)))
            width *= 2
            Th = np.arange(base_k, base_k + R, dtype=np.int64)[:, None]

            par = procs > 1 and R >= 2 and nT > (1 << 16)
            if par:
                shm_a = shared_memory.SharedMemory(create=True, size=max(R * m, 8))
                shm_s = shared_memory.SharedMemory(create=True, size=max(R * m * 8, 8))
                alive = np.ndarray((R, m), dtype=np.bool_, buffer=shm_a.buf)
                sup = np.ndarray((R, m), dtype=np.int64, buffer=shm_s.buf)
                alive[...] = aliveg
                nw = min(procs, R)
                step = -(-R // nw)
                bounds = [(i, min(i + step, R)) for i in range(0, R, step)]
                pool = mp.get_context("fork").Pool(
                    len(bounds), initializer=_mk_init,
                    initargs=((nT, m, R, shm_T.name, shm_a.name, shm_s.name),))
            else:
                alive = np.broadcast_to(aliveg, (R, m)).copy()
                sup = np.zeros((R, m), dtype=np.int64)

            try:
                while True:
                    rounds += 1
                    if par:
                        pool.map(_mk_rows, bounds)
                    else:
                        for r in range(R):
                            av = alive[r]
                            w = (av[a] & av[b] & av[c]).astype(np.int64)
                            sup[r] = 0
                            np.add.at(sup[r], a, w)
                            np.add.at(sup[r], b, w)
                            np.add.at(sup[r], c, w)
                    B = alive & (sup <= Th)
                    if not B.any():
                        break
                    alive &= ~B
                dead = ~alive
                hit = aliveg & dead.any(0)
                K[hit] = (dead.argmax(0) + base_k)[hit]
                aliveg = aliveg & alive[R - 1]
            finally:
                if par:
                    pool.close(); pool.join()
                    for shm in (shm_a, shm_s):
                        shm.close(); shm.unlink()
            base_k += R
    finally:
        if shm_T is not None:
            shm_T.close(); shm_T.unlink()
    return K, rounds


def run(path, load=0.45, procs=1, chunk=0):
    t = timer()
    S, E, D, nV = load_mtx(path)
    t_load = timer() - t

    t = timer()
    s, e, d, o = orient(S, E, D, nV)
    m = int(s.size)
    codes = cantor(s, e)
    table = Cuckoo(codes, np.arange(m, dtype=np.int64), load=load)
    t_build = timer() - t

    t = timer()
    T = triangles(s, e, d, o, table, m, procs)
    t_tri = timer() - t

    t = timer()
    K, rounds = (peel_multik(T, m, chunk, procs) if chunk
                 else peel(T, m, procs))
    t_peel = timer() - t

    return {
        "file": os.path.basename(path),
        "vertex_count": nV,
        "edge_count": int(S.size),
        "undirected_edges": m,
        "triangle_count": int(T.shape[0]),
        "K_max": int(K.max()) if K.size else 0,
        "K_avg": round(float(K.mean()), 2) if K.size else 0.0,
        "load_time_sec": round(t_load, 4),
        "table_build_time_sec": round(t_build, 4),
        "triangle_time_sec": round(t_tri, 4),
        "peeling_time_sec": round(t_peel, 4),
        "total_time_sec": round(t_build + t_tri + t_peel, 4),
        "rounds": rounds,
        "table_passes": table.passes,
        "table_bytes": table.bytes(),
        "load_factor": load,
        "procs": procs,
        "chunk": chunk,
    }, K


CSV_HEADER = ("dataset,vertices,edges,undirected_edges,triangle_count,"
              "table_build_time_sec,triangle_time_sec,peeling_time_sec,"
              "total_time_sec,rounds,table_passes,procs,chunk,k_max,k_avg\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--load", type=float, default=0.45,
                    help="cuckoo table load factor (lower = fewer passes)")
    ap.add_argument("--procs", type=int, default=1,
                    help="worker processes for the support pass")
    ap.add_argument("--chunk", type=int, default=0,
                    help="multi-k run-axis width (0 = one k at a time)")
    ap.add_argument("--check", action="store_true",
                    help="verify trussness against ktruss_hash.py")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    res, K = run(args.graph, args.load, args.procs, args.chunk)

    if args.check:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "kh", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ktruss_hash.py"))
        kh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kh)
        ref = kh.ktruss_hash(args.graph, procs=1)
        for field in ("triangle_count", "K_max"):
            if int(ref[field]) != int(res[field]):
                raise SystemExit(f"{field} mismatch: cantor={res[field]} "
                                 f"ktruss_hash={ref[field]}")
        if abs(float(ref["K_avg"]) - float(res["K_avg"])) > 0.01:
            raise SystemExit(f"K_avg mismatch: cantor={res['K_avg']} "
                             f"ktruss_hash={ref['K_avg']}")
        res["checked_against"] = "ktruss_hash.py"

    if args.json or not args.csv_out:
        print(json.dumps(res, indent=4))

    if args.csv_out:
        new = not os.path.exists(args.csv_out)
        with open(args.csv_out, "a") as fh:
            if new:
                fh.write(CSV_HEADER)
            fh.write(f"{res['file']},{res['vertex_count']},{res['edge_count']},"
                     f"{res['undirected_edges']},{res['triangle_count']},"
                     f"{res['table_build_time_sec']},{res['triangle_time_sec']},"
                     f"{res['peeling_time_sec']},{res['total_time_sec']},"
                     f"{res['rounds']},{res['table_passes']},{res['procs']},{res['chunk']},"
                     f"{res['K_max']},{res['K_avg']}\n")


if __name__ == "__main__":
    main()
