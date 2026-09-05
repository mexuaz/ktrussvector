#!/usr/bin/env python3
"""PyTorch k-truss: Cantor/Cuckoo triangulation with the Algorithm 7 peel.

GPU counterpart of ktruss/ktruss_cantor_scale_arr.py. Same formulation,
same three phases, same deviations from the book algorithm -- only the array
library changes:

  * orient by degree, Cantor-pair each edge, hold the codes in a Cuckoo table,
    and close every wedge with two vectorised probes;
  * peel with Algorithm 7 (`ps-peeling`) of Chapter 4: carry the triangle-edge
    incidence array T, count it by run lengths, peel, and *shrink* T by the
    triangles the peeled edges destroyed, so round cost tracks the triangles
    still alive rather than the original triangle count.

What is deliberately NOT ported. The NumPy file spends most of its length on a
worker-process peel: T is cut on run boundaries so each worker owns a
contiguous range of edges, writes only its own K entries, and never takes part
in a reduction. That structure exists because multiprocessing makes any
cross-worker sum cost O(nw * m) per round -- 155 GB of reduction traffic at 96
workers on soc-Slashdot0811. None of it applies here. A CUDA kernel launch
already fans the whole incidence array across the device, so the segment split,
the shared-memory plumbing, `_worker_count()` and the `workers` column all
collapse into "one segment, one stream". This file is therefore the NumPy
file's SERIAL path with torch tensors, which is the whole point: the GPU costs
a change of array library, not a change of algorithm.

The same three deviations from Algorithm 7 as the NumPy version, for the same
reasons:

(1) `T_c == k` is `T_c <= k`. Removing a batch of edges can drop another edge's
    count by more than one, so a count can step from k+2 to k-1 without ever
    equalling k; under `== k` such an edge matches no branch again, is never
    assigned a truss value and never leaves T, and the outer loop does not
    terminate.
(2) `K[T_h] = K[T_h] + 1` is INERT -- every edge in T_h is later overwritten by
    `K[T_k] = k` -- but costs a scatter over every surviving edge every round.
    Off by default, restored with --alg7-increment, under which the result is
    bit-identical.
(3) Counts come from run boundaries of a sorted T, not `unique(T)` or a
    bincount: O(|T|) with no term in the edge count m, and compaction (which
    only deletes) preserves the sortedness this relies on.

Edges in no triangle never enter T and keep truss value 0, which agrees with
ktruss_cantor.py peeling them in its k=0 round.

What a round costs on a GPU. Every round is a handful of full-width passes over
the live incidence array -- a run-boundary scan, a comparison, two gathers and
a compaction -- so the arithmetic is exactly the NumPy version's. The one cost
the CPU version does not pay is a device synchronisation per round: the loop
has to read back whether anything peeled before it can decide to raise k. That
makes the ROUND COUNT, not the triangle count, what a graph is punished for
here, which is the same conclusion ktruss_torch.py reached from the other
direction.

Correctness is checked, not assumed. --check re-runs the whole pipeline on the
CPU with ktruss_cantor.py (NumPy) and compares the triangle SET and the
trussness of every edge elementwise, so a disagreement in either the
triangulation or the peel is caught.

Usage:
    ktruss_cantor_arr_torch.py <graph.mtx> [--device cuda] [--load F]
                               [--budget N] [--repeat N] [--alg7-increment]
                               [--check] [--json] [--csv F]
"""

import argparse
import json
import os
import sys
from timeit import default_timer as timer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphio import load_mtx  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ktruss_cantor as kc  # noqa: E402

# Odd multipliers for the two independent hash functions, identical to
# ktruss_cantor.Cuckoo. Odd keeps the low bits from collapsing, and the two
# differ so a pair colliding in one table is unlikely to collide in the other.
_H0 = 0x9E3779B97F4A7C15 & ((1 << 62) - 1)
_H1 = 0xC2B2AE3D27D4EB4F & ((1 << 62) - 1)
_MASK = (1 << 62) - 1


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def cantor(u, v):
    """Injective code for the undirected edge {u, v}.

    pi(a,b) = (a+b)(a+b+1)/2 + b on the ORDERED pair (min, max), so {u,v} and
    {v,u} land on the same code without a separate canonicalisation step.
    """
    a = torch.minimum(u, v)
    b = torch.maximum(u, v)
    s = a + b
    return (s * (s + 1)) // 2 + b


class Cuckoo:
    """Two-table Cuckoo hash over int64 keys, built and probed in batches.

    ktruss_cantor.Cuckoo with one change forced by the GPU: insertion
    alternates tables, every homeless key is scattered into the current table,
    and both the losers of a same-slot clash and the residents they displaced
    retry in the other table next pass. The NumPy version decides who won by
    scattering key and val and reading the key back; that is not safe on CUDA,
    where the two scatters are independent kernels. _build() therefore resolves
    each slot on a separate owner array first and writes key and val from the
    single element that owns it -- see the comment there.
    """

    EMPTY = -1

    def __init__(self, keys, vals, dev, load=0.45, max_pass=200):
        self.dev = dev
        for attempt in range(4):
            if self._build(keys, vals, load / (2 ** attempt), max_pass):
                self.load_used = load / (2 ** attempt)
                return
        raise RuntimeError("cuckoo build did not converge after 4 rehashes")

    def _build(self, keys, vals, load, max_pass):
        n = max(int(keys.numel() / load), 8)
        self.m = n | 1                            # odd size, better spread
        self.key = [torch.full((self.m,), self.EMPTY, dtype=torch.int64,
                               device=self.dev) for _ in range(2)]
        self.val = [torch.zeros(self.m, dtype=torch.int64, device=self.dev)
                    for _ in range(2)]
        # Which source element owns each slot this pass; see the loop below.
        owner = torch.empty(self.m, dtype=torch.int64, device=self.dev)

        k, v = keys, vals
        t, self.passes = 0, 0
        for _ in range(max_pass):
            if k.numel() == 0:
                return True
            self.passes += 1
            pos = self._slot(t, k)
            src = torch.arange(k.numel(), dtype=torch.int64, device=self.dev)

            # Read the residents BEFORE scattering, then re-insert whoever got
            # displaced. Writing only into empty slots would deadlock, and
            # scattering without capturing the residents would lose them
            # silently -- every later probe for such a key would miss and
            # triangles would quietly go uncounted.
            prev_k = self.key[t][pos]
            prev_v = self.val[t][pos]

            # Resolve the same-slot clash ONCE, on an owner array, and derive
            # BOTH writes from the winner. The NumPy version can scatter key
            # and val independently and read the winner back, because CPU
            # index_put_ applies duplicate indices in order and so the same
            # element wins both. On CUDA those are two separate kernels and
            # each picks its winner independently: slot p can end up holding
            # element i's KEY beside element j's VALUE. Such a slot still
            # probes as a hit and returns the WRONG edge id, which does not
            # change the triangle COUNT at all -- job 20016225 preflighted
            # 551724 triangles against NumPy's 551724 with the two sets
            # unequal. pos[w] below is duplicate-free, so both writes land
            # from the one element that owns the slot.
            owner.fill_(-1)
            owner[pos] = src
            mine = owner[pos] == src
            w = src[mine]
            self.key[t][pos[w]] = k[w]
            self.val[t][pos[w]] = v[w]

            # A key equal to the slot's new resident needs no retry: the table
            # already answers for it. That is NumPy's "last writer wins" for
            # duplicate keys, and without it such a key would be retried for
            # ever. The Cantor codes here are unique, so it never fires.
            won = mine | (self.key[t][pos] == k)
            lost = ~won                           # lost a same-slot clash
            eject = mine & (prev_k != self.EMPTY) & (prev_k != k)
            k = torch.cat((k[lost], prev_k[eject]))
            v = torch.cat((v[lost], prev_v[eject]))
            t ^= 1
        return k.numel() == 0

    def _slot(self, t, k):
        # keys are already well spread by the pairing, so a multiply-shift is
        # enough; the mask keeps the product non-negative in int64
        h = _H0 if t == 0 else _H1
        return ((k * h) & _MASK) % self.m

    def lookup(self, q):
        """Vectorised probe: (found, value) for every query, two gathers."""
        p0 = self._slot(0, q)
        hit0 = self.key[0][p0] == q
        p1 = self._slot(1, q)
        hit1 = self.key[1][p1] == q
        val = torch.where(hit0, self.val[0][p0], self.val[1][p1])
        return hit0 | hit1, val

    def bytes(self):
        return sum(a.numel() * a.element_size() for a in self.key + self.val)


def orient(S, E, nV, dev):
    """Keep only u < v, so each undirected edge appears once.

    load_mtx already sorts by (source, destination) and the mask preserves that
    order, but the two stable sorts are kept: they are the lexsort the NumPy
    version does, they cost one pass, and they mean this function does not
    depend on a guarantee made in another file.
    """
    St = torch.from_numpy(S.astype(np.int64)).to(dev)
    Et = torch.from_numpy(E.astype(np.int64)).to(dev)
    keep = St < Et
    s, e = St[keep], Et[keep]
    idx = torch.argsort(e, stable=True)
    s, e = s[idx], e[idx]
    idx = torch.argsort(s, stable=True)
    s, e = s[idx], e[idx]
    d = torch.bincount(s, minlength=nV).to(torch.int64)
    o = torch.cat((torch.zeros(1, dtype=torch.int64, device=dev),
                   torch.cumsum(d, 0)[:-1]))
    return s, e, d, o


def triangles(s, e, d, o, table, budget=1 << 24):
    """Every triangle once, as a triple of undirected edge ids.

    The graph is oriented (u<v), so walking w over adj(v) from the edge (u,v)
    and probing for (u,w) finds each triangle exactly once: u < v < w. No
    deduplication pass is needed.

    The wedge expansion is the memory peak of the whole program, so it is cut
    into blocks whose expanded length stays under `budget`. The block bounds
    come from a searchsorted over the cumulative wedge count rather than the
    NumPy version's element-at-a-time accumulation -- same blocks, but chosen
    without a Python loop over edges.
    """
    dev = s.device
    nE = e.numel()
    if nE == 0:
        return torch.zeros((0, 3), dtype=torch.int64, device=dev)

    cnt_all = d[e]
    cw = torch.cumsum(cnt_all, 0)
    cw_h = cw.cpu()                               # bounds are host-side control

    out = []
    lo = 0
    while lo < nE:
        base = int(cw_h[lo - 1]) if lo else 0
        hi = int(torch.searchsorted(cw_h, base + budget, right=True))
        hi = min(max(hi, lo + 1), nE)             # a single fat edge still fits
        total = int(cw_h[hi - 1]) - base
        if total:
            cnt = cnt_all[lo:hi]
            grp = torch.repeat_interleave(
                torch.arange(hi - lo, device=dev), cnt, output_size=total)
            excl = torch.cumsum(cnt, 0) - cnt
            pos = o[e[lo:hi]][grp] + (torch.arange(total, device=dev)
                                      - excl[grp])
            found, idx_uw = table.lookup(cantor(s[lo + grp], e[pos]))
            if bool(found.any()):
                out.append(torch.stack((lo + grp[found], pos[found],
                                        idx_uw[found]), 1))
        lo = hi

    if not out:
        return torch.zeros((0, 3), dtype=torch.int64, device=dev)
    return torch.cat(out) if len(out) > 1 else out[0]


def _runs(Tv):
    """(start, value, length) of every run in a sorted array.

    On the sorted incidence array this is exactly Algorithm 7's
    `T_u, T_c = unique(T)`, at O(|T|) and with no array of size m.
    """
    n = Tv.numel()
    if n == 0:
        e = torch.zeros(0, dtype=torch.int64, device=Tv.device)
        return e, e, e
    head = torch.ones(1, dtype=torch.bool, device=Tv.device)
    b = torch.nonzero(torch.cat((head, Tv[1:] != Tv[:-1])),
                      as_tuple=True)[0]
    end = torch.full((1,), n, dtype=torch.int64, device=Tv.device)
    return b, Tv[b], torch.diff(torch.cat((b, end)))


def peel_arr(G, m, dev, increment=False):
    """Trussness per undirected edge. Returns (K, rounds).

    Algorithm 7 over a shrinking incidence array: count the runs of a sorted T,
    peel every edge whose count is <= k, kill the triangles those edges were
    in, and compact them out of T so the next round only touches survivors.
    """
    K = torch.zeros(m, dtype=torch.int64, device=dev)
    nT = int(G.shape[0])
    if nT == 0:
        return K, 0

    # Sorted incidence array plus the triangle each incidence belongs to.
    Tv = torch.cat((G[:, 0], G[:, 1], G[:, 2]))
    Tt = torch.arange(nT, dtype=torch.int64, device=dev).repeat(3)
    order = torch.argsort(Tv, stable=True)
    Tv, Tt = Tv[order], Tt[order]

    alive_tri = torch.ones(nT, dtype=torch.bool, device=dev)
    Ev = _runs(Tv)[1].clone()          # alive edges; edges absent keep K = 0

    k, rounds = 1, 0
    while Ev.numel():
        while True:
            rounds += 1
            _, u, c = _runs(Tv)

            peel = c <= k                         # T_c <= k, see docstring (1)
            Tk = u[peel]
            if u.numel():
                # alive here but no incidence left: count 0, so peel now
                # rather than strand the edge
                i = torch.searchsorted(u, Ev).clamp_(max=u.numel() - 1)
                drop = Ev[u[i] != Ev]
            else:
                drop = Ev
            if Tk.numel() + drop.numel() == 0:
                break

            K[Tk] = k
            if drop.numel():
                K[drop] = k
            surv = u[~peel]                       # T_h
            if increment and surv.numel():
                K[surv] += 1                      # Algorithm 7 line 8; inert
            Ev = surv

            if Tk.numel():
                # Expand the per-run peel mask to per-incidence, kill those
                # triangles, then drop their incidences. Deletion preserves
                # sortedness, which the next round relies on.
                dead = torch.repeat_interleave(peel, c, output_size=Tv.numel())
                alive_tri[Tt[dead]] = False
                keep = alive_tri[Tt]
                Tv, Tt = Tv[keep], Tt[keep]
        k += 1

    return K, rounds


def run(path, device="cuda", load=0.45, budget=1 << 24, increment=False,
        csr=None):
    """Same phase breakdown and result keys as ktruss_cantor_scale_arr.run()."""
    dev = torch.device(device)
    if csr is None:
        t = timer()
        csr = load_mtx(path)
        t_load = timer() - t
    else:
        t_load = 0.0
    S, E, _D, nV = csr

    if dev.type == "cuda":                        # pay context creation first
        torch.zeros(1, device=dev).add_(1)
        _sync(dev)

    t = timer()
    s, e, d, o = orient(S, E, nV, dev)
    m = int(s.numel())
    table = Cuckoo(cantor(s, e),
                   torch.arange(m, dtype=torch.int64, device=dev), dev,
                   load=load)
    _sync(dev)
    t_build = timer() - t

    t = timer()
    G = triangles(s, e, d, o, table, budget)
    _sync(dev)
    t_tri = timer() - t

    t = timer()
    K, rounds = peel_arr(G, m, dev, increment)
    _sync(dev)
    t_peel = timer() - t

    Kh = K.cpu().numpy()
    return {
        "file": os.path.basename(path),
        "vertex_count": nV,
        "edge_count": int(S.size),
        "undirected_edges": m,
        "triangle_count": int(G.shape[0]),
        "K_max": int(Kh.max()) if Kh.size else 0,
        "K_avg": round(float(Kh.mean()), 2) if Kh.size else 0.0,
        "load_time_sec": round(t_load, 4),
        "table_build_time_sec": round(t_build, 4),
        "triangle_time_sec": round(t_tri, 4),
        "peeling_time_sec": round(t_peel, 4),
        "total_time_sec": round(t_build + t_tri + t_peel, 4),
        "rounds": rounds,
        "alg7_increment": int(bool(increment)),
        "table_passes": table.passes,
        "table_bytes": table.bytes(),
        "load_factor": load,
        "budget": budget,
        "device": dev.type,
        "device_name": (torch.cuda.get_device_name(dev) if dev.type == "cuda"
                        else "cpu"),
        "torch": torch.__version__,
    }, Kh, G


def check(path, res, Kh, G, load):
    """Assert the triangle SET and every trussness match NumPy's, elementwise.

    The reference is ktruss_cantor.py run end to end on the CPU, so a
    disagreement in the triangulation and a disagreement in the peel are both
    caught -- and reported separately, because they mean different bugs.
    """
    S, E, D, nV = kc.load_mtx(path)
    s, e, d, o = kc.orient(S, E, D, nV)
    m = int(s.size)
    table = kc.Cuckoo(kc.cantor(s, e), np.arange(m, dtype=np.int64), load=load)
    Gref = kc.triangles(s, e, d, o, table, m, 1)
    Kref, _ = kc.peel(Gref, m, 1)

    def canon(T):
        T = np.sort(np.asarray(T), axis=1)
        return T[np.lexsort((T[:, 2], T[:, 1], T[:, 0]))] if T.shape[0] else T

    Gt = canon(G.cpu().numpy())
    Gr = canon(Gref)
    if Gt.shape != Gr.shape or not np.array_equal(Gt, Gr):
        raise SystemExit(
            f"triangle mismatch on {res['file']}: torch={Gt.shape[0]} "
            f"cantor={Gr.shape[0]}")
    if not np.array_equal(Kh, Kref):
        bad = np.flatnonzero(Kh != Kref)
        raise SystemExit(
            f"trussness mismatch on {res['file']}: {bad.size} of {m} edges "
            f"differ, first at edge {bad[0]} "
            f"(torch={Kh[bad[0]]}, cantor={Kref[bad[0]]})")
    res["checked_against"] = "ktruss_cantor.py"


FIELDS = ("dataset", "variant", "device", "device_name", "budget", "load",
          "table_build_sec", "triangle_sec", "peel_sec", "total_sec",
          "setup_sec", "load_sec", "rounds", "table_passes", "triangles",
          "k_max", "k_avg", "vertices", "undirected_edges", "alg7_increment",
          "torch")


def _row(path, res):
    return {
        "dataset": os.path.basename(path), "variant": "arr-torch",
        "device": res["device"], "device_name": res["device_name"],
        "budget": res["budget"], "load": res["load_factor"],
        "table_build_sec": res["table_build_time_sec"],
        "triangle_sec": res["triangle_time_sec"],
        "peel_sec": res["peeling_time_sec"],
        "total_sec": res["total_time_sec"],
        "setup_sec": round(res["table_build_time_sec"]
                           + res["triangle_time_sec"], 4),
        "load_sec": res["load_time_sec"],
        "rounds": res["rounds"], "table_passes": res["table_passes"],
        "triangles": res["triangle_count"], "k_max": res["K_max"],
        "k_avg": res["K_avg"], "vertices": res["vertex_count"],
        "undirected_edges": res["undirected_edges"],
        "alg7_increment": res["alg7_increment"], "torch": res["torch"],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--load", type=float, default=0.45,
                    help="cuckoo table load factor")
    ap.add_argument("--budget", type=int, default=1 << 24,
                    help="wedge expansion block ceiling")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat the run, keep the fastest total")
    ap.add_argument("--alg7-increment", action="store_true",
                    help="restore Algorithm 7's inert K[T_h] += 1")
    ap.add_argument("--check", action="store_true",
                    help="assert triangles and K match ktruss_cantor.py")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    t = timer()
    csr = load_mtx(args.graph)
    t_load = round(timer() - t, 4)

    best = None
    for _ in range(max(1, args.repeat)):
        res, Kh, G = run(args.graph, args.device, args.load, args.budget,
                         args.alg7_increment, csr=csr)
        res["load_time_sec"] = t_load
        if best is None or res["total_time_sec"] < best[0]["total_time_sec"]:
            best = (res, Kh, G)
    res, Kh, G = best

    if args.check:
        check(args.graph, res, Kh, G, args.load)

    row = _row(args.graph, res)
    if args.json:
        print(json.dumps(res, indent=4))
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
