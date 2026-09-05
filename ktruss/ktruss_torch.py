#!/usr/bin/env python3
"""PyTorch k-truss, with the unrolled-threshold peel of the vectorised k-core.

GPU counterpart of ktruss/ktruss_hash.py, and of the C++ `seg` kernel that
lives in the gpu-nucleus repository.
The trussness of an edge is the largest k such that the edge survives repeated
removal of every edge whose support (triangle count) is <= k.

The reference recomputes support from scratch on the residual graph every
round, which is correct but re-enumerates every wedge each time -- and the
memory of that work says the bottleneck is the ROUND COUNT, not the residual
size. Both peels here therefore keep the reference's semantics but pay the
enumeration only once:

  * Triangles are enumerated up front into a fixed list of edge-id triples.
    Which triples exist is a property of the input graph and never changes;
    only which of their edges are still alive does. So a round is a scan of
    that list, not a re-expansion of the graph.
  * `base` walks k = 0, 1, 2, ... exactly like ktruss_hash.py.
  * `multi` is the k-core trick of Chapter 3 (kcore_kcd_torch.py in the
    gpu-nucleus repository). Peeling at a fixed
    threshold j from the whole graph removes exactly the edges of trussness
    <= j, so the smallest j whose run removes e IS trussness(e). The runs are
    independent, so a chunk of thresholds is peeled in ONE set of tensor ops
    with a leading run axis and the chunk costs the deepest single run rather
    than the sum. Unlike k-core, though, a round here is O(|triangles|) rather
    than a handful of tiny kernels, so the run axis multiplies real work --
    whether that pays is the question this file exists to answer.

Support convention matches ktruss_hash.py: one value per UNDIRECTED edge, and
sup[e] is the number of triangles containing e.

Usage:
    ktruss_torch.py <graph.mtx> [--outer base,multi] [--check] [--device cuda]
                    [--chunk N] [--budget N] [--repeat N] [--csv-out F]
"""

import argparse
import json
import os
import sys
from timeit import default_timer as timer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphio import load_mtx, offsets  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Ceiling on chunk * triangle_count: the multi peel holds an (R, nT) weight
# vector per round, so this is what keeps a wide run axis from exhausting the
# device on triangle-dense graphs.
RUN_TRIANGLE_CAP = 200_000_000


def load_csr(path):
    t = timer()
    S, E, CD, nV = load_mtx(path)
    O = offsets(CD)
    return S, E, CD, O, nV, int(S.size), timer() - t


def build_index(S, E, CD, O, nV, dev):
    """Edge keys, reverse-edge map and directed -> undirected edge ids.

    key(s,d) = mxx[s] + d with mxx the prefix sum of (max_neighbour(s) + 1).
    The edge list is sorted by (src,dst), so the key array is sorted too and an
    edge lookup is one searchsorted rather than a hash probe -- the same
    formulation ktruss_hash.py uses.
    """
    St = torch.from_numpy(S.astype(np.int64)).to(dev)
    Et = torch.from_numpy(E.astype(np.int64)).to(dev)
    Dt = torch.from_numpy(CD.astype(np.int64)).to(dev)
    Ot = torch.from_numpy(O[:nV].astype(np.int64)).to(dev)
    nE = St.numel()

    val = torch.zeros(nV, dtype=torch.int64, device=dev)
    nz = Dt > 0
    val[nz] = Et[Ot[nz] + Dt[nz] - 1]           # largest neighbour of each v
    span = torch.where(nz, val + 1, torch.zeros_like(val))
    mxx = torch.zeros(nV + 1, dtype=torch.int64, device=dev)
    torch.cumsum(span, 0, out=mxx[1:])
    key = mxx[St] + Et

    # reverse edge, then one undirected id per pair
    rev = torch.searchsorted(key, mxx[Et] + St)
    can = St < Et
    uid = torch.zeros(nE, dtype=torch.int64, device=dev)
    m = int(can.sum())
    ids = torch.arange(m, dtype=torch.int64, device=dev)
    uid[can] = ids
    uid[rev[can]] = ids
    return St, Et, Dt, Ot, val, mxx, key, uid, m


def triangles(St, Et, Dt, Ot, val, mxx, key, uid, nE, budget, dev):
    """Every triangle once, as a sorted triple of undirected edge ids.

    A wedge s -> d -> x closes into a triangle when (s,x) is an edge. Rooting
    the expansion at each directed edge (s,d) finds each triangle six times
    (once per directed wedge), so the triples are sorted and deduplicated.
    """
    out = []
    step = max(1, budget // max(1, int(Dt.max())))
    for lo in range(0, nE, step):
        hi = min(lo + step, nE)
        s = St[lo:hi]
        d = Et[lo:hi]
        cnt = Dt[d]
        total = int(cnt.sum())
        if total == 0:
            continue
        # expand adj(d): grp indexes back into the [lo,hi) edge slice
        grp = torch.repeat_interleave(
            torch.arange(hi - lo, device=dev), cnt, output_size=total)
        excl = torch.cumsum(cnt, 0) - cnt
        pos = Ot[d][grp] + (torch.arange(total, device=dev) - excl[grp])
        dn = Et[pos]
        s_g = s[grp]

        # past val[s] no edge (s, dn) can exist, so drop those before probing
        keep = dn <= val[s_g]
        if not bool(keep.any()):
            continue
        pos, dn, s_g, grp = pos[keep], dn[keep], s_g[keep], grp[keep]

        q = mxx[s_g] + dn
        idx = torch.searchsorted(key, q).clamp_(max=nE - 1)
        hit = key[idx] == q
        if not bool(hit.any()):
            continue
        # the triangle's three edges: (s,d), (d,dn), (s,dn)
        tri = torch.stack((uid[lo + grp[hit]], uid[pos[hit]], uid[idx[hit]]), 1)
        out.append(torch.sort(tri, dim=1).values)

    if not out:
        return torch.zeros((0, 3), dtype=torch.int64, device=dev)
    tri = torch.cat(out) if len(out) > 1 else out[0]
    return torch.unique(tri, dim=0)


def peel(T, m, dev, outer="base", chunk=8):
    """Return (K, rounds). K[e] is the trussness of undirected edge e.

    A round rebuilds support by scanning the triangle list and counting only
    the triangles whose three edges are all still alive -- the same "recompute
    from scratch" semantics as ktruss_hash.py, but over a list that was built
    once instead of a wedge expansion repeated every round.
    """
    K = torch.zeros(m, dtype=torch.int64, device=dev)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    nT = T.shape[0]
    rounds = 0

    if outer == "base":
        sup = torch.zeros(m, dtype=torch.int64, device=dev)
        alive = torch.ones(m, dtype=torch.bool, device=dev)
        live = m
        k = 0
        while live:
            rounds += 1
            # weight 0/1 per triangle: no compaction, so no data dependent size
            w = (alive[a] & alive[b] & alive[c]).to(torch.int64)
            sup.zero_()
            sup.index_add_(0, a, w)
            sup.index_add_(0, b, w)
            sup.index_add_(0, c, w)
            B = alive & (sup <= k)
            n = int(B.sum())
            if n == 0:
                k += 1
                continue
            K[B] = k
            alive &= ~B
            live -= n
        return K, rounds

    # ---- multi: a chunk of thresholds peeled together on a leading run axis
    aliveg = torch.ones(m, dtype=torch.bool, device=dev)   # global survivors
    base_k, width = 0, chunk
    while bool(aliveg.any()):
        # The per-round weight vector is R * nT int64, which is what actually
        # bounds the run axis here -- on a dense graph nT dwarfs the edge count,
        # so a chunk that would be harmless for k-core is not harmless here.
        R = max(1, min(width, RUN_TRIANGLE_CAP // max(nT, 1)))
        Th = torch.arange(base_k, base_k + R, dtype=torch.int64,
                          device=dev).view(R, 1)
        alive = aliveg.unsqueeze(0).expand(R, m).clone()
        # index views, not copies: scatter_add_ walks them by stride
        ia = a.unsqueeze(0).expand(R, nT)
        ib = b.unsqueeze(0).expand(R, nT)
        ic = c.unsqueeze(0).expand(R, nT)
        sup = torch.zeros(R, m, dtype=torch.int64, device=dev)
        while True:
            rounds += 1
            w = (alive[:, a] & alive[:, b] & alive[:, c]).to(torch.int64)
            sup.zero_()
            sup.scatter_add_(1, ia, w)
            sup.scatter_add_(1, ib, w)
            sup.scatter_add_(1, ic, w)
            B = alive & (sup <= Th)
            if int(B.sum()) == 0:
                break
            alive &= ~B
        # lowest threshold that removed an edge is its trussness
        dead = ~alive
        hit = aliveg & dead.any(0)
        K[hit] = (dead.to(torch.uint8).argmax(0) + base_k)[hit]
        aliveg = aliveg & alive[R - 1]
        base_k += R
        width *= 2
    return K, rounds


def ktruss(csr, device="cuda", outer="base", chunk=8, budget=1 << 24):
    S, E, CD, O, nV, nE, load = csr
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.zeros(1, device=dev).add_(1)
        torch.cuda.synchronize()

    t = timer()
    St, Et, Dt, Ot, val, mxx, key, uid, m = build_index(S, E, CD, O, nV, dev)
    T = triangles(St, Et, Dt, Ot, val, mxx, key, uid, nE, budget, dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    setup = timer() - t

    t0 = timer()
    K, rounds = peel(T, m, dev, outer, chunk)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = timer() - t0

    Kh = K.cpu().numpy()
    return Kh, {
        "outer": outer, "vertex_count": nV, "edge_count": nE,
        "undirected_edges": m, "triangle_count": int(T.shape[0]),
        "K_max": int(Kh.max()) if Kh.size else 0,
        "K_avg": round(float(Kh.mean()), 2) if Kh.size else 0.0,
        "peeling_time_sec": round(elapsed, 4),
        "setup_time_sec": round(setup, 4),
        "load_time_sec": round(load, 4),
        "rounds": rounds, "chunk": chunk,
        "device_name": (torch.cuda.get_device_name(dev) if dev.type == "cuda"
                        else "cpu"),
        "torch": torch.__version__,
    }


CSV_HEADER = ("dataset,outer,peeling_time_sec,speedup,rounds,setup_time_sec,"
              "triangle_count,k_max,k_avg,vertices,edges,undirected_edges,"
              "chunk,device_name\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outer", default="base,multi")
    ap.add_argument("--chunk", type=int, default=8,
                    help="thresholds peeled together by 'multi'; memory scales "
                         "with chunk * triangle_count")
    ap.add_argument("--budget", type=int, default=1 << 24)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    csr = load_csr(args.graph)
    name = os.path.basename(args.graph)
    variants = [v for v in args.outer.split(",") if v]
    bad = [v for v in variants if v not in ("base", "multi")]
    if bad:
        raise SystemExit(f"unknown outer loop(s): {','.join(bad)}")

    ref, rows = None, []
    for v in variants:
        best = None
        for _ in range(max(1, args.repeat)):
            Kh, r = ktruss(csr, args.device, v, args.chunk, args.budget)
            if best is None or r["peeling_time_sec"] < best["peeling_time_sec"]:
                best = r
        if args.check:
            if ref is None:
                ref = Kh
            elif not np.array_equal(Kh, ref):
                bad_n = int((Kh != ref).sum())
                raise SystemExit(f"outer {v}: trussness mismatch on {bad_n} edges")
        rows.append(best)

    base = next((r["peeling_time_sec"] for r in rows if r["outer"] == "base"),
                rows[0]["peeling_time_sec"])
    if args.json:
        for r in rows:
            r["speedup"] = round(base / r["peeling_time_sec"], 3)
        print(json.dumps(rows, indent=4))
    else:
        print(f"{name}  nV={rows[0]['vertex_count']}  m={rows[0]['undirected_edges']}"
              f"  triangles={rows[0]['triangle_count']}  setup={rows[0]['setup_time_sec']}s"
              f"  {rows[0]['device_name']}")
        print(f"{'outer':>8} {'time_sec':>10} {'speedup':>9} {'rounds':>8} {'k_max':>7}")
        for r in rows:
            print(f"{r['outer']:>8} {r['peeling_time_sec']:10.4f} "
                  f"{base / r['peeling_time_sec']:8.2f}x {r['rounds']:8d} "
                  f"{r['K_max']:7d}")

    if args.csv_out:
        new = not os.path.exists(args.csv_out)
        with open(args.csv_out, "a") as fh:
            if new:
                fh.write(CSV_HEADER)
            for r in rows:
                fh.write(f"{name},{r['outer']},{r['peeling_time_sec']},"
                         f"{round(base / r['peeling_time_sec'], 3)},{r['rounds']},"
                         f"{r['setup_time_sec']},{r['triangle_count']},"
                         f"{r['K_max']},{r['K_avg']},{r['vertex_count']},"
                         f"{r['edge_count']},{r['undirected_edges']},"
                         f"{r['chunk']},{r['device_name']}\n")


if __name__ == "__main__":
    main()
