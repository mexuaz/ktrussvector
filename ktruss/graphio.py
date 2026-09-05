#!/usr/bin/env python3
"""Graph input shared by the Python nucleus implementations.

These grew inside ktruss_hash.py; they live here now because every Python
implementation in this directory needs them. Every file in ktruss/ reads its
input through this module, so the .mtx parsing and the index conventions are
defined in exactly one place. (In the gpu-nucleus repository the k-core
implementations share it too.)

Conventions, which the callers depend on:

  * The CSR is UNDIRECTED and deduplicated -- every edge appears in both
    directions, so `D` sums to 2m and `E` has one entry per directed arc. Self
    loops are dropped.
  * `S`/`E` are sorted by (source, destination), so a vertex's neighbours are
    the contiguous slice `E[O[v] : O[v] + D[v]]` and are themselves sorted.
  * `O = offsets(D)` has nV+1 entries; `O[-1]` is the arc count.
"""

import os

# Importing numpy under OMP_PROC_BIND pins this process to a single core, and
# forked workers inherit that mask; clear it before numpy pulls in libgomp.
for _v in ("OMP_PROC_BIND", "OMP_PLACES"):
    os.environ.pop(_v, None)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

VTYPE = np.uint32
ETYPE = np.uint64


def multi_arange(start, count):
    """Concatenation of arange(start[i], start[i] + count[i]) for every i."""
    nz = count > 0
    start = start[nz]
    count = count[nz].astype(np.int64)
    if start.size == 0:
        return np.empty(0, dtype=np.int64)

    total = int(count.sum())
    # Reset positions: where each block begins in the flat vector.
    ri = np.zeros(count.size, dtype=np.int64)
    ri[1:] = np.cumsum(count[:-1])

    incr = np.ones(total, dtype=np.int64)
    incr[ri] = start
    incr[ri[1:]] += 1 - (start[:-1].astype(np.int64) + count[:-1])
    return np.cumsum(incr)


def offsets(deg):
    o = np.zeros(deg.size + 1, dtype=ETYPE)
    np.cumsum(deg, out=o[1:])
    return o


def load_mtx(path):
    """Read an .mtx / .edges edge list into a sorted, deduplicated undirected CSR.

    MatrixMarket puts a `rows cols nnz` header on the first non-comment line,
    which has to be skipped. The .edges files in this collection do not -- they
    carry their dimensions inside a `%` comment instead, so their first
    non-comment line is already an edge and dropping it would silently lose one.
    The banner is what distinguishes them.
    """
    with open(path, "r") as fh:
        has_header = fh.readline().lstrip("%").lstrip().lower().startswith(
            "matrixmarket")

        # Hand the file itself to loadtxt rather than a list of lines: the big
        # inputs here run to tens of millions of rows and materialising them as
        # Python strings costs several GB before numpy sees any of it.
        fh.seek(0)
        pos = fh.tell()
        line = fh.readline()
        while line and line[:1] in "#%":
            pos = fh.tell()
            line = fh.readline()
        if has_header:          # the `rows cols nnz` line is not an edge
            pos = fh.tell()
        fh.seek(pos)
        raw = np.loadtxt(fh, dtype=np.int64, usecols=(0, 1), ndmin=2)
    raw = raw[raw[:, 0] != raw[:, 1]]

    both = np.vstack((raw, raw[:, ::-1]))
    nV = int(both.max()) + 1

    lin = both[:, 0].astype(np.int64) * nV + both[:, 1]
    lin = np.unique(lin)

    S = (lin // nV).astype(VTYPE)
    E = (lin % nV).astype(VTYPE)
    D = np.bincount(S, minlength=nV).astype(VTYPE)
    return S, E, D, nV
