# ktrussvector

Vectorised $k$-truss decomposition in a Python array library — no GPU code of our
own, no C++, no compiled extension. Triangle enumeration and edge peeling are both
expressed as whole-array gathers, scatters and prefix sums, so the multi-core and
SIMD parallelism comes from NumPy's own primitives. The identical formulation runs
unchanged under PyTorch, on a CPU and on a GPU.

The method in one paragraph: orient the edge list on vertex identifier, map each
ordered endpoint pair to a single integer with a **Cantor pairing**, and hold those
codes in a **Cuckoo hash table** built by whole-array passes rather than by scalar
insertion. A triangle on edge $(u,v)$ is then a one-hop pair that is also a direct
edge, which is two vectorised gathers and an equality test — the *probe*. Peeling
runs over a flat triangle–edge incidence array in one of two ways: **peel$_m$**
carries a batch of truss thresholds on a leading run axis and advances them all in
one set of array operations, and **peel$_a$** sorts the incidence array once so it
can shrink as triangles die and be cut into segments that own disjoint edges.

---

## The implementations

Everything lives in [ktruss/](ktruss/), a flat directory — the modules import each
other as siblings.

| File | Backend | What it is |
|---|---|---|
| [`graphio.py`](ktruss/graphio.py) | — | `.mtx` / `.edges` reader, CSR construction, and the `multi_arange` primitive. Every other file reads its input through this one. |
| [`ktruss_cantor.py`](ktruss/ktruss_cantor.py) | NumPy | The core file: Cantor pairing, the vectorised Cuckoo build, the probe, and **both** the single-threshold `peel()` and the unrolled **`peel_multik()`** (peel$_m$). |
| [`ktruss_cantor_scale.py`](ktruss/ktruss_cantor_scale.py) | NumPy | Scaling harness for the above. `--variant peel \| multik \| multik-uncapped`, one CSV row per configuration. Sets the thread limits *before* NumPy is imported, which is the only point at which they bind. |
| [`ktruss_cantor_scale_arr.py`](ktruss/ktruss_cantor_scale_arr.py) | NumPy | **peel$_a$**: the same triangulation, imported unchanged, with the sorted incidence array that shrinks as triangles die and is partitioned across worker processes with no cross-worker reduction. |
| [`ktruss_cantor_arr_torch.py`](ktruss/ktruss_cantor_arr_torch.py) | PyTorch | peel$_a$ ported to torch tensors, `--device cpu` or `--device cuda`. This is the file behind the three-backend table: the GPU costs a change of array library, not a change of algorithm. |
| [`ktruss_torch.py`](ktruss/ktruss_torch.py) | PyTorch | The unrolled-threshold peel (`--outer multi`) on the GPU, over a searchsorted key probe. `--outer base` is the one-threshold-at-a-time control. |
| [`ktruss_hash.py`](ktruss/ktruss_hash.py) | NumPy | The alternative way to close a wedge: `multi_arange` neighbourhood expansion resolved by `searchsorted`, segmented on vertex boundaries. This is what the Cuckoo probe is measured against. |
| [`ktruss_hash_scale.py`](ktruss/ktruss_hash_scale.py) | NumPy | Its scaling harness. `--mode support` (one full wedge pass) or `--mode peel` (the complete decomposition). |

### The pipeline, stage by stage

| Stage | Code |
|---|---|
| Orient the edge list, keeping $u < v$ so each undirected edge appears once | `ktruss_cantor.orient()` |
| Map each ordered endpoint pair to one integer | `ktruss_cantor.cantor()` |
| Build the two Cuckoo tables by whole-array passes, not scalar insertion | `ktruss_cantor.Cuckoo._build()` |
| Close every wedge with two gathers and an equality test — the *probe* | `ktruss_cantor.triangles()` |
| Peel, one truss threshold per round | `ktruss_cantor.peel()` |
| Peel, $R$ thresholds at once on a leading run axis — peel$_m$ | `ktruss_cantor.peel_multik()`; `ktruss_torch.py --outer multi` |
| Peel over a shrinking sorted incidence array, segmented across workers — peel$_a$ | `ktruss_cantor_scale_arr.peel_arr()`; `ktruss_cantor_arr_torch.peel_arr()` |
| The whole thing end to end, timed per phase | `ktruss_cantor.run()` |

---

## Requirements

Python 3.13. The NumPy path needs nothing but NumPy:

```bash
pip install -r requirements.txt          # numpy==2.4.2
```

The PyTorch files need torch as well; `--device cpu` works without any CUDA:

```bash
pip install -r requirements-torch.txt    # numpy==2.4.2, torch==2.13.0
```

Both pins are the versions the measurements below were taken under.

### Datasets

Not distributed here — they are public, and all six come from the Stanford
[SNAP](https://snap.stanford.edu) collection, the
[Laboratory for Web Algorithmics](http://law.di.unimi.it/datasets.php),
[KONECT](http://konect.cc), and [Network Repository](https://networkrepository.com).
Every script takes the graph as its first positional argument, so they can live
anywhere.

`graphio.load_mtx()` accepts MatrixMarket (`.mtx`, with the `rows cols nnz` banner)
and bare edge lists (`.edges`, dimensions in a `%` comment). It makes the graph
undirected and simple, drops self-loops and duplicate edges, and sorts by
(source, destination); isolated vertices carry no edges and so never appear.

---

## Running

Every file is a standalone script. Times are printed per phase — Cuckoo table
build, probe, peel — and the graph load is excluded from all of them.

```bash
# NumPy, one process, peel_a, checked edge for edge against ktruss_cantor.peel()
python3 ktruss/ktruss_cantor_scale_arr.py datasets/soc-digg.mtx --cores 32 --check --json

# NumPy, the unrolled peel_m, checked against the searchsorted implementation
python3 ktruss/ktruss_cantor.py datasets/soc-digg.mtx --chunk 8 --check --json

# a scaling row, written straight to CSV
python3 ktruss/ktruss_cantor_scale.py datasets/soc-digg.mtx --cores 32 --variant multik --csv out.csv

# PyTorch, same algorithm, on one GPU (or --device cpu)
python3 ktruss/ktruss_cantor_arr_torch.py datasets/soc-digg.mtx --device cuda --repeat 3 --csv out.csv

# the multi_arange + searchsorted wedge pass, for comparison
python3 ktruss/ktruss_hash_scale.py datasets/soc-digg.mtx --cores 32 --mode support
```

`--check` is worth using on any new machine. All variants produce identical truss
values, and this is what asserts it:

| File | What `--check` compares |
|---|---|
| `ktruss_cantor.py` | triangle count, $k_{\max}$ and $k_{\mathrm{avg}}$ against `ktruss_hash.py`, an independent implementation with a different wedge kernel. |
| `ktruss_cantor_scale_arr.py` | every edge's truss value, elementwise, against `ktruss_cantor.peel()` on the same triangles — so the comparison isolates the peel. |
| `ktruss_cantor_arr_torch.py` | the triangle **set** and every edge's truss value, elementwise, against `ktruss_cantor.py` run end to end on the CPU — so a disagreement in the triangulation and one in the peel are caught separately. |

**Threads.** The thread count has to be fixed before the array library is imported,
so each configuration must be a *fresh process*. The `_scale` harnesses read
`--cores` straight off `argv` and set `OMP_NUM_THREADS` (and the MKL / OpenBLAS /
VecLib / NumExpr equivalents) ahead of every other import; setting them afterwards
has no effect. They also clear Slurm's `OMP_PROC_BIND` / `OMP_PLACES`, which
otherwise pin the process to a single core and flatten every scaling curve.

---

## Results

### Hardware and software

| | |
|---|---|
| **CPU** | One socket of a Nibi node: two 96-core Intel Xeon 6972P (Granite Rapids), simultaneous multithreading disabled, 755 GB. Each configuration confined to one socket and given the stated number of cores. |
| **GPU** | One NVIDIA H100 80GB HBM3, CUDA 13.2. |
| **Software** | Python 3.13.2, NumPy 2.4.2, PyTorch 2.13.0. GCC 12.3 toolchain (`StdEnv/2023`). |
| **Timing** | Seconds, per phase: Cuckoo table build, probe, peel. Reading the graph from disk and building the flattened adjacency list are excluded. *Total* is the sum of the three phases of one run of one variant, so every column of a row comes from the same run. Repeats take the minimum. |

No compiled code of ours is involved anywhere: the parallelism is whatever the
array library's threading, the `multiprocessing` worker pools of peel$_a$, or the
device itself provides.

### The graphs measured

After preprocessing — undirected, simple, isolated vertices dropped — so $|V|$
counts vertices of non-zero degree and $|E|$ undirected edges. $k_{\max}$ and
$k_{\mathrm{avg}}$ are the largest and mean *truss* value.

| Dataset | \|V\| | \|E\| | Size | d_avg | d_max | Triangles | k_max | k_avg |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| soc-Slashdot0811 | 77K | 469K | 9 MB | 12.1 | 2,539 | 552K | 33 | 1.69 |
| soc-BlogCatalog | 89K | 2.1M | 20 MB | 47.2 | 9,444 | 51.2M | 99 | 32.13 |
| soc-FourSquare | 639K | 3.2M | 38 MB | 10.1 | 106,218 | 21.7M | 36 | 11.16 |
| amazon-2008 | 735K | 3.5M | 67 MB | 9.6 | 1,077 | 4.5M | 9 | 2.70 |
| soc-digg | 771K | 5.9M | 72 MB | 15.3 | 17,643 | 62.7M | 71 | 13.63 |
| cit-patent | 3.8M | 16.5M | 244 MB | 8.8 | 793 | 7.5M | 34 | 0.81 |

### One triangulation, two peels (NumPy)

Seconds at one core, then what 32 cores buy each column. Each speed-up is against
**that column's own** one-core time, so the two peel columns are not comparable
with each other as ratios — for absolute cost, read the three columns on the left.

| Graph | probe (1c) | peel_m (1c) | peel_a (1c) | probe ×32 | peel_m ×32 | peel_a ×32 | total_m ×32 | total_a ×32 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| soc-Slashdot0811 | 1.483 | 7.886 | 0.9359 | 3.7x | 2.5x | 1.0x | 2.6x | 1.9x |
| soc-BlogCatalog | 29.84 | 2580 | 982.51 | 6.1x | 2.1x | 7.6x | 2.1x | 7.5x |
| soc-FourSquare | 485.02 | 273.50 | 102.52 | 17.6x | 4.3x | 7.0x | 8.2x | 12.8x |
| amazon-2008 | 3.118 | 14.07 | 5.100 | 3.6x | 2.5x | 2.0x | 2.5x | 2.6x |
| soc-digg | 94.82 | 5640 | 1576 | 15.6x | 1.6x | 8.8x | 1.6x | 8.7x |
| cit-patent | 13.32 | 235.96 | 13.37 | 6.3x | 5.3x | 2.8x | 4.9x | 2.6x |

peel$_a$ wins on absolute time on every graph, by 2.6× (soc-BlogCatalog) to 18×
(cit-patent) at one core, and it scales where peel$_m$ does not: on soc-digg, 8.8×
from 32 cores against peel$_m$'s 1.6×. Peeling at $R$ thresholds at once costs
$R \times n_T$ per round with nothing shrinking, so on a triangle-dense graph the
width cap holds $R$ far below the core count. The one graph where the run axis pays
is cit-patent, which has the fewest triangles per edge of the six. On
soc-Slashdot0811 peel$_a$ gains nothing from 32 cores: its peel is 431 short rounds,
process dispatch costs more than a round is worth, and the implementation declines
to spawn workers.

### Two ways to close the same wedge (NumPy)

The Cuckoo probe against the `multi_arange` + `searchsorted` expansion, both at a
four-million-wedge budget. Speed-ups are against each column's own one-core time.

| Graph | Cuckoo (1c) | m-arange (1c) | Cuckoo ×16 | m-arange ×16 | Cuckoo ×32 | m-arange ×32 |
|---|--:|--:|--:|--:|--:|--:|
| soc-Slashdot0811 | 1.483 | 4.933 | 3.3x | 13.4x | 3.7x | 22.9x |
| soc-BlogCatalog | 29.84 | 193.51 | 5.3x | 14.9x | 6.1x | 27.2x |
| soc-FourSquare | 485.02 | 3037 | 12.0x | 15.1x | 17.6x | 27.3x |
| amazon-2008 | 3.118 | 4.944 | 4.8x | 13.5x | 3.6x | 20.5x |
| soc-digg | 94.82 | 285.25 | 9.5x | 15.4x | 15.6x | 27.4x |
| cit-patent | 13.32 | 31.21 | 5.6x | 11.2x | 6.3x | 17.8x |

Serially the Cuckoo table wins everywhere, by 1.6–6.5×. But m-arange scales far
better, because its expansion is segmented on vertex boundaries and each segment
accumulates into a disjoint slice of the support array with no locks and no
reduction — by 32 cores it has overtaken on soc-Slashdot0811 and amazon-2008. It
does not overtake on soc-FourSquare, whose 106,218-degree vertex forces the
expansion into 18,820 segments.

### One formulation, three backends (peel$_a$)

The NumPy column is the segmented, multi-process peel; both PyTorch columns are
that same algorithm's *serial* path, so the last two differ only in the device.
Times are the table build, the probe and the peel summed.

| Dataset | NumPy, 32c | PyTorch, 32c | PyTorch, H100 | GPU vs PyTorch CPU | GPU vs NumPy |
|---|--:|--:|--:|--:|--:|
| soc-Slashdot0811 | 1.353 | 1.356 | 0.1749 | 7.8x | 7.7x |
| soc-BlogCatalog | 135.51 | 498.50 | 8.227 | 60.6x | 16.5x |
| soc-FourSquare | 46.53 | 139.69 | 3.331 | 41.9x | 14.0x |
| amazon-2008 | 3.962 | 5.381 | 0.1154 | 46.6x | 34.3x |
| soc-digg | 190.00 | 657.01 | 12.71 | 51.7x | 14.9x |
| cit-patent | 12.80 | 5.989 | 0.3599 | 16.6x | 35.6x |

The truss values, and the round counts that produce them, are identical in all
three. With the library and the code held fixed, the device is worth 7.8–60.6×;
against the fastest CPU result, 7.7–35.6×. PyTorch is the slower of the two CPU
backends on five of the six graphs, and gives up most on the two with the deepest
peels — soc-BlogCatalog at 2,277 rounds and soc-digg at 2,972 — which are also the
two where the segmented NumPy peel is furthest ahead. cit-patent is the exception:
PyTorch is 2.1× the faster CPU backend there because its Cuckoo build costs 1.9 s
against NumPy's 5.8 and its peel is short enough for that to decide the row.

---

## License

BSD 3-Clause. See [LICENSE](LICENSE).
