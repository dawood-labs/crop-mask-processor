# crop-mask-processor

Production pipeline for the FAO crop-mask de-overlap job: reads the per-crop
district shapefiles from Google Cloud Storage, removes the overlap between
crops, clips each layer to its district boundary, and writes clean single-crop
shapefiles plus an acreage workbook back to GCS.

It replaces a single-threaded local script that could not finish a large
district. On the 2025 dataset the largest district (RAHIM YAR KHAN, 635 MB of
input) went from **not completing a single step in 20 minutes** to **6 minutes
end to end**, and the whole 66-district run is parallel and memory-bounded.

---

## What it does

For every district, in crop-priority order (Fall Maize > Sugarcane > Cotton >
Rice):

1. **De-overlap** — each crop has every higher-priority crop erased from it.
2. **Clip** — the result is cut to the district polygon from the FAO boundary
   shapefile.
3. **Dissolve** to singleparts, drop polygons of 0.5 acre or less, and drop the
   whole layer if what remains totals 200 acres or less.
4. **Write** a shapefile carrying a single `int32` column, `predicted`
   (Rice 7, Cotton 2, Fall Maize 3016, Sugarcane 1).

Areas are always measured in UTM 42N (`EPSG:32642`); the output CRS is
configurable and defaults to `EPSG:4326`.

## Quick start

```bash
pip install -e .

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# see what would run, without processing anything
cropmask -c config/default.yaml --dry-run

# process five districts as a test
cropmask -c config/default.yaml --limit 5

# full run
cropmask -c config/default.yaml
```

Outputs land under `output_uri` mirroring the input layout, next to
`acreage_report.xlsx`:

```
<output_uri>/
    Rice/Punjab/SAHIWAL/SAHIWAL_Rice.shp
    Cotton/Sindh/BADIN/BADIN_Cotton.shp
    ...
    acreage_report.xlsx
```

## Configuration

Everything lives in `config/default.yaml`; every key can be overridden on the
command line (`cropmask --help`). The ones that matter most:

| Key | Meaning |
|---|---|
| `input_uri` / `output_uri` | GCS prefixes for input and output |
| `boundary_uri`, `boundary_field` | district polygons and the name column |
| `workers` | `null` auto-sizes the pool from CPU count and free RAM |
| `memory_fraction` | share of *available* RAM the run may claim (default 0.70) |
| `memory_per_input_byte` | RAM assumed per byte of input, used for admission control |
| `min_polygon_acres`, `min_total_acres` | the two spec thresholds |
| `keep_intermediates` | write the per-step QA layers (~10x output size) |
| `only_provinces`, `only_districts`, `limit` | restrict a run |

## How it scales

**Parallelism.** Districts are independent, so each one is a task in a process
pool. Workers are recycled every `max_tasks_per_child` districts so a large
district does not leave its worker permanently inflated.

**Sizing itself.** `workers: null` reads the usable CPU count (respecting
cgroup quotas and CPU affinity, not just `os.cpu_count()`) and the available
RAM (respecting a cgroup memory limit), then takes whichever binds first.

**Staying inside memory.** A worker count alone is not enough when districts
range from 100 bytes to 635 MB. Admission is driven by an explicit RAM budget:
each district gets an estimated peak, and it is only submitted when that
estimate fits in the memory not already claimed by running districts. Large
districts are scheduled first so the run does not end with one giant district
holding a single core while the rest idle, and a district too large to share
the machine runs on its own. If system memory pressure crosses 88% the
scheduler stops admitting work until it drops.

If a worker is killed anyway, the pool is rebuilt at half the size and the
remaining districts are retried rather than the run dying.

**Disk.** Each district is staged to local scratch, processed, uploaded, and
its scratch deleted, so local disk stays at roughly one district per worker
instead of the whole dataset.

## Why it is fast

The original script was slow because of its algorithm, not its lack of
threads. Two changes did nearly all of the work:

**1. Six chained overlays collapse into three differences.** The spec describes
a chain — subtract Fall Maize from everything, then Sugarcane_D from Rice_D and
Cotton_D, then Cotton_D2 from Rice_D2. Set algebra reduces that chain to a
plain crop-priority partition:

```
Sugarcane_final = Sugarcane - Maize
Cotton_final    = Cotton    - (Maize u Sugarcane)
Rice_final      = Rice      - (Maize u Sugarcane u Cotton)
```

Same geometry, one pass per crop instead of six chained `gpd.overlay` calls.

**2. R-tree pre-filtering instead of whole-layer GEOS calls.** `gpd.overlay`
and `GeoDataFrame.dissolve` both cost super-linearly in total vertex count.
Measured on the 2025 Cotton layer for RAHIM YAR KHAN (10,902 features,
3.6M vertices), only **1,612 pairs of features actually intersect** — the
polygons are almost entirely disjoint, so a full dissolve spends nearly all of
its time re-noding geometry that never needed merging.

So:

* `erase` only unions the erasers that genuinely intersect each target, and
  passes untouched features straight through with no GEOS call at all.
* `dissolve` unions only the *connected components* of the touch graph
  (found with an R-tree query plus `scipy.sparse.csgraph`), leaving isolated
  polygons alone.
* `clip` skips the intersection entirely for features fully inside the mask.

Layers are also read with `pyogrio` geometry-only (attributes are discarded by
the spec anyway) and reprojected as raw arrays, skipping GeoDataFrame overhead.

## Correctness

The optimisation is only worth anything if the numbers do not move.
`scripts/verify_equivalence.py` runs the legacy implementation and the fast one
over an identical subset of a real district and prints the acreages side by
side:

```bash
python scripts/verify_equivalence.py /path/to/local/2025 BADIN 0.25
```

Measured on the 2025 data:

| District | Crop | Legacy acres | Fast acres | Difference |
|---|---|---|---|---|
| BADIN | Sugarcane | 1,014.56 | 1,014.56 | 0.00 |
| BADIN | Rice | 72,088.00 | 72,088.00 | 0.00 |
| SAHIWAL | Fall Maize | 2,816.49 | 2,816.49 | 0.00 |
| SAHIWAL | Cotton | 271.30 | 271.30 | 0.00 |
| SAHIWAL | Rice | 5,928.29 | 5,928.47 | +0.003% |

The one non-zero difference is floating-point noding noise on a polygon sitting
almost exactly on the 0.5-acre cut-off. The fast path performs fewer chained
overlays, so it accumulates *less* of this error, not more.

## The report

`acreage_report.xlsx` is written next to the outputs:

| Sheet | Contents |
|---|---|
| `Detail` | every district/crop: input polygons and acres, acreage after each stage, final polygons and acres, status |
| `Summary_by_Crop` | per-crop totals, acres lost, percentage retained |
| `Final_Acres_by_District` | district x crop pivot of final acreage |
| `Deleted_under_threshold` | layers dropped by the 200-acre rule |
| `Missing_or_Error` | missing inputs and failures, with the reason |
| `Timings` | seconds and peak RSS per district — use this to tune `memory_per_input_byte` |
| `Input_Anomalies` | duplicate, incomplete or unrecognised inputs |
| `Run_Info` | the exact settings the run used |

## Input layout

The pipeline tolerates a variable folder depth. It anchors on the district
names in the boundary shapefile rather than on fixed path positions, so all of
these resolve:

```
<root>/Cotton/2025/Punjab/RAHIM YAR KHAN/RAHIM YAR KHAN_2025_COTTON.shp
<root>/Cotton/Punjab/SAHIWAL/sahiwal_cotton.shp
<root>/Cotton/2025/Sindh/MIRPUR KHAS/mirpurkhas_sep_4_cotton1/mirpurkhas_sep_4_cotton1.shp
```

Where a district folder holds more than one shapefile the **largest** is used,
and the choice is recorded in `Input_Anomalies`. This matters on the 2025 data:
MIRPUR KHAS / Cotton contains both a real 28 MB layer and a 100-byte
`..._Emptyshapefile.shp` placeholder, and picking alphabetically — as the
original script did — silently selected the empty one.

## Development

```bash
pip install -e .
python -m pytest tests/ -v
```

## Repository layout

```
src/cropmask/
    cli.py          argument parsing, logging, run summary
    config.py       YAML + CLI configuration
    constants.py    crop order, predicted codes, aliases
    discovery.py    GCS listing -> per-district work plan
    geometry.py     the fast erase / clip / dissolve primitives
    io_layers.py    geometry-only shapefile read and write
    pipeline.py     the FAO specification for one district
    worker.py       stage from GCS, process, publish, clean up
    runner.py       memory-aware parallel scheduling
    report.py       the Excel workbook
    resources.py    CPU and RAM detection, worker planning
scripts/
    verify_equivalence.py   legacy vs fast, on real data
```
