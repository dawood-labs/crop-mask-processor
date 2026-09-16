# crop-mask-processor

FAO crop-mask de-overlap pipeline. It reads the per-crop district shapefiles
from Google Cloud Storage, removes the overlap between crops, clips each layer
to its district boundary, and writes clean single-crop shapefiles plus an
acreage workbook back to GCS.

---

# Part 1 — Getting it running

Follow these in order. You should be able to go from nothing to a finished test
run in about fifteen minutes.

## Step 1. Get the code

```bash
git clone git@github.com:dawood-labs/crop-mask-processor.git
cd crop-mask-processor
```

## Step 2. Install it

You need Python 3.11 or newer. Check first:

```bash
python3 --version
```

Then install:

```bash
pip install -e .
```

The `-e` means "editable": if you change the source, the change takes effect
immediately without reinstalling.

Check it worked:

```bash
cropmask --help
```

If you see the help text, you are set up. If you get `command not found`, your
`~/.local/bin` is probably not on `PATH`; either add it, or run the tool as
`python3 -m cropmask.cli` everywhere below instead of `cropmask`.

The pipeline needs **shapely 2.1 or newer** (it uses a faster, more accurate way
of repairing broken polygons that older versions do not have). Confirm which one
Python actually picks up:

```bash
python3 -c "import shapely; print(shapely.__version__)"
```

If it prints `2.0.x`, run `pip install --user "shapely>=2.1"`.

## Step 3. Give it access to GCS

The pipeline needs a service-account key file to read and write the bucket.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/your-key.json
```

Put that line in your `~/.bashrc` so you do not have to repeat it every session.

> **Never commit the key file.** `.gitignore` already blocks `*.json` for this
> reason. If you are ever asked to "just add the key so it works", the answer is
> no — use the environment variable.

## Step 4. Look before you leap

Always do this first on a new season. It lists exactly what *would* be
processed and then stops without touching anything:

```bash
cropmask -c config/default.yaml --year 2025 --dry-run
```

You should see something like:

```
66 district(s), 3.72 GiB total
    635.3 MB  Punjab     RAHIM YAR KHAN            crops: Cotton, Fall Maize, Rice, Sugarcane
    347.6 MB  Punjab     FAISALABAD                crops: Cotton, Fall Maize, Rice, Sugarcane
    ...
```

**Read the district count.** If it says 0, or a number you did not expect, stop
and find out why before running anything — it usually means the folder layout
in the bucket is not what you assumed. Also read any `anomaly:` warnings at the
bottom; those are inputs the pipeline had to make a decision about.

## Step 5. Test on a few districts

Never go straight to a full run. Process the five smallest districts first:

```bash
cropmask -c config/default.yaml --year 2025 --limit 5 \
         --output-uri gs://farmdar_data_catalog/fao/crop_processing/output_test/2025
```

Note the different `--output-uri`: test runs write somewhere separate so they
cannot mix with real results.

This takes a couple of minutes. At the end you get a summary:

```
Layers written  : 8
Dropped layers  : 12
Errors          : 0
Total acreage   : 45231.88
Elapsed         : 1.9 min
```

**Errors must be 0.** If not, open the report's `Missing_or_Error` sheet and
read the `reason` column before going further.

## Step 6. Run the full season

```bash
cropmask -c config/default.yaml --year 2025
```

A full season takes roughly 5 to 15 minutes; 2021 to 2024 took between 4 and
13. You can leave it; if your connection drops, see "If a run dies" below.

To run several seasons back to back without watching them:

```bash
python3 scripts/run_seasons.py 2018 2019 2020
```

It checks each season's files first, runs the seasons one after another, and
writes `~/cropmask_runs/run_<year>.log` for each plus a one-line result per
season in `~/cropmask_runs/seasons_summary.txt`. A season that ends with errors
does not stop the queue.

## Step 7. Collect the results

```
gs://farmdar_data_catalog/fao/crop_processing/output/2025/
    Rice/Punjab/SAHIWAL/SAHIWAL_Rice.shp
    Cotton/Sindh/BADIN/BADIN_Cotton.shp
    ...
    acreage_report_2025.xlsx
```

Each output shapefile carries a single integer column, `predicted`:
Rice `7`, Cotton `2`, Fall Maize `3016`, Sugarcane `1`.

---

# Part 2 — Running a different season

Everything is driven by `--year`. You do not edit the config file.

```bash
cropmask -c config/default.yaml --year 2019
```

That reads `.../crop_processing/2019` and writes `.../output/2019`. The config
contains `{year}` placeholders that get filled in:

```yaml
year: 2025
input_uri:   "gs://farmdar_data_catalog/fao/crop_processing/{year}"
output_uri:  "gs://farmdar_data_catalog/fao/crop_processing/output/{year}"
```

**Run one season at a time.** A single season already uses the whole machine,
so running two at once just makes both slower.

---

# Part 3 — When something goes wrong

## If a run dies halfway

Run the exact same command again. Every district writes a small marker when it
finishes, so a second run skips what is already done:

```
Resuming: 47 district(s) already finished, 19 left to do
```

Districts that *errored* deliberately do not get a marker, so they are retried.

To genuinely redo everything from scratch, add `--overwrite`.

## If you get "no CRS (.prj missing)"

A shapefile is missing its `.prj` sidecar, so there is no way to know what its
coordinates mean. The pipeline refuses to guess — the same numbers could be
degrees or metres, placing the data in completely different parts of the world.
This has happened in 2019, 2021, 2022 and 2025.

Run the checker first; it only reports:

```bash
python3 scripts/fix_missing_prj.py --year 2021
```

For each missing `.prj` it looks for a candidate (a misnamed `.prj` in the same
folder, or the one every other layer of that crop and season uses) and checks
that the layer's coordinates, read that way, land inside its own district. Lines
marked `OK` passed; `SKIP` means a person needs to look. When you are happy:

```bash
python3 scripts/fix_missing_prj.py --year 2021 --apply
cropmask -c config/default.yaml --year 2021
```

The second command resumes, so only the districts that failed are processed.

Districts where a crop could not be read still publish the crops that are not
affected. Crops are cleaned in priority order, so if Cotton cannot be read, Rice
in that district is held back too - it cannot be cleaned against the Cotton.

## If a file is in the wrong province folder

A district must only appear under its own province folder. If a copy turns up
under the other province too (it happened with SANGHAR in 2023 and MIANWALI in
2024), the pipeline uses the province recorded in the district boundary file,
ignores the stray file, logs a warning, and lists it on the `Input_Anomalies`
sheet. Nothing is merged or guessed. Ask the data team to remove the stray copy,
or to move it if it turns out to be the right version.

## If a district is slow

Some districts take a few minutes; that is normal and has little to do with file
size. What costs time is a very large polygon: about one layer in three has a
single shape of more than 10,000 points, and GUJRANWALA's rice regularly has one
of over a million. That district is usually the last to finish, at around five
minutes.

Any single geometry step slower than 5 seconds is logged with where it happened
and how big the shapes were:

```
SLOW erase took 11.7s in Punjab / GUJRANWALA / Rice erase (target 1,353,947 verts, erasers 32 verts)
```

One of those per large district is expected. To see where a whole season's time
went, after it finishes:

```bash
python3 scripts/hotspots.py ~/cropmask_runs/run_2024.log /tmp/cropmask/2024/acreage_report_2024.xlsx
```

It ranks time per stage (read, erase, clip, dissolve) and the slowest individual
steps. If one step keeps coming top across seasons, that is the thing to
optimise.

## If you are worried about memory

You should not have to think about it — the pipeline sizes itself. But if you
want to be cautious on a smaller machine:

```bash
cropmask -c config/default.yaml --year 2025 --workers 4 --memory-fraction 0.5
```

## Where the logs are

```
/tmp/cropmask/2025/cropmask.log
```

Worker processes write there too, so a traceback from a failed district is in
that file and not only on your screen.

---

# Part 4 — Reading the workbook

`acreage_report_<year>.xlsx` sits next to the outputs. Eight sheets:

| Sheet | What it is for |
|---|---|
| `Detail` | One row per district per crop. Acreage after *every* stage, so you can see where area went |
| `Summary_by_Crop` | Per-crop totals and what percentage survived |
| `Final_Acres_by_District` | District x crop grid — the table FAO wants |
| `Deleted_under_threshold` | Layers dropped by the 200-acre rule, with the reason |
| `Missing_or_Error` | Anything that failed, and why. **Check this first** |
| `Timings` | Seconds and peak memory per district |
| `Input_Anomalies` | Duplicate, incomplete or odd input files |
| `Run_Info` | Every setting the run used, so a number can be reproduced later |

## How to read the Detail sheet

Follow one row across and you can see exactly where acreage went:

```
district  crop     input_union  after_difference  after_clip  final   lost
BADIN     Cotton      31633.53          29123.49    29106.37  28303.15  3330.38
```

* started at **31,633** acres
* after de-overlap **29,123** → **2,510 acres went to Sugarcane**, which outranks Cotton
* after clipping **29,106** → 17 acres were outside the district boundary
* final **28,303** → the rest went to polygons under 0.5 acre

Compare that with Sugarcane in the same district, which barely changes: nothing
outranks it there, so nothing cuts it. That contrast is the de-overlap working.

## A sanity check worth doing every time

Open `Summary_by_Crop` and look at `retained_pct`. It should generally *fall*
down the crop priority order:

```
Fall Maize  >  Sugarcane  >  Cotton  >  Rice
```

Fall Maize is never cut by anything, Rice is cut by all three. If Rice retains
more than Fall Maize, something is wrong — investigate before sending anything
out.

---

# Part 5 — What the pipeline actually does

The problem it solves: the same field can appear in two crop masks at once. If
you just add the areas up, that field gets counted twice.

So each crop is given a priority, and higher-priority crops win:

```
Fall Maize  >  Sugarcane  >  Cotton  >  Rice
```

| Crop | What it keeps |
|---|---|
| Fall Maize | everything |
| Sugarcane | its own area **minus** Maize |
| Cotton | its own area **minus** (Maize + Sugarcane) |
| Rice | its own area **minus** (Maize + Sugarcane + Cotton) |

Then, per crop: clip to the district boundary, dissolve, split multipart into
singlepart, drop polygons of 0.5 acre or less, and drop the whole layer if what
remains totals 200 acres or less. All areas are measured in UTM 42N
(`EPSG:32642`); the shapefiles are written in `EPSG:4326`.

All four crops of a district are processed together in one worker, because the
de-overlap needs all four in memory at once. Districts are independent, so
*districts* are what run in parallel.

---

# Part 6 — Notes for whoever maintains this

## Do not change these without a reason

**`metric_crs: EPSG:32642`.** FAO specifies UTM 42N. It has a known quirk — the
zone's central meridian is 69°E, so eastern Punjab sits 4-5° outside it and its
acreage reads about 0.3-0.5% high relative to Sindh. An equal-area projection
would remove that, and would also stop matching the figures FAO expects. Leave
it alone.

**The area comparison in `filter_by_area`.** It compares the *unrounded* acreage
against the threshold. Rounding first makes the real cut-off 0.505 acres, and
the bias only ever deletes land.

**`make_valid(method="structure")`.** Repairing broken polygons with the default
"linework" method is four times slower on the largest shapes, and it gets area
wrong in some cases: a hole touching its outer edge is filled back in and counted
as crop. Across all of 2024 the two methods differ by 0.02 acres, but structure
is the right one.

**Province comes from the boundary file.** A district found under two province
folders is resolved using the boundary file's `province` column, and the stray
file is ignored and reported rather than processed as a second district.

**File lookup by full path, never by filename.** Crop shapefiles are routinely
named after their district — in the 2025 data, 58 of 66 districts have two or
more crops whose files are both called `<DISTRICT>.shp`, differing only in
which crop folder they sit in. Matching on the filename loads one crop's
geometry for another, and the output still looks completely plausible.

## Before you trust a change

```bash
python3 -m pytest tests/ -v
```

96 tests. Many of them exist because a specific bug shipped once; the docstring
says which.

To check a change against a real district before running a season, process it
locally and compare with the last run's acreage:

```bash
python3 scripts/profile_district.py --year 2017 GUJRANWALA
```

It prints per-crop stage timings and a `match` column against the previous
result.

To check the fast implementation still agrees with the original, on real data:

```bash
python3 scripts/verify_equivalence.py /path/to/local/2025 BADIN 0.25
```

It runs both implementations over the same subset and prints the acreages side
by side. They should match to 0.00%.

## Why it is fast

The original script could not finish a single large district. Two changes did
almost all of the work.

**The six chained overlays collapse into three differences.** The spec describes
a chain — subtract Maize from everything, then Sugarcane_D from Rice_D and
Cotton_D, then Cotton_D2 from Rice_D2. Set algebra reduces that to the plain
priority partition in Part 5: same geometry, one pass per crop.

**R-tree pre-filtering instead of whole-layer GEOS calls.** Measured on the 2025
Cotton layer for RAHIM YAR KHAN — 10,902 features, 3.6M vertices — only **1,612
pairs of features actually intersect anything**. The polygons are almost
entirely disjoint, so a full dissolve spends nearly all its time re-noding
geometry that never needed merging. So `erase` unions only the erasers that
genuinely touch each target, `dissolve` unions only the connected components of
the touch graph, and `clip` skips the intersection for features fully inside the
mask.

**Very large polygons are never processed more than once per step.** An eraser
is cut down to the target's bounding box before use; intersects queries always
prepare the larger geometry; and unions add the largest shape last, once. Before
this, a single 218,000-acre sugarcane polygon was re-processed for each of 1,005
cotton fields in RAHIM YAR KHAN 2017, and the district did not finish in 50
minutes. It now takes 3.

## Layout

```
src/cropmask/
    cli.py          arguments, logging, the run summary
    config.py       YAML + CLI configuration, {year} substitution
    constants.py    crop order, predicted codes, folder-name aliases
    discovery.py    GCS listing -> per-district work plan
    geometry.py     the fast erase / clip / dissolve primitives
    io_layers.py    geometry-only shapefile read and write
    pipeline.py     the FAO specification, for one district
    worker.py       stage from GCS, process, publish, clean up
    runner.py       memory-aware parallel scheduling
    state.py        completion markers, for resuming
    report.py       the Excel workbook
    resources.py    CPU and RAM detection
    calibration.py  learns real memory cost during the run
    telemetry.py    logs slow geometry steps and per-stage timings
tests/              96 tests
scripts/
    run_seasons.py          run several seasons back to back
    fix_missing_prj.py      verify and supply missing .prj files
    hotspots.py             rank a season's slowest steps
    profile_district.py     run one district locally, with timings and an acreage check
    verify_equivalence.py   old implementation vs new, on real data
    benchmark_legacy.py     time the original script against this one
```
