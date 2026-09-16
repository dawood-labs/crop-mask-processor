#!/usr/bin/env python3
"""Find shapefiles with no .prj and supply one - only when the evidence is solid.

A shapefile without a .prj has coordinates but no statement of what they mean,
and the pipeline refuses to guess: the same numbers could be degrees or metres.
This happens every season or two (2025 HYDERABAD, 2019 PAKPATTAN, 2021 KHANEWAL,
VEHARI and NAROWAL), so the check that was being done by hand is done here.

For each shapefile missing a .prj, a candidate is chosen and then verified:

1. Candidate: a .prj in the same folder under a different name. PAKPATTAN 2019
   had one - three files had been renamed with a "1" prefix and two had not.
   Failing that, the .prj that every other layer of the same crop and season
   uses, provided they all agree.
2. Verification: the layer's coordinates, read as that CRS, must fall inside the
   bounding box of its own district boundary (plus a small margin). A layer in
   metres read as degrees lands nowhere near Pakistan, so this cannot pass by
   accident.

Only if both hold is a .prj written, as a new object next to the shapefile;
nothing existing is modified. Anything else is reported for a person to look at.

    python scripts/fix_missing_prj.py --year 2021            # report only
    python scripts/fix_missing_prj.py --year 2021 --apply    # write the .prj files
"""

from __future__ import annotations

import argparse
import collections
import sys
import tempfile
from pathlib import Path, PurePosixPath

#: Degrees of slack around the district's bounding box. Crop masks are clipped
#: to districts later, so a layer can overhang its boundary slightly.
MARGIN_DEG = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--apply", action="store_true", help="write the .prj files")
    ap.add_argument("--config", default=str(Path(__file__).parent.parent / "config/default.yaml"))
    args = ap.parse_args()

    import pyogrio
    from pyproj import CRS

    from cropmask import gcs
    from cropmask.config import Config
    from cropmask.discovery import _locate, detect_crop, norm_name

    cfg = Config.load(args.config, year=args.year)
    src = gcs.GcsPath.parse(cfg.input_uri)
    client = gcs.get_client(cfg.credentials_json)
    bucket = client.bucket(src.bucket)
    work = Path(tempfile.mkdtemp(prefix="prjfix_"))

    boundary_path = gcs.fetch_to_local(cfg.boundary_uri, work / "boundary", cfg.credentials_json)
    boundary = pyogrio.read_dataframe(str(boundary_path), columns=[cfg.boundary_field]).to_crs("EPSG:4326")
    district_bounds = {}
    for name, geom in zip(boundary[cfg.boundary_field], boundary.geometry):
        key = norm_name(name)
        b = geom.bounds
        if key in district_bounds:
            o = district_bounds[key]
            b = (min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
        district_bounds[key] = b

    blobs = gcs.list_blobs(cfg.input_uri, cfg.credentials_json)
    stems: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for name, _ in blobs:
        ext = PurePosixPath(name).suffix.lower()
        if ext:
            stems[name[: -len(ext)]][ext] = name

    # the .prj each crop normally uses this season
    usual: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for stem, files in stems.items():
        if ".shp" in files and ".prj" in files:
            rel = stem[len(src.prefix):].lstrip("/").split("/")
            crop = detect_crop(rel[:-1])
            if crop:
                usual[crop][bucket.blob(files[".prj"]).download_as_text()] += 1

    missing = sorted(s for s, f in stems.items() if ".shp" in f and ".prj" not in f)
    print(f"{args.year}: {len(missing)} shapefile(s) without a .prj\n")
    fixed = skipped = 0

    for stem in missing:
        rel_parts = stem[len(src.prefix):].lstrip("/").split("/")
        label = "/".join(rel_parts)
        crop = detect_crop(rel_parts[:-1])
        located = _locate(rel_parts + ["x"], set(district_bounds))
        folder = stem.rsplit("/", 1)[0]

        # 1. a misnamed .prj beside it
        siblings = [f[".prj"] for s, f in stems.items()
                    if s.rsplit("/", 1)[0] == folder and ".prj" in f and s != stem]
        if len(siblings) == 1:
            candidate, source = bucket.blob(siblings[0]).download_as_text(), f"own folder ({PurePosixPath(siblings[0]).name})"
        elif crop and len(usual[crop]) == 1:
            candidate, source = next(iter(usual[crop])), f"all other {crop} layers this season"
        else:
            print(f"SKIP  {label}\n      no single candidate .prj (siblings {len(siblings)}, "
                  f"distinct {crop} .prj files {len(usual.get(crop, {}))})")
            skipped += 1
            continue

        if located is None:
            print(f"SKIP  {label}\n      could not tell which district this is")
            skipped += 1
            continue
        district = norm_name(located[1])

        local = work / norm_name(label).replace(" ", "_")
        local.mkdir(parents=True, exist_ok=True)
        for ext in (".shp", ".shx", ".dbf"):
            if ext in stems[stem]:
                bucket.blob(stems[stem][ext]).download_to_filename(str(local / f"layer{ext}"))
        (local / "layer.prj").write_text(candidate)

        try:
            crs = CRS.from_wkt(candidate)
            layer = pyogrio.read_dataframe(str(local / "layer.shp"), columns=[])
            lx0, ly0, lx1, ly1 = layer.set_crs(crs, allow_override=True).to_crs("EPSG:4326").total_bounds
        except Exception as exc:
            print(f"SKIP  {label}\n      could not read with the candidate CRS: {exc}")
            skipped += 1
            continue

        dx0, dy0, dx1, dy1 = district_bounds[district]
        m = MARGIN_DEG
        inside = lx0 >= dx0 - m and ly0 >= dy0 - m and lx1 <= dx1 + m and ly1 <= dy1 + m
        detail = (f"      candidate from {source}: {crs.name}\n"
                  f"      layer    [{lx0:.4f}, {ly0:.4f}, {lx1:.4f}, {ly1:.4f}]\n"
                  f"      district [{dx0:.4f}, {dy0:.4f}, {dx1:.4f}, {dy1:.4f}]  {located[1]}")
        if not inside:
            print(f"SKIP  {label}  (coordinates do not sit inside the district)\n{detail}")
            skipped += 1
            continue

        target = stem + ".prj"
        if args.apply:
            if bucket.blob(target).exists():
                print(f"SKIP  {label}  (a .prj appeared meanwhile)")
                skipped += 1
                continue
            bucket.blob(target).upload_from_string(candidate, content_type="text/plain")
            print(f"FIXED {label}\n{detail}")
        else:
            print(f"OK    {label}  (would write .prj; rerun with --apply)\n{detail}")
        fixed += 1

    verb = "written" if args.apply else "verifiable"
    print(f"\n{fixed} {verb}, {skipped} need a person")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
