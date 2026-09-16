"""Turn a flat GCS object listing into a per-district work plan.

The on-disk layout is not perfectly regular. Real 2025 paths look like::

    <root>/Cotton/2025/Punjab/RAHIM YAR KHAN/RAHIM YAR KHAN_2025_COTTON.shp
    <root>/Cotton/2025/Sindh/MIRPUR KHAS/mirpurkhas_sep_4_.../mirpurkhas_....shp

so the province/district pair sits at a depth that varies per file. Rather than
hard-coding positions we anchor on the district names from the boundary
shapefile, which is the one authoritative list of districts the run must cover.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .constants import CROP_ALIASES, CROP_ORDER, SHAPEFILE_EXTENSIONS, SHAPEFILE_REQUIRED

log = logging.getLogger(__name__)


def norm_name(value: str) -> str:
    """``TANDO-ALLAHYAR`` / ``Tando Allahyar_`` -> one comparable key."""
    text = str(value).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_crop(parts) -> str | None:
    """Identify the crop from path components."""
    alias_map = {norm_name(a): crop for crop, al in CROP_ALIASES.items() for a in al}
    for part in parts:
        hit = alias_map.get(norm_name(part))
        if hit:
            return hit
    joined = norm_name(" ".join(str(p) for p in parts))
    # "Fall Maize" first: "maize" must win over a stray "rice" elsewhere in the path.
    for crop in CROP_ORDER:
        for alias in CROP_ALIASES[crop]:
            if norm_name(alias) in joined:
                return crop
    return None


@dataclass
class CropInput:
    crop: str
    #: Blob name of the chosen ``.shp``.
    shp_blob: str
    #: Every sidecar that belongs to it, including the ``.shp`` itself.
    blobs: list[str]
    size_bytes: int
    #: Other candidate shapefiles found in the same folder (kept for the report).
    rejected: list[str] = field(default_factory=list)


@dataclass
class DistrictTask:
    province: str
    district: str
    crops: dict[str, CropInput] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.province, self.district)

    @property
    def size_bytes(self) -> int:
        return sum(c.size_bytes for c in self.crops.values())

    def blobs(self) -> list[str]:
        return [b for c in self.crops.values() for b in c.blobs]


def _locate(rel_parts: list[str], known_districts: set[str]) -> tuple[str, str] | None:
    """Find ``(province, district)`` in a relative path's directory components.

    Scans from the deepest folder outwards so that a shapefile nested inside an
    extra sub-directory still resolves to the district folder above it.
    """
    dirs = rel_parts[:-1]  # drop the filename
    for i in range(len(dirs) - 1, 0, -1):
        if norm_name(dirs[i]) in known_districts:
            return dirs[i - 1].strip(), dirs[i].strip()
    return None


def build_index(
    blobs: list[tuple[str, int]],
    root_prefix: str,
    known_districts: set[str],
    district_provinces: dict[str, str] | None = None,
) -> tuple[dict[tuple[str, str], DistrictTask], list[dict]]:
    """Group a blob listing into one :class:`DistrictTask` per district.

    ``district_provinces`` maps a normalised district name to its normalised
    province, as the boundary file records it. It is used only to resolve a
    district found under more than one province folder.

    Returns the tasks plus a list of anomalies worth putting in the report.
    """
    root_prefix = root_prefix.strip("/")
    by_folder: dict[tuple[str, str, str, str], dict[str, list[tuple[str, int]]]] = {}
    anomalies: list[dict] = []
    unmatched: list[str] = []

    for name, size in blobs:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in SHAPEFILE_EXTENSIONS:
            continue

        rel = name[len(root_prefix):].lstrip("/") if root_prefix else name
        parts = rel.split("/")
        if len(parts) < 3:
            continue

        located = _locate(parts, known_districts)
        if located is None:
            if suffix == ".shp":
                unmatched.append(name)
            continue
        province, district = located

        crop = detect_crop(parts[:-1])
        if crop is None:
            if suffix == ".shp":
                anomalies.append({"issue": "unrecognised crop", "blob": name})
            continue

        # Group by the shapefile's own stem so sidecars stay with their .shp.
        stem = name[: -len(suffix)]
        key = (crop, province, district, stem)
        by_folder.setdefault(key, {}).setdefault(suffix, []).append((name, size))

    by_folder = _drop_misplaced_province_folders(
        by_folder, district_provinces or {}, anomalies
    )

    # Collapse stems into one chosen shapefile per (crop, district).
    candidates: dict[tuple[str, str, str], list[CropInput]] = {}
    for (crop, province, district, stem), files in by_folder.items():
        present = set(files)
        if not set(SHAPEFILE_REQUIRED).issubset(present):
            missing = sorted(set(SHAPEFILE_REQUIRED) - present)
            anomalies.append(
                {"issue": f"incomplete shapefile (missing {','.join(missing)})",
                 "blob": stem + ".shp", "crop": crop,
                 "province": province, "district": district}
            )
            continue
        all_blobs = [n for group in files.values() for n, _ in group]
        size = sum(s for group in files.values() for _, s in group)
        candidates.setdefault((crop, province, district), []).append(
            CropInput(crop=crop, shp_blob=stem + ".shp", blobs=all_blobs, size_bytes=size)
        )

    tasks: dict[tuple[str, str], DistrictTask] = {}
    for (crop, province, district), options in candidates.items():
        # Several shapefiles in one district folder: the 2025 data pairs a real
        # layer with a 100-byte "Emptyshapefile" placeholder, so take the
        # largest rather than the alphabetically first.
        options.sort(key=lambda c: c.size_bytes, reverse=True)
        chosen = options[0]
        if len(options) > 1:
            chosen.rejected = [o.shp_blob for o in options[1:]]
            anomalies.append(
                {"issue": f"{len(options)} shapefiles found, used largest",
                 "crop": crop, "province": province, "district": district,
                 "used": chosen.shp_blob,
                 "ignored": " | ".join(chosen.rejected)}
            )
        # Key on the normalised names, not the raw folder strings. The bucket
        # spells the same district differently across crop trees
        # ("RAHIM YAR KHAN" vs "Rahim Yar Khan"), and keying on the raw text
        # splits one district into two tasks: neither erases the other's crops,
        # both write to the same completion marker, and the workbook counts the
        # district twice.
        key = (norm_name(province), norm_name(district))
        task = tasks.setdefault(key, DistrictTask(province, district))
        task.crops[crop] = chosen

    if unmatched:
        log.warning(
            "%d shapefile(s) did not match any boundary district and were skipped; "
            "first: %s", len(unmatched), unmatched[0],
        )
        anomalies.append(
            {"issue": "no matching district in boundary shapefile",
             "blob": f"{len(unmatched)} file(s), e.g. {unmatched[0]}"}
        )

    return tasks, anomalies


def _drop_misplaced_province_folders(by_folder, district_provinces, anomalies):
    """Keep one province folder per district; report and ignore the others.

    A district filed under two province folders was being split into two
    independent tasks. The stray one was processed on its own - never erased by
    the district's other crops - and published as an extra, un-de-overlapped
    layer that the report then counted twice. It happened in 2023 (a byte-for-byte
    copy of SANGHAR's rice under Punjab) and 2024 (a second, different version of
    MIANWALI's rice under Sindh).

    The province the boundary file records wins. Failing that, the folder holding
    most of the district's crops wins. A stray file is never merged in as an
    alternative candidate: with two genuinely different versions, as in MIANWALI,
    that would silently pick whichever is larger. It is ignored and reported, so
    a person decides.

    Districts found under a single province folder are untouched, whatever that
    folder is called.
    """
    provinces_by_district: dict[str, dict[str, set[str]]] = {}
    for (crop, province, district, _stem) in by_folder:
        provinces_by_district.setdefault(norm_name(district), {}) \
            .setdefault(norm_name(province), set()).add(crop)

    keep: dict[str, str | None] = {}
    for district, provinces in provinces_by_district.items():
        if len(provinces) == 1:
            continue
        official = district_provinces.get(district)
        if official in provinces:
            keep[district] = official
            continue
        ranked = sorted(provinces.items(), key=lambda kv: len(kv[1]), reverse=True)
        if len(ranked[1][1]) == len(ranked[0][1]):
            keep[district] = None      # a tie: nothing to go on
            anomalies.append({
                "issue": "district found under several province folders with no way "
                         "to tell which is right - district skipped",
                "district": district,
                "provinces": " | ".join(sorted(provinces)),
            })
            log.error("%s is filed under %s and cannot be resolved; skipped",
                      district, sorted(provinces))
            continue
        keep[district] = ranked[0][0]

    if not keep:
        return by_folder

    kept = {}
    for key, files in by_folder.items():
        crop, province, district, stem = key
        chosen = keep.get(norm_name(district), norm_name(province))
        if norm_name(province) == chosen:
            kept[key] = files
            continue
        shp = stem + ".shp"
        anomalies.append({
            "issue": "shapefile in the wrong province folder - ignored",
            "crop": crop, "province": province, "district": district,
            "ignored": shp,
            "district_province": chosen or "unresolved",
        })
        log.warning("ignoring %s: %s belongs under %s, not %s",
                    shp, district, chosen or "an unresolved province", province)
    return kept


def select_tasks(
    tasks: dict[tuple[str, str], DistrictTask],
    only_provinces: list[str],
    only_districts: list[str],
    limit: int = 0,
) -> list[DistrictTask]:
    """Apply the province/district filters and the test ``limit``."""
    selected = list(tasks.values())

    if only_provinces:
        want = {norm_name(p) for p in only_provinces}
        selected = [t for t in selected if norm_name(t.province) in want]
    if only_districts:
        want = {norm_name(d) for d in only_districts}
        selected = [t for t in selected if norm_name(t.district) in want]

    # Smallest first when sampling, so a --limit test run stays quick.
    selected.sort(key=lambda t: t.size_bytes)
    if limit:
        selected = selected[:limit]
    return selected
