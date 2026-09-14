"""Per-district completion markers, so a long run can be resumed.

A full 2025 run takes long enough that losing it to a spot-instance
interruption, a dropped connection or a bad district near the end is a real
cost. After a district is uploaded, a small JSON marker is written alongside
the outputs; on the next run those districts are skipped and their report rows
are read back from the marker, so the workbook still covers everything.

Existence of the output shapefiles is deliberately *not* used as the signal: a
district whose layers were all dropped by the 200-acre rule produces no
shapefiles at all, and would be reprocessed on every run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from . import gcs
from .discovery import norm_name

log = logging.getLogger(__name__)

STATE_DIRNAME = "_state"


def marker_name(output_prefix: str, province: str, district: str) -> str:
    safe = f"{norm_name(province)}__{norm_name(district)}".replace(" ", "_")
    prefix = f"{output_prefix}/" if output_prefix else ""
    return f"{prefix}{STATE_DIRNAME}/{safe}.json"


@dataclass
class CompletedDistrict:
    province: str
    district: str
    records: list[dict]
    seconds: float
    peak_rss_mb: float


def write_marker(
    result,
    output_uri: str,
    work_dir,
    credentials_json: str | None = None,
) -> None:
    """Record that a district finished. Never allowed to fail the district."""
    from pathlib import Path

    dest = gcs.GcsPath.parse(output_uri)
    payload = {
        "province": result.province,
        "district": result.district,
        "seconds": result.seconds,
        "peak_rss_mb": result.peak_rss_mb,
        "records": result.records,
    }
    try:
        local = Path(work_dir) / f"_marker_{norm_name(result.district)}.json".replace(" ", "_")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(payload, default=str), encoding="utf-8")
        gcs.upload_many(
            [(local, marker_name(dest.prefix, result.province, result.district))],
            dest.bucket,
            credentials_json=credentials_json,
        )
        local.unlink(missing_ok=True)
    except Exception as exc:
        log.warning("could not write completion marker for %s: %s",
                    result.district, exc)


def load_completed(
    output_uri: str, credentials_json: str | None = None
) -> dict[tuple[str, str], CompletedDistrict]:
    """Every district already finished under ``output_uri``."""
    dest = gcs.GcsPath.parse(output_uri)
    prefix = f"{dest.prefix}/{STATE_DIRNAME}" if dest.prefix else STATE_DIRNAME

    try:
        blobs = gcs.list_blobs(f"gs://{dest.bucket}/{prefix}", credentials_json)
    except Exception as exc:
        log.debug("no resumable state at %s: %s", prefix, exc)
        return {}

    client = gcs.get_client(credentials_json)
    bucket = client.bucket(dest.bucket)
    done: dict[tuple[str, str], CompletedDistrict] = {}

    for name, _ in blobs:
        if not name.endswith(".json"):
            continue
        try:
            payload = json.loads(bucket.blob(name).download_as_bytes())
            entry = CompletedDistrict(
                province=payload["province"],
                district=payload["district"],
                records=payload.get("records", []),
                seconds=float(payload.get("seconds") or 0.0),
                peak_rss_mb=float(payload.get("peak_rss_mb") or 0.0),
            )
        except Exception as exc:
            log.warning("ignoring unreadable marker %s: %s", name, exc)
            continue
        done[(norm_name(entry.province), norm_name(entry.district))] = entry

    return done


def filter_pending(tasks: list, completed: dict) -> tuple[list, list]:
    """Split tasks into ``(still_to_do, already_done)``."""
    pending, skipped = [], []
    for task in tasks:
        key = (norm_name(task.province), norm_name(task.district))
        if key in completed:
            skipped.append(completed[key])
        else:
            pending.append(task)
    return pending, skipped
