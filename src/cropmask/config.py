"""Runtime configuration: YAML file + environment + CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # ---- data location ---------------------------------------------------
    #: ``gs://bucket/prefix`` holding ``<crop>/.../<province>/<district>/*.shp``
    input_uri: str = ""
    #: ``gs://bucket/prefix`` the results are written under.
    output_uri: str = ""
    #: District boundary polygons (``gs://`` or local path).
    boundary_uri: str = ""
    #: Column in the boundary shapefile carrying the district name.
    boundary_field: str = "districts"
    #: Service-account JSON. Falls back to GOOGLE_APPLICATION_CREDENTIALS / ADC.
    credentials_json: str | None = None

    # ---- processing ------------------------------------------------------
    #: Projected CRS used for every geometry op and all area maths.
    metric_crs: str = "EPSG:32642"
    #: CRS the final shapefiles are written in (None = keep metric_crs).
    output_crs: str | None = "EPSG:4326"
    #: Polygons with area <= this are dropped.
    min_polygon_acres: float = 0.5
    #: A whole crop layer is dropped when its total area <= this.
    min_total_acres: float = 200.0

    # ---- scaling ---------------------------------------------------------
    #: Worker processes. ``0``/None = decide from CPU and RAM at runtime.
    workers: int | None = None
    #: Fraction of *available* RAM the run is allowed to occupy.
    memory_fraction: float = 0.70
    #: RAM assumed per byte of input shapefile, on top of the fixed per-worker
    #: baseline. Calibrated on the 2025 data, where the worst district needed
    #: 6.7x; see the report's Timings sheet to re-tune.
    memory_per_input_byte: float = 8.0
    #: Never plan a worker with less than this much RAM (bytes).
    min_worker_memory: int = 2 * 1024**3
    #: Pool size as a multiple of the CPU count. Slight overcommit keeps cores
    #: busy while other workers are downloading from / uploading to GCS.
    cpu_oversubscribe: float = 1.25
    #: Replace the static memory estimate with measured peak RSS as the run
    #: progresses, so the scheduler packs the machine as tightly as is safe.
    adaptive_memory: bool = True
    #: Threads used for GCS transfers inside one worker.
    transfer_threads: int = 8
    #: Recycle a worker process after this many districts so peak allocations
    #: are returned to the OS. 0 = keep workers for the whole run.
    max_tasks_per_child: int = 4

    # ---- behaviour -------------------------------------------------------
    #: Re-process districts whose outputs already exist in the output bucket.
    overwrite: bool = False
    #: Keep the per-step QA layers (roughly 10x the output volume).
    keep_intermediates: bool = False
    #: Restrict the run (empty = everything).
    only_provinces: list[str] = field(default_factory=list)
    only_districts: list[str] = field(default_factory=list)
    #: Process at most N districts (smallest first). 0 = no limit.
    limit: int = 0
    #: Local scratch directory for staging GCS data.
    work_dir: str = "/tmp/cropmask"
    #: Name of the Excel workbook written next to the outputs.
    report_name: str = "acreage_report.xlsx"
    #: Seconds a single district may take before it is abandoned. 0 = no limit.
    district_timeout: int = 5400
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> "Config":
        """Build a config from an optional YAML file plus keyword overrides.

        Overrides whose value is ``None`` are ignored, so an unset CLI flag
        never clobbers a value coming from the YAML file.
        """
        data: dict[str, Any] = {}
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")

        data.update({k: v for k, v in overrides.items() if v is not None and k in known})

        cfg = cls(**data)
        cfg._resolve_env()
        cfg.validate()
        return cfg

    def _resolve_env(self) -> None:
        if not self.credentials_json:
            self.credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if self.credentials_json:
            # Let every library in the process (gcsfs, GDAL/VSI) see the key too.
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_json

    def validate(self) -> None:
        for name in ("input_uri", "output_uri", "boundary_uri"):
            if not getattr(self, name):
                raise ValueError(f"config.{name} is required")
        if self.credentials_json and not Path(self.credentials_json).exists():
            raise FileNotFoundError(f"credentials_json not found: {self.credentials_json}")
        if not 0.1 <= self.memory_fraction <= 0.95:
            raise ValueError("memory_fraction must be between 0.1 and 0.95")
        if self.min_polygon_acres < 0 or self.min_total_acres < 0:
            raise ValueError("acreage thresholds must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
