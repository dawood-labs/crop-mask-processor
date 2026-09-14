"""Google Cloud Storage helpers.

Everything the pipeline reads and writes lives in GCS; local disk is only ever
used as per-district scratch space that is deleted as soon as the district is
done. That keeps the instance's disk footprint flat no matter how many
districts are processed.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from google.api_core import exceptions as gexc
from google.api_core import retry
from google.cloud import storage

log = logging.getLogger(__name__)

#: Retry transient GCS failures; the default policy does not cover 5xx on upload.
_RETRY = retry.Retry(
    predicate=retry.if_exception_type(
        gexc.TooManyRequests,
        gexc.InternalServerError,
        gexc.BadGateway,
        gexc.ServiceUnavailable,
        gexc.GatewayTimeout,
        ConnectionError,
    ),
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    timeout=600.0,
)

# One client per process. The GCS client is not fork-safe, so each worker
# process builds its own on first use instead of inheriting the parent's.
_local = threading.local()
_client_pid: int | None = None
_client: storage.Client | None = None
_client_lock = threading.Lock()


@dataclass(frozen=True)
class GcsPath:
    bucket: str
    prefix: str

    @classmethod
    def parse(cls, uri: str) -> "GcsPath":
        if not uri.startswith("gs://"):
            raise ValueError(f"not a gs:// URI: {uri}")
        rest = uri[5:]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"missing bucket in URI: {uri}")
        return cls(bucket=bucket, prefix=prefix.strip("/"))

    def child(self, *parts: str) -> "GcsPath":
        extra = "/".join(p.strip("/") for p in parts if p)
        prefix = f"{self.prefix}/{extra}" if self.prefix else extra
        return GcsPath(self.bucket, prefix)

    def __str__(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}" if self.prefix else f"gs://{self.bucket}"


def is_gcs(uri: str) -> bool:
    return str(uri).startswith("gs://")


def get_client(credentials_json: str | None = None) -> storage.Client:
    """Process-local GCS client, rebuilt automatically after a fork."""
    global _client, _client_pid
    pid = os.getpid()
    with _client_lock:
        if _client is None or _client_pid != pid:
            key = credentials_json or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if key and Path(key).exists():
                _client = storage.Client.from_service_account_json(key)
            else:
                _client = storage.Client()  # ADC / metadata server
            _client_pid = pid
    return _client


def list_blobs(uri: str, credentials_json: str | None = None) -> list[tuple[str, int]]:
    """Every object under ``uri`` as ``(full_name, size_bytes)``."""
    gp = GcsPath.parse(uri)
    client = get_client(credentials_json)
    prefix = f"{gp.prefix}/" if gp.prefix else None
    return [
        (b.name, b.size or 0)
        for b in client.list_blobs(gp.bucket, prefix=prefix)
        if not b.name.endswith("/")
    ]


def download_many(
    bucket: str,
    names: list[str],
    dest_root: Path,
    strip_prefix: str = "",
    threads: int = 8,
    credentials_json: str | None = None,
) -> list[Path]:
    """Download objects in parallel, mirroring their layout under ``dest_root``."""
    client = get_client(credentials_json)
    bkt = client.bucket(bucket)
    dest_root = Path(dest_root)

    def _one(name: str) -> Path:
        rel = name[len(strip_prefix):].lstrip("/") if strip_prefix else name
        out = dest_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        bkt.blob(name).download_to_filename(str(out), retry=_RETRY)
        return out

    if not names:
        return []
    with ThreadPoolExecutor(max_workers=min(threads, len(names))) as ex:
        return list(ex.map(_one, names))


def upload_many(
    files: list[tuple[Path, str]],
    bucket: str,
    threads: int = 8,
    credentials_json: str | None = None,
) -> int:
    """Upload ``(local_path, blob_name)`` pairs. Returns the byte count."""
    if not files:
        return 0
    client = get_client(credentials_json)
    bkt = client.bucket(bucket)

    def _one(item: tuple[Path, str]) -> int:
        local, name = item
        blob = bkt.blob(name)
        blob.upload_from_filename(str(local), retry=_RETRY)
        return local.stat().st_size

    with ThreadPoolExecutor(max_workers=min(threads, len(files))) as ex:
        return sum(ex.map(_one, files))


def upload_dir(
    local_dir: Path,
    dest: GcsPath,
    threads: int = 8,
    credentials_json: str | None = None,
) -> int:
    local_dir = Path(local_dir)
    items = [
        (p, f"{dest.prefix}/{p.relative_to(local_dir).as_posix()}".lstrip("/"))
        for p in local_dir.rglob("*")
        if p.is_file()
    ]
    return upload_many(items, dest.bucket, threads, credentials_json)


def blob_exists(uri: str, credentials_json: str | None = None) -> bool:
    gp = GcsPath.parse(uri)
    return get_client(credentials_json).bucket(gp.bucket).blob(gp.prefix).exists()


def fetch_to_local(uri: str, dest_dir: Path, credentials_json: str | None = None) -> Path:
    """Fetch a single object (and, for a .shp, its sidecars) to ``dest_dir``."""
    from .constants import SHAPEFILE_EXTENSIONS

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not is_gcs(uri):
        return Path(uri)

    gp = GcsPath.parse(uri)
    client = get_client(credentials_json)
    bkt = client.bucket(gp.bucket)

    names = [gp.prefix]
    if gp.prefix.lower().endswith(".shp"):
        stem = gp.prefix[:-4]
        names = [
            b.name
            for b in client.list_blobs(gp.bucket, prefix=stem)
            if Path(b.name).suffix.lower() in SHAPEFILE_EXTENSIONS
        ] or names

    for name in names:
        out = dest_dir / Path(name).name
        bkt.blob(name).download_to_filename(str(out), retry=_RETRY)

    return dest_dir / Path(gp.prefix).name


def rmtree_quiet(path: Path) -> None:
    """Delete scratch space; never let cleanup failure kill a run."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("could not remove %s: %s", path, exc)
