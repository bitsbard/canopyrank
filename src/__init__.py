"""Shared helpers for the CanopyRank pipeline."""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

USER_AGENT = "CanopyRank/0.1 (https://github.com/bitsbard/canopyrank)"


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent


def load_region_config(config_path: str | Path) -> dict[str, Any]:
    """Load a region YAML config, resolving the path against cwd then repo root."""
    path = Path(config_path)
    if not path.is_file():
        candidate = project_root() / path
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Region config not found: {config_path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Region config must be a mapping: {path}")
    data["_config_path"] = str(path.resolve())
    return data


def repo_path(config: dict[str, Any], *parts: str) -> Path:
    """Join ``parts`` onto the repository root."""
    del config  # config is accepted so call sites stay consistent
    return project_root().joinpath(*parts)


def configured_path(config: dict[str, Any], key: str) -> Path:
    """Resolve ``paths.<key>`` from the region config against the repo root."""
    rel = config["paths"][key]
    path = Path(rel)
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_parent(path: Path) -> Path:
    """Create parent directories for ``path`` and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download_url(url: str, dest: Path, timeout: int = 120) -> Path:
    """Download ``url`` to ``dest`` using a descriptive, browser-like User-Agent."""
    ensure_parent(dest)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            dest.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Failed to download {url} (HTTP {exc.code}). "
            f"The host may be blocking automated requests — try downloading "
            f"manually in a browser and placing the file at {dest}."
        ) from exc
    return dest


def unzip_archive(archive: Path, dest_dir: Path) -> Path:
    """Extract a zip archive and return ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    return dest_dir


def find_shapefile(directory: Path) -> Path:
    """Return the first ``.shp`` under ``directory`` (recursive)."""
    matches = sorted(directory.rglob("*.shp"))
    if not matches:
        raise FileNotFoundError(f"No shapefile found under {directory}")
    return matches[0]


def arcgis_query_geojson(
    layer_url: str,
    where: str = "1=1",
    out_fields: str = "*",
    page_size: int = 2000,
    out_sr: int = 4326,
    timeout: int = 120,
) -> dict[str, Any]:
    """Page through an ArcGIS feature layer and return a GeoJSON FeatureCollection."""
    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = (
            f"where={urllib.parse.quote(where)}"
            f"&outFields={urllib.parse.quote(out_fields)}"
            f"&f=geojson"
            f"&returnGeometry=true"
            f"&outSR={out_sr}"
            f"&resultOffset={offset}"
            f"&resultRecordCount={page_size}"
        )
        url = f"{layer_url.rstrip('/')}/query?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("error"):
            raise RuntimeError(f"ArcGIS query failed for {layer_url}: {payload['error']}")
        batch = payload.get("features") or []
        features.extend(batch)
        exceeded = payload.get("exceededTransferLimit") or payload.get("properties", {}).get(
            "exceededTransferLimit"
        )
        if len(batch) < page_size and not exceeded:
            break
        if not batch:
            break
        offset += len(batch)
    return {"type": "FeatureCollection", "features": features}


def add_root_to_syspath() -> None:
    """Put the repo root on ``sys.path`` so ``import src`` works from CLI scripts."""
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
