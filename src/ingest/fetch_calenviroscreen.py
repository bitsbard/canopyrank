#!/usr/bin/env python3
"""Load CalEnviroScreen and clip it to the county boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    configured_path,
    download_url,
    ensure_parent,
    find_shapefile,
    load_region_config,
    unzip_archive,
)

try:
    from fetch_parcels import load_county_boundary
except ImportError:
    from src.ingest.fetch_parcels import load_county_boundary  # noqa: E402


def load_calenviroscreen(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Download (if needed) and read CalEnviroScreen tract polygons."""
    ces_cfg = config["calenviroscreen"]
    raw_zip = configured_path(config, "data_raw") / "calenviroscreen4.zip"
    extract_dir = configured_path(config, "data_raw") / "calenviroscreen4"
    if not raw_zip.is_file():
        try:
            download_url(ces_cfg["shapefile_url"], raw_zip, timeout=180)
        except Exception as exc:  # noqa: BLE001 — surface the official URL
            raise RuntimeError(
                "Failed to download CalEnviroScreen shapefile. "
                f"Get it from {ces_cfg['shapefile_url']} "
                f"or the feature service {ces_cfg['feature_service']} "
                "(https://oehha.ca.gov/calenviroscreen/download-data)."
            ) from exc
    if not extract_dir.exists() or not any(extract_dir.rglob("*.shp")):
        unzip_archive(raw_zip, extract_dir)
    gdf = gpd.read_file(find_shapefile(extract_dir))
    return gdf


def run(config: dict[str, Any]) -> Path:
    """Clip CalEnviroScreen tracts to the county and write a GeoPackage.

    Returns
    -------
    pathlib.Path
        Path to ``data/interim/calenviroscreen.gpkg``.
    """
    ces_cfg = config["calenviroscreen"]
    boundary = load_county_boundary(config)
    ces = load_calenviroscreen(config).to_crs(config["crs"]["working"])
    county_field = ces_cfg.get("county_field")
    county_name = ces_cfg.get("county_name")
    if county_field and county_field in ces.columns and county_name:
        filtered = ces[
            ces[county_field].astype(str).str.contains(str(county_name), case=False, na=False)
        ]
        if not filtered.empty:
            ces = filtered
    ces = gpd.overlay(ces, boundary[["geometry"]], how="intersection", keep_geom_type=True)
    percentile = ces_cfg["percentile_field"]
    if percentile not in ces.columns:
        raise KeyError(
            f"CalEnviroScreen percentile field {percentile!r} not in columns: {list(ces.columns)}. "
            "See the data dictionary at https://oehha.ca.gov/calenviroscreen/download-data"
        )
    out = configured_path(config, "calenviroscreen")
    ensure_parent(out)
    ces.to_file(out, driver="GPKG")
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Load CalEnviroScreen and join/clip to county geography."
    )
    parser.add_argument(
        "--region",
        default="config/santa_cruz.yaml",
        help="Path to the region YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_region_config(args.region)
    out = run(config)
    print(f"Wrote county CalEnviroScreen tracts to {out}")


if __name__ == "__main__":
    main()
