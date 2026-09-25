#!/usr/bin/env python3
"""Compute per-parcel zonal statistics for LST, canopy, and impervious rasters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import configured_path, ensure_parent, load_region_config  # noqa: E402


def _require_raster(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required raster not found: {path}. {hint}")
    return path


def _mean_stats(geoms: gpd.GeoSeries, raster: Path, nodata: float | None) -> list[float]:
    stats = zonal_stats(
        geoms,
        str(raster),
        stats=["mean"],
        nodata=nodata,
        geojson_out=False,
        all_touched=True,
    )
    return [row["mean"] if row["mean"] is not None else np.nan for row in stats]


def run(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Return parcels with ``lst_mean``, ``canopy_pct``, and ``impervious_pct``.

    Parameters
    ----------
    config:
        Region config mapping.
    """
    parcels_path = configured_path(config, "parcels")
    if not parcels_path.is_file():
        raise FileNotFoundError(
            f"Cleaned parcels not found: {parcels_path}. "
            "Run: python src/ingest/fetch_parcels.py --region <config>"
        )
    parcels = gpd.read_file(parcels_path)
    lst_path = _require_raster(
        configured_path(config, "landsat_lst"),
        "Run: python src/ingest/fetch_landsat.py --region <config>",
    )
    canopy_path = _require_raster(
        configured_path(config, "canopy_raster"),
        "Run: python src/ingest/fetch_canopy.py --region <config>",
    )
    impervious_path = _require_raster(
        configured_path(config, "nlcd_impervious"),
        "Run: python src/ingest/fetch_canopy.py --region <config>",
    )

    geoms = parcels.geometry
    parcels = parcels.copy()
    parcels["lst_mean"] = _mean_stats(geoms, lst_path, nodata=-9999)
    parcels["canopy_pct"] = _mean_stats(geoms, canopy_path, nodata=-9999)
    parcels["impervious_pct"] = _mean_stats(geoms, impervious_path, nodata=-9999)
    parcels["area_m2"] = parcels.geometry.area
    parcels["lst_mean"] = pd.to_numeric(parcels["lst_mean"], errors="coerce")
    parcels["canopy_pct"] = pd.to_numeric(parcels["canopy_pct"], errors="coerce").clip(0, 100)
    parcels["impervious_pct"] = pd.to_numeric(parcels["impervious_pct"], errors="coerce").clip(0, 100)

    out = configured_path(config, "zonal_stats")
    ensure_parent(out)
    parcels.to_file(out, driver="GPKG")
    return parcels


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Per-parcel zonal stats for LST and canopy.")
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
    gdf = run(config)
    print(f"Wrote zonal stats for {len(gdf)} parcels to {configured_path(config, 'zonal_stats')}")


if __name__ == "__main__":
    main()
