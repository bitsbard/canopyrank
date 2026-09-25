#!/usr/bin/env python3
"""Pull NLCD land cover / impervious rasters (LiDAR canopy when configured)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import configured_path, ensure_parent, load_region_config  # noqa: E402

try:
    from fetch_landsat import (
        county_ee_geometry,
        export_ee_image_geotiff,
        initialize_earth_engine,
    )
except ImportError:
    from src.ingest.fetch_landsat import (  # noqa: E402
        county_ee_geometry,
        export_ee_image_geotiff,
        initialize_earth_engine,
    )


def _write_canopy_from_landcover(landcover_path: Path, canopy_path: Path, forest_classes: list[int]) -> Path:
    """Encode forest NLCD classes as 100% canopy, all other valid pixels as 0."""
    with rasterio.open(landcover_path) as src:
        data = src.read(1)
        nodata = src.nodata
        canopy = np.zeros(data.shape, dtype=np.float32)
        valid = np.ones(data.shape, dtype=bool)
        if nodata is not None:
            valid = data != nodata
        forest = np.isin(data, forest_classes)
        canopy[forest] = 100.0
        canopy[~valid] = -9999.0
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1, nodata=-9999.0)
        ensure_parent(canopy_path)
        with rasterio.open(canopy_path, "w", **meta) as dst:
            dst.write(canopy, 1)
    return canopy_path


def run(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Fetch canopy and impervious rasters for the study area.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        Land-cover, impervious, and canopy-percent GeoTIFF paths.
    """
    canopy_cfg = config["canopy"]
    landcover_path = configured_path(config, "nlcd_landcover")
    impervious_path = configured_path(config, "nlcd_impervious")
    canopy_path = configured_path(config, "canopy_raster")
    lidar_path = canopy_cfg.get("lidar_path")
    use_lidar = canopy_cfg.get("source") == "lidar" and lidar_path

    if use_lidar:
        src_path = Path(lidar_path)
        if not src_path.is_absolute():
            src_path = ROOT / src_path
        if not src_path.is_file():
            raise FileNotFoundError(
                f"canopy.source is 'lidar' but raster is missing: {src_path}. "
                "Santa Cruz County does not currently publish a ready-to-use tree-canopy "
                "GeoTIFF. Inventory public LiDAR at "
                "https://coast.noaa.gov/dataviewer/#/lidar/search/ "
                "or fall back to NLCD via canopy.source: nlcd "
                f"({canopy_cfg['nlcd']['mrlc_url']})."
            )
        ensure_parent(canopy_path)
        shutil.copy2(src_path, canopy_path)

    initialize_earth_engine(config)
    import ee

    region = county_ee_geometry(config)
    nlcd_cfg = canopy_cfg["nlcd"]
    nlcd_col = ee.ImageCollection(nlcd_cfg["ee_collection"]).filter(
        ee.Filter.eq("system:index", str(nlcd_cfg["year"]))
    )
    if int(nlcd_col.size().getInfo()) == 0:
        raise RuntimeError(
            f"NLCD image not found for year {nlcd_cfg['year']} in "
            f"{nlcd_cfg['ee_collection']}. Direct download: {nlcd_cfg['mrlc_url']}"
        )
    nlcd = nlcd_col.first()
    landcover = nlcd.select(nlcd_cfg["landcover_band"]).clip(region).unmask(0).toUint8()
    impervious = nlcd.select(nlcd_cfg["impervious_band"]).clip(region).unmask(-9999).toFloat()
    scale = int(config["landsat"]["scale_m"])
    crs = config["crs"]["working"]
    export_ee_image_geotiff(landcover, config, landcover_path, region, scale, crs, nodata=0)
    export_ee_image_geotiff(impervious, config, impervious_path, region, scale, crs, nodata=-9999.0)
    if not use_lidar:
        _write_canopy_from_landcover(
            landcover_path,
            canopy_path,
            list(nlcd_cfg["forest_classes"]),
        )
    return landcover_path, impervious_path, canopy_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch NLCD land cover / impervious (or local LiDAR canopy)."
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
    landcover, impervious, canopy = run(config)
    print(f"Wrote land cover to {landcover}")
    print(f"Wrote impervious to {impervious}")
    print(f"Wrote canopy percent to {canopy}")


if __name__ == "__main__":
    main()
