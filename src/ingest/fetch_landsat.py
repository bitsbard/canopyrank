#!/usr/bin/env python3
"""Pull a cloud-masked summer Landsat 8/9 LST composite via Earth Engine."""

from __future__ import annotations

import geopandas as gpd
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import configured_path, ensure_parent, load_region_config  # noqa: E402

try:
    from fetch_parcels import load_county_boundary
except ImportError:  # package-style import
    from src.ingest.fetch_parcels import load_county_boundary  # noqa: E402


def initialize_earth_engine(config: dict[str, Any]) -> None:
    """Initialize the Earth Engine client from environment variables.

    Reads (see ``earth_engine`` in the region config):

    * ``EE_PROJECT`` / ``GOOGLE_CLOUD_PROJECT`` — Cloud project id
    * ``GOOGLE_APPLICATION_CREDENTIALS`` — service-account JSON
    * ``EARTHENGINE_TOKEN`` — optional OAuth access token
    """
    import ee
    from google.oauth2.credentials import Credentials as OAuthCredentials

    ee_cfg = config["earth_engine"]
    project = os.environ.get(ee_cfg["project_env"]) or os.environ.get("GOOGLE_CLOUD_PROJECT")
    token = os.environ.get(ee_cfg["token_env"])
    creds_path = os.environ.get(ee_cfg["credentials_env"])

    if not project and not creds_path and not token:
        raise RuntimeError(
            "Earth Engine is not authenticated. Set "
            f"{ee_cfg['project_env']} (or GOOGLE_CLOUD_PROJECT) and either "
            f"{ee_cfg['credentials_env']} or {ee_cfg['token_env']}. "
            "Register a free account at https://earthengine.google.com/"
        )

    kwargs: dict[str, Any] = {}
    if project:
        kwargs["project"] = project
    if token:
        credentials = OAuthCredentials(token=token)
        ee.Initialize(credentials=credentials, **kwargs)
        return
    if creds_path and not Path(creds_path).is_file():
        raise FileNotFoundError(
            f"{ee_cfg['credentials_env']}={creds_path} is not a file."
        )
    ee.Initialize(**kwargs)


def county_ee_geometry(config: dict[str, Any]) -> Any:
    """Return an ``ee.Geometry`` for the study area (land only), in EPSG:4326."""
    import ee

    parcels_path = configured_path(config, "parcels")
    if parcels_path.is_file():
        # Prefer the parcel layer's dissolved extent — land only, no open water.
        parcels = gpd.read_file(parcels_path).to_crs(config["crs"]["geographic"])
        boundary = gpd.GeoDataFrame(
            geometry=[parcels.union_all().convex_hull], crs=parcels.crs
        )
    else:
        boundary = load_county_boundary(config).to_crs(config["crs"]["geographic"])

    geojson = boundary.__geo_interface__
    return ee.FeatureCollection(geojson).geometry()


def _mask_and_scale_lst(image: Any, config: dict[str, Any]) -> Any:
    """Cloud-mask a Landsat Collection-2 L2 image and return LST in °C."""
    landsat = config["landsat"]
    qa = image.select(landsat["qa_band"])
    # QA_PIXEL bits: 1 dilated cloud, 3 cloud, 4 cloud shadow
    clear = (
        qa.bitwiseAnd(1 << 1)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    lst = (
        image.select(landsat["thermal_band"])
        .multiply(landsat["st_scale"])
        .add(landsat["st_offset"])
        .subtract(273.15)
        .rename("LST")
        .updateMask(clear)
    )
    return lst.copyProperties(image, ["system:time_start"])


def build_summer_lst_composite(config: dict[str, Any], region: Any) -> Any:
    """Median summer Landsat 8/9 LST composite, cloud-masked, clipped to ``region``."""
    import ee

    landsat = config["landsat"]
    months = landsat["months"]
    collection = None
    for asset in landsat["collections"]:
        col = (
            ee.ImageCollection(asset)
            .filterBounds(region)
            .filterDate(landsat["start_date"], landsat["end_date"])
            .filter(ee.Filter.calendarRange(int(months[0]), int(months[-1]), "month"))
            .filter(ee.Filter.lt("CLOUD_COVER", landsat["max_cloud_cover"]))
            .map(lambda img, cfg=config: _mask_and_scale_lst(img, cfg))
        )
        collection = col if collection is None else collection.merge(col)
    if collection is None:
        raise RuntimeError("No Landsat collections configured.")
    size = collection.size().getInfo()
    if size == 0:
        raise RuntimeError(
            "No Landsat 8/9 scenes matched the summer/cloud filters for this region. "
            "Adjust landsat.start_date / end_date / max_cloud_cover in the region config."
        )
    return collection.median().clip(region).unmask(-9999).toFloat()


def export_ee_image_geotiff(
    image: Any,
    config: dict[str, Any],
    out_path: Path,
    region: Any,
    scale_m: int,
    crs: str,
    nodata: float = -9999.0,
) -> Path:
    """Download an EE image as a GeoTIFF via tiled ``computePixels`` calls."""
    import ee
    from pyproj import Transformer

    # Earth Engine's Geometry.transform() does not reliably reproject FeatureCollection-
    # derived geometries server-side, so compute the working-CRS bounding box locally.
    coords = region.bounds(maxError=1).coordinates().getInfo()[0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    tile_px = int(config["earth_engine"].get("export_tile_px", 1024))
    width_px = int(math.ceil((xmax - xmin) / scale_m))
    height_px = int(math.ceil((ymax - ymin) / scale_m))
    if width_px <= 0 or height_px <= 0:
        raise RuntimeError(f"Invalid export size {width_px}x{height_px} for {out_path}")

    tmp_dir = out_path.parent / f".{out_path.stem}_tiles"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tile_paths: list[Path] = []

    y0 = ymax
    row = 0
    while y0 > ymin:
        tile_h = min(tile_px, int(math.ceil((y0 - ymin) / scale_m)))
        x0 = xmin
        col = 0
        while x0 < xmax:
            tile_w = min(tile_px, int(math.ceil((xmax - x0) / scale_m)))
            request = {
                "expression": image,
                "fileFormat": "GEO_TIFF",
                "grid": {
                    "dimensions": {"width": tile_w, "height": tile_h},
                    "affineTransform": {
                        "scaleX": scale_m,
                        "shearX": 0,
                        "translateX": x0,
                        "shearY": 0,
                        "scaleY": -scale_m,
                        "translateY": y0,
                    },
                    "crsCode": crs,
                },
            }
            raw = ee.data.computePixels(request)
            tile_path = tmp_dir / f"tile_{row}_{col}.tif"
            if isinstance(raw, (bytes, bytearray)):
                tile_path.write_bytes(bytes(raw))
            else:
                # Some API versions return a dict/array; write via rasterio.
                arr = np.array(raw, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                transform = from_origin(x0, y0, scale_m, scale_m)
                with rasterio.open(
                    tile_path,
                    "w",
                    driver="GTiff",
                    height=arr.shape[-2],
                    width=arr.shape[-1],
                    count=arr.shape[0],
                    dtype=arr.dtype,
                    crs=crs,
                    transform=transform,
                    nodata=nodata,
                ) as dst:
                    dst.write(arr)
            tile_paths.append(tile_path)
            x0 += tile_w * scale_m
            col += 1
        y0 -= tile_h * scale_m
        row += 1

    ensure_parent(out_path)
    if len(tile_paths) == 1:
        out_path.write_bytes(tile_paths[0].read_bytes())
    else:
        datasets = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = rio_merge(datasets, nodata=nodata)
        meta = datasets[0].meta.copy()
        for ds in datasets:
            ds.close()
        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "nodata": nodata,
            }
        )
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
    return out_path


def run(config: dict[str, Any]) -> Path:
    """Build and export the summer LST composite.

    Returns
    -------
    pathlib.Path
        Path to the GeoTIFF under ``data/interim/``.
    """
    initialize_earth_engine(config)
    region = county_ee_geometry(config)
    image = build_summer_lst_composite(config, region)
    out = configured_path(config, "landsat_lst")
    export_ee_image_geotiff(
        image,
        config,
        out,
        region,
        scale_m=int(config["landsat"]["scale_m"]),
        crs=config["crs"]["working"],
    )
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch a cloud-masked summer Landsat 8/9 LST composite."
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
    print(f"Wrote Landsat LST composite to {out}")


if __name__ == "__main__":
    main()
