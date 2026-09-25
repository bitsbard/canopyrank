#!/usr/bin/env python3
"""Load, clean, and reproject county parcel boundaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    arcgis_query_geojson,
    configured_path,
    download_url,
    ensure_parent,
    find_shapefile,
    load_region_config,
    unzip_archive,
)


def load_county_boundary(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Load the study-area polygon from a local shapefile or Census TIGER/cartographic files.

    Parameters
    ----------
    config:
        Region config mapping.

    Returns
    -------
    geopandas.GeoDataFrame
        Single- (or multi-) polygon county boundary in the working CRS.
    """
    boundary_cfg = config["boundary"]
    working_crs = config["crs"]["working"]
    shapefile = boundary_cfg.get("shapefile")
    if shapefile:
        path = Path(shapefile)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(
                f"County boundary shapefile not found: {path}. "
                "Set boundary.shapefile or rely on boundary.fips / boundary.tiger_url."
            )
        gdf = gpd.read_file(path)
    else:
        fips = str(boundary_cfg.get("fips") or config["region"]["fips"])
        state_fp, county_fp = fips[:2], fips[2:]
        tiger_url = boundary_cfg["tiger_url"]
        raw_zip = configured_path(config, "data_raw") / "tiger_counties.zip"
        extract_dir = configured_path(config, "data_raw") / "tiger_counties"
        if not raw_zip.is_file():
            download_url(tiger_url, raw_zip)
        if not extract_dir.exists() or not any(extract_dir.rglob("*.shp")):
            unzip_archive(raw_zip, extract_dir)
        gdf = gpd.read_file(find_shapefile(extract_dir))
        state_col = "STATEFP" if "STATEFP" in gdf.columns else "STATEFP20"
        county_col = "COUNTYFP" if "COUNTYFP" in gdf.columns else "COUNTYFP20"
        gdf = gdf[(gdf[state_col].astype(str) == state_fp) & (gdf[county_col].astype(str) == county_fp)]
        if gdf.empty:
            raise RuntimeError(
                f"No county polygon matched FIPS {fips} in {tiger_url}."
            )
    gdf = _clean_geometries(gdf).to_crs(working_crs)
    out = configured_path(config, "county_boundary")
    ensure_parent(out)
    gdf.to_file(out, driver="GeoJSON")
    return gdf


def _clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop empty geometries and repair invalid ones."""
    cleaned = gdf.copy()
    cleaned = cleaned[cleaned.geometry.notna() & ~cleaned.geometry.is_empty]
    cleaned["geometry"] = cleaned.geometry.make_valid()
    cleaned = cleaned.explode(index_parts=False)
    geom_types = cleaned.geometry.geom_type
    cleaned = cleaned[geom_types.isin(["Polygon", "MultiPolygon"])]
    dissolved_types = cleaned.geometry.geom_type
    if (dissolved_types == "Polygon").any() or (dissolved_types == "MultiPolygon").any():
        cleaned = cleaned[cleaned.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if cleaned.empty:
        raise RuntimeError("No valid polygon geometries remain after cleaning.")
    return cleaned.reset_index(drop=True)


def load_parcels(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Fetch parcel polygons from a local file or the configured ArcGIS layer."""
    parcel_cfg = config["parcels"]
    local_path = parcel_cfg.get("local_path")
    if local_path:
        path = Path(local_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"Parcel file not found: {path}. "
                f"Download from {parcel_cfg['url']} or update parcels.local_path."
            )
        parcels = gpd.read_file(path)
    else:
        url = parcel_cfg["url"]
        fields = parcel_cfg.get("out_fields") or ["*"]
        out_fields = ",".join(fields) if fields != ["*"] else "*"
        collection = arcgis_query_geojson(
            url,
            out_fields=out_fields,
            page_size=int(parcel_cfg.get("page_size", 2000)),
        )
        if not collection["features"]:
            raise RuntimeError(
                "Parcel query returned no features. "
                f"Confirm the service is public: {url}/query"
            )
        parcels = gpd.GeoDataFrame.from_features(collection, crs=config["crs"]["geographic"])
    return parcels


def run(config: dict[str, Any]) -> Path:
    """Clean, reproject, clip, and write parcels to ``data/interim/``.

    Returns
    -------
    pathlib.Path
        Path to the written GeoPackage.
    """
    boundary = load_county_boundary(config)
    parcels = load_parcels(config)
    parcels = _clean_geometries(parcels).to_crs(config["crs"]["working"])
    parcels = gpd.overlay(parcels, boundary[["geometry"]], how="intersection", keep_geom_type=True)
    parcels = parcels[~parcels.geometry.is_empty].copy()
    id_field = config["parcels"]["id_field"]
    if id_field in parcels.columns:
        parcels = parcels.drop_duplicates(subset=[id_field], keep="first")
    acres_field = config["parcels"].get("acres_field")
    if acres_field and acres_field in parcels.columns:
        parcels[acres_field] = pd.to_numeric(parcels[acres_field], errors="coerce")
    out = configured_path(config, "parcels")
    ensure_parent(out)
    parcels.to_file(out, driver="GPKG")
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Fetch and clean county parcel boundaries.")
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
    print(f"Wrote cleaned parcels to {out}")


if __name__ == "__main__":
    main()
