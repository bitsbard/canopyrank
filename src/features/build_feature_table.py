#!/usr/bin/env python3
"""Join per-parcel zonal stats with CalEnviroScreen into one feature table."""

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

from src import configured_path, ensure_parent, load_region_config  # noqa: E402


def _ensure_inputs(config: dict[str, Any]) -> None:
    """Run upstream ingest/feature steps when their outputs are missing."""
    ingest_dir = str(ROOT / "src" / "ingest")
    features_dir = str(ROOT / "src" / "features")
    for path in (ingest_dir, features_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    if not configured_path(config, "parcels").is_file():
        from fetch_parcels import run as fetch_parcels

        fetch_parcels(config)
    if not configured_path(config, "landsat_lst").is_file():
        from fetch_landsat import run as fetch_landsat

        fetch_landsat(config)
    if not configured_path(config, "canopy_raster").is_file() or not configured_path(
        config, "nlcd_impervious"
    ).is_file():
        from fetch_canopy import run as fetch_canopy

        fetch_canopy(config)
    if not configured_path(config, "calenviroscreen").is_file():
        from fetch_calenviroscreen import run as fetch_ces

        fetch_ces(config)
    if not configured_path(config, "zonal_stats").is_file():
        from zonal_stats import run as zonal_run

        zonal_run(config)


def _keyword_hit(series: pd.Series, keywords: list[str]) -> pd.Series:
    """Return a boolean series true when any keyword appears (case-insensitive)."""
    text = series.fillna("").astype(str).str.lower()
    hit = pd.Series(False, index=series.index)
    for word in keywords:
        hit = hit | text.str.contains(str(word).lower(), regex=False)
    return hit


def add_feasibility(parcels: gpd.GeoDataFrame, config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Flag public land, vacant lots, and right-of-way from assessor attributes."""
    cfg = config["parcels"]
    out = parcels.copy()
    use_desc = cfg.get("use_desc_field")
    park_field = cfg.get("park_field")
    gp_parks = cfg.get("gp_parks_field")

    public = pd.Series(False, index=out.index)
    vacant = pd.Series(False, index=out.index)
    row = pd.Series(False, index=out.index)

    if park_field and park_field in out.columns:
        public = public | out[park_field].fillna("").astype(str).str.strip().ne("")
    if gp_parks and gp_parks in out.columns:
        public = public | out[gp_parks].fillna("").astype(str).str.strip().ne("")
    if use_desc and use_desc in out.columns:
        public = public | _keyword_hit(out[use_desc], cfg.get("public_keywords") or [])
        vacant = vacant | _keyword_hit(out[use_desc], cfg.get("vacant_keywords") or [])
        row = row | _keyword_hit(out[use_desc], cfg.get("row_keywords") or [])

    out["is_public"] = public.astype(int)
    out["is_vacant"] = vacant.astype(int)
    out["is_right_of_way"] = row.astype(int)
    out["feasible"] = ((out["is_public"] + out["is_vacant"] + out["is_right_of_way"]) > 0).astype(int)
    return out


def run(config: dict[str, Any]) -> Path:
    """Join zonal stats and CalEnviroScreen percentiles; write ``data/processed/``.

    Returns
    -------
    pathlib.Path
        Path to the feature GeoPackage.
    """
    _ensure_inputs(config)
    parcels = gpd.read_file(configured_path(config, "zonal_stats"))
    ces = gpd.read_file(configured_path(config, "calenviroscreen"))
    ces_cfg = config["calenviroscreen"]
    keep_cols = ["geometry", ces_cfg["percentile_field"]]
    tract_field = ces_cfg.get("tract_field")
    if tract_field and tract_field in ces.columns:
        keep_cols.append(tract_field)
    ces_join = ces[keep_cols].copy()
    ces_join = ces_join.rename(columns={ces_cfg["percentile_field"]: "ces_percentile"})
    if tract_field and tract_field in ces_join.columns:
        ces_join = ces_join.rename(columns={tract_field: "ces_tract"})

    joined = gpd.sjoin(parcels, ces_join, how="left", predicate="intersects")
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    id_field = config["parcels"]["id_field"]
    if id_field in joined.columns:
        joined["ces_percentile"] = pd.to_numeric(joined["ces_percentile"], errors="coerce")
        joined = joined.sort_values("ces_percentile", ascending=False)
        joined = joined.drop_duplicates(subset=[id_field], keep="first")
    joined["ces_percentile"] = pd.to_numeric(joined["ces_percentile"], errors="coerce")
    joined = add_feasibility(joined, config)

    out = configured_path(config, "feature_table")
    ensure_parent(out)
    joined.to_file(out, driver="GPKG")
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build the per-parcel feature table.")
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
    print(f"Wrote feature table to {out}")


if __name__ == "__main__":
    main()
