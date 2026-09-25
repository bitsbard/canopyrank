#!/usr/bin/env python3
"""Rank parcels by heat-reduction potential, EJ weight, and feasibility."""

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


def _ensure_predictions(config: dict[str, Any]) -> Path:
    """Return the predictions path, running downscale (and train) if needed."""
    pred_path = configured_path(config, "predictions")
    if pred_path.is_file():
        return pred_path
    model_dir = str(Path(__file__).resolve().parent)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    if not configured_path(config, "model").is_file():
        from train import run as train_run

        train_run(config)
    from downscale import run as downscale_run

    return downscale_run(config)


def rank(config: dict[str, Any], top_n: int) -> gpd.GeoDataFrame:
    """Compute ranking scores and return the top-N parcels.

    Score = predicted heat-reduction potential × EJ percentile weight × feasibility.
    """
    pred_path = _ensure_predictions(config)
    gdf = gpd.read_file(pred_path)
    ej_field = config["ranking"]["ej_percentile_field"]
    if ej_field not in gdf.columns:
        raise KeyError(f"EJ field {ej_field!r} missing from predictions table.")
    if "heat_reduction_potential" not in gdf.columns:
        raise KeyError("heat_reduction_potential missing; re-run src/model/downscale.py")
    if "feasible" not in gdf.columns:
        raise KeyError("feasible missing; re-run src/features/build_feature_table.py")

    out = gdf.copy()
    ej = pd.to_numeric(out[ej_field], errors="coerce").fillna(0).clip(0, 100) / 100.0
    heat = pd.to_numeric(out["heat_reduction_potential"], errors="coerce").fillna(0)
    feasible = pd.to_numeric(out["feasible"], errors="coerce").fillna(0)
    out["ej_weight"] = ej
    out["rank_score"] = heat * ej * feasible
    ranked = out.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    if top_n > 0:
        ranked = ranked.loc[ranked["rank"] <= top_n].copy()
    return ranked


def run(config: dict[str, Any], top_n: int) -> Path:
    """Write top-N ranked parcels as GeoJSON.

    Returns
    -------
    pathlib.Path
        Path to ``outputs/ranked_parcels.geojson``.
    """
    ranked = rank(config, top_n=top_n)
    geographic = config["crs"]["geographic"]
    ranked = ranked.to_crs(geographic)
    out = configured_path(config, "ranked_geojson")
    ensure_parent(out)
    ranked.to_file(out, driver="GeoJSON")
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Rank parcels for tree-planting priority and write GeoJSON."
    )
    parser.add_argument(
        "--region",
        default="config/santa_cruz.yaml",
        help="Path to the region YAML config.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top-ranked parcels to export (default: ranking.top_n in config).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_region_config(args.region)
    top_n = args.top_n if args.top_n is not None else int(config["ranking"]["top_n"])
    out = run(config, top_n=top_n)
    print(f"Wrote top {top_n} ranked parcels to {out}")


if __name__ == "__main__":
    main()
