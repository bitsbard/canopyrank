#!/usr/bin/env python3
"""Apply the trained downscaling model to every parcel in the study area."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import configured_path, ensure_parent, load_region_config  # noqa: E402


def load_model(config: dict[str, Any]) -> XGBRegressor:
    """Load the trained XGBRegressor from the path in the region config."""
    model_path = configured_path(config, "model")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run: python src/model/train.py --region <config>"
        )
    model = XGBRegressor()
    model.load_model(model_path)
    return model


def run(config: dict[str, Any]) -> Path:
    """Write parcel-level predicted LST for the full feature table.

    Returns
    -------
    pathlib.Path
        Path to the predictions GeoPackage.
    """
    features_path = configured_path(config, "feature_table")
    if not features_path.is_file():
        raise FileNotFoundError(
            f"Feature table not found: {features_path}. "
            "Run: python src/features/build_feature_table.py --region <config>"
        )
    gdf = gpd.read_file(features_path)
    features = list(config["model"]["features"])
    missing = [col for col in features if col not in gdf.columns]
    if missing:
        raise KeyError(f"Feature table is missing model columns: {missing}")

    model = load_model(config)
    x = gdf[features].replace([np.inf, -np.inf], np.nan)
    complete = x.notna().all(axis=1)
    preds = np.full(len(gdf), np.nan, dtype=np.float32)
    if complete.any():
        preds[complete.to_numpy()] = model.predict(x.loc[complete, features])
    gdf = gdf.copy()
    gdf["lst_pred"] = preds

    # Heat-reduction potential: predicted cooling if canopy is raised to the target.
    target_pct = float(config["ranking"]["canopy_target_pct"])
    x_cf = x.copy()
    if "canopy_pct" in x_cf.columns:
        x_cf["canopy_pct"] = np.maximum(x_cf["canopy_pct"], target_pct)
    cf = np.full(len(gdf), np.nan, dtype=np.float32)
    if complete.any():
        cf[complete.to_numpy()] = model.predict(x_cf.loc[complete, features])
    gdf["lst_pred_target_canopy"] = cf
    gdf["heat_reduction_potential"] = gdf["lst_pred"] - gdf["lst_pred_target_canopy"]
    gdf["heat_reduction_potential"] = gdf["heat_reduction_potential"].clip(lower=0)

    out = configured_path(config, "predictions")
    ensure_parent(out)
    gdf.to_file(out, driver="GPKG")
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Apply the LST downscaling model to all parcels."
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
    print(f"Wrote parcel-level predicted LST to {out}")


if __name__ == "__main__":
    main()
