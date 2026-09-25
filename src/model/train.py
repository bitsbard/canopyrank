#!/usr/bin/env python3
"""Train an XGBoost regressor that predicts parcel LST from canopy/impervious features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import yaml
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import configured_path, ensure_parent, load_region_config  # noqa: E402


def spatial_quadrant_mask(gdf: gpd.GeoDataFrame, test_quadrant: int) -> tuple[np.ndarray, np.ndarray]:
    """Split parcels by geographic quadrant using median x/y of centroids.

    Quadrants (relative to median easting/northing):

    * 0 — southwest
    * 1 — southeast
    * 2 — northwest
    * 3 — northeast

    A spatial split is required because LST is strongly spatially autocorrelated;
    a row-random split would leak nearby parcels into both sets and inflate R².

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Boolean train mask and test mask.
    """
    centroids = gdf.geometry.centroid
    x_mid = float(centroids.x.median())
    y_mid = float(centroids.y.median())
    east = (centroids.x >= x_mid).to_numpy()
    north = (centroids.y >= y_mid).to_numpy()
    quadrant = east.astype(int) + 2 * north.astype(int)
    test = quadrant == int(test_quadrant)
    train = ~test
    if train.sum() == 0 or test.sum() == 0:
        raise RuntimeError(
            f"Spatial split produced an empty set (train={train.sum()}, test={test.sum()}). "
            "Choose a different model.test_quadrant."
        )
    return train, test


def load_training_frame(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Load the processed feature table, building it if needed."""
    path = configured_path(config, "feature_table")
    if not path.is_file():
        features_dir = str(ROOT / "src" / "features")
        if features_dir not in sys.path:
            sys.path.insert(0, features_dir)
        from build_feature_table import run as build_features

        build_features(config)
    return gpd.read_file(configured_path(config, "feature_table"))


def run(config: dict[str, Any]) -> Path:
    """Fit the downscaling model, report R² / importances, and save the booster.

    Returns
    -------
    pathlib.Path
        Path to the serialized XGBoost model.
    """
    model_cfg = config["model"]
    features = list(model_cfg["features"])
    target = model_cfg["target"]
    gdf = load_training_frame(config)
    frame = gdf[features + [target, "geometry"]].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])
    if frame.empty:
        raise RuntimeError("No complete rows available for training. Re-run the feature pipeline.")

    train_mask, test_mask = spatial_quadrant_mask(frame, int(model_cfg["test_quadrant"]))
    x_train = frame.loc[train_mask, features]
    y_train = frame.loc[train_mask, target]
    x_test = frame.loc[test_mask, features]
    y_test = frame.loc[test_mask, target]

    model = XGBRegressor(
        n_estimators=int(model_cfg["n_estimators"]),
        max_depth=int(model_cfg["max_depth"]),
        learning_rate=float(model_cfg["learning_rate"]),
        subsample=float(model_cfg.get("subsample", 0.8)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
        min_child_weight=float(model_cfg.get("min_child_weight", 1)),
        objective="reg:squarederror",
        random_state=int(model_cfg["random_state"]),
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    r2 = float(r2_score(y_test, y_hat))
    importances = {
        name: float(score) for name, score in zip(features, model.feature_importances_)
    }
    metrics = {
        "r2_spatial_holdout": r2,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "test_quadrant": int(model_cfg["test_quadrant"]),
        "features": features,
        "target": target,
        "feature_importances": importances,
    }
    model_path = configured_path(config, "model")
    metrics_path = configured_path(config, "model_metrics")
    ensure_parent(model_path)
    model.save_model(model_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metrics, handle, sort_keys=False)

    print(f"Spatial hold-out R² (quadrant {model_cfg['test_quadrant']}): {r2:.4f}")
    print("Feature importances:")
    for name, score in sorted(importances.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {score:.4f}")
    print(f"Wrote model to {model_path}")
    print(f"Wrote metrics to {metrics_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a spatially held-out XGBoost LST downscaling model."
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
    run(config)


if __name__ == "__main__":
    main()
