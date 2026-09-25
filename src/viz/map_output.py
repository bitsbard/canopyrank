#!/usr/bin/env python3
"""Render ranked parcels as an interactive Folium map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
from branca.colormap import LinearColormap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import configured_path, ensure_parent, load_region_config  # noqa: E402


def _tooltip_fields(gdf: gpd.GeoDataFrame, config: dict[str, Any]) -> list[str]:
    """Choose attribute columns that exist on the ranked GeoJSON."""
    preferred = [
        config["parcels"]["id_field"],
        "rank",
        "rank_score",
        "heat_reduction_potential",
        "ces_percentile",
        "canopy_pct",
        "impervious_pct",
        "lst_pred",
        "feasible",
        "is_public",
        "is_vacant",
        "is_right_of_way",
    ]
    return [col for col in preferred if col in gdf.columns]


def run(config: dict[str, Any], input_path: Path, output_path: Path | None = None) -> Path:
    """Write an HTML map color-coded by planting rank.

    Parameters
    ----------
    config:
        Region config mapping.
    input_path:
        Ranked parcels GeoJSON.
    output_path:
        Optional HTML destination; defaults to ``paths.ranked_map``.
    """
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Ranked GeoJSON not found: {input_path}. "
            "Run: python src/model/rank_parcels.py --top-n 100"
        )
    gdf = gpd.read_file(input_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(config["crs"]["geographic"])
    gdf = gdf.to_crs(config["crs"]["geographic"])
    if "rank" not in gdf.columns:
        raise KeyError("Input GeoJSON must include a 'rank' column.")

    centroid = gdf.geometry.union_all().centroid
    viz = config.get("viz") or {}
    fmap = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=int(viz.get("zoom_start", 11)),
        tiles=viz.get(
            "tiles",
            "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        ),
        attr=viz.get(
            "tiles_attr",
            '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
            'contributors © <a href="https://carto.com/attributions">CARTO</a>',
        ),
    )
    vmin = float(gdf["rank"].min())
    vmax = float(gdf["rank"].max())
    colormap = LinearColormap(
        colors=["#7a0177", "#c51b8a", "#f768a1", "#fbb4b9", "#feebe2"],
        vmin=vmin,
        vmax=vmax,
        caption="Planting priority rank (1 = highest)",
    )
    colormap.add_to(fmap)

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        rank_val = feature["properties"].get("rank", vmax)
        return {
            "fillColor": colormap(rank_val),
            "color": "#4a044e",
            "weight": 0.6,
            "fillOpacity": 0.75,
        }

    folium.GeoJson(
        json.loads(gdf.to_json()),
        name="Ranked parcels",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=_tooltip_fields(gdf, config)),
    ).add_to(fmap)
    folium.LayerControl().add_to(fmap)

    dest = output_path or configured_path(config, "ranked_map")
    ensure_parent(dest)
    fmap.save(str(dest))
    return dest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Render ranked parcels as a Folium HTML map.")
    parser.add_argument(
        "--region",
        default="config/santa_cruz.yaml",
        help="Path to the region YAML config.",
    )
    parser.add_argument(
        "--input",
        default="outputs/ranked_parcels.geojson",
        help="Ranked parcels GeoJSON (Quick Start: outputs/ranked_parcels.geojson).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HTML output path (default: paths.ranked_map in the region config).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_region_config(args.region)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_path = Path(args.output) if args.output else None
    if output_path is not None and not output_path.is_absolute():
        output_path = ROOT / output_path
    dest = run(config, input_path=input_path, output_path=output_path)
    print(f"Wrote interactive map to {dest}")


if __name__ == "__main__":
    main()
