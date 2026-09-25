# CanopyRank

**Open-source, parcel-level urban heat prioritization.**

CanopyRank turns public satellite, canopy, and parcel data into a ranked list of *where to plant trees next* — down to individual lots, not census tracts. Built for city foresters, urban planners, and environmental justice organizations who need an actionable list, not another heat map.

> Most urban heat island tools stop at 30m raster resolution or census-tract granularity. CanopyRank downscales to the parcel level and ranks sites by heat-reduction potential, environmental-justice priority, and feasibility — using only free, public data sources.

---

## Why

Cities and counties routinely pay consultants for heat-vulnerability studies that produce PDFs, not pipelines. Meanwhile the underlying data — Landsat thermal imagery, canopy cover, parcel boundaries, EJ screening indices — is public and free. CanopyRank is the missing glue: an open, reproducible pipeline from raw public data to a ranked, mappable planting list.

## Features

- **Parcel-level downscaling** — regresses coarse (30m) land surface temperature against fine-grained canopy/impervious/building features to estimate heat at the individual parcel level
- **EJ-weighted ranking** — combines predicted heat-reduction potential with CalEnviroScreen / EPA EJScreen priority scores
- **Feasibility filtering** — flags parcels by ownership type (public land, vacant lot, right-of-way) so output is actionable, not theoretical
- **Fully open data stack** — Landsat/ECOSTRESS (Earth Engine), NLCD or county LiDAR canopy, county assessor parcels, CalEnviroScreen/EJScreen
- **Exportable output** — ranked parcels as GeoJSON, viewable in any GIS tool or as an interactive Folium map

## Architecture

```
Raw public data → Zonal stats per parcel → Downscaling model → EJ-weighted ranking → GeoJSON / map
```

| Stage | Library | Input | Output |
|---|---|---|---|
| Ingest | `earthengine-api` | Landsat thermal band | Cloud-masked LST composite |
| Ingest | `geopandas` | County parcel shapefiles | Cleaned parcel geometries |
| Features | `rasterstats` | LST + canopy rasters | Per-parcel zonal statistics |
| Model | `xgboost` | Feature table | Predicted parcel-level LST |
| Ranking | `pandas` | Predictions + EJ index | Ranked parcel list |
| Output | `folium` | Ranked GeoJSON | Interactive map |

## Installation

```bash
git clone https://github.com/bitsbard/canopyrank.git
cd canopyrank
pip install -r requirements.txt
```

Requires a free [Google Earth Engine](https://earthengine.google.com/) account for satellite data access.

## Quick Start

```bash
# 1. Configure your study area (county/city boundary + data sources)
python src/ingest/fetch_landsat.py --region config/santa_cruz.yaml

# 2. Build the parcel feature table
python src/features/build_feature_table.py --region config/santa_cruz.yaml

# 3. Train the downscaling model and rank parcels
python src/model/train.py
python src/model/rank_parcels.py --top-n 100

# 4. Generate the interactive map
python src/viz/map_output.py --input outputs/ranked_parcels.geojson
```

Output: `outputs/ranked_parcels.geojson` and an interactive HTML map of the top-ranked planting sites.

## Data Sources

All data sources used are free and public:

- [Landsat 8/9](https://earthexplorer.usgs.gov/) via Google Earth Engine
- [NLCD Land Cover](https://www.mrlc.gov/) (fallback canopy data)
- County LiDAR canopy layers, where published
- County assessor parcel boundaries
- [CalEnviroScreen](https://oehha.ca.gov/calenviroscreen) / [EPA EJScreen](https://www.epa.gov/ejscreen)

## Extending to a New Region

CanopyRank is region-agnostic. To run it for a new county, add a config file specifying the boundary, parcel data source, and canopy source — see `config/santa_cruz.yaml` for the template.

## Roadmap

- [ ] Support for ECOSTRESS as a higher-resolution LST alternative
- [ ] Pre-built config files for additional CA counties
- [ ] Batch processing for multi-county comparisons
- [ ] Web-based interface for non-technical users

## Contributing

Issues and pull requests welcome. This project is in active early development — feedback on methodology (especially the downscaling approach) is particularly useful.

## License

MIT

## Citation

If you use CanopyRank in research or municipal planning work, please cite this repository.

---

Built by [Jared Mills](https://github.com/bitsbard) — feedback and collaborators welcome.
