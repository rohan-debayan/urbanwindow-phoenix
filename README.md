# UrbanWindow 3D: Phoenix

*How much green do people see from their windows?*

This notebook estimates a **window-view green ratio** for every floor of every building in Phoenix. It renders eye-level views from Google's Photorealistic 3D Tiles, segments them with a DeepLabV3-ResNet50 model, and uses the result to study urban heat and social equity, with a city-wide stress test and a Houston comparison. It uses Anvil HPC alongside.

I-GUIDE Summer School 2026, Team 3 (Seeing Green from Indoors).

## Run it

Open `phoenix_urbanwindow_iguide.ipynb` on the I-GUIDE platform (the `geoai` kernel) or any Jupyter environment, and install the extra packages once:

```
%pip install -r requirements.txt
```

The notebook downloads its dataset on the first run (one Setup cell), so no large data files live in this repository.

## Contents

| File | What it is |
|------|------------|
| `phoenix_urbanwindow_iguide.ipynb` | The analysis notebook, runs top to bottom |
| `uw_kit.py` | Helper functions the notebook calls (viewpoint geometry, segmentation, analysis, plots) |
| `requirements.txt` | Packages beyond the geoai kernel |

## Method and credits

The UrbanWindow 3D method and the original Houston case study are by **Zongrong Li**, GEAR Lab, Texas A&M (advisor Dr. Lei Zou): *Seeing Green from Indoors in 3D* (SSRN abstract 6766522). This is the Phoenix extension.

**Data sources:** Google Photorealistic 3D Tiles (Cesium ion), GlobalBuildingAtlas, Sentinel-2 NDVI, Landsat LST, CDC/ATSDR SVI 2022, Census TIGER.
