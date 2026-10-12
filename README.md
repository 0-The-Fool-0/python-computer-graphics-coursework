# Python Computer Graphics Coursework

Independent Python demonstrations of rasterization, 2D line clipping, and 3D wireframe transformations. The algorithm implementations live in `processing/`; each UI has its own entry point.

## Topics and entry points

| Entry point | Topic |
|---|---|
| `app_raster.py` | Raster line-drawing methods and visual comparison |
| `app_clipping.py` | Line clipping against a rectangle, including the Liang–Barsky algorithm |
| `app_3d.py` | Transformations of a 3D wireframe model |
| `processing/raster.py` | Rasterization algorithms |
| `processing/clipping.py` | Clipping geometry |
| `processing/transform3d.py` | 3D model and transformation math |

## Requirements

Use the repository's dependency list:

```bash
python -m pip install -r requirements.txt
```

## Run a demonstration

The entry points are separate Streamlit apps. From the repository root, launch one at a time:

```bash
streamlit run app_raster.py
streamlit run app_clipping.py
streamlit run app_3d.py
```

These commands are based on the inspected project structure; the apps and tests have not been run in this environment.

## Scope and limitations

This is an academic algorithm collection, not a packaged graphics engine. The applications should be checked for edge cases and the instructions refined after a successful local run. Course redistribution requirements should be confirmed before making the repository public.