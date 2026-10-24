# Antigravity Workspace Rules: Flood Mapping & Exposure Estimation

1. Environment & Stack:
   - Python 3.11, PyTorch 2.x, segmentation_models_pytorch, Hugging Face transformers,
     rasterio, geopandas, osmnx, numpy, pandas, fastapi, uvicorn.

2. Data Conventions & Label Masking:
   - Ground truth values: 1 = Flooded/Water, 0 = Non-Flooded/Dry Land, -1 = Ignore/Invalid.
   - Pixels marked as -1 MUST be strictly masked out from loss calculations, gradient updates,
     confusion matrices, and evaluation metrics.
   - NEVER report plain overall accuracy. All evaluations must report Flood-Class IoU,
     Flood F1-score, Precision, Recall, and Mean IoU.

3. Cross-Region Generalization Integrity:
   - The Bolivia flood event chips are strictly held-out for out-of-region evaluation.
   - NEVER use the Bolivia split for training, validation, loss computation, hyperparameter
     tuning, or early stopping. Model selection uses Validation Flood IoU ONLY.

4. Coordinate Reference Systems & Spatial Operations:
   - NEVER hard-code a single UTM projection (e.g., EPSG:32619). Derive the local UTM zone
     dynamically per chip using `geopandas.estimate_utm_crs()` or raster metadata.
   - WorldPop exposure must be computed via fractional subpixel aggregation. Never resample
     population count rasters using nearest-neighbor, bilinear, or cubic interpolation.

5. Execution Safety & Reproducibility:
   - All training and evaluation scripts must support a `--smoke` flag:
     `--smoke` runs 8 chips for 2 epochs with batch size 2, finishing in < 120 seconds.
   - Every experiment must use a fixed seed and read from a YAML configuration in `configs/`.
   - All reported figures and tables must be generated programmatically from `results/*.json`.
   - Do NOT commit raw GeoTIFF datasets or `.pth` model checkpoints to Git.
