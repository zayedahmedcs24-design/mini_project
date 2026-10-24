"""
Geospatial exposure estimation: Fractional WorldPop population aggregation.

Methodological Principle:
Instead of crude nearest-neighbor downsampling (which discards fine 10m flood boundaries),
this module calculates the subpixel flood fraction f_c in [0.0, 1.0] inside each 100m
WorldPop grid cell, and multiplies it by the cell population count.

Assumption:
This calculation assumes that human population is distributed uniformly within
each 100m x 100m grid cell.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from rasterio.crs import CRS


def aggregate_fractional_population(
    pop_grid: np.ndarray,
    flood_fraction_grid: np.ndarray,
) -> Dict[str, float]:
    """
    Computes exposed population given aligned arrays of population counts
    and flood fractions.

    Args:
        pop_grid: 2D array of population counts per cell.
        flood_fraction_grid: 2D array of flood area fractions in [0.0, 1.0].

    Returns:
        Dict with keys:
            - exposed_population: Total estimated number of people in inundated areas
            - total_population: Total population within the analyzed grid
            - exposure_percentage: Percentage of population exposed
    """
    # Replace NaNs or negative nodata with 0.0
    valid_pop = np.where((np.isnan(pop_grid)) | (pop_grid < 0), 0.0, pop_grid)
    valid_frac = np.clip(np.where(np.isnan(flood_fraction_grid), 0.0, flood_fraction_grid), 0.0, 1.0)

    # Exposed population per cell: Pop_c * Fraction_c
    cell_exposure = valid_pop * valid_frac
    total_exposed = float(np.sum(cell_exposure))
    total_pop = float(np.sum(valid_pop))

    pct_exposed = (total_exposed / total_pop * 100.0) if total_pop > 0 else 0.0

    return {
        "exposed_population": total_exposed,
        "total_population": total_pop,
        "exposure_percentage": pct_exposed,
    }


def compute_worldpop_exposure(
    flood_mask: np.ndarray,
    flood_transform: Affine,
    flood_crs: Union[str, CRS],
    worldpop_path: Union[str, Path],
) -> Dict[str, float]:
    """
    Computes population exposure by reprojecting a 10m flood mask to a 100m WorldPop
    raster using area-averaging resampling.

    Args:
        flood_mask: 2D array of binary flood predictions (1 = water, 0 = dry).
        flood_transform: Affine transform of the flood mask.
        flood_crs: Coordinate reference system of the flood mask (e.g. 'EPSG:4326').
        worldpop_path: Filepath to the WorldPop population count GeoTIFF.

    Returns:
        Dict with exposure metrics.
    """
    worldpop_path = Path(worldpop_path)
    if not worldpop_path.exists():
        raise FileNotFoundError(f"WorldPop raster not found at: {worldpop_path}")

    with rasterio.open(worldpop_path) as pop_src:
        # Windowed read around flood bounds
        from rasterio.windows import from_bounds
        bounds = rasterio.transform.array_bounds(
            height=flood_mask.shape[0],
            width=flood_mask.shape[1],
            transform=flood_transform,
        )

        # Reproject bounds to WorldPop CRS if different
        from rasterio.warp import transform_bounds
        pop_crs = pop_src.crs
        pop_bounds = transform_bounds(flood_crs, pop_crs, *bounds)

        window = from_bounds(*pop_bounds, transform=pop_src.transform)
        # Pad window slightly to prevent edge clipping
        window = window.round_offsets().round_lengths()

        pop_data = pop_src.read(1, window=window)
        win_transform = pop_src.window_transform(window)

        # Destination array for flood fraction inside WorldPop cells
        flood_fraction = np.zeros_like(pop_data, dtype=np.float32)

        # Area-averaging reprojection: averages 10m subpixels into 100m destination cells
        reproject(
            source=flood_mask.astype(np.float32),
            destination=flood_fraction,
            src_transform=flood_transform,
            src_crs=flood_crs,
            dst_transform=win_transform,
            dst_crs=pop_crs,
            resampling=Resampling.average,
        )

    return aggregate_fractional_population(pop_data, flood_fraction)
