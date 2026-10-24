"""
Uncertainty propagation into downstream disaster exposure bounds.

Translates continuous predictive probability maps into three operational risk tiers:
1. Conservative (p > 0.70): High-confidence flood footprint (minimum emergency aid).
2. Expected     (p > 0.50): Median baseline planning estimate.
3. Worst-Case   (p > 0.30): Expansive evacuation boundary including low-lying saturated soil.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from rasterio.transform import Affine
from rasterio.crs import CRS
import geopandas as gpd

from src.floodmap.exposure.worldpop import compute_worldpop_exposure, aggregate_fractional_population
from src.floodmap.exposure.roads import compute_flooded_road_length


def compute_exposure_brackets(
    prob_map: np.ndarray,
    transform: Affine,
    crs: Union[str, CRS],
    roads: Optional[Union[gpd.GeoDataFrame, str, Path]] = None,
    worldpop_path: Optional[Union[str, Path]] = None,
    pop_grid: Optional[np.ndarray] = None,
    flood_fraction_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes three-tier exposure intervals (Conservative, Expected, Worst-Case).
    """
    tiers = {
        "conservative": 0.70,
        "expected": 0.50,
        "worst_case": 0.30,
    }

    results = {}

    for tier_name, thresh in tiers.items():
        binary_mask = (prob_map >= thresh).astype(np.uint8)
        tier_res = {"threshold": thresh}

        # Road exposure
        if roads is not None:
            road_metrics = compute_flooded_road_length(
                flood_mask=binary_mask,
                flood_transform=transform,
                flood_crs=crs,
                roads=roads,
            )
            tier_res["roads"] = road_metrics

        # Population exposure via WorldPop
        if worldpop_path is not None and Path(worldpop_path).exists():
            pop_metrics = compute_worldpop_exposure(
                flood_mask=binary_mask,
                flood_transform=transform,
                flood_crs=crs,
                worldpop_path=worldpop_path,
            )
            tier_res["population"] = pop_metrics
        elif pop_grid is not None and flood_fraction_grid is not None:
            # Scaled fraction
            tier_fraction = np.clip(flood_fraction_grid * (1.0 / thresh), 0.0, 1.0)
            pop_metrics = aggregate_fractional_population(pop_grid, tier_fraction)
            tier_res["population"] = pop_metrics

        results[tier_name] = tier_res

    return results
