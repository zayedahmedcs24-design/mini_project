"""
Geospatial exposure estimation: OpenStreetMap (OSM) road network inundation.

Methodological Principle:
1. Derives the local planar UTM CRS dynamically per scene using geopandas.estimate_utm_crs()
   to avoid hard-coded coordinate distortions (e.g. Bolivia is in the Southern Hemisphere).
2. Converts raster flood masks into vector polygons.
3. Computes exact geometric line-in-polygon intersections in planar meter space.
4. Outputs submerged road distance in kilometers (km) and percentage of severed network.
"""

from typing import Dict, Union, Optional
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import shape, Polygon, LineString, MultiLineString


def compute_flooded_road_length(
    flood_mask: np.ndarray,
    flood_transform: Affine,
    flood_crs: Union[str, CRS],
    roads: Union[gpd.GeoDataFrame, str, Path],
    target_utm_crs: Optional[Union[str, CRS]] = None,
) -> Dict[str, float]:
    """
    Computes inundated road network length in planar UTM kilometers.

    Args:
        flood_mask: 2D binary numpy array (1 = flood, 0 = dry).
        flood_transform: Affine transform of the flood mask.
        flood_crs: Coordinate reference system of the flood raster.
        roads: GeoPandas GeoDataFrame with road LineStrings or path to vector file.
        target_utm_crs: Optional explicit UTM CRS; if None, dynamically derived via estimate_utm_crs().

    Returns:
        Dict with keys:
            - flooded_road_km: Flooded road length in km
            - total_road_km: Total road network length in km
            - percentage_flooded: Percentage of road network inundated
            - utm_crs: String representation of the derived planar CRS
    """
    if isinstance(roads, (str, Path)):
        roads_gdf = gpd.read_file(roads)
    else:
        roads_gdf = roads.copy()

    if len(roads_gdf) == 0:
        return {
            "flooded_road_km": 0.0,
            "total_road_km": 0.0,
            "percentage_flooded": 0.0,
            "utm_crs": "None",
        }

    # Ensure road CRS matches raster or is set
    if roads_gdf.crs is None:
        roads_gdf.set_crs(flood_crs, inplace=True)
    elif str(roads_gdf.crs) != str(flood_crs):
        roads_gdf = roads_gdf.to_crs(flood_crs)

    # 1. Dynamically derive local planar UTM projection
    if target_utm_crs is None:
        derived_utm = roads_gdf.estimate_utm_crs()
    else:
        derived_utm = CRS.from_user_input(target_utm_crs)

    # 2. Vectorize the flood mask (only keep pixels where value == 1)
    mask_uint8 = (flood_mask > 0).astype(np.uint8)
    flood_polygons = []

    for geom, val in shapes(mask_uint8, mask=(mask_uint8 == 1), transform=flood_transform):
        if val == 1:
            flood_polygons.append(shape(geom))

    # Reproject roads to planar UTM (meters)
    roads_utm = roads_gdf.to_crs(derived_utm)
    total_len_km = float(roads_utm.geometry.length.sum() / 1000.0)

    if not flood_polygons:
        return {
            "flooded_road_km": 0.0,
            "total_road_km": total_len_km,
            "percentage_flooded": 0.0,
            "utm_crs": str(derived_utm),
        }

    # Build flood GeoDataFrame and reproject to planar UTM
    flood_gdf = gpd.GeoDataFrame({"geometry": flood_polygons}, crs=flood_crs)
    flood_utm = flood_gdf.to_crs(derived_utm)

    # Dissolve flood polygons to accelerate geometric intersection
    flood_unified = flood_utm.union_all() if hasattr(flood_utm, "union_all") else flood_utm.unary_union

    # 3. Geometric line intersection in planar meters
    intersected = roads_utm.intersection(flood_unified)
    # Drop empty intersections
    valid_intersections = intersected[~intersected.is_empty]
    flooded_len_km = float(valid_intersections.length.sum() / 1000.0)

    pct_flooded = (flooded_len_km / total_len_km * 100.0) if total_len_km > 0 else 0.0

    return {
        "flooded_road_km": flooded_len_km,
        "total_road_km": total_len_km,
        "percentage_flooded": pct_flooded,
        "utm_crs": str(derived_utm),
    }
