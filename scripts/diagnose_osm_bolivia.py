"""
Diagnostic script for OSMnx road querying on Bolivia chips.
Tests:
1. Standard query, bbox, endpoint, and error for 2 chips (Bolivia_103757 and Bolivia_129334).
2. (a) 5 km buffer around each chip.
3. (b) Custom filter for highway|track|path classes.
4. (c) Alternative Overpass endpoints.
"""

import sys
import rasterio
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox

CHIPS = ["Bolivia_103757", "Bolivia_129334"]
ENDPOINTS = [
    ("Default (overpass-api.de)", "https://overpass-api.de/api"),
    ("Kumi Systems", "https://overpass.kumi.systems/api"),
    ("Mail.ru", "https://maps.mail.ru/osm/tools/overpass/api"),
    ("Private Coffee", "https://overpass.private.coffee/api"),
]

def get_chip_bbox(chip_id: str):
    tif_path = f"data/raw/LabelHand/{chip_id}_LabelHand.tif"
    with rasterio.open(tif_path) as src:
        b = src.bounds
        return (b.left, b.bottom, b.right, b.top)

def get_buffered_bbox(bbox, buffer_km=5.0):
    # In Bolivia (~14S latitude), 1 deg lat ~ 111 km, 1 deg lon ~ 107 km
    d_lat = buffer_km / 111.0
    d_lon = buffer_km / 107.0
    left, bottom, right, top = bbox
    return (left - d_lon, bottom - d_lat, right + d_lon, top + d_lat)

def get_query_str(bbox, tags):
    polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
    coord_strs = ox._overpass._make_overpass_polygon_coord_strs(polygon)
    return ox._overpass._create_overpass_features_query(coord_strs[0], tags)

def run_diagnostics():
    print("=" * 80)
    print("   OSMNX BOLIVIA ROAD QUERY DIAGNOSTIC REPORT")
    print("=" * 80)

    for chip_id in CHIPS:
        bbox = get_chip_bbox(chip_id)
        buf_bbox = get_buffered_bbox(bbox, 5.0)

        print(f"\n" + "#" * 80)
        print(f"CHIP: {chip_id}")
        print(f"Bounding Box (W, S, E, N): {bbox}")
        print(f"Buffered Box (+5km) (W, S, E, N): {buf_bbox}")
        print("#" * 80)

        # Baseline Query
        print("\n--- 1. Baseline Query ---")
        ox.settings.overpass_url = "https://overpass-api.de/api"
        ox.settings.requests_timeout = 10
        print(f"Endpoint: {ox.settings.overpass_url}")
        print(f"Bbox: {bbox}")
        print("Tags: {'highway': True}")
        q_str = get_query_str(bbox, {"highway": True})
        print(f"Overpass QL Query:\n{q_str.strip()}")
        try:
            gdf = ox.features_from_bbox(bbox=bbox, tags={"highway": True})
            lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
            print(f"Result: SUCCESS! Found {len(lines)} road features (total {len(gdf)} features).")
        except Exception as e:
            print(f"Result: ERROR -> {type(e).__name__}: {e}")

        # Test (a): 5 km Buffer
        print("\n--- 2. Test (a): 5 km Buffer ---")
        print(f"Buffered Bbox: {buf_bbox}")
        q_str_buf = get_query_str(buf_bbox, {"highway": True})
        print(f"Overpass QL Query (5km buffer):\n{q_str_buf.strip()}")
        try:
            gdf = ox.features_from_bbox(bbox=buf_bbox, tags={"highway": True})
            lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
            print(f"Result (5km buffer): SUCCESS! Found {len(lines)} road features.")
        except Exception as e:
            print(f"Result (5km buffer): ERROR -> {type(e).__name__}: {e}")

        # Test (b): Custom filter for highway|track|path classes
        print("\n--- 3. Test (b): Custom filter (highway|track|path) ---")
        custom_tags = {"highway": ["track", "path", "residential", "unclassified", "service", "road", "primary", "secondary", "tertiary"]}
        print(f"Tags: {custom_tags}")
        q_str_custom = get_query_str(bbox, custom_tags)
        print(f"Overpass QL Query (custom filter):\n{q_str_custom.strip()}")
        try:
            gdf = ox.features_from_bbox(bbox=bbox, tags=custom_tags)
            lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
            print(f"Result (custom filter): SUCCESS! Found {len(lines)} road features.")
        except Exception as e:
            print(f"Result (custom filter): ERROR -> {type(e).__name__}: {e}")

        # Test (c): Alternative Overpass Endpoints
        print("\n--- 4. Test (c): Alternative Overpass Endpoints ---")
        for name, url in ENDPOINTS:
            ox.settings.overpass_url = url
            ox.settings.requests_timeout = 8
            print(f"Testing [{name}] -> {url}...")
            try:
                gdf = ox.features_from_bbox(bbox=bbox, tags={"highway": True})
                lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
                print(f"  Result: SUCCESS! {len(lines)} road lines returned.")
            except Exception as e:
                print(f"  Result: ERROR -> {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_diagnostics()
