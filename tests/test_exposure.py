"""
Unit tests for geospatial disaster exposure estimation modules:
- Fractional WorldPop population aggregation
- OSM road network inundation length in planar metric space
- Three-tier uncertainty exposure brackets (Conservative, Expected, Worst-case)
"""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import LineString, Point
from rasterio.transform import from_origin

from src.floodmap.exposure.worldpop import aggregate_fractional_population
from src.floodmap.exposure.roads import compute_flooded_road_length
from src.floodmap.exposure.summary import compute_exposure_brackets


class TestWorldPopAggregation:
    def test_aggregate_fractional_population_exact_math(self):
        # 2x2 grid with known population: [100, 200; 300, 400] => total = 1000
        pop_grid = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
        # Fractions: [0.0, 0.5; 1.0, 0.25]
        # Expected exposed: 100*0.0 + 200*0.5 + 300*1.0 + 400*0.25 = 0 + 100 + 300 + 100 = 500
        flood_fraction = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

        res = aggregate_fractional_population(pop_grid, flood_fraction)
        assert pytest.approx(res["total_population"], rel=1e-5) == 1000.0
        assert pytest.approx(res["exposed_population"], rel=1e-5) == 500.0
        assert pytest.approx(res["exposure_percentage"], rel=1e-5) == 50.0

    def test_aggregate_fractional_population_nan_and_negative_handling(self):
        # Grids with NaNs and negative nodata values
        pop_grid = np.array([[np.nan, 200.0], [-9999.0, 300.0]], dtype=np.float32)
        flood_fraction = np.array([[0.5, np.nan], [0.8, 0.5]], dtype=np.float32)

        res = aggregate_fractional_population(pop_grid, flood_fraction)
        # Valid cells:
        # (0, 1): pop 200, frac NaN -> 0.0 => 0 exposed
        # (1, 1): pop 300, frac 0.5 => 150 exposed
        # Total valid pop = 500
        assert pytest.approx(res["total_population"], rel=1e-5) == 500.0
        assert pytest.approx(res["exposed_population"], rel=1e-5) == 150.0
        assert pytest.approx(res["exposure_percentage"], rel=1e-5) == 30.0

    def test_aggregate_fractional_population_extremes(self):
        pop_grid = np.ones((5, 5), dtype=np.float32) * 10.0  # 250 total
        zero_flood = np.zeros((5, 5), dtype=np.float32)
        full_flood = np.ones((5, 5), dtype=np.float32)

        res_zero = aggregate_fractional_population(pop_grid, zero_flood)
        assert res_zero["exposed_population"] == 0.0
        assert res_zero["exposure_percentage"] == 0.0

        res_full = aggregate_fractional_population(pop_grid, full_flood)
        assert res_full["exposed_population"] == 250.0
        assert res_full["exposure_percentage"] == 100.0


class TestRoadInundation:
    @pytest.fixture
    def synthetic_raster_and_road(self):
        # 1000m x 1000m scene (100x100 pixels, 10m res)
        # Origin at UTM 19N: (500000, 4650000)
        transform = from_origin(500000, 4650000, 10, 10)
        flood_mask = np.zeros((100, 100), dtype=np.uint8)
        # Upper 50% is inundated (rows 0 to 49)
        flood_mask[:50, :] = 1

        # Vertical road from top to bottom (length = 1000m = 1.0 km)
        road = LineString([(500500, 4650000), (500500, 4649000)])
        roads_gdf = gpd.GeoDataFrame(geometry=[road], crs="EPSG:32619")

        return flood_mask, transform, "EPSG:32619", roads_gdf

    def test_compute_flooded_road_length_exact_half(self, synthetic_raster_and_road):
        flood_mask, transform, crs, roads_gdf = synthetic_raster_and_road
        res = compute_flooded_road_length(
            flood_mask=flood_mask,
            flood_transform=transform,
            flood_crs=crs,
            roads=roads_gdf,
            target_utm_crs="EPSG:32619",
        )

        assert pytest.approx(res["total_road_km"], rel=1e-3) == 1.0
        assert pytest.approx(res["flooded_road_km"], rel=1e-3) == 0.5
        assert pytest.approx(res["percentage_flooded"], rel=1e-3) == 50.0
        assert "32619" in res["utm_crs"]

    def test_compute_flooded_road_length_empty_gdf(self, synthetic_raster_and_road):
        flood_mask, transform, crs, _ = synthetic_raster_and_road
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:32619")

        res = compute_flooded_road_length(
            flood_mask=flood_mask,
            flood_transform=transform,
            flood_crs=crs,
            roads=empty_gdf,
        )
        assert res["total_road_km"] == 0.0
        assert res["flooded_road_km"] == 0.0
        assert res["percentage_flooded"] == 0.0

    def test_compute_flooded_road_length_dry_scene(self, synthetic_raster_and_road):
        _, transform, crs, roads_gdf = synthetic_raster_and_road
        all_dry = np.zeros((100, 100), dtype=np.uint8)

        res = compute_flooded_road_length(
            flood_mask=all_dry,
            flood_transform=transform,
            flood_crs=crs,
            roads=roads_gdf,
            target_utm_crs="EPSG:32619",
        )
        assert pytest.approx(res["total_road_km"], rel=1e-3) == 1.0
        assert res["flooded_road_km"] == 0.0
        assert res["percentage_flooded"] == 0.0

    def test_compute_flooded_road_length_fully_flooded(self, synthetic_raster_and_road):
        _, transform, crs, roads_gdf = synthetic_raster_and_road
        all_water = np.ones((100, 100), dtype=np.uint8)

        res = compute_flooded_road_length(
            flood_mask=all_water,
            flood_transform=transform,
            flood_crs=crs,
            roads=roads_gdf,
            target_utm_crs="EPSG:32619",
        )
        assert pytest.approx(res["total_road_km"], rel=1e-3) == 1.0
        assert pytest.approx(res["flooded_road_km"], rel=1e-3) == 1.0
        assert pytest.approx(res["percentage_flooded"], rel=1e-3) == 100.0


class TestExposureBrackets:
    def test_exposure_brackets_monotonicity(self):
        # 100x100 probability surface with decreasing gradient from top to bottom
        transform = from_origin(500000, 4650000, 10, 10)
        prob_map = np.zeros((100, 100), dtype=np.float32)
        prob_map[:30, :] = 0.85   # Conservative tier (p >= 0.70)
        prob_map[30:50, :] = 0.60 # Expected tier (p >= 0.50)
        prob_map[50:70, :] = 0.40 # Worst-case tier (p >= 0.30)
        prob_map[70:, :] = 0.10   # Unflooded dry land

        road = LineString([(500500, 4650000), (500500, 4649000)])
        roads_gdf = gpd.GeoDataFrame(geometry=[road], crs="EPSG:32619")

        brackets = compute_exposure_brackets(
            prob_map=prob_map,
            transform=transform,
            crs="EPSG:32619",
            roads=roads_gdf,
        )

        assert "conservative" in brackets
        assert "expected" in brackets
        assert "worst_case" in brackets

        c_km = brackets["conservative"]["roads"]["flooded_road_km"]
        e_km = brackets["expected"]["roads"]["flooded_road_km"]
        w_km = brackets["worst_case"]["roads"]["flooded_road_km"]

        # Strict physical invariant: Conservative <= Expected <= Worst-Case
        assert c_km <= e_km <= w_km
        assert pytest.approx(c_km, rel=1e-2) == 0.3
        assert pytest.approx(e_km, rel=1e-2) == 0.5
        assert pytest.approx(w_km, rel=1e-2) == 0.7
