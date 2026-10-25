from __future__ import annotations

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

from ch3_pipeline.phase3b import CameraMetadata, graph_quality, nearest_edge_candidates, transform_xy, validate_metadata


def test_crs_conversion_and_invalid_coordinates():
    lon,lat=transform_xy(425000,565000,"EPSG:27700")
    assert -3 < lon < 0 and 54 < lat < 56
    errors=validate_metadata([CameraMetadata("A",100,200,"EPSG:4326")])
    assert any(item.startswith("invalid_coordinate") for item in errors)


def test_duplicate_ids_multiple_coordinates_and_provenance():
    records=[CameraMetadata("A",55,-1,"EPSG:4326",source_file="x",source_type="csv",extraction_method="verified",evidence_grade="A"),
             CameraMetadata("A",55.1,-1.1,"EPSG:4326",evidence_grade="B")]
    errors=validate_metadata(records)
    assert any(item.startswith("duplicate_camera_id") for item in errors)
    assert any(item.startswith("multiple_coordinates") for item in errors)
    assert any(item.startswith("incomplete_provenance") for item in errors)


def test_nearest_edge_ambiguity_divided_oneway_and_missing_maxspeed():
    cameras=gpd.GeoDataFrame([{"camera_id":"SYNTH","geometry":Point(-1.6,55.0)}],crs="EPSG:4326")
    edges=gpd.GeoDataFrame([
        {"u":1,"v":2,"key":0,"ref":"A1","highway":"trunk","oneway":True,"maxspeed":"50 mph","geometry":LineString([(-1.601,54.999),(-1.601,55.001)])},
        {"u":3,"v":4,"key":0,"ref":"A1","highway":"trunk","oneway":True,"maxspeed":None,"geometry":LineString([(-1.599,55.001),(-1.599,54.999)])},
    ],crs="EPSG:4326")
    result=nearest_edge_candidates(cameras,edges,k=2,ambiguity_metres=20)
    assert len(result)==2
    assert result["junction_ambiguity"].all()
    assert result["divided_road_ambiguity"].all()
    assert result["oneway"].all()
    assert result["maxspeed"].isna().sum()==1


def test_graph_quality_and_privacy_safe_fields():
    G=nx.MultiDiGraph(crs="EPSG:4326")
    G.add_edge(1,2,length=10,highway="primary",oneway=True,maxspeed="30 mph")
    G.add_edge(2,1,length=10,highway="primary",oneway=False)
    quality=graph_quality(G)
    assert quality["directed_edge_count"]==2
    assert quality["maxspeed_edge_count"]==1
    assert not any("vehicle" in key or "token" in key for key in quality)
