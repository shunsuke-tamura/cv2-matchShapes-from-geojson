import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon


geojson_path = "test2.geojson"
gdf = gpd.read_file(geojson_path)


def load_contour_by_NAME_JA(name: str) -> np.ndarray:
    polygon = gdf[gdf["NAME_JA"] == name].iloc[0]["geometry"]
    if type(polygon) == MultiPolygon:
        polygons = polygon.geoms
        polygon = max(polygons, key=lambda p: p.area)
    return np.array(polygon.exterior.coords)


# トルクメニスタン、ウズベキスタン、アフガニスタン、パキスタンの比較はわかりやすいかも
country1_contour = load_contour_by_NAME_JA("パキスタン")
country2_contour = load_contour_by_NAME_JA("トルクメニスタン")

similarity_score = cv2.matchShapes(
    country1_contour, country2_contour, cv2.CONTOURS_MATCH_I1, 0
)
print(similarity_score)
