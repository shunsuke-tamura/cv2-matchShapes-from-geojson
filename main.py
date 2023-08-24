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


country1_contour = load_contour_by_NAME_JA("ナミビア")
country2_contour = load_contour_by_NAME_JA("アンゴラ")

similarity_score = cv2.matchShapes(
    country1_contour, country2_contour, cv2.CONTOURS_MATCH_I1, 0
)
print(similarity_score)
