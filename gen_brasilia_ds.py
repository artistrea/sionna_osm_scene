import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt

import osmnx as ox
import geopandas as gpd

# Consulta pelo nome do Plano Piloto
place_q = "Plano Piloto, Brasília, DF, Brasil"
tags = {'building': True}
gdf = ox.features_from_place(place_q, tags=tags)

gdf_poly = gdf[gdf.geometry.type.isin(['Polygon','MultiPolygon'])]

buildings = gdf_poly

streets = ox.graph_from_place(place_q, network_type='drive')
edges = ox.graph_to_gdfs(streets, nodes=False)

fig, ax = plt.subplots(figsize=(12,12))

edges.plot(ax=ax, linewidth=0.5, edgecolor='gray')

# Suponha que já tenha plano_piloto_polygon
building_gdf = gpd.GeoDataFrame(geometry=[buildings], crs="EPSG:4326")

# Plota o contorno
building_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue')

ax.set_aspect('equal')
ax.axis('off')
plt.show()

