import osmnx as ox
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import shapely as shp
import scipy
from lib.utils import memoizer


def bounding_polygon(buildings, street_edges):
    minx, miny, maxx, maxy = bounds(buildings, street_edges)
    return shp.box(minx, miny, maxx, maxy, ccw=True)


def bounds(buildings: gpd.GeoDataFrame, street_edges: gpd.GeoDataFrame):
    minx0, miny0, maxx0, maxy0 = buildings.total_bounds
    minx1, miny1, maxx1, maxy1 = street_edges.total_bounds

    minx = min(minx0, minx1)
    miny = min(miny0, miny1)
    maxx = max(maxx0, maxx1)
    maxy = max(maxy0, maxy1)

    return minx, miny, maxx, maxy

def get_building_height(
    building,
    rng: np.random.Generator,
    building_low: float,
    building_high: float,
):
    """Gets height from a geodataframe row"""
    # if math.isnan(float(building['building:levels'])):
    #     if not isinstance(building["addr:street"], str):
    #         building_height = 7.5
    #     elif "CLN " in building["addr:street"] or "CLS " in building["addr:street"]:
    #         building_height = 8.5
    #     elif "EQN " in building["addr:street"] or "EQS " in building["addr:street"]:
    #         building_height = 7.5
    #     elif "SQN " in building["addr:street"] or "SQS " in building["addr:street"]:
    #         building_height = 18
    #     else:
    #         building_height = 3.5
    # else:
    #     building_height = (int(building['building:levels']) + 1) * 3.5
    building_height = rng.random() * (building_high - building_low) + building_low

    return building_height
    # return 20

def get_building_pilotis_bool(
    building,
    rng: np.random.Generator,
    pilotis_probability: float
):
    """Returns a bool for if a building has pilotis from a geodataframe row"""
    # if math.isnan(float(building['building:levels'])):
    #     if not isinstance(building["addr:street"], str):
    #         building_height = 7.5
    #     elif "CLN " in building["addr:street"] or "CLS " in building["addr:street"]:
    #         building_height = 8.5
    #     elif "EQN " in building["addr:street"] or "EQS " in building["addr:street"]:
    #         building_height = 7.5
    #     elif "SQN " in building["addr:street"] or "SQS " in building["addr:street"]:
    #         building_height = 18
    #     else:
    #         building_height = 3.5
    # else:
    #     building_height = (int(building['building:levels']) + 1) * 3.5
    has_pilotis = rng.uniform() < pilotis_probability

    return has_pilotis
    # return 20


def place2poly(
    place_query,
) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
    """From a place query returns its building and street
    polygons.
    """
    tags = {'building': True}
    buildings = ox.features_from_place(place_query, tags=tags)

    if buildings.empty:
        raise ValueError("Place does not contain any buildings")

    # get rid of bad labeling people do
    # and only accept geometry with area as building
    buildings = buildings.loc[
        buildings.geometry.notnull() &
        buildings.geometry.type.isin(['Polygon','MultiPolygon'])
    ]

    streets = ox.graph_from_place(place_query, network_type='drive')

    # get in local UTM coordinate system
    streets = ox.project_graph(streets)

    street_edges_utm = ox.graph_to_gdfs(streets, nodes=False)

    buildings_utm = buildings.to_crs(street_edges_utm.crs)

    return buildings_utm, street_edges_utm


def get_tx_points_on_street_level(
    buildings, street_edges
):
    candidate_points = []

    from tqdm import tqdm
    # generate candidates
    for line in tqdm(street_edges.geometry, desc="Getting candidate tx points"):
        if isinstance(line, shp.LineString):
            length = line.length
            step = 20  # [m]
            distances = np.arange(0, length + 1e-6, step)
            for d in distances:
                pt = line.interpolate(d)
                # make sure point is not inside any building
                if not buildings.geometry.contains(pt).any():
                    candidate_points.append(pt)

    candidate_points_array = np.array([(pt.x, pt.y) for pt in candidate_points])

    tree = scipy.spatial.cKDTree(candidate_points_array)
    keep_mask = np.ones(len(candidate_points), dtype=bool)
    min_dist = 20

    for i, pt in enumerate(candidate_points):
        if not keep_mask[i]:
            continue
        # query: todos os pontos dentro de min_dist
        idxs = tree.query_ball_point((pt.x, pt.y), r=min_dist)
        # marca todos exceto o primeiro como False
        idxs = [j for j in idxs if j > i]
        keep_mask[idxs] = False

    selected_points = candidate_points_array[keep_mask]

    gdf = gpd.GeoDataFrame(
        dict(height=[1.5 for _ in selected_points]),
        geometry=[shp.Point(xy[0], xy[1], 1.5) for xy in selected_points],
        crs=street_edges.crs
    )

    return gdf

def get_tx_points_on_buildings(
    buildings, street_edges
):
    candidate_points = []
    tx_hs = []

    # prepare structure to query buildings and check whether a tx is inside
    # any
    polys = []
    heights = []

    for idx, row in buildings.iterrows():
        poly = row.geometry
        if poly is None:
            continue
        if poly.geom_type == "MultiPolygon":
            for subpoly in poly.geoms:
                polys.append(subpoly)
                heights.append(row["height"])
        else:
            polys.append(poly)
            heights.append(row["height"])

    tree = shp.STRtree(polys)

    from tqdm import tqdm
    # generate candidates
    step = 12  # distance between candidates on top of building
    clearance = 1.0  # 1 meter clearance when checking if building intersects with tx

    for idx, poly in tqdm(enumerate(polys), total=len(polys), desc="Getting candidate tx points"):
        tx_height = heights[idx] + 3.0
        line = poly.exterior
        length = line.length
        distances = np.arange(0, length - step, step)

        for d in distances:
            pt2d = line.interpolate(d)

            # check nearby polygons
            nearby = tree.query(pt2d.buffer(clearance))
            covered = False

            for j in nearby:
                if j == idx:
                    continue

                # and if height of that building is "too much"
                if heights[j] >= tx_height - 1:
                    covered = True
                    break

            # adds 2d point
            candidate_points.append(pt2d)
            tx_hs.append(tx_height)

    gdf = gpd.GeoDataFrame(
        dict(height=[h for h in tx_hs]),
        geometry=candidate_points,
        crs=buildings.crs,
    )

    return gdf

def get_tx_points(
    buildings, street_edges
):
    return get_tx_points_on_buildings(buildings, street_edges)
    # return get_tx_points_on_street_level(buildings, street_edges)


def brasilia_osm_polys(
    force_update=False,
    building_high=19.8,
    building_low=6.6,
    pilotis_probability=0.5,
):
    place_query = "Plano Piloto, Brasília, DF, Brasil"
    BRASILIA_OSM_POLYS_PATH = Path("./BRASILIA/")
    buildings_memd, buildings_mem, has_b_memd = memoizer(
        pickle.load, pickle.dump, BRASILIA_OSM_POLYS_PATH / "buildings.pkl"
    )
    streets_memd, streets_mem, has_s_memd = memoizer(
        pickle.load, pickle.dump, BRASILIA_OSM_POLYS_PATH / "street_edges.pkl"
    )
    tx_pos_memd, tx_pos_mem, has_tx_memd = memoizer(
        pickle.load, pickle.dump, BRASILIA_OSM_POLYS_PATH / "tx_pos.pkl"
    )
    has_all_memd = has_b_memd() and has_s_memd() and has_tx_memd()

    if force_update or not has_all_memd:
        buildings, streets = place2poly(place_query)

        xmin, xmax = 0., 2e6
        ymin, ymax = 0., 8.2590e6

        cut_box = shp.geometry.box(xmin, ymin, xmax, ymax)

        cut_gdf = gpd.GeoDataFrame([1], geometry=[cut_box], crs=buildings.crs)
        minx, miny, maxx, maxy = cut_gdf.bounds
        # remove useless stuff on top
        buildings, streets = gpd.clip(buildings, cut_gdf), gpd.clip(streets, cut_gdf)
        # TODO: move to top of everything
        rng = np.random.default_rng(100)

        buildings['height'] = [
            get_building_height(row, rng, building_low, building_high)
                for _, row in buildings.iterrows()
        ]
        buildings['has_pilotis'] = [
            get_building_pilotis_bool(row, rng, pilotis_probability)
                for _, row in buildings.iterrows()
        ]

        tx_pos = get_tx_points(buildings, streets)

        minx, miny, _, __ = bounds(buildings, streets)

        transl = lambda g: shp.affinity.translate(g, xoff=-minx, yoff=-miny)
        streets['geometry'] = streets['geometry'].apply(transl)
        buildings['geometry'] = buildings['geometry'].apply(transl)
        tx_pos['geometry'] = tx_pos['geometry'].apply(transl)

        streets_mem(streets)
        buildings_mem(buildings)
        tx_pos_mem(tx_pos)
    else:
        buildings = buildings_memd()
        streets = streets_memd()
        tx_pos = tx_pos_memd()

    return buildings, streets, tx_pos


if __name__ == "__main__":
    buildings, edges, tx_pos = brasilia_osm_polys(
        force_update=True
    )

    # tx_pos = get_tx_points(buildings, edges)

    fig, ax = plt.subplots(figsize=(12,12))

    # xmin, xmax = 0., 2e6
    # ymin, ymax = 0., 8.2590e6

    # cut_box = shp.geometry.box(xmin, ymin, xmax, ymax)

    # cut_gdf = gpd.GeoDataFrame([1], geometry=[cut_box], crs=buildings.crs)
    # cut_gdf.plot(ax=ax, linewidth=0.5, edgecolor='green', facecolor='lightgreen')

    edges.plot(ax=ax, linewidth=0.5, edgecolor='gray')

    buildings.plot(ax=ax, edgecolor='black', facecolor='lightblue')

    tx_pos.plot(ax=ax, edgecolor='red', facecolor='pink')

    ax.set_aspect('equal')
    # ax.axis('off')
    plt.show()
