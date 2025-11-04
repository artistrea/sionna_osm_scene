# TODO: refactor everything

from lib.osm2poly import brasilia_osm_polys, bounding_polygon, bounds
from dataclasses import dataclass, field
import pickle
from lib.utils import memoizer
from itertools import product
import cv2
import rasterio as rio
import skimage as ski
import shapely as shp
import shapely.plotting
import pyvista as pv
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import geopandas as gpd


@dataclass
class DatasetItem():
    buildings_frame: Path | str = None
    # axis aligned bounding box
    aabb: tuple[tuple[float, float], tuple[float, float]] = None
    tx_frames: list[Path | str] = field(default_factory=list)
    tx_positions: list[tuple[float, float]] = field(default_factory=list)
    gain_frames: list[Path | str] = field(default_factory=list)

@dataclass
class Dataset():
    max_tx_in_frame: int = 0
    n_sets: int = 0
    items: list[DatasetItem] = field(default_factory=list)


def polys2xml(
    scene,
    buildings,
    street_edges,
    mesh_dir: Path
):
    add_floor_to_scene(scene, buildings, street_edges, mesh_dir)
    add_buildings_to_scene(scene, buildings, street_edges, mesh_dir)

def clean_polygon(geom):
    """Return a valid polygon or multipolygon."""
    geom = shp.make_valid(geom)
    if geom.is_empty:
        return None
    # Fix self-intersections and invalid geometries
    geom = geom.buffer(0)
    if geom.is_empty:
        return None
    if geom.geom_type == "MultiPolygon":
        # Keep only the largest component (optional)
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom

def get_building_ring(building_polygon: shp.Polygon):
    building_polygon = clean_polygon(building_polygon)
    exterior_coords = building_polygon.exterior.coords
    oriented_coords = list(exterior_coords)
    # Ensure counterclockwise orientation
    # if building_polygon.exterior.is_ccw:
    #     oriented_coords.reverse()
    points = [(coord[0], coord[1]) for coord in oriented_coords]
    return points

def add_buildings_to_scene(scene, buildings, street_edges, mesh_dir):
    ground_polygon = bounding_polygon(buildings, street_edges)
    # do not center, so that images and xml scene both have
    # the same coord system
    center_x = 0
    center_y = 0

    buildings_list = buildings.to_dict('records')
    from tqdm import tqdm
    for idx, building in tqdm(enumerate(buildings_list),
                              total=len(buildings_list),
                              desc="Creating building meshes"):
        # Convert building geometry to a shapely polygon
        building_points = get_building_ring(
            shp.geometry.shape(building['geometry'])
        )
        building_height = building["height"]
        building_ground_height = 0
        z_coordinates = np.full(len(building_points), building_ground_height)
        # bounding polygon
        footprint_plane = points_2d_to_poly(
            building_points, building_ground_height,
        )
        footprint_plane = footprint_plane.triangulate()
        # print("footprint_plane", footprint_plane)
        # print("(0, 0, building_height)", (0, 0, building_height))
        footprint_3D = footprint_plane.extrude((0, 0, building_height), capping=True)
        footprint_3D.save(str(mesh_dir / f"building_{idx}.ply"))
        local_mesh = o3d.io.read_triangle_mesh(str(mesh_dir / f"building_{idx}.ply"))
        o3d.io.write_triangle_mesh(str(mesh_dir / f"building_{idx}.ply"), local_mesh)
        material_type = "mat-itu_concrete"
        # Add shape elements for PLY files in the folder
        sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-building_{idx}")
        ET.SubElement(sionna_shape, "string", name="filename", value=str(mesh_dir / f"building_{idx}.ply"))
        bsdf_ref = ET.SubElement(sionna_shape, "ref", id= material_type, name="bsdf")
        ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")


def points_2d_to_poly(points, z):
    """Convert a sequence of 2d coordinates to a polydata with a polygon."""
    faces = [len(points), *range(len(points))]
    poly = pv.PolyData([p + (z,) for p in points], faces=faces)
    return poly

def add_floor_to_scene(
    scene,
    buildings,
    street_edges,
    mesh_dir: Path
):
    ground_polygon = bounding_polygon(buildings, street_edges)
    # do not center, so that images and xml scene both have
    # the same coord system
    center_x = 0
    center_y = 0

    # TODO: better code
    z_coordinates = np.full(len(ground_polygon.exterior.coords), 0)  # Assuming the initial Z coordinate is zmin
    exterior_coords = ground_polygon.exterior.coords
    oriented_coords = list(exterior_coords)
    # Ensure counterclockwise orientation
    if ground_polygon.exterior.is_ccw:
        oriented_coords.reverse()
    points = [(coord[0]-center_x, coord[1]-center_y) for coord in oriented_coords]
    # bounding polygon
    boundary_points_polydata = points_2d_to_poly(points, z_coordinates[0])
    edge_polygon = boundary_points_polydata
    footprint_plane = edge_polygon.delaunay_2d()
    footprint_plane.points[:] = (footprint_plane.points - footprint_plane.center)*1.5 + footprint_plane.center
    pv.save_meshio(str(mesh_dir / "ground.ply"),footprint_plane)

    material_type = "mat-itu_concrete"
    sionna_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-ground")
    ET.SubElement(
        sionna_shape, "string", name="filename",
        value=str(mesh_dir / "ground.ply")
    )
    bsdf_ref = ET.SubElement(sionna_shape, "ref", id=material_type, name="bsdf")
    ET.SubElement(sionna_shape, "boolean", name="face_normals",value="true")


def create_default_xml_scene():
    # Set up default values for resolution
    spp_default = 4096
    resx_default = 1024
    resy_default = 768

    # Define camera settings
    camera_settings = {
        "rotation": (0, 0, -90),  # Assuming Z-up orientation
        "fov": 42.854885
    }

    # Define material colors. This is RGB 0-1 formar https://rgbcolorpicker.com/0-1
    material_colors = {
        "mat-itu_concrete": (0.539479, 0.539479, 0.539480),
        "mat-itu_marble": (0.701101, 0.644479, 0.485150),
        "mat-itu_metal": (0.219526, 0.219526, 0.254152),
        "mat-itu_wood": (0.043, 0.58, 0.184),
        "mat-itu_wet_ground": (0.91,0.569,0.055),
    }

    scene = ET.Element("scene", version="2.1.0")
    # Add defaults
    ET.SubElement(scene, "default", name="spp", value=str(spp_default))
    ET.SubElement(scene, "default", name="resx", value=str(resx_default))
    ET.SubElement(scene, "default", name="resy", value=str(resy_default))
    # Add integrator
    integrator = ET.SubElement(scene, "integrator", type="path")
    ET.SubElement(integrator, "integer", name="max_depth", value="12")

    # Define materials
    for material_id, rgb in material_colors.items():
        bsdf_twosided = ET.SubElement(scene, "bsdf", type="twosided", id=material_id)
        bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
        ET.SubElement(bsdf_diffuse, "rgb", value=f"{rgb[0]} {rgb[1]} {rgb[2]}", name="reflectance")

    # Add emitter
    emitter = ET.SubElement(scene, "emitter", type="constant", id="World")
    ET.SubElement(emitter, "rgb", value="1.000000 1.000000 1.000000", name="radiance")

    # Add camera (sensor)
    sensor = ET.SubElement(scene, "sensor", type="perspective", id="Camera")
    ET.SubElement(sensor, "string", name="fov_axis", value="x")
    ET.SubElement(sensor, "float", name="fov", value=str(camera_settings["fov"]))
    ET.SubElement(sensor, "float", name="principal_point_offset_x", value="0.000000")
    ET.SubElement(sensor, "float", name="principal_point_offset_y", value="-0.000000")
    ET.SubElement(sensor, "float", name="near_clip", value="0.100000")
    ET.SubElement(sensor, "float", name="far_clip", value="10000.000000")
    sionna_transform = ET.SubElement(sensor, "transform", name="to_world")
    ET.SubElement(sionna_transform, "rotate", x="1", angle=str(camera_settings["rotation"][0]))
    ET.SubElement(sionna_transform, "rotate", y="1", angle=str(camera_settings["rotation"][1]))
    ET.SubElement(sionna_transform, "rotate", z="1", angle=str(camera_settings["rotation"][2]))
    camera_position = np.array([0, 0, 100])  # Adjust camera height
    ET.SubElement(sionna_transform, "translate", value=" ".join(map(str, camera_position)))
    sampler = ET.SubElement(sensor, "sampler", type="independent")
    ET.SubElement(sampler, "integer", name="sample_count", value="$spp")
    film = ET.SubElement(sensor, "film", type="hdrfilm")
    ET.SubElement(film, "integer", name="width", value="$resx")
    ET.SubElement(film, "integer", name="height", value="$resy")

    return scene

def create_full_scene(
    scene_dir: Path,
    buildings: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
):
    scene_dir.mkdir(exist_ok=True)
    scene = create_default_xml_scene()
    meshes_dir = scene_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)
    polys2xml(scene, buildings, edges, meshes_dir)
    # Create and write the XML file
    tree = ET.ElementTree(scene)
    xml_string = ET.tostring(scene, encoding="utf-8")
    xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="    ")  # Adjust the indent as needed

    with open(scene_dir / "scene.xml", "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_pretty)


def load_dataset(
    base_path: Path,
    force_update=False,
) -> Dataset:
    """Due to usage of pickle, you must import Dataset and DatasetItem
    for this function to work correctly.

        from lib.poly2scenes import Dataset, DatasetItem
    """
    # TODO: use something other than pickle
    dataset_memd, dataset_mem, has_dataset_memd = memoizer(
        pickle.load, pickle.dump, base_path / "dataset.pkl"
    )
    if has_dataset_memd() and not force_update:
        return dataset_memd()

    # TODO: pass base path here
    buildings, street_edges, tx_pos = brasilia_osm_polys()

    # TODO: pass base path here
    dataset = create_dataset_scenarios(
        buildings, street_edges, tx_pos,
        force_update=True
    )

    dataset_mem(dataset)

    return dataset

def create_dataset_scenarios(
    buildings, street_edges, tx_pos,
    force_update=False
) -> Dataset:
    minx, miny, maxx, maxy = bounds(buildings, street_edges)
    minx, miny, maxx, maxy = round(minx), round(miny), round(maxx), round(maxy)
    height, width = maxy - miny + 1, maxx - minx + 1

    step = 256
    end_size = 256

    from rasterio.features import rasterize

    tx_memd, tx_mem, has_tx_memd = memoizer(
        pickle.load, pickle.dump, Path("BRASILIA/ds/all_memd_txs.pkl")
    )
    buildings_memd, buildings_mem, has_b_memd = memoizer(
        pickle.load, pickle.dump, Path("BRASILIA/ds/buildings.pkl")
    )

    if has_tx_memd() and not force_update:
        tx_frame = tx_memd()
    else:
        shapes = [(row["geometry"], 255*(row["height"] - 6.6) / (19.8 - 6.6 + 3.0)) for i, row in tx_pos.iterrows()]
        tx_frame = np.zeros((height, width), dtype=np.uint8)
        rasterize(shapes, out=tx_frame)
        tx_mem(tx_frame)

    if has_b_memd() and not force_update:
        buildings_frame = buildings_memd()
    else:
        shapes = [(row["geometry"], 255 * (row["height"] - 6.6) / (19.8 - 6.6 + 3.0)) for i, row in buildings.iterrows()]
        buildings_frame = np.zeros((height, width), dtype=np.uint8)
        rasterize(shapes, out=buildings_frame)
        buildings_mem(buildings_frame)

    dataset = Dataset()

    from tqdm import tqdm
    id = 0
    Path("BRASILIA/ds/buildings").mkdir(exist_ok=True)
    Path("BRASILIA/ds/tx_pos").mkdir(exist_ok=True)

    for i, j in tqdm(list(product(range(0, width, step), range(0, height, step)))):
        txs = tx_frame[j:j+end_size, i:i+end_size]
        if txs.shape != (end_size, end_size):
            # print("wrong shape at (i, j) = ", (i, j))
            continue
        if not np.any(txs):
            # print("ignoring since there is no tx in possible grid split")
            continue
        builds = buildings_frame[j:j+end_size, i:i+end_size]
        if np.sum(builds != 0) < 0.1 * 256**2:
            # print("ignoring since there aren't enough buildings on scene")
            continue

        cv2.imwrite(
            f"BRASILIA/ds/buildings/{id}.png",
            builds
        )
        dsf = DatasetItem()
        dsf.buildings_frame = f"BRASILIA/ds/buildings/{id}.png"
        # axis aligned bounding box
        dsf.aabb = ((i, j), (i + end_size, j + end_size))

        this_tx_frame = np.zeros_like(txs)
        txs_y, txs_x = np.where(txs)
        dataset.max_tx_in_frame = max(dataset.max_tx_in_frame, len(txs_y))
        dataset.n_sets += len(txs_y)

        for tx_k in range(len(txs_x)):
            this_tx_frame[txs_y[tx_k], txs_x[tx_k]] = txs[txs_y[tx_k], txs_x[tx_k]]
            cv2.imwrite(
                f"BRASILIA/ds/tx_pos/{id}_{tx_k}.png",
                this_tx_frame,
            )
            dsf.tx_frames.append(f"BRASILIA/ds/tx_pos/{id}_{tx_k}.png")
            height = txs[txs_y[tx_k], txs_x[tx_k]] * (19.8 - 6.6 + 3.0) / 255 + 6.6
            pos = (txs_x[tx_k] + i + 0.5, txs_y[tx_k] + j + 0.5, height)
            dsf.tx_positions.append(pos)
            this_tx_frame[txs_y[tx_k], txs_x[tx_k]] = 0

        dataset.items.append(dsf)

        id += 1

    return dataset


def clear_dataset():
    ds_path = Path("BRASILIA/ds")
    if ds_path.exists():
        shutil.rmtree(ds_path)


if __name__ == "__main__":
    # create_full_scene(
    #     Path("./BRASILIA/scene"),
    #     buildings, edges
    # )

    # ds = load_dataset(
    #     Path("./BRASILIA/scene"),
    #     force_update=True
    # )
    # print(ds.n_sets)
    buildings, street_edges, tx_pos = brasilia_osm_polys()

    # print(buildings)
    ds = load_dataset(
        Path("./BRASILIA/ds"),
        # force_update=True
    )

    # print(f"{ds.items[0].aabb=}")
    # print(f"{ds.items[0].buildings_frame=}")
    # # print(f"{buildings.geometry=}")
    # aabb = ds.items[0].aabb
    # xy0, xy1 = aabb
    # # cut_box = shp.geometry.box(xy1[0], xy1[1], xy0[0], xy0[1])
    # # cut_box = shp.geometry.box(xy0[1], xy0[0], xy1[1], xy1[0])

    # # cut_box = shp.geometry.box(xy0[0], xy0[1], xy1[0], xy1[1])
    # # cut_gdf = gpd.GeoDataFrame([1], geometry=[cut_box], crs=buildings.crs)

    # # cut_buildings = gpd.clip(buildings, cut_gdf)
    # id = 544449619
    # # print("buildings.keys()", [str(x) for x in buildings.keys()])
    # print("buildings")
    # print(buildings)
    # # cut_buildings = buildings.iloc[id]
    # # cut_buildings = cut_buildings.iloc[1:2]
    # # print(f"{cut_buildings.geometry=}")
    # # print(cut_buildings.geometry)
    # print()
    # ids = buildings.index.get_level_values('id')
    # cut_buildings = buildings.iloc[ids == id]
    # buildings_list = cut_buildings.to_dict('records')
    # # exit()
    # for b in buildings_list:
    #     building_points = get_building_ring(b["geometry"])
    #     print("aa")
    # print(f"{building_points=}")
    # poly_after = shp.Polygon(building_points)

    # building_height = 20
    # building_ground_height = 0
    # z_coordinates = np.full(len(building_points), building_ground_height)
    # # bounding polygon
    # boundary_points_polydata = points_2d_to_poly(building_points, building_ground_height)
    # edge_polygon = boundary_points_polydata
    # footprint_plane = edge_polygon
    # footprint_plane = footprint_plane.triangulate()
    # footprint_3D = footprint_plane.extrude((0, 0, building_height), capping=True)
    # footprint_3D.plot()
    # footprint_3D.save(str(mesh_dir / f"building_{idx}.ply"))

    # exit()

    # fig, ax = plt.subplots(figsize=(12,12))
    # cut_buildings.plot(ax=ax, edgecolor='black', facecolor='lightblue')
    # shp.plotting.plot_polygon(poly_after, ax)

    # ax.set_aspect('equal')
    # # ax.axis('off')
    # plt.xlim((xy0[0], xy1[0]))
    # plt.ylim((xy0[1], xy1[1]))
    # plt.show()
    # # bs = buildings[xy0[0]:xy1[0], xy0[1]:xy1[1]]
