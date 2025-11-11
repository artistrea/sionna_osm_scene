import numpy as np
from tqdm import tqdm
import mitsuba as mi
import matplotlib as mpl
import matplotlib.pyplot as plt
from lib.poly2scenes import load_dataset, Dataset, DatasetItem
from pathlib import Path
import cv2
import drjit as dr

import sionna.rt

from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver, load_mesh, watt_to_dbm, transform_mesh,\
                      cpx_abs_square

if __name__ == "__main__":
    import warnings
    warnings.simplefilter('ignore')
    print("Mitsuba version:", mi.__version__)
    # scene = load_scene("BRASILIA/scene/scene.xml") # Load empty scene
    # scene.tx_array = PlanarArray(num_rows=1,
    #                              num_cols=1,
    #                              pattern="iso",
    #                              polarization="V")
    # scene.rx_array = scene.tx_array

    # tx = Transmitter(name=f"tx{1}",
    #                   position=[
    #                       150.0,
    #                       1700.0,
    #                       1.5
    #                   ],
    #                   orientation=[0, 0, 0],
    #                   power_dbm=0)
    # scene.add(tx)

    # ds = load_dataset(Path("BRASILIA/ds/"))

    # # mini, mindist2 = -1, np.inf
    # # for i, item in enumerate(ds.items):
    # #     dist = item.aabb[0][0]**2 + item.aabb[0][0]**2
    # #     if dist < mindist2:
    # #         mindist2 = dist
    # #         mini = i
    # # print(f"{mini=}")
    # item = ds.items[0]
    # print(f"{item.buildings_frame=}")
    # print(f"{item.aabb=}")

    # my_cam = Camera(position=[-1, 1600.0, 3e2], look_at=[256,1700.0,0])

    # # Render scene with new camera*
    # scene.render(camera=my_cam, resolution=[650, 500], num_samples=2*128); # Increase num_samples to increase image quality
    # # rm.show(metric="path_gain", tx=0)
    # plt.show()

    # exit()

    Path("BRASILIA/ds/gains/IRT2").mkdir(exist_ok=True, parents=True)

    no_preview = True # Toggle to False to use the preview widget
                      # instead of rendering for scene visualization

    scene = load_scene("BRASILIA/scene/scene.xml") # Load empty scene

    scene.frequency = 3.5e9
    scene.bandwidth = 20e6

    # Configure antenna arrays for all transmitters and receivers
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array

    ds_path = Path("BRASILIA/ds/")
    ds = load_dataset(ds_path)
    print(f"using dataset at {ds_path}")
    print("ds: ")
    print("\tmax_tx_in_frame", ds.max_tx_in_frame)
    print("\tn_sets", ds.n_sets)
    # print("\tmax_height", ds.max_height)
    # print("\tmin_height", ds.min_height)
    print("\tmax_path_gain", ds.max_path_gain)
    print("\tmin_path_gain", ds.min_path_gain)

    initial = 0
    for i, item in tqdm(enumerate(ds.items[initial:]), total=len(ds.items), desc="Running RayTracing scenes", initial=initial):
        # sionna can't handle too many tx at same time, so we make do with this:
        step = 10
        tx_id = 0

        # Compute radio map
        rm_solver = RadioMapSolver()
        range_x = item.aabb[1][0] - item.aabb[0][0]
        range_y = item.aabb[1][1] - item.aabb[0][1]
        size = [range_x, range_y]
        # print(f"{item.aabb=}")
        # print(f"{size=}")
        center_x = (item.aabb[0][0] + item.aabb[1][0]) / 2
        center_y = (item.aabb[0][1] + item.aabb[1][1]) / 2
        # center_x, center_y = center_y, center_x
        center = [center_x, center_y, 1.5]

        windows_start_i = list(range(0, len(item.tx_positions), step))
        for wind_s in tqdm(windows_start_i, desc="Different tx positions"):
            tx_to_consider = item.tx_positions[wind_s:wind_s+step]
            for j, tx_pos in enumerate(tx_to_consider):
                # Define and add a first transmitter to the scene
                tx = Transmitter(name=f"tx{j}",
                                  position=[
                                      float(tx_pos[0]),
                                      float(tx_pos[1]),
                                      float(tx_pos[2]),
                                  ],
                                  orientation=[0, 0, 0],
                                  power_dbm=0)
                scene.add(tx)
                # break

            # print(f"{center=}")
            rm = rm_solver(scene,
                           max_depth=2,           # Maximum number of ray scene interactions
                           samples_per_tx=10**8, # If you increase: less noise, but more memory required
                           cell_size=(1, 1),      # Resolution of the radio map
                           center=center,         # Center of the radio map
                           size=size,             # Total size of the radio map
                           orientation=[0, 0, 0], # Orientation of the radio map, e.g., could be also vertical
                           refraction=False,
                           specular_reflection=True,
                           diffuse_reflection=False,
                           diffraction=True,
                       )

                # Metrics have the shape
                # [num_tx, num_cells_y, num_cells_x]

            # if rm.path_gain.shape != (len(tx_to_consider), range_y, range_x):
            #     raise Exception("wrong rm.path_gain.shape")

            # print(f"{np.min(rm.path_gain)=}")
            # print(f"{np.max(rm.path_gain)=}")
            # print(f"{rm.path_gain.shape=}")
            # print(f"{rm.path_gain=}")
            with np.errstate(divide='ignore', invalid='ignore'):
                gain = 10 * np.log10(rm.path_gain)
            # print(f"{np.min(gain)=}")
            # print(f"{np.max(gain)=}")
            # print(f"{gain=}")
            # Create new camera with different configuration
            # my_cam = Camera(position=[center_x+1e-9, center_y-1,512], look_at=center)

            # # Render scene with new camera*
            # scene.render(camera=my_cam, resolution=[650, 500], num_samples=512).savefig(
            #     f"BRASILIA/ds/gains/IRT2/scene-{i}.png"
            # )
            # plt.show()

            for j, tx_pos in enumerate(tx_to_consider):
                # print(f"{tx_pos=}")
                # norm_gain = np.copy(gain[j])
                norm_gain = np.copy(gain[j])
                # cv2.imwrite(f"BRASILIA/ds/gains/IRT2/q9_raw_{i}_{j}.tiff", norm_gain)
                min_g = ds.min_path_gain
                max_g = ds.max_path_gain
                norm_gain[norm_gain < min_g] = min_g
                norm_gain[norm_gain > max_g] = max_g
                norm_gain = (norm_gain - min_g) / (max_g-min_g)

                # print(f"{np.min(norm_gain)=}")
                # print(f"{np.max(norm_gain)=}")

                norm_gain = (norm_gain * 255).astype(np.uint8)

                print(f"{np.min(norm_gain)=}")
                print(f"{np.max(norm_gain)=}")

                cv2.imwrite(f"BRASILIA/ds/gains/IRT2/{i}_{tx_id}.png", norm_gain)
                tx_id += 1
                # rm.show(metric="path_gain", tx=0).savefig(
                # rm.show(metric="path_gain", tx=j).savefig(
                #     f"BRASILIA/ds/gains/IRT2/sim_{i}_{tx_id}.png"
                # )
                # # plt.imsave(f"BRASILIA/ds/gains/IRT2/plt_{i}_{j}.png", norm_gain)
                # plt.close("all")

                # print(f'{rm.path_gain.shape=}') # Path gain
                # print(f'{rm.rss.shape=}') # RSS
                # print(f'{rm.sinr.shape=}') # SINR

                # The location of all cell centers in the global coordinate system of the scene
                # can be accessed via:
                # [num_cells_y, num_cells_x, 3]
                # print(f'{rm.cell_centers.shape=}')
                # print(f'{rm.cell_centers=}')
                # print(f'{center=}')

                scene.remove(f"tx{j}")
                # if j == 10:
                #     exit()
                # exit()
            # exit()
