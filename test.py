import numpy as np
import mitsuba as mi
import matplotlib as mpl
import matplotlib.pyplot as plt
import sionna.rt
import cv2

from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver, load_mesh, watt_to_dbm, transform_mesh,\
                      cpx_abs_square

def config_scene(num_rows, num_cols):
    """Load and configure a scene"""
    # scene = load_scene(sionna.rt.scene.floor_wall)
    scene = load_scene()
    # print(sionna.rt.scene.floor_wall)
    scene.bandwidth=100e6

    # # Configure antenna arrays for all transmitters and receivers
    # scene.tx_array = PlanarArray(num_rows=num_rows,
    #                              num_cols=num_cols,
    #                              pattern="tr38901",
    #                              polarization="V")

    # scene.rx_array = PlanarArray(num_rows=1,
    #                              num_cols=1,
    #                              pattern="iso",
    #                              polarization="V")
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array

    # Place transmitters
    positions = np.array(
                 [
                  [0., 0., 11.0]
                  # [-150.3, 21.63, 42.5],
                  # [-125.1, 9.58, 42.5],
                  # [-104.5, 54.94, 42.5],
                  # [-128.6, 66.73, 42.5],
                  # [172.1, 103.7, 24],
                  # [232.8, -95.5, 17],
                  # [80.1, 193.8, 21]
                 ])
    look_ats = np.array(
                [
                 [1.,1.,1.0],
                 # [-90, -80, 0],
                 # [-16.5, 75.8, 0],
                 # [-164, 153.7, 0],
                 # [247, 92, 0],
                 # [211, -180, 0],
                 # [126.3, 194.7, 0]
                ])
    power_dbms = [
        0,
        # 23,
        # 23,
        # 23,
        # 23,
        # 23,
        # 23
    ]

    for i, position in enumerate(positions):
        scene.add(Transmitter(name=f'tx{i}',
                              position=position,
                              look_at=look_ats[i],
                              power_dbm=power_dbms[i]))

    return scene

# Load and configure scene
num_rows=8
num_cols=2
scene_etoile = config_scene(num_rows, num_cols)

rm_solver = RadioMapSolver()
# Compute the SINR map
rm = rm_solver(scene_etoile,
                      max_depth=2,
                      samples_per_tx=10**6,
                      cell_size=(1, 1),
                      orientation=[0,0,0],
                      center=[0,0,1.0],
                      size=[80*1, 80*1],
                      refraction=False,
                      diffraction=True,
                  )

# cam = Camera(position=[0,-2,15],
#                  look_at=[0,0,0])
# scene_etoile.render(camera=cam,
#                     radio_map=rm,
#                     rm_metric="path_gain",
#                     # rm_vmin=-10,
#                     # rm_vmax=60
#                 );

# rm.show(metric="path_gain", tx=0)
rm.show(metric="path_gain")

gain = 10 * np.log10(rm.path_gain[0])

norm_gain = np.copy(gain)
min_g = -128.
max_g = 0.
norm_gain[norm_gain < min_g] = min_g
norm_gain = (norm_gain - min_g) / (max_g - min_g)

norm_gain = (norm_gain * 255).astype(np.uint8)

cv2.imwrite("grayscale.png", norm_gain)

# plt.show()
