from lib.poly2scenes import create_full_scene, load_dataset, Dataset, DatasetItem
from lib.osm2poly import brasilia_osm_polys
import matplotlib.pyplot as plt
from pathlib import Path

# PLOT_STUFF = True
PLOT_STUFF = False

buildings, street_edges, tx_pos = brasilia_osm_polys(
    # force_update=True
)

if PLOT_STUFF:
    fig, ax = plt.subplots(figsize=(12,12))

    street_edges.plot(ax=ax, linewidth=0.5, edgecolor='gray')

    buildings.plot(ax=ax, edgecolor='black', facecolor='lightblue')

    tx_pos.plot(ax=ax, edgecolor='red', facecolor='pink')

    ax.set_aspect('equal')
    # ax.axis('off')
    plt.show()

# creates full sionna scene
# create_full_scene(
#     Path("./BRASILIA/scene"),
#     buildings, street_edges
# )

ds = load_dataset(
    Path("./BRASILIA/ds"),
    # force_update=True,
)

print("Number of building configurations:", len(ds.items))
print("Number of samples:", ds.n_sets)
print("Maximum number of tx in a single scene:", ds.max_tx_in_frame)

for frame in ds.items:
    print("frame.aabb", frame.aabb)
    print("frame.buildings_frame", frame.buildings_frame)
    print("len(frame.tx_frames)", len(frame.tx_frames))
    print(f"{frame.tx_positions[0]=}")
    # print("frame.tx_frames", frame.tx_frames)
    break
