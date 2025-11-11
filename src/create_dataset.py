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
create_full_scene(
    Path("./BRASILIA/scene"),
    buildings, street_edges
)

ds = load_dataset(
    Path("./BRASILIA/ds"),
    force_update=True,
)

a = False
total = 0
res = {
    "train": [],
    "val": [],
    "test": [],
    "max_path_gain": ds.max_path_gain,
    "min_path_gain": ds.min_path_gain,
    "min_height": ds.min_height,
    "max_height": ds.max_height,
}
validation = False
test = False

for i in range(len(ds.items)):
    for j in range(len(ds.items[i].tx_frames)):
        id = (i, j)
        if not Path(f"./BRASILIA/ds/gains/IRT2/{i}_{j}.png").exists():
            a = True
            break
        total+=1
        if test:
            res["test"].append(id)
        elif validation:
            res["val"].append(id)
        else:
            res["train"].append(id)

    if total > 43413 * 0.8:
        validation = True
    if total > 43413 * 0.9:
        test = True
    if a:
        break
import json

with open(Path(f"./BRASILIA/ds/ids.json"), "w") as fp:
    json.dump(res, fp)

print("total", total)
print("0.1 * 43413", 0.1 * 43413)
print("0.8 * 43413", 0.8 * 43413)
print("test", len(res["test"]))
print("val", len(res["val"]))
print("train", len(res["train"]))
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
