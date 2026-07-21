"""Plot snow squall target locations on a simple US map."""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

CSV_PATH = "training_dataset/targets/snow_squall_targets.csv"

df = pd.read_csv(CSV_PATH)

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
ax.add_feature(cfeature.LAKES, facecolor="lightblue")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES, edgecolor="gray", linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

ax.scatter(
    df["target_lon"],
    df["target_lat"],
    s=8,
    c="crimson",
    alpha=0.5,
    transform=ccrs.PlateCarree(),
)

ax.set_title(f"Snow Squall Target Locations (n={len(df)})")

plt.savefig("training_dataset/targets/snow_squall_map.png", dpi=150, bbox_inches="tight")
plt.show()
