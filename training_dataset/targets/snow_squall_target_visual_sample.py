#!/usr/bin/env python3
"""
Visually spot-check snow squall target extraction.

Draws a random sample of `n` rows from snow_squall_targets.csv and, for each,
re-fetches the *actual* warning polygon (IEM VTEC archive) and the *actual*
NEXRAD Level II volume scan (Google Cloud mirror) used at extraction time, then
plots:

  * the warning polygon boundary
  * the base reflectivity sweep from the radar volume scan
  * the extracted target point (max-reflectivity location)
  * the radar site location

Everything plotted is driven by columns already in the CSV -- warning_id
(-> wfo/year/etn, used to refetch the polygon), issued_utc, volume_scan_time_utc
(the scan to fetch), and target_lat/target_lon (the extracted point) -- so this
serves as an independent visual check that snow_squall_target_extraction.py did
what it claims.

Usage
-----
    python snow_squall_target_visual_sample.py --n 6
    python snow_squall_target_visual_sample.py --n 12 --seed 42 --outdir samples
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely.geometry import shape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snow_squall_target_extraction import fetch_closest_volume_file  # noqa: E402

log = logging.getLogger("snow_squall_target_visual_sample")

VTEC_EVENT_URL = "https://mesonet.agron.iastate.edu/geojson/vtec_event.py"
REQUEST_TIMEOUT = 60


def parse_warning_id(warning_id: str) -> tuple[str, str, str]:
    """'BGM-SQ-W-2020-1' -> ('BGM', '2020', '1')."""
    wfo, _, _, year, etn = warning_id.split("-")
    return wfo, year, etn


def fetch_warning_polygon(wfo: str, year: str, etn: str):
    """Refetch the warning polygon geometry straight from the IEM VTEC archive."""
    params = {
        "wfo": wfo,
        "year": year,
        "phenomena": "SQ",
        "significance": "W",
        "etn": etn,
    }
    resp = requests.get(VTEC_EVENT_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    if not features:
        return None
    return shape(features[0]["geometry"])


def plot_sample(row: pd.Series, outdir: str) -> None:
    warning_id = row["warning_id"]
    wfo, year, etn = parse_warning_id(warning_id)
    scan_time = datetime.fromisoformat(row["volume_scan_time_utc"]).astimezone(timezone.utc)

    log.info("[%s] fetching warning polygon", warning_id)
    polygon = fetch_warning_polygon(wfo, year, etn)
    if polygon is None:
        log.warning("[%s] no polygon returned by IEM, skipping", warning_id)
        return

    log.info("[%s] fetching radar volume for %s near %s", warning_id, row["radar_site"], scan_time)
    volume_path = fetch_closest_volume_file(row["radar_site"], scan_time)
    if volume_path is None:
        log.warning("[%s] no radar volume found, skipping", warning_id)
        return

    import pyart

    try:
        radar = pyart.io.read_nexrad_archive(volume_path)
        try:
            display = pyart.graph.RadarMapDisplay(radar)

            fig = plt.figure(figsize=(8, 8))
            minx, miny, maxx, maxy = polygon.bounds
            pad = 0.6
            display.plot_ppi_map(
                "reflectivity",
                sweep=0,
                fig=fig,
                vmin=-20,
                vmax=70,
                cmap="NWSRef",
                min_lon=minx - pad,
                max_lon=maxx + pad,
                min_lat=miny - pad,
                max_lat=maxy + pad,
                colorbar_label="Reflectivity (dBZ)",
            )
            ax = display.ax

            xs, ys = polygon.exterior.xy if polygon.geom_type == "Polygon" else polygon.geoms[0].exterior.xy
            ax.plot(xs, ys, color="black", linewidth=2, transform=ccrs.PlateCarree(), label="Warning polygon")

            ax.plot(
                row["target_lon"], row["target_lat"],
                marker="*", markersize=18, color="red", markeredgecolor="black",
                transform=ccrs.PlateCarree(), label="Extracted target (max dBZ)",
            )
            ax.plot(
                row["radar_lon"], row["radar_lat"],
                marker="^", markersize=10, color="blue", markeredgecolor="black",
                transform=ccrs.PlateCarree(), label="Radar site",
            )
            ax.legend(loc="upper right", fontsize=8)

            ax.set_title(
                f"{warning_id}  ({row['radar_site']})\n"
                f"issued {row['issued_utc']}\n"
                f"scan {row['volume_scan_time_utc']}  |  {row['max_reflectivity_dbz']:.1f} dBZ",
                fontsize=10,
            )

            out_path = os.path.join(outdir, f"{warning_id}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("[%s] saved %s", warning_id, out_path)
        finally:
            del radar
    finally:
        try:
            os.unlink(volume_path)
            os.rmdir(os.path.dirname(volume_path))
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=os.path.join(os.path.dirname(__file__), "snow_squall_targets.csv"))
    parser.add_argument("--n", type=int, default=6, help="Number of random targets to sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible samples")
    parser.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(__file__), "visual_samples"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    for _, row in sample.iterrows():
        try:
            plot_sample(row, args.outdir)
        except Exception:
            log.exception("[%s] failed to plot", row["warning_id"])

    log.info("Done. Plots written to %s", args.outdir)


if __name__ == "__main__":
    main()
