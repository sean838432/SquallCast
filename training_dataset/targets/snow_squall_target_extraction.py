#!/usr/bin/env python3
"""
Extract point-location "snow squall targets" from NWS Snow Squall Warnings.

For every Snow Squall Warning (VTEC phenomena=SQ, significance=W) issued in the
continental US between a start date (default 2020-01-01) and today, this script:

  1. Downloads the warning polygon + issuance time from the IEM VTEC archive.
  2. Finds the nearest WSR-88D (NEXRAD) radar to the warning.
  3. Fetches that radar's Level II volume scan closest in time to issuance
     (from the public "gcp-public-data-nexrad-l2" archive) and reads it with Py-ART.
  4. Locates the lat/lon of maximum reflectivity within the warning polygon,
     on the lowest elevation sweep.
  5. Appends one row per warning to a CSV: warning id, issuance time, radar site,
     lat/lon of peak reflectivity, and the reflectivity value.

Data sources
------------
  * Warnings : IEM VTEC GIS archive (mesonet.agron.iastate.edu), documented at
               https://mesonet.agron.iastate.edu/info/datasets/vtec.html
  * Radar sites : NWS API, https://api.weather.gov/radar/stations
  * Radar data  : Google Cloud public dataset "gcp-public-data-nexrad-l2", which
               mirrors NOAA's Level II archive as hourly tarballs of complete volume
               files. (The equivalent AWS bucket "noaa-nexrad-level2" does not permit
               anonymous bucket *listing* -- only GetObject with a key you already
               know -- so listing is done against the GCS mirror instead.)

Requirements
------------
    pip install requests pandas geopandas shapely pyproj arm_pyart

Usage
-----
    python snow_squall_target_extraction.py --output snow_squall_targets.csv
    python snow_squall_target_extraction.py --start 2023-01-01 --end 2023-03-01 --limit 20
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import base as shapely_base

log = logging.getLogger("snow_squall_target_extraction")

IEM_WATCHWARN_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
NWS_RADAR_STATIONS_URL = "https://api.weather.gov/radar/stations"
GCS_LIST_URL = "https://storage.googleapis.com/storage/v1/b/gcp-public-data-nexrad-l2/o"
GCS_DOWNLOAD_URL = "https://storage.googleapis.com/download/storage/v1/b/gcp-public-data-nexrad-l2/o/{key}"

REQUEST_TIMEOUT = 60
HTTP_RETRIES = 3
HTTP_RETRY_SLEEP = 5
POLITE_SLEEP = 0.5  # seconds between radar downloads, to be a good API citizen

CSV_FIELDS = [
    "warning_id",
    "wfo",
    "etn",
    "issued_utc",
    "expired_utc",
    "polygon_centroid_lat",
    "polygon_centroid_lon",
    "radar_site",
    "radar_lat",
    "radar_lon",
    "volume_scan_time_utc",
    "max_reflectivity_dbz",
    "target_lat",
    "target_lon",
]


def http_get(url: str, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            log.warning("GET %s failed (attempt %d/%d): %s", url, attempt, HTTP_RETRIES, exc)
            time.sleep(HTTP_RETRY_SLEEP)
    raise last_exc


# --------------------------------------------------------------------------- #
# 1. Snow squall warnings
# --------------------------------------------------------------------------- #

@dataclass
class SnowSquallWarning:
    warning_id: str
    wfo: str
    etn: str
    issued: datetime
    expired: datetime
    geometry: shapely_base.BaseGeometry


def fetch_snow_squall_warnings(start: date, end: date) -> list[SnowSquallWarning]:
    """Download all SQ.W (Snow Squall Warning) polygons issued in [start, end).

    Queried one calendar year at a time to keep each IEM response small.
    """
    import geopandas as gpd

    warnings: list[SnowSquallWarning] = []
    seen_keys: set[tuple] = set()

    cursor = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)

    while cursor < end_dt:
        chunk_end = min(datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc), end_dt)
        params = {
            "accept": "shapefile",
            "sts": cursor.strftime("%Y-%m-%dT%H:%MZ"),
            "ets": chunk_end.strftime("%Y-%m-%dT%H:%MZ"),
            "phenomena": "SQ",
            "significance": "W",
            "limitps": "yes",
        }
        log.info("Fetching SQ.W warnings %s -> %s", params["sts"], params["ets"])
        resp = http_get(IEM_WATCHWARN_URL, params=params)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "wwa.zip")
            with open(zip_path, "wb") as f:
                f.write(resp.content)
            try:
                gdf = gpd.read_file(f"zip://{zip_path}")
            except Exception:
                log.info("No warnings found for %s -> %s", params["sts"], params["ets"])
                cursor = chunk_end
                continue

        for _, row in gdf.iterrows():
            key = (row["WFO"], row["ETN"], row["VTEC_YR"], row["ISSUED"])
            if key in seen_keys:
                continue  # duplicate row from multi-county UGC expansion
            seen_keys.add(key)

            issued = datetime.strptime(row["ISSUED"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            expired = datetime.strptime(row["EXPIRED"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            warning_id = f"{row['WFO']}-SQ-W-{row['VTEC_YR']}-{row['ETN']}"
            warnings.append(
                SnowSquallWarning(
                    warning_id=warning_id,
                    wfo=row["WFO"],
                    etn=str(row["ETN"]),
                    issued=issued,
                    expired=expired,
                    geometry=row["geometry"],
                )
            )

        cursor = chunk_end

    warnings.sort(key=lambda w: w.issued)
    log.info("Found %d snow squall warnings between %s and %s", len(warnings), start, end)
    return warnings


# --------------------------------------------------------------------------- #
# 2. Radar sites
# --------------------------------------------------------------------------- #

def fetch_radar_stations() -> pd.DataFrame:
    """Return CONUS WSR-88D radar sites as a DataFrame with id, lat, lon."""
    resp = http_get(NWS_RADAR_STATIONS_URL)
    data = resp.json()
    rows = []
    for feat in data["features"]:
        props = feat["properties"]
        if props.get("stationType") != "WSR-88D":
            continue
        lon, lat = feat["geometry"]["coordinates"][:2]
        rows.append({"site": props["id"], "lat": lat, "lon": lon})
    df = pd.DataFrame(rows)
    log.info("Loaded %d WSR-88D radar sites", len(df))
    return df


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_radar(lat: float, lon: float, stations: pd.DataFrame) -> pd.Series:
    dists = stations.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
    return stations.loc[dists.idxmin()]


# --------------------------------------------------------------------------- #
# 3. NEXRAD Level II retrieval (GCS mirror of NOAA archive)
# --------------------------------------------------------------------------- #

def _gcs_list(prefix: str) -> list[dict]:
    items: list[dict] = []
    params = {"prefix": prefix, "maxResults": 1000}
    while True:
        resp = http_get(GCS_LIST_URL, params=params)
        data = resp.json()
        items.extend(data.get("items", []))
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return items


def _hour_tar_candidates(site: str, target: datetime) -> list[dict]:
    """List the hourly tarballs for the target hour and its neighbors."""
    candidates: list[dict] = []
    for delta_hours in (0, -1, 1):
        t = target + timedelta(hours=delta_hours)
        prefix = f"{t:%Y/%m/%d}/{site}/"
        for item in _gcs_list(prefix):
            name = item["name"].rsplit("/", 1)[-1]
            if not name.endswith(".tar"):
                continue
            candidates.append(item)
    # de-dupe (adjacent-day queries can overlap)
    uniq = {item["name"]: item for item in candidates}
    return list(uniq.values())


def _volume_time_from_name(name: str) -> Optional[datetime]:
    # e.g. KTLX20220106_003722_V06.ar2v
    base = name.rsplit("/", 1)[-1]
    if "_MDM" in base or not base.endswith(".ar2v"):
        return None
    try:
        parts = base.replace(".ar2v", "").split("_")
        ymd_and_site = parts[0]
        hms = parts[1]
        ymd = ymd_and_site[-8:]
        return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (IndexError, ValueError):
        return None


def fetch_closest_volume_file(site: str, target: datetime, max_gap: timedelta = timedelta(minutes=30)) -> Optional[str]:
    """Download the NEXRAD Level II volume scan closest in time to `target`.

    Returns a path to a local temp file, or None if nothing close enough was found.
    """
    tars = _hour_tar_candidates(site, target)
    if not tars:
        log.warning("No Level II data available for %s near %s", site, target)
        return None

    best_member = None
    best_tar_path = None
    best_gap = None

    for tar_item in tars:
        key = tar_item["name"]
        tar_hour_start = datetime.strptime(key.rsplit("/", 1)[-1].split("_")[-2], "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
        if abs((tar_hour_start - target).total_seconds()) > 3600 + max_gap.total_seconds():
            continue

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
            resp = http_get(
                GCS_DOWNLOAD_URL.format(key=requests.utils.quote(key, safe="")),
                params={"alt": "media", "generation": tar_item["generation"]},
            )
            tmp_tar.write(resp.content)
            tmp_tar_path = tmp_tar.name

        replaced_best = False
        try:
            with tarfile.open(tmp_tar_path) as tf:
                for member in tf.getmembers():
                    vt = _volume_time_from_name(member.name)
                    if vt is None:
                        continue
                    gap = abs((vt - target).total_seconds())
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_member = member.name
                        replaced_best = True
        except tarfile.TarError as exc:
            log.warning("Could not read tar %s: %s", key, exc)
            os.unlink(tmp_tar_path)
            continue

        if replaced_best:
            if best_tar_path and os.path.exists(best_tar_path):
                os.unlink(best_tar_path)
            best_tar_path = tmp_tar_path
        else:
            os.unlink(tmp_tar_path)

    if best_member is None or best_gap is None or best_gap > max_gap.total_seconds():
        log.warning("No volume scan within %s of %s at %s", max_gap, site, target)
        if best_tar_path and os.path.exists(best_tar_path):
            os.unlink(best_tar_path)
        return None

    out_dir = tempfile.mkdtemp()
    with tarfile.open(best_tar_path) as tf:
        tf.extract(best_member, path=out_dir)
    os.unlink(best_tar_path)
    return os.path.join(out_dir, best_member)


# --------------------------------------------------------------------------- #
# 4. Peak reflectivity within the warning polygon
# --------------------------------------------------------------------------- #

def find_peak_reflectivity(volume_path: str, polygon: shapely_base.BaseGeometry, sweep: int = 0):
    """Return (lat, lon, dbz, scan_time) of the max reflectivity gate inside `polygon`."""
    import pyart

    radar = pyart.io.read_nexrad_archive(volume_path)
    try:
        refl = radar.get_field(sweep, "reflectivity")
        lats, lons, _ = radar.get_gate_lat_lon_alt(sweep)

        minx, miny, maxx, maxy = polygon.bounds
        bbox_mask = (lons >= minx) & (lons <= maxx) & (lats >= miny) & (lats <= maxy)
        if not np.any(bbox_mask):
            return None

        inside = np.zeros_like(bbox_mask)
        idx = np.where(bbox_mask)
        inside[idx] = shapely.contains_xy(polygon, lons[idx], lats[idx])
        if not np.any(inside):
            return None

        refl_masked = np.where(inside, np.ma.filled(refl, np.nan), np.nan)
        if np.all(np.isnan(refl_masked)):
            return None

        flat_idx = np.nanargmax(refl_masked)
        row, col = np.unravel_index(flat_idx, refl_masked.shape)

        scan_time = pyart.util.datetime_from_radar(radar)
        return float(lats[row, col]), float(lons[row, col]), float(refl_masked[row, col]), scan_time
    finally:
        del radar


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def load_already_processed(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["warning_id"])
        return set(df["warning_id"])
    except Exception:
        return set()


def process_warning(w: SnowSquallWarning, stations: pd.DataFrame) -> Optional[dict]:
    centroid = w.geometry.centroid
    radar = nearest_radar(centroid.y, centroid.x, stations)

    volume_path = fetch_closest_volume_file(radar["site"], w.issued)
    if volume_path is None:
        return None

    try:
        result = find_peak_reflectivity(volume_path, w.geometry)
    finally:
        try:
            os.unlink(volume_path)
            os.rmdir(os.path.dirname(volume_path))
        except OSError:
            pass

    if result is None:
        log.warning("No reflectivity gates found inside polygon for %s", w.warning_id)
        return None

    lat, lon, dbz, scan_time = result
    return {
        "warning_id": w.warning_id,
        "wfo": w.wfo,
        "etn": w.etn,
        "issued_utc": w.issued.isoformat(),
        "expired_utc": w.expired.isoformat(),
        "polygon_centroid_lat": centroid.y,
        "polygon_centroid_lon": centroid.x,
        "radar_site": radar["site"],
        "radar_lat": radar["lat"],
        "radar_lon": radar["lon"],
        "volume_scan_time_utc": scan_time.isoformat() if scan_time else "",
        "max_reflectivity_dbz": dbz,
        "target_lat": lat,
        "target_lon": lon,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD), UTC")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), UTC. Default: today")
    parser.add_argument("--output", type=str, default="snow_squall_targets.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N warnings (for testing)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else datetime.now(timezone.utc).date()

    stations = fetch_radar_stations()
    warnings = fetch_snow_squall_warnings(start, end)
    if args.limit:
        warnings = warnings[: args.limit]

    already_done = load_already_processed(args.output)
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0

    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

        for i, w in enumerate(warnings, 1):
            if w.warning_id in already_done:
                continue
            log.info("[%d/%d] Processing %s (issued %s)", i, len(warnings), w.warning_id, w.issued)
            try:
                row = process_warning(w, stations)
            except Exception:
                log.exception("Failed to process %s", w.warning_id)
                row = None

            if row:
                writer.writerow(row)
                f.flush()
                log.info(
                    "  -> target %.4f, %.4f (%.1f dBZ) via %s",
                    row["target_lat"], row["target_lon"], row["max_reflectivity_dbz"], row["radar_site"],
                )
            time.sleep(POLITE_SLEEP)

    log.info("Done. Results written to %s", args.output)


if __name__ == "__main__":
    main()
