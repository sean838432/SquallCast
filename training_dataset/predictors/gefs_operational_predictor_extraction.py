"""
Extract GEFS vertical-profile predictors at each snow squall target location.

For every target in `snow_squall_targets.csv`, this script pulls a GEFS
control-run "Day 1" forecast profile at the nearest gridpoint:

  * Initialization : 12Z the calendar day *before* the warning was issued
  * Forecast hour   : whichever GEFS lead time (f000, f003, ... or f000,
                       f006, ... depending on era) falls closest to the
                       actual warning issuance time, so each event uses a
                       ~12-36 hour lead time rather than a fixed one.
  * Member          : control run (gec00)
  * Fields          : HGT, TMP, RH, UGRD, VGRD on mandatory pressure levels,
                       plus 2 m TMP/RH/DPT(derived), 10 m UGRD/VGRD, PRMSL,
                       SNOD, WEASD, APCP (accumulated since prior fcst hour),
                       CSNOW (categorical snow flag).

Data source
-----------
NOAA GEFS operational archive on AWS S3 (public, unsigned):
    https://noaa-gefs-pds.s3.amazonaws.com/index.html

The bucket layout changed with the GEFSv12 upgrade (2020-09-23):
    pre  : gefs.YYYYMMDD/HH/pgrb2a/gec00.tHHz.pgrb2af{FF|FFF}
    post : gefs.YYYYMMDD/HH/atmos/pgrb2ap5/gec00.tHHz.pgrb2a.0p50.f{FFF}
Both layouts expose the same "a" (pressure-level + surface) product on a
0.5-degree grid, so this script tries the post-upgrade path first and falls
back to the pre-upgrade path.

Requirements
------------
    pip install pandas numpy requests xarray
    conda install -c conda-forge pygrib eccodes

Usage
-----
    python gefs_operational_predictor_extraction.py \
        --targets /home/sean834/squallcast/training_dataset/targets/snow_squall_targets.csv \
        --output-dir gefs_profiles
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr

log = logging.getLogger("gefs_operational_predictor_extraction")

BASE_URL = "https://noaa-gefs-pds.s3.amazonaws.com"
UPGRADE_DATE = date(2020, 9, 23)  # GEFSv12 operational upgrade

MEMBER = "gec00"  # control run

# Available forecast-hour steps for the pgrb2a product, by era.
# pre-upgrade  (GEFSv11) : 6-hourly, 0-384h
# post-upgrade (GEFSv12) : 3-hourly to 240h, then 6-hourly to 384h
PRE_UPGRADE_FCST_HOURS = list(range(0, 384 + 1, 6))
POST_UPGRADE_FCST_HOURS = list(range(0, 240 + 1, 3)) + list(range(246, 384 + 1, 6))

REQUEST_TIMEOUT = 120
HTTP_RETRIES = 3
HTTP_RETRY_SLEEP = 5

PROFILE_VARS = ["HGT", "TMP", "RH", "UGRD", "VGRD"]
# union of mandatory pressure levels (mb) present across PROFILE_VARS
PROFILE_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 100, 50, 10]

# (grib shortName, grib typeOfLevel, grib level) -> output variable name
SURFACE_FIELDS = {
    ("2t", "heightAboveGround", 2): "TMP_2m",
    ("2r", "heightAboveGround", 2): "RH_2m",
    ("10u", "heightAboveGround", 10): "UGRD_10m",
    ("10v", "heightAboveGround", 10): "VGRD_10m",
    ("prmsl", "meanSea", 0): "PRMSL",
    ("sde", "surface", 0): "SNOD",
    ("sdwe", "surface", 0): "WEASD",
    ("tp", "surface", 0): "APCP",
    ("csnow", "surface", 0): "CSNOW",
}


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


def http_head_ok(url: str) -> bool:
    try:
        resp = requests.head(url, timeout=REQUEST_TIMEOUT)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------- #
# URL construction
# --------------------------------------------------------------------------- #

def gefs_grib_url(init_dt: datetime, forecast_hour: int) -> str:
    """Return the URL of the gec00 pgrb2a file for this init time + lead hour.

    Tries the post-GEFSv12 layout first, falls back to the pre-upgrade one.
    """
    ymd = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")

    post_url = (
        f"{BASE_URL}/gefs.{ymd}/{hh}/atmos/pgrb2ap5/"
        f"{MEMBER}.t{hh}z.pgrb2a.0p50.f{forecast_hour:03d}"
    )
    pre_url = (
        f"{BASE_URL}/gefs.{ymd}/{hh}/pgrb2a/"
        f"{MEMBER}.t{hh}z.pgrb2af{forecast_hour:02d}"
    )

    if init_dt.date() >= UPGRADE_DATE:
        candidates = [post_url, pre_url]
    else:
        candidates = [pre_url, post_url]

    for url in candidates:
        if http_head_ok(url):
            return url

    raise FileNotFoundError(
        f"No GEFS pgrb2a file found for init {init_dt.isoformat()} f{forecast_hour:03d} at either layout"
    )


def init_datetime_for_event(issued_utc: datetime) -> datetime:
    """12Z the calendar day before the warning was issued."""
    day_before = issued_utc.date() - timedelta(days=1)
    return datetime(day_before.year, day_before.month, day_before.day, 12, tzinfo=timezone.utc)


def nearest_forecast_hour(init_dt: datetime, issued_utc: datetime) -> tuple[int, float]:
    """Return (forecast_hour, lead_hours) for the GEFS lead time closest to issuance.

    `lead_hours` is the *actual* (fractional) time between init and issuance;
    `forecast_hour` is the nearest available GEFS archive step to that lead.
    """
    lead_hours = (issued_utc - init_dt).total_seconds() / 3600.0
    available = PRE_UPGRADE_FCST_HOURS if init_dt.date() < UPGRADE_DATE else POST_UPGRADE_FCST_HOURS
    forecast_hour = min(available, key=lambda fh: abs(fh - lead_hours))
    return forecast_hour, lead_hours


# --------------------------------------------------------------------------- #
# GRIB download + decode
# --------------------------------------------------------------------------- #

@dataclass
class GefsProfile:
    init_time: datetime
    valid_time: datetime
    forecast_hour: int
    lead_hours: float
    grid_lat: float
    grid_lon: float
    profile: dict  # var -> {level: value}
    surface: dict  # var -> value


def download_grib(url: str) -> str:
    resp = http_get(url)
    fd, path = tempfile.mkstemp(suffix=".grib2")
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path


def nearest_index(grb, lat: float, lon: float) -> tuple[int, int]:
    """Analytic nearest-neighbor index into a regular lat/lon GRIB grid."""
    lat0 = grb["latitudeOfFirstGridPointInDegrees"]
    lon0 = grb["longitudeOfFirstGridPointInDegrees"]
    dlat = grb["jDirectionIncrementInDegrees"]
    dlon = grb["iDirectionIncrementInDegrees"]
    ny = grb["Ny"]
    nx = grb["Nx"]

    lon_wrapped = lon % 360.0
    lon0_wrapped = lon0 % 360.0

    # latitude decreases from lat0 as j increases (standard for GEFS pgrb2a)
    j = int(round((lat0 - lat) / dlat))
    i = int(round(((lon_wrapped - lon0_wrapped) % 360.0) / dlon))

    j = min(max(j, 0), ny - 1)
    i = min(max(i, 0), nx - 1)
    return j, i


def extract_profile(grib_path: str, lat: float, lon: float) -> tuple[dict, dict, float, float, Optional[datetime]]:
    """Read the gec00 pgrb2a file and pull profile + surface fields at (lat, lon)."""
    import pygrib

    profile: dict = {v: {} for v in PROFILE_VARS}
    surface: dict = {}
    grid_lat = grid_lon = None
    valid_time = None

    with pygrib.open(grib_path) as grbs:
        for grb in grbs:
            valid_time = valid_time or grb.validDate.replace(tzinfo=timezone.utc)

            if grb.typeOfLevel == "isobaricInhPa" and grb.shortName in PROFILE_VARS_SHORTNAMES:
                var = PROFILE_VARS_SHORTNAMES[grb.shortName]
                level = int(grb.level)
                if level not in PROFILE_LEVELS:
                    continue
                j, i = nearest_index(grb, lat, lon)
                profile[var][level] = float(grb.values[j, i])
                if grid_lat is None:
                    grid_lat, grid_lon = float(grb.latlons()[0][j, i]), float(grb.latlons()[1][j, i])
                continue

            key = (grb.shortName, grb.typeOfLevel, int(grb.level))
            if key in SURFACE_FIELDS:
                out_name = SURFACE_FIELDS[key]
                j, i = nearest_index(grb, lat, lon)
                surface[out_name] = float(grb.values[j, i])
                if grid_lat is None:
                    grid_lat, grid_lon = float(grb.latlons()[0][j, i]), float(grb.latlons()[1][j, i])

    return profile, surface, grid_lat, grid_lon, valid_time


PROFILE_VARS_SHORTNAMES = {
    "gh": "HGT",
    "t": "TMP",
    "r": "RH",
    "u": "UGRD",
    "v": "VGRD",
}


def dewpoint_from_t_rh(temp_k: float, rh_pct: float) -> float:
    """Magnus-formula dewpoint (K) from temperature (K) and relative humidity (%)."""
    t_c = temp_k - 273.15
    rh = max(min(rh_pct, 100.0), 0.1)
    a, b = 17.625, 243.04
    gamma = (a * t_c / (b + t_c)) + np.log(rh / 100.0)
    dewpoint_c = (b * gamma) / (a - gamma)
    return dewpoint_c + 273.15


# --------------------------------------------------------------------------- #
# NetCDF output
# --------------------------------------------------------------------------- #

def build_dataset(warning_row: pd.Series, gp: GefsProfile) -> xr.Dataset:
    data_vars = {}
    for var in PROFILE_VARS:
        values = [gp.profile[var].get(lvl, np.nan) for lvl in PROFILE_LEVELS]
        data_vars[var] = ("level", np.array(values, dtype=np.float32))

    for name, value in gp.surface.items():
        data_vars[name] = ((), np.float32(value))

    if "TMP_2m" in gp.surface and "RH_2m" in gp.surface:
        data_vars["DPT_2m"] = ((), np.float32(dewpoint_from_t_rh(gp.surface["TMP_2m"], gp.surface["RH_2m"])))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"level": np.array(PROFILE_LEVELS, dtype=np.int32)},
        attrs={
            "warning_id": warning_row["warning_id"],
            "wfo": warning_row["wfo"],
            "issued_utc": str(warning_row["issued_utc"]),
            "target_lat": float(warning_row["target_lat"]),
            "target_lon": float(warning_row["target_lon"]),
            "gefs_init_time_utc": gp.init_time.isoformat(),
            "gefs_valid_time_utc": gp.valid_time.isoformat() if gp.valid_time else "",
            "gefs_forecast_hour": gp.forecast_hour,
            "gefs_lead_hours": gp.lead_hours,
            "gefs_member": MEMBER,
            "gefs_grid_lat": gp.grid_lat,
            "gefs_grid_lon": gp.grid_lon,
            "source": "https://noaa-gefs-pds.s3.amazonaws.com",
        },
    )
    ds["level"].attrs["units"] = "hPa"
    ds["HGT"].attrs.update(units="gpm", long_name="Geopotential height")
    ds["TMP"].attrs.update(units="K", long_name="Temperature")
    ds["RH"].attrs.update(units="%", long_name="Relative humidity")
    ds["UGRD"].attrs.update(units="m/s", long_name="U-component of wind")
    ds["VGRD"].attrs.update(units="m/s", long_name="V-component of wind")
    return ds


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def load_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["issued_utc"] = pd.to_datetime(df["issued_utc"], utc=True)
    return df


def process_all(targets: pd.DataFrame, output_dir: str, limit: Optional[int] = None, overwrite: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if limit:
        targets = targets.iloc[:limit]

    # cache the decoded grib file per init date, since many warnings on the
    # same calendar day share the same GEFS initialization
    cache_url: Optional[str] = None
    cache_path: Optional[str] = None

    total = len(targets)
    n_ok = n_fail = 0

    for i, row in enumerate(targets.itertuples(index=False), 1):
        warning_id = row.warning_id
        issued_utc = row.issued_utc.to_pydatetime()
        init_dt = init_datetime_for_event(issued_utc)
        forecast_hour, lead_hours = nearest_forecast_hour(init_dt, issued_utc)

        out_name = (
            f"{warning_id}"
            f"_issued{issued_utc:%Y%m%dT%H%M%SZ}"
            f"_gefs{init_dt:%Y%m%dT%Hz}"
            f"_f{forecast_hour:03d}.nc"
        )
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path) and not overwrite:
            log.info("[%d/%d] %s already extracted, skipping", i, total, warning_id)
            continue

        log.info(
            "[%d/%d] %s issued %s -> GEFS init %s, lead %.1fh -> f%03d",
            i, total, warning_id, issued_utc, init_dt, lead_hours, forecast_hour,
        )

        try:
            url = gefs_grib_url(init_dt, forecast_hour)

            if url != cache_url:
                if cache_path and os.path.exists(cache_path):
                    os.unlink(cache_path)
                log.info("  Downloading %s", url)
                cache_path = download_grib(url)
                cache_url = url

            profile, surface, grid_lat, grid_lon, valid_time = extract_profile(
                cache_path, row.target_lat, row.target_lon
            )
            gp = GefsProfile(
                init_time=init_dt,
                valid_time=valid_time,
                forecast_hour=forecast_hour,
                lead_hours=lead_hours,
                grid_lat=grid_lat,
                grid_lon=grid_lon,
                profile=profile,
                surface=surface,
            )

            row_series = pd.Series(row._asdict())
            ds = build_dataset(row_series, gp)
            ds.to_netcdf(out_path)
            n_ok += 1
            log.info("  -> wrote %s (gridpoint %.3f, %.3f)", out_path, grid_lat, grid_lon)
        except Exception:
            n_fail += 1
            log.exception("  Failed to extract %s", warning_id)

    if cache_path and os.path.exists(cache_path):
        os.unlink(cache_path)

    log.info("Done. %d succeeded, %d failed, %d skipped", n_ok, n_fail, total - n_ok - n_fail)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--targets",
        type=str,
        default="/home/sean834/squallcast/training_dataset/targets/snow_squall_targets.csv",
    )
    parser.add_argument("--output-dir", type=str, default="gefs_profiles")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N targets (for testing)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    targets = load_targets(args.targets)
    process_all(targets, args.output_dir, limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
