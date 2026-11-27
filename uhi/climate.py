"""Air temperature since 1981: station series, homogenised series and the SPARTACUS grid.

The monthly station series and SPARTACUS come from the GeoSphere Austria Data
Hub (open, no key, CC BY 4.0). The homogenised HISTALP series come from
www.zamg.ac.at/histalp (no key; "downloadable for non-commercial research
without restrictions", to be acknowledged with Auer et al. 2007).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://dataset.api.hub.geosphere.at/v1"
STATIONS = f"{API}/station/historical/klima-v2-1m"
SPARTACUS = f"{API}/timeseries/historical/spartacus-v3-1m-1km"
# Monthly means: of the daily mean, of the daily minima, of the daily maxima.
STATION_PARAMETERS = {"tl_mittel": "mean_c", "tlmin_mittel": "min_c", "tlmax_mittel": "max_c"}
SPARTACUS_PARAMETERS = {"TM": "mean_c", "TN": "min_c", "TX": "max_c"}
MEASURES = ("mean_c", "min_c", "max_c")

HISTALP = "https://www.zamg.ac.at/histalp/dataset/station/csv.php"
HISTALP_MISSING = 999999
HISTALP_UNIT_C = 0.1  # values are tenths of a degree
# name: (station key, abbreviation, role). Hohe Warte is the only Vienna station with a
# homogenised monthly series up to now; the three others are the nearest low-lying ones.
HISTALP_STATIONS = {
    "Wien-Hohe Warte": ("_152", "WIE", "urban"),
    "Eisenstadt": ("_269", "EST", "rural"),
    "Retz": ("_115", "RET", "rural"),
    "St. Pölten": ("_221", "SPO", "rural"),
}
POINTS_PER_REQUEST = 25


def _get(url: str, params, path: Path) -> str:
    if not path.exists():
        resp = requests.get(url, params=params, timeout=300)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return path.read_text()


def station_metadata(cache: Path) -> pd.DataFrame:
    """All stations of the monthly dataset: id, name, lon, lat, altitude."""
    meta = json.loads(_get(f"{STATIONS}/metadata", None, cache / "climate" / "stations_metadata.json"))
    names = {p["name"] for p in meta["parameters"]}
    missing = set(STATION_PARAMETERS) - names
    if missing:
        raise SystemExit(f"Monthly station dataset lacks {sorted(missing)}")
    stations = pd.DataFrame(meta["stations"])
    stations["id"] = stations["id"].astype(str)
    return stations[["id", "name", "lon", "lat", "altitude"]]


def parse_station_csv(text: str) -> pd.DataFrame:
    """The API's CSV as time (UTC), station (str), substation (str), mean_c, min_c, max_c."""
    df = pd.read_csv(io.StringIO(text))
    missing = ({"time", "station"} | set(STATION_PARAMETERS)) - set(df.columns)
    if missing:
        raise SystemExit(f"Station CSV lacks {sorted(missing)}; has {list(df.columns)}")
    out = pd.DataFrame(
        {"time": pd.to_datetime(df["time"], utc=True), "station": df["station"].astype(str)}
    )
    sub = df["substation"] if "substation" in df.columns else pd.Series(np.nan, index=df.index)
    # The physical site behind a merged series; empty where the hub gives none.
    out["substation"] = [
        "" if pd.isna(v) else str(int(v)) for v in pd.to_numeric(sub, errors="coerce")
    ]
    for source, name in STATION_PARAMETERS.items():
        out[name] = pd.to_numeric(df[source], errors="coerce")
    return out


def fetch_station_monthly(station_ids: list[str], first: int, last: int, cache: Path) -> pd.DataFrame:
    """Monthly air temperature of the given stations for `first`-`last`."""
    frames = []
    for i in range(0, len(station_ids), 20):
        chunk = station_ids[i : i + 20]
        params = {
            "parameters": ",".join(STATION_PARAMETERS),
            "station_ids": ",".join(chunk),
            "start": f"{first}-01-01",
            "end": f"{last}-12-31",
            "output_format": "csv",
        }
        path = cache / "climate" / f"stations_{first}-{last}_{chunk[0]}-{chunk[-1]}_{len(chunk)}.csv"
        frames.append(parse_station_csv(_get(STATIONS, params, path)))
    return pd.concat(frames, ignore_index=True)


def parse_spartacus_csv(text: str) -> pd.DataFrame:
    """The API's CSV as time (UTC), lat, lon (of the grid cell), mean_c, min_c, max_c."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.split(" [")[0] for c in df.columns]  # "TM [degC]" -> "TM"
    missing = ({"time", "lat", "lon"} | set(SPARTACUS_PARAMETERS)) - set(df.columns)
    if missing:
        raise SystemExit(f"SPARTACUS CSV lacks {sorted(missing)}; has {list(df.columns)}")
    out = pd.DataFrame({"time": pd.to_datetime(df["time"], utc=True), "lat": df["lat"], "lon": df["lon"]})
    for source, name in SPARTACUS_PARAMETERS.items():
        out[name] = pd.to_numeric(df[source], errors="coerce")
    return out


def fetch_spartacus(points: list[tuple[float, float]], first: int, last: int, cache: Path, tag: str) -> pd.DataFrame:
    """Monthly gridded air temperature at the 1 km cells nearest to `points` (lat, lon)."""
    frames = []
    for i in range(0, len(points), POINTS_PER_REQUEST):
        chunk = points[i : i + POINTS_PER_REQUEST]
        params = [("parameters", ",".join(SPARTACUS_PARAMETERS)), ("start", f"{first}-01-01"),
                  ("end", f"{last}-12-31"), ("output_format", "csv")]
        params += [("lat_lon", f"{lat:.4f},{lon:.4f}") for lat, lon in chunk]
        path = cache / "climate" / f"spartacus_{tag}_{first}-{last}_{len(points)}_{i:04d}.csv"
        frames.append(parse_spartacus_csv(_get(SPARTACUS, params, path)))
    return pd.concat(frames, ignore_index=True)


def lattice(bounds: tuple[float, float, float, float], spacing: float) -> list[tuple[float, float]]:
    """Regular points (x, y) inside `bounds`, `spacing` apart, starting half a step in."""
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx + spacing / 2, maxx, spacing)
    ys = np.arange(miny + spacing / 2, maxy, spacing)
    return [(float(x), float(y)) for y in ys for x in xs]


def pick_series(monthly: pd.DataFrame, stations: pd.DataFrame, summers: pd.DataFrame, min_summers: int) -> pd.DataFrame:
    """One series per station name: the id with the most complete summers.

    The data hub lists long stations twice, once under a merged id and once per
    physical site. `summers` has `station`, `year` (complete summers only).
    Returns the stations kept, with `summers` and the number of physical sites
    (`sites`) that contributed, a sign of relocations.
    """
    counts = summers.groupby("station")["year"].nunique().rename("summers")
    table = stations.merge(counts, left_on="id", right_index=True)
    table = table[table["summers"] >= min_summers]
    table = table.assign(id_number=table["id"].astype(int)).sort_values(
        ["summers", "id_number"], ascending=[False, True]
    ).drop(columns="id_number")
    table = table.drop_duplicates("name").reset_index(drop=True)
    summer_rows = monthly[monthly["time"].dt.month.isin((6, 7, 8))].dropna(subset=["mean_c"])
    sites = summer_rows[summer_rows["substation"] != ""].groupby("station")["substation"]
    table["sites"] = table["id"].map(sites.nunique()).fillna(1).astype(int)
    changes = {}
    for sid, g in summer_rows[summer_rows["substation"] != ""].groupby("station"):
        g = g.sort_values("time")
        moved = g["substation"] != g["substation"].shift()
        changes[sid] = [int(y) for y in g.loc[moved, "time"].dt.year.iloc[1:]]
    table["site_change_years"] = table["id"].map(lambda s: changes.get(s, []))
    return table


def parse_histalp_csv(text: str) -> pd.DataFrame:
    """A HISTALP station export as year and June-August mean in °C (NaN if a month is missing)."""
    rows = [line for line in text.splitlines() if line and not line.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), sep=";")
    missing = {"year", "jun", "jul", "aug"} - set(df.columns)
    if missing:
        raise SystemExit(f"HISTALP CSV lacks {sorted(missing)}; has {list(df.columns)}")
    months = df[["jun", "jul", "aug"]].apply(pd.to_numeric, errors="coerce")
    months = months.where(months != HISTALP_MISSING)
    return pd.DataFrame(
        {"year": df["year"].astype(int), "mean_c": months.mean(axis=1, skipna=False) * HISTALP_UNIT_C}
    )


def fetch_histalp(first: int, last: int, cache: Path) -> pd.DataFrame:
    """Homogenised summer means: one column per HISTALP station, one row per year."""
    series = {}
    for name, (key, abbr, _) in HISTALP_STATIONS.items():
        path = cache / "climate" / f"histalp_{abbr}_{first}-{last}.csv"
        if not path.exists():
            form = {"country": "AT", "station": key, "statabbr": abbr, "parameter": "T01",
                    "years": f"{first} - {last}", "exportCSV": "export Data as CSV"}
            resp = requests.post(HISTALP, data=form, timeout=120)
            resp.raise_for_status()
            if "year;" not in resp.text:
                raise SystemExit(f"HISTALP returned no table for {name}: {resp.text[:200]!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resp.content)
        series[name] = parse_histalp_csv(path.read_text()).set_index("year")["mean_c"]
    return pd.DataFrame(series).sort_index()
