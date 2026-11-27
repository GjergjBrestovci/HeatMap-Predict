# Datasets for HeatMap-Predict

Which data the plan in the [README](../README.md) needs, where to get it, and what it costs in
accounts and keys.

The plan has three parts, referred to below as:

- **P1 – Parks vs sealed areas:** satellite heat maps and vegetation plus ground measurements,
  comparing small/medium/large parks with sealed control areas.
- **P2 – Climate vs urban expansion:** historical satellite and weather data, to separate
  long-term warming from local land-use change.
- **P3 – ML model:** predict how temperature shifts when a neighbourhood's layout changes.

Access column: **open** = no account or key; **account** = free registration needed;
**key** = free registration plus an API token.

Endpoints marked ✔ were called successfully from this repo on 2026-11-26. Everything else is
from the providers' documentation and should be re-checked when first used.

## Overview

| # | Dataset | Variable | Resolution | Time range | Access | Serves |
|---|---------|----------|------------|------------|--------|--------|
| 1 | Landsat 8/9 C2 L2 | surface temperature, reflectance | 30 m (thermal 100 m native) | 2013– | open ✔ | P1, P3 |
| 2 | Landsat 4/5/7 C2 L2 | surface temperature, reflectance | 30 m (thermal 60–120 m native) | 1982–2024 | open | P2 |
| 3 | MODIS MOD11/MYD11 | LST day and night | 1 km | 2000– | open ✔ (PC) / account (NASA) | P2 |
| 4 | ECOSTRESS LST | LST at varying times of day | 70 m | 2018– | account | P1, P3 |
| 5 | Sentinel-2 L2A | reflectance → NDVI | 10 m | 2015– | open ✔ (PC) / account (CDSE) | P1, P3 |
| 6 | GeoSphere station data (klima-v2, TAWES) | air temperature, humidity, wind, radiation | points, 10 min–daily | 1775– (Hohe Warte) | open ✔ | P1, P2, P3 |
| 7 | GeoSphere SPARTACUS | daily min/max air temperature, precipitation | 1 km | 1961– | open ✔ | P2 |
| 8 | GeoSphere INCA | hourly air temperature, wind, humidity, radiation | 1 km | 2011– | open ✔ | P1, P3 |
| 9 | ERA5 / ERA5-Land | reanalysis (2 m temperature etc.) | ~31 km / ~9 km | 1940– / 1950– | key (CDS) | P2 |
| 10 | Stadt Wien: Öffentlich zugängliche Grünflächen | park polygons with names | vector | current | open ✔ | P1 |
| 11 | Stadt Wien: Realnutzungskartierung | land use per block | vector | 1981–2024 (17 editions) | open ✔ | P1, P2, P3 |
| 12 | Stadt Wien: Flächenmehrzweckkarte (FMZK) | buildings, traffic, green and water surfaces | vector, cadastral scale | current | open ✔ | P1, P3 |
| 13 | Stadt Wien: Baumkataster | street and park trees | points | current | open ✔ | P3 |
| 14 | Stadt Wien: building info / surface model | building height and type | vector / 0.5–1 m raster | current | open | P3 |
| 15 | Copernicus HRL Imperviousness | sealed surface share | 10–20 m (100 m aggregates) | 2006–2018, every 3 years | account | P1, P2, P3 |
| 16 | Copernicus Urban Atlas | land use, street tree layer, building height | vector, 0.25 ha minimum unit | 2006, 2012, 2018 | account | P2 |
| 17 | GHSL built-up surface | built-up area per cell | 100 m (10 m for 2018) | 1975–2030, 5-year steps | open | P2 |
| 18 | OpenStreetMap | roads, railways, land use, buildings | vector | current (history available) | open | P1, P3 |

## Satellite land surface temperature

### 1. Landsat 8/9 Collection 2 Level-2 (Surface Temperature)

- **Source:** USGS. Easiest access: Microsoft Planetary Computer (PC) STAC API,
  `https://planetarycomputer.microsoft.com/api/stac/v1`, collection `landsat-c2-l2` ✔.
  Also USGS EarthExplorer and the AWS `usgs-landsat` bucket.
- **Covers:** band `lwir11` (ST_B10) as surface temperature in Kelvin
  (`DN * 0.00341802 + 149.0`); red and NIR reflectance for NDVI; `qa_pixel` for clouds.
  30 m grid, but the thermal sensor measures at 100 m and is resampled. Each satellite
  revisits every 16 days; the two together give 8 days. Overpass about 09:45 UTC (late
  morning local time). Vienna sits on WRS-2 path/row 190/026, 189/026 and 189/027.
  2013 to now.
- **Licence:** public domain (USGS).
- **Download:** STAC search, then read the cloud-optimised GeoTIFFs directly with rasterio.
  PC signs URLs anonymously via the `planetary-computer` package. **Open.** EarthExplorer
  needs a free USGS account; the AWS bucket is requester-pays and needs an AWS account.
- **Serves:** P1 (the main heat map), P3 (training target).
- **Known limits, seen in the prototype:** about 9 % of the Vienna window has no surface
  temperature in every scene, including cloud-free ones. The gaps are identical between
  scenes, which points to missing emissivity input (the product uses ASTER GED) rather than
  clouds. One native thermal pixel is 1 ha, so pocket parks are measured mostly by their
  surroundings. Only a morning overpass exists; the night-time heat island is not visible.

### 2. Landsat 4/5/7 Collection 2 Level-2

- **Source / download / licence:** same as above, same `landsat-c2-l2` collection (filter on
  `platform`). **Open.**
- **Covers:** Landsat 4/5 TM 1982–2012 (thermal 120 m native), Landsat 7 ETM+ 1999–2024
  (60 m native; striped data gaps from 2003 on because of the scan-line corrector failure).
- **Serves:** P2. This is the only free archive with sub-kilometre LST going back to the 1980s.
- **Caveat:** sensors differ, so trends across Landsat 5 → 7 → 8 need care. Compare
  differences within a scene (park minus sealed area) rather than absolute temperatures.
- **Checked 2026-11-27 ✔:** Landsat 5 has 16–22 summer items per year over Vienna from 1984.
  The thermal asset is called `lwir` (not `lwir11`), with the same scale and offset. The
  overpass time moved from about 09:05 UTC in 1986 to 09:35 in 2010.

### 3. MODIS LST (MOD11A1/MYD11A1 daily, MOD11A2/MYD11A2 8-day)

- **Source:** NASA LP DAAC. On PC as `modis-11A1-061` and `modis-11A2-061` ✔.
- **Covers:** day and night LST at 1 km. Terra since 2000 (about 10:30 and 22:30 local),
  Aqua since 2002 (about 13:30 and 01:30). Global. Both satellites are near end of life;
  VIIRS (VNP21, 750 m–1 km, 2012–) is the successor.
- **Licence:** open, no restrictions (NASA).
- **Download:** PC STAC, **open**. NASA Earthdata (AppEEARS, `earthaccess`) needs a free
  **account**.
- **Serves:** P2: a 25-year daily series with night-time values, for city-vs-surroundings
  heat island trends. Too coarse for individual parks.

### 4. ECOSTRESS LST (ECO_L2T_LSTE)

- **Source:** NASA/JPL instrument on the ISS, distributed by LP DAAC:
  <https://lpdaac.usgs.gov/> , request tool <https://appeears.earthdatacloud.nasa.gov/>.
- **Covers:** LST and emissivity at 70 m, since mid-2018. The ISS orbit means overpass times
  drift through the day and night, so it captures afternoon and night scenes that Landsat
  cannot. Revisit is irregular (roughly every 1–5 days, many scenes cloudy or partial).
- **Licence:** open, no restrictions (NASA).
- **Download:** AppEEARS area request or `earthaccess` in Python. Needs a NASA Earthdata
  **account**.
- **Serves:** P1 (night-time park cooling), P3 (time-of-day as a model feature).

## Vegetation / NDVI

### 5. Sentinel-2 Level-2A

- **Source:** ESA/Copernicus. On PC as `sentinel-2-l2a` ✔. Official portal: Copernicus Data
  Space Ecosystem (CDSE), <https://dataspace.copernicus.eu/>.
- **Covers:** surface reflectance, 10 m for red (B04) and NIR (B08), revisit about 5 days
  (2–3 days at Vienna's latitude with overlapping orbits). 2015 to now (L2A over Europe from 2017).
  Vienna is in tile 33UWP (and 33UXP for the eastern edge).
- **Licence:** free and open, attribution required (Copernicus Sentinel data terms).
- **Download:** PC STAC, **open**. CDSE needs a free **account**.
- **Serves:** P1 and P3. NDVI at 10 m resolves pocket parks, tree rows and courtyards that
  Landsat's 30 m blurs.
- **Landsat NDVI** (datasets 1 and 2) comes with the thermal scene at no extra cost, is
  perfectly co-registered with LST, and reaches back to the 1980s. Use it for P2 and wherever
  LST and NDVI must come from the same moment.

## Ground stations and gridded climate (GeoSphere Austria)

All GeoSphere products below come from the Data Hub, <https://data.hub.geosphere.at/>, through
one REST API: `https://dataset.api.hub.geosphere.at/v1/` ✔. **Open**, no key. Licence
**CC BY 4.0** (cite GeoSphere Austria and the dataset DOI). Formats: CSV, GeoJSON, NetCDF for
grids. The list of all resources is at `/v1/datasets`; each has a `/metadata` endpoint with
parameters and stations. Large requests are rejected by size, so split by year.

### 6. Station data: `klima-v2-*` and `tawes-v1-10min`

- **Covers:** quality-controlled station series at 10 min, hourly, daily, monthly and yearly
  steps (`klima-v2-10min`, `-1h`, `-1d`, `-1m`, `-1y`), and the raw 10-minute feed (`tawes`).
  Air temperature, humidity, wind, sunshine, radiation, precipitation.
- **Vienna stations active now** ✔: Hohe Warte (id 105 / 5904, records from 1775), Innere
  Stadt (5925, from 1985), Donaufeld (5935), Unterlaa (107 / 5917), Mariabrunn (106 / 5805),
  Stammersdorf (4115), Jubiläumswarte (5802). Schwechat airport and Groß-Enzersdorf serve as
  rural references just outside the city.
- **Download example** ✔:
  `/v1/station/historical/klima-v2-1d?parameters=tl_mittel,tlmax&station_ids=105&start=2026-08-09&end=2026-08-09&output_format=csv`
- **Serves:** P1 (air temperature at satellite overpass time; Innere Stadt vs Hohe Warte vs
  outskirts as the classic heat island contrast), P2 (long series), P3 (weather features).
- **Monthly series** (`klima-v2-1m`: `tl_mittel`, `tlmin_mittel`, `tlmax_mittel`) ✔: long
  stations appear under a merged id (105 Hohe Warte, 31 Groß-Enzersdorf) and again per
  physical site; the `substation` column says which site a month comes from. The merged
  series are not homogenised: Hohe Warte minus Groß-Enzersdorf drops by about 0.5 °C when
  the latter changes site in 1992/93. HISTALP (`histalp-v1-1y`) is homogenised but annual
  only and ends in 2022.
- **Caveat:** seven stations cannot validate individual parks. Satellites measure surface
  temperature, stations measure air at 2 m; these differ by 10 °C and more on a sunny day.

### 7. SPARTACUS (`spartacus-v3-1d-1km`, also monthly/quarterly/yearly; v2 still served)

- **Covers:** gridded daily minimum and maximum air temperature, precipitation and sunshine
  duration for Austria at 1 km, from 1961, interpolated from stations.
- **Download:** `/v1/grid/historical/spartacus-v3-1d-1km` with a `bbox` (NetCDF), or
  `/v1/timeseries/historical/...` for points (CSV; 25 points and 46 years of the monthly
  `spartacus-v3-1m-1km` in one request worked ✔). Monthly parameters `TM`, `TN`, `TX`.
- **Serves:** P2: the regional climate baseline for Vienna and its rural surroundings since
  1961, independent of satellites.
- **Caveat:** built from stations, so it shows the urban effect only where stations exist.
  It is a background-climate dataset, not a heat map.

### 8. INCA (`inca-v2-1h-1km`)

- **Covers:** hourly analysis of temperature, humidity, wind and global radiation at 1 km,
  from 2011.
- **Serves:** P1 and P3: weather conditions at the hour of each satellite overpass, for every
  location rather than only at stations.

### 9. ERA5 and ERA5-Land

- **Source:** ECMWF / Copernicus Climate Change Service, Climate Data Store (CDS),
  <https://cds.climate.copernicus.eu/>.
- **Covers:** global hourly reanalysis. ERA5: about 31 km, 1940 to now. ERA5-Land: about 9 km,
  1950 to now. 2 m temperature, wind, radiation, soil moisture and many more.
- **Licence:** CC BY 4.0 (check the licence shown on each CDS dataset page and accept it there
  before the first download).
- **Download:** `cdsapi` Python client. Needs a free ECMWF **account and API key**
  (`~/.cdsapirc`).
- **Without a key:** PC hosts `era5-pds` ✔, but only for 1979–2020. The Open-Meteo archive API
  (<https://open-meteo.com/>) serves ERA5 and ERA5-Land point series with no key, free for
  non-commercial use.
- **Serves:** P2: the large-scale warming signal with no urban influence at all (Vienna is a
  handful of grid cells). Subtracting it from city observations is one way to isolate the
  local effect. SPARTACUS covers the same need at higher resolution for 1961 onward; ERA5
  adds 1940–1960 and variables SPARTACUS lacks.

## Vienna open data (Stadt Wien OGD)

Catalogue: <https://www.data.gv.at/> (filter on publisher "Stadt Wien"). All layers below are
served by one WFS, **open**, no key ✔:

```
https://data.wien.gv.at/daten/geo?service=WFS&request=GetFeature&version=1.1.0
  &typeName=ogdwien:<LAYER>&srsName=EPSG:4326&outputFormat=json
```

`outputFormat` can also be `shape-zip` or `csv`. `geopandas.read_file(url)` reads it directly.
Licence **CC BY 4.0**, attribution "Datenquelle: Stadt Wien – data.wien.gv.at".
Layer names are from the WFS capabilities document on 2026-11-26.

### 10. Öffentlich zugängliche Grünflächen: `OEFFGRUENFLOGD`

- **Covers:** 1,936 polygons of publicly accessible green space with name (`T_LANG`), area
  (`FLAECHE`, m²) and type flags (`P_PK` park, `P_WLD` forest, `P_WAS` water, …).
- **Serves:** P1: named park outlines, and the area attribute gives the small/medium/large
  classes. Related: `PARKINFOOGD` (1,052 park points with facilities such as water features).

### 11. Realnutzungskartierung: `REALNUT2024OGD`, and 16 earlier editions

- **Covers:** land use per block in about 30 classes on three levels, for example
  "Bahnhöfe und Bahnanlagen", "Straßenraum unbegrünt", "dichtes Wohn(misch)gebiet",
  "Parkplätze u. Parkhäuser", "Park, Grünanlage". The 2022 edition has 18,644 polygons.
- **Editions on the WFS:** 1981, 1985, 1991, 1994, 1997, 2001, 2003, 2005, 2007/08, 2009,
  2012, 2014, 2016, 2018, 2020, 2022, 2024.
- **Serves:** P1 (picks the sealed control areas by class instead of by hand), P2 (40 years of
  mapped urban expansion from one source), P3 (land-use class as a feature and as the thing
  that scenarios change).
- **Caveat:** the class scheme changed twice (checked 2026-11-27 ✔). Editions 1981–2005 have
  only a two-letter code (`NUTCD`, about 43 codes) and no legend is published with the data
  or on the printed maps; `uhi/history.py` translates them by overlap with the 2007/08
  edition. Editions 2007/08–2022 name the class in `NUTZUNG_LEVEL1`–`3`, 2024 in
  `LEV1`–`3`. In 1981–1991 about half of the polygons have no code: they are block outlines
  lying on top of the coded ones. Streets are not mapped before 2007/08 (about 4,000 ha).
  Layer names are irregular: `REALNUT200708OGD`, `REALNUT2016GOGD`.

### 12. Flächenmehrzweckkarte (FMZK)

- **Covers:** Vienna's large-scale digital base map as separate layers: buildings
  (`FMZKGEBOGD`), traffic surfaces (`FMZKVERKEHR1OGD`, `FMZKVERKEHR2OGD`), green surfaces
  (`FMZKGRUEN1OGD`, `FMZKGRUEN2OGD`), water (`FMZKGEWOGD`), and `FMZKBKMOGD`.
- **Serves:** P1 and P3: the most detailed picture of what is sealed. Rasterising it to the
  Landsat grid gives the share of building, road and green per pixel, which is the natural
  feature set for the model and the layer a "replace the car park with a park" scenario edits.
- **Caveat:** the full city is large. Request by `bbox` or download the district tiles from
  data.gv.at. Attribute meanings were not inspected yet.

### 13. Baumkataster: `BAUMKATOGD`

- **Covers:** 233,731 city-managed trees ✔ with species, planting year, trunk girth, height and
  crown diameter. Private gardens and forests are not included.
- **Serves:** P3: street-tree canopy, which no land-use map captures.

### 14. Building information and surface model

- **Covers:** `GEBAEUDEINFOOGD` and `GEBAEUDETYPOGD` (building attributes), and the city's
  digital surface and terrain models (Oberflächenmodell / Geländemodell, raster tiles on
  data.gv.at). Height = surface minus terrain.
- **Serves:** P3: building height and density drive shading and heat storage, and are needed
  to model "add a housing block".
- **To check:** tile sizes and which attributes carry height.

Also look on data.gv.at for the city's own heat products (Stadtklimaanalyse, Urban Heat
Vulnerability Index) and the air-quality network's temperature readings. They were not
verified for this document.

## Europe-wide and global layers (for history and for comparison)

### 15. Copernicus High Resolution Layer: Imperviousness

- **Source:** Copernicus Land Monitoring Service, <https://land.copernicus.eu/>.
- **Covers:** share of sealed surface per pixel, 0–100 %. Reference years 2006, 2009, 2012,
  2015 (20 m) and 2018 (10 m), plus change layers. Europe.
- **Licence:** free and open, attribution required. **Download:** needs a free EU Login
  **account**.
- **Serves:** a ready-made, continuous sealed-surface measure for P1 and P3, and five
  consistent time steps for P2.

### 16. Copernicus Urban Atlas

- **Source:** as above. **Covers:** land use for Vienna's functional urban area, 2006, 2012,
  2018, with change layers, a street tree layer, and building heights (2012, core city).
- **Access:** **account** (EU Login). **Serves:** P2, and it extends beyond the city boundary,
  which the Vienna OGD layers do not.

### 17. GHSL built-up surface (GHS-BUILT-S)

- **Source:** EU Joint Research Centre,
  <https://human-settlement.emergency.copernicus.eu/download.php>.
- **Covers:** built-up area per 100 m cell, 1975–2030 in 5-year steps, derived from Landsat and
  Sentinel. Global.
- **Licence:** CC BY 4.0. **Download:** GeoTIFF tiles over plain HTTP, **open** ✔. Vienna is
  in tile `R4_C20` of the 100 m Mollweide grid, about 25 MB per epoch. It counts building
  footprints, not roads or car parks.
- **Serves:** P2: urban expansion on the same time axis as the Landsat thermal archive, and
  covering the suburbs in Lower Austria.

### 18. OpenStreetMap

- **Source:** Overpass API or Geofabrik extracts (<https://download.geofabrik.de/europe/austria.html>).
- **Licence:** ODbL (attribution and share-alike for derived databases). **Open.**
- **Serves:** named features the city layers lack (for example the Gürtel as a road), and data
  outside the city boundary.

## Where to start

1. **Landsat 8/9 surface temperature (1):** the heat map itself, open, and already working in
   the prototype.
2. **Öffentlich zugängliche Grünflächen + Realnutzungskartierung (10, 11):** define parks and
   sealed controls from official data, and the Realnutzung editions back to 1981 carry P2.
3. **Sentinel-2 NDVI (5):** vegetation at 10 m, same access route as Landsat.
4. **GeoSphere stations (6):** the only ground truth, and the longest record.
5. **SPARTACUS (7):** the climate baseline for P2, from the same API as the stations.

All five are open without any account. The first account worth creating is NASA Earthdata,
for ECOSTRESS night-time scenes.

## Accounts and keys

| Needed for | What | Cost |
|------------|------|------|
| ECOSTRESS; MODIS/VIIRS from NASA | NASA Earthdata login, <https://urs.earthdata.nasa.gov/> | free |
| ERA5 / ERA5-Land from CDS | ECMWF account + API key, <https://cds.climate.copernicus.eu/> | free |
| HRL Imperviousness, Urban Atlas | EU Login, <https://land.copernicus.eu/> | free |
| Sentinel-2 from the official portal (optional) | CDSE account, <https://dataspace.copernicus.eu/> | free |
| Landsat from EarthExplorer (optional) | USGS ERS account | free |

Nothing in the list costs money. Google Earth Engine would give most satellite datasets in one
place, but needs a Google Cloud project and is free only for non-commercial use; it is not
required.
