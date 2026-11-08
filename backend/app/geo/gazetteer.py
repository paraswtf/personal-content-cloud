"""Offline gazetteer (docs/ARCHITECTURE.md section 25): GeoNames cities15000 with the admin-1 and
country names, loaded lazily into numpy arrays on first use. place_for(lat, lon) answers the nearest
city within RADIUS_KM as a Place (city, admin, country, display) or None. Without the files, or with
GEO_ENABLED=false, available() is False and every lookup answers None; download() fetches the three
files into MODELS_DIR/geo (scripts/download_models.py calls it on every start).

GeoNames data is CC BY 4.0 (geonames.org); the attribution sits in the README, the map footer and
Settings > System."""
from __future__ import annotations

import csv
import io
import logging
import math
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

RADIUS_KM = 100.0
NEAR_FACTOR = 1.5   # a bigger place within this multiple of the nearest distance names the spot
NEAR_KM = 5.0       # or within this many kilometres
EARTH_KM = 6371.0088
CITIES_ZIP = "cities15000.zip"
CITIES_TXT = "cities15000.txt"
ADMIN1 = "admin1CodesASCII.txt"
COUNTRIES = "countryInfo.txt"
FILES = (CITIES_ZIP, ADMIN1, COUNTRIES)
BASE_URL = "https://download.geonames.org/export/dump/"
DOWNLOAD_TIMEOUT_S = 120


@dataclass(frozen=True)
class Place:
    city: str
    admin: str | None
    country: str | None
    lat: float
    lon: float

    @property
    def display(self) -> str:
        return ", ".join(p for p in (self.city, self.admin, self.country) if p)


def geo_dir() -> Path:
    return settings.models_dir / "geo"


def missing_files() -> list[str]:
    d = geo_dir()
    return [name for name in FILES if not (d / name).is_file()]


def available() -> bool:
    """The files are present and lookups are allowed."""
    return bool(settings.geo_enabled) and not missing_files()


class Gazetteer:
    """Cities sorted by latitude, so a lookup scans only the latitude band that can be within reach."""

    def __init__(self, names: list[str], admins: list[str | None], countries: list[str | None],
                 lats: np.ndarray, lons: np.ndarray, populations: np.ndarray | None = None) -> None:
        order = np.argsort(lats, kind="stable")
        self.names = [names[i] for i in order]
        self.admins = [admins[i] for i in order]
        self.countries = [countries[i] for i in order]
        self.lats = lats[order]
        self.lons = lons[order]
        pops = populations if populations is not None else np.zeros(len(names), dtype=np.float64)
        self.populations = pops[order]
        self.lat_rad = np.radians(self.lats)
        self.lon_rad = np.radians(self.lons)

    def __len__(self) -> int:
        return len(self.names)

    def nearest(self, lat: float, lon: float, radius_km: float = RADIUS_KM) -> Place | None:
        """The place that names the spot: the nearest city within radius_km, unless a bigger one sits
        about as close (within NEAR_FACTOR times the nearest distance, or NEAR_KM), so a photo in a
        district of Paris says Paris, not the district."""
        if not len(self.names) or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return None
        band = math.degrees(radius_km / EARTH_KM) + 1e-9
        lo = int(np.searchsorted(self.lats, lat - band, side="left"))
        hi = int(np.searchsorted(self.lats, lat + band, side="right"))
        if hi <= lo:
            return None
        plat, plon = math.radians(lat), math.radians(lon)
        dlat = self.lat_rad[lo:hi] - plat
        dlon = self.lon_rad[lo:hi] - plon
        a = np.sin(dlat / 2) ** 2 + math.cos(plat) * np.cos(self.lat_rad[lo:hi]) * np.sin(dlon / 2) ** 2
        km = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        i = int(np.argmin(km))
        d0 = float(km[i])
        if d0 > radius_km:
            return None
        near = km <= max(d0 * NEAR_FACTOR, NEAR_KM)
        pops = np.where(near, self.populations[lo:hi], -1.0)
        best = int(np.argmax(pops))
        if pops[best] <= self.populations[lo + i]:
            best = i
        j = lo + best
        return Place(city=self.names[j], admin=self.admins[j], country=self.countries[j],
                     lat=float(self.lats[j]), lon=float(self.lons[j]))


def _read_admin1(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 2 and row[0] and not row[0].startswith("#"):
                out[row[0]] = row[1]
    return out


def _read_countries(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 5 and row[0] and not row[0].startswith("#"):
                out[row[0]] = row[4]
    return out


def _parse(d: Path) -> Gazetteer:
    admin1 = _read_admin1(d / ADMIN1)
    countries = _read_countries(d / COUNTRIES)
    names: list[str] = []
    admins: list[str | None] = []
    cnames: list[str | None] = []
    lats: list[float] = []
    lons: list[float] = []
    pops: list[float] = []
    with zipfile.ZipFile(d / CITIES_ZIP) as zf:
        member = next((n for n in zf.namelist() if n.endswith(".txt")), CITIES_TXT)
        with zf.open(member) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"), delimiter="\t",
                                quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) < 11 or not row[1]:
                    continue
                try:
                    lat, lon = float(row[4]), float(row[5])
                except ValueError:
                    continue
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    continue
                cc = row[8].strip()
                names.append(row[1].strip())
                admins.append(admin1.get(f"{cc}.{row[10].strip()}") if row[10].strip() else None)
                cnames.append(countries.get(cc))
                lats.append(lat)
                lons.append(lon)
                try:
                    pops.append(float(row[14]) if len(row) > 14 and row[14].strip() else 0.0)
                except ValueError:
                    pops.append(0.0)
    return Gazetteer(names, admins, cnames, np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64),
                     np.asarray(pops, dtype=np.float64))


_lock = threading.Lock()
_loaded: Gazetteer | None = None
_loaded_from: Path | None = None
_warned = False


def load() -> Gazetteer | None:
    """The gazetteer, parsed once per process (and again when MODELS_DIR changes); None when unavailable."""
    global _loaded, _loaded_from, _warned
    if not available():
        if not _warned and settings.geo_enabled:
            log.info("places are off: %s missing in %s", ", ".join(missing_files()), geo_dir())
            _warned = True
        return None
    d = geo_dir()
    with _lock:
        if _loaded is not None and _loaded_from == d:
            return _loaded
        try:
            _loaded = _parse(d)
            _loaded_from = d
            log.info("gazetteer loaded: %d places from %s", len(_loaded), d)
        except Exception:  # noqa: BLE001 - a broken download must not take the worker down
            log.exception("could not load the gazetteer from %s", d)
            _loaded = None
            _loaded_from = None
        return _loaded


def reset() -> None:
    """Forget the parsed data (tests, and after a download)."""
    global _loaded, _loaded_from, _warned
    with _lock:
        _loaded = None
        _loaded_from = None
        _warned = False


def place_for(lat: float | None, lon: float | None, radius_km: float = RADIUS_KM) -> Place | None:
    if lat is None or lon is None:
        return None
    g = load()
    if g is None:
        return None
    return g.nearest(float(lat), float(lon), radius_km)


def download(progress: Callable[[str], None] | None = None) -> list[str]:
    """Fetch the missing files into MODELS_DIR/geo through a .part file; returns the names fetched.
    Does nothing when GEO_ENABLED is false. Raises on a failed fetch (the caller reports it)."""
    if not settings.geo_enabled:
        return []
    d = geo_dir()
    fetched: list[str] = []
    for name in missing_files():
        d.mkdir(parents=True, exist_ok=True)
        part = d / f"{name}.part"
        if progress:
            progress(f"downloading places data: {name}")
        req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "personal-content-cloud"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp, part.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if part.stat().st_size == 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: empty download")
        part.replace(d / name)
        fetched.append(name)
    if fetched:
        reset()
    return fetched
