#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract pixel centers from grav_32.1.tif as points (WGS84 geometry),
and sample SRTM15Plus.tif ('Elev') at the same locations.

Output: ESRI Shapefile (geometry in EPSG:4326) with attributes:
  lon, lat, src_epsg (=4326), grav_epsg, srtm_epsg, utm_e, utm_n, utm_zone, utm_hem, grav, elev

Usage:
  python extract_grav_srtm_to_shp.py grav_32.1.tif SRTM15Plus.tif out.shp
  python extract_grav_srtm_to_shp.py grav.tif srtm15.tif out.shp --step 4
  # If a raster has no CRS but you know it:
  python extract_grav_srtm_to_shp.py grav.tif srtm15.tif out.shp --assume-crs-grav 4326
"""

import argparse
import os
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import rasterio
from rasterio.windows import Window
import fiona
from fiona.crs import from_epsg
from shapely.geometry import Point, mapping
from pyproj import Transformer, CRS


def iter_windows(width: int, height: int, chunk: int = 1024) -> Iterable[Window]:
    for row_off in range(0, height, chunk):
        h = min(chunk, height - row_off)
        for col_off in range(0, width, chunk):
            w = min(chunk, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


def utm_zone_from_lon(lon: float) -> int:
    return max(1, min(60, int((lon + 180.0) // 6.0) + 1))


def get_utm_transformer(lon: float, lat: float,
                        cache: Dict[Tuple[int, str], Transformer]) -> Tuple[Transformer, int, str]:
    zone = utm_zone_from_lon(lon)
    hem = 'N' if lat >= 0.0 else 'S'
    epsg = 32600 + zone if hem == 'N' else 32700 + zone
    key = (zone, hem)
    if key not in cache:
        cache[key] = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return cache[key], zone, hem


def looks_like_geographic(ds: rasterio.io.DatasetReader) -> bool:
    b = ds.bounds
    px_w = abs(ds.transform.a)
    px_h = abs(ds.transform.e)
    return (b.left >= -360 and b.right <= 360 and b.bottom >= -180 and b.top <= 180) and (px_w <= 1.0 and px_h <= 1.0)


def get_epsg_int_from_crs(crs: CRS, assumed: Optional[int] = None) -> Optional[int]:
    # 1) direct EPSG
    try:
        epsg = crs.to_epsg()
        if epsg is not None:
            return int(epsg)
    except Exception:
        pass
    # 2) authority tuple (EPSG/XXXX or OGC/CRS84)
    try:
        auth = crs.to_authority()
        if auth:
            name, code = (auth[0] or "").upper(), str(auth[1])
            if name == "EPSG" and code.isdigit():
                return int(code)
            if name in ("OGC", "CRS") and code.upper() in ("CRS84", "84"):
                return 4326
    except Exception:
        pass
    # 3) equals WGS84
    try:
        if crs.equals(CRS.from_epsg(4326)):
            return 4326
    except Exception:
        pass
    # 4) fallback
    if assumed is not None:
        return int(assumed)
    return None


def get_crs_or_assume(ds: rasterio.io.DatasetReader, name: str, assume_epsg: Optional[int]) -> CRS:
    if ds.crs is not None:
        return ds.crs
    if assume_epsg is not None:
        print(f"[WARN] {name} has no CRS. Assuming EPSG:{assume_epsg}.")
        return CRS.from_epsg(int(assume_epsg))
    if looks_like_geographic(ds):
        print(f"[WARN] {name} has no CRS but looks geographic. Assuming EPSG:4326.")
        return CRS.from_epsg(4326)
    raise ValueError(f"{name} raster has no CRS. Set --assume-crs-{name.lower()} or fix the file.")


def main():
    ap = argparse.ArgumentParser(description="Extract gravity + SRTM15+ to Shapefile (WGS84).")
    ap.add_argument("grav_tif", help="Path to grav_32.1.tif (gravity raster)")
    ap.add_argument("srtm_tif", help="Path to SRTM15Plus.tif (elevation/bathymetry raster)")
    ap.add_argument("out_shp", help="Output ESRI Shapefile path, e.g., output.shp")
    ap.add_argument("--band-grav", type=int, default=1, help="Band index for gravity raster (default: 1)")
    ap.add_argument("--band-srtm", type=int, default=1, help="Band index for SRTM raster (default: 1)")
    ap.add_argument("--step", type=int, default=1, help="Stride: sample every Nth pixel (default: 1)")
    ap.add_argument("--chunk", type=int, default=1024, help="Processing chunk size (px)")
    ap.add_argument("--grav-nodata", type=float, default=None, help="Override grav nodata (exclude if matched)")
    ap.add_argument("--srtm-nodata", type=float, default=None, help="Override srtm nodata (record None)")
    ap.add_argument("--min-grav", type=float, default=None, help="Keep only grav >= this")
    ap.add_argument("--max-grav", type=float, default=None, help="Keep only grav <= this")
    ap.add_argument("--assume-crs-grav", type=int, default=None, help="Assume EPSG for GRAV if missing (e.g., 4326)")
    ap.add_argument("--assume-crs-srtm", type=int, default=None, help="Assume EPSG for SRTM if missing (e.g., 4326)")
    args = ap.parse_args()

    if args.step < 1:
        raise ValueError("--step must be >= 1")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_shp)) or ".", exist_ok=True)

    with rasterio.open(args.grav_tif) as gds, rasterio.open(args.srtm_tif) as sds:
        # Validate bands
        if not (1 <= args.band_grav <= gds.count):
            raise ValueError(f"grav band {args.band_grav} out of range (1..{gds.count})")
        if not (1 <= args.band_srtm <= sds.count):
            raise ValueError(f"srtm band {args.band_srtm} out of range (1..{sds.count})")

        # CRS
        grav_crs = get_crs_or_assume(gds, "GRAV", args.assume_crs_grav)
        srtm_crs = get_crs_or_assume(sds, "SRTM", args.assume_crs_srtm)

        grav_epsg = get_epsg_int_from_crs(grav_crs, args.assume_crs_grav)
        srtm_epsg = get_epsg_int_from_crs(srtm_crs, args.assume_crs_srtm)
        geom_epsg = 4326  # output shapefile geometry CRS

        # Nodata
        grav_nodata = args.grav_nodata if args.grav_nodata is not None else gds.nodatavals[args.band_grav - 1]
        srtm_nodata = args.srtm_nodata if args.srtm_nodata is not None else sds.nodatavals[args.band_srtm - 1]

        # Transformers
        to_wgs = Transformer.from_crs(grav_crs, f"EPSG:{geom_epsg}", always_xy=True)
        to_srtm = Transformer.from_crs(grav_crs, srtm_crs, always_xy=True) if grav_crs != srtm_crs else None

        # Shapefile schema
        schema = {
            "geometry": "Point",
            "properties": {
                "lon": "float",
                "lat": "float",
                "src_epsg": "int",    # geometry EPSG (4326)
                "grav_epsg": "int",   # source gravity EPSG (or -1)
                "srtm_epsg": "int",   # source SRTM EPSG (or -1)
                "utm_e": "float",
                "utm_n": "float",
                "utm_zone": "int",
                "utm_hem": "str:1",
                "grav": "float",
                "elev": "float",
            },
        }
        shp_crs = from_epsg(geom_epsg)

        # Cache UTM transformers
        utm_cache: Dict[Tuple[int, str], Transformer] = {}

        with fiona.open(args.out_shp, "w", driver="ESRI Shapefile",
                        crs=shp_crs, schema=schema, encoding="utf-8") as sink:

            for win in iter_windows(gds.width, gds.height, args.chunk):
                arr = gds.read(args.band_grav, window=win, masked=False)

                valid = np.ones_like(arr, dtype=bool)
                if grav_nodata is not None and not np.isnan(grav_nodata):
                    valid &= arr != grav_nodata
                if args.min_grav is not None:
                    valid &= arr >= args.min_grav
                if args.max_grav is not None:
                    valid &= arr <= args.max_grav

                if args.step > 1:
                    grid = np.zeros_like(valid, dtype=bool)
                    grid[::args.step, ::args.step] = True
                    valid &= grid

                if not np.any(valid):
                    continue

                rows, cols = np.where(valid)
                rows_full = rows + int(win.row_off)
                cols_full = cols + int(win.col_off)

                xs, ys = rasterio.transform.xy(gds.transform, rows_full, cols_full, offset="center")
                xs = np.asarray(xs, dtype=float)
                ys = np.asarray(ys, dtype=float)

                lons, lats = to_wgs.transform(xs, ys)

                if to_srtm is not None:
                    xs_srtm, ys_srtm = to_srtm.transform(xs, ys)
                else:
                    xs_srtm, ys_srtm = xs, ys

                srtm_vals = np.fromiter(
                    (float(v[0]) for v in sds.sample(zip(xs_srtm, ys_srtm), indexes=args.band_srtm)),
                    dtype=float,
                    count=len(xs_srtm),
                )

                feats: List[dict] = []
                gvals = arr[rows, cols].astype(float)
                for lon, lat, gval, sval in zip(lons, lats, gvals, srtm_vals):
                    elev = None
                    if srtm_nodata is None or (not np.isnan(srtm_nodata) and sval != srtm_nodata):
                        elev = float(sval)

                    utm_tr, zone, hem = get_utm_transformer(lon, lat, utm_cache)
                    easting, northing = utm_tr.transform(lon, lat)

                    feats.append({
                        "type": "Feature",
                        "geometry": mapping(Point(float(lon), float(lat))),
                        "properties": {
                            "lon": float(lon),
                            "lat": float(lat),
                            "src_epsg": int(geom_epsg),                         # always 4326
                            "grav_epsg": int(grav_epsg) if grav_epsg else -1,  # gravity EPSG or -1
                            "srtm_epsg": int(srtm_epsg) if srtm_epsg else -1,  # SRTM EPSG or -1
                            "utm_e": float(easting),
                            "utm_n": float(northing),
                            "utm_zone": int(zone),
                            "utm_hem": hem,
                            "grav": float(gval),
                            "elev": elev,
                        },
                    })

                if feats:
                    sink.writerecords(feats)

    print(f"Done. Wrote: {args.out_shp}")


if __name__ == "__main__":
    main()
