#!/usr/bin/env python3
"""
Constroi o indice tabular do dataset xBD/xView2 (GeoTIFF) para a EDA.

Varre `geotiffs/{split}/labels/*.json` + os GeoTIFFs correspondentes e escreve
tabelas Parquet em `cache/`:

  images.parquet        1 linha por imagem (22.068 linhas) - metadados da cena,
                        header do raster, tamanho em disco e contagens por classe.
  buildings.parquet     1 linha por poligono de edificacao - classe, area em px
                        e em m2, bbox e centroide.
  band_stats.parquet    estatisticas de pixel por banda em uma amostra aleatoria
                        de imagens (para normalizacao e checagem de dtype).
  registration.parquet  deslocamento do centroide pre->pos do MESMO predio
                        (mesmo uid), por cena - proxy de co-registro.

Uso:
    python scripts/build_index.py                       # tudo, com defaults
    python scripts/build_index.py --workers 8 --sample-n 400
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from shapely import wkt as shapely_wkt
from shapely.errors import GEOSException

# ----------------------------------------------------------------------------- config

SPLITS = ("tier1", "tier3", "hold", "test")

# Vocabulario de classes do xBD. `no-subtype` = poligono de imagem pre-desastre,
# que so tem a localizacao do predio (sem rotulo de dano).
DAMAGE_CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed", "un-classified")
NO_SUBTYPE = "no-subtype"

# graus -> metros (WGS84, aproximacao local; erro < 0.5% nas latitudes do dataset)
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


# ----------------------------------------------------------------------------- helpers


def parse_stem(stem: str) -> tuple[str, str, str]:
    """`hurricane-florence_00000006_post_disaster` -> (disaster, scene_id, phase)."""
    parts = stem.split("_")
    phase = "post" if "post" in parts else "pre"
    disaster = parts[0]
    scene_id = "_".join(parts[:2])
    return disaster, scene_id, phase


def meters_per_pixel(res_x_deg: float, res_y_deg: float, lat_deg: float) -> tuple[float, float]:
    """Converte a resolucao do GeoTIFF (graus/px, EPSG:4326) para metros/px."""
    mx = res_x_deg * M_PER_DEG_LON_EQ * math.cos(math.radians(lat_deg))
    my = res_y_deg * M_PER_DEG_LAT
    return mx, my


def _scan_one(args: tuple[str, str]) -> tuple[dict, list[dict]] | None:
    """Processa um par (label JSON, GeoTIFF). Roda em processo worker."""
    split, label_path = args
    lp = Path(label_path)
    stem = lp.stem
    tif = lp.parent.parent / "images" / f"{stem}.tif"

    try:
        with open(lp) as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {"__error__": f"{lp}: {exc}"}, []

    meta = doc.get("metadata", {})
    disaster, scene_id, phase = parse_stem(stem)

    rec: dict = {
        "split": split,
        "disaster": meta.get("disaster", disaster),
        "disaster_type": meta.get("disaster_type"),
        "scene_id": scene_id,
        "phase": phase,
        "stem": stem,
        "sensor": meta.get("sensor"),
        "catalog_id": meta.get("catalog_id"),
        "capture_date": meta.get("capture_date"),
        "gsd": meta.get("gsd"),
        "pan_resolution": meta.get("pan_resolution"),
        "off_nadir_angle": meta.get("off_nadir_angle"),
        "sun_azimuth": meta.get("sun_azimuth"),
        "sun_elevation": meta.get("sun_elevation"),
        "target_azimuth": meta.get("target_azimuth"),
        "meta_width": meta.get("width"),
        "meta_height": meta.get("height"),
        "json_bytes": lp.stat().st_size,
    }

    # ---- header do raster (nao le os pixels, so o cabecalho) ----
    lat_center = np.nan
    mx = my = np.nan
    try:
        rec["tif_bytes"] = tif.stat().st_size
        with rasterio.open(tif) as src:
            b = src.bounds
            lat_center = (b.bottom + b.top) / 2.0
            mx, my = meters_per_pixel(src.res[0], src.res[1], lat_center)
            rec.update(
                {
                    "tif_exists": True,
                    "height": src.height,
                    "width": src.width,
                    "bands": src.count,
                    "dtype": src.dtypes[0],
                    "crs": str(src.crs),
                    "nodata": src.nodata,
                    "compression": str(src.compression),
                    "res_x_deg": src.res[0],
                    "res_y_deg": src.res[1],
                    "lon_center": (b.left + b.right) / 2.0,
                    "lat_center": lat_center,
                    "m_per_px_x": mx,
                    "m_per_px_y": my,
                }
            )
    except (OSError, RasterioIOError) as exc:
        rec.update({"tif_exists": False, "tif_bytes": 0, "read_error": str(exc)[:200]})

    # ---- poligonos ----
    feats = doc.get("features", {}).get("xy", [])
    px_area_m2 = mx * my if np.isfinite(mx) and np.isfinite(my) else np.nan

    buildings: list[dict] = []
    counts = dict.fromkeys(DAMAGE_CLASSES, 0)
    counts[NO_SUBTYPE] = 0
    total_area_px = 0.0
    n_invalid = 0

    for feat in feats:
        props = feat.get("properties", {})
        subtype = props.get("subtype", NO_SUBTYPE)
        counts[subtype] = counts.get(subtype, 0) + 1
        try:
            geom = shapely_wkt.loads(feat["wkt"])
        except (GEOSException, KeyError, TypeError):
            n_invalid += 1
            continue
        if geom.is_empty:
            n_invalid += 1
            continue

        area_px = geom.area
        total_area_px += area_px
        minx, miny, maxx, maxy = geom.bounds
        cx, cy = geom.centroid.x, geom.centroid.y
        buildings.append(
            {
                "split": split,
                "disaster": rec["disaster"],
                "disaster_type": rec["disaster_type"],
                "scene_id": scene_id,
                "phase": phase,
                "uid": props.get("uid"),
                "subtype": subtype,
                "area_px": area_px,
                "area_m2": area_px * px_area_m2,
                "perimeter_px": geom.length,
                "bbox_w": maxx - minx,
                "bbox_h": maxy - miny,
                "cx": cx,
                "cy": cy,
                "is_valid": geom.is_valid,
            }
        )

    n_px = (rec.get("width") or 0) * (rec.get("height") or 0)
    rec.update(
        {
            "n_buildings": len(feats),
            "n_invalid_geom": n_invalid,
            "building_area_px": total_area_px,
            "building_cover_frac": (total_area_px / n_px) if n_px else np.nan,
            **{f"n_{c}": counts.get(c, 0) for c in (*DAMAGE_CLASSES, NO_SUBTYPE)},
        }
    )
    return rec, buildings


def _band_stats_one(path: str) -> dict | None:
    """Le os pixels de UMA imagem e devolve estatisticas por banda."""
    try:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # (bands, h, w)
    except (OSError, RasterioIOError):
        return None
    flat = arr.reshape(arr.shape[0], -1)
    out = {
        "stem": Path(path).stem,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "n_bands": arr.shape[0],
        "frac_zero": float((flat == 0).mean()),
        "fits_uint8": bool(flat.min() >= 0 and flat.max() <= 255),
    }
    for i in range(arr.shape[0]):
        out[f"mean_b{i}"] = float(flat[i].mean())
        out[f"std_b{i}"] = float(flat[i].std())
    return out


# ----------------------------------------------------------------------------- etapas


def scan_labels(root: Path, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks: list[tuple[str, str]] = []
    for split in SPLITS:
        lab_dir = root / split / "labels"
        if not lab_dir.is_dir():
            print(f"  ! split ausente: {lab_dir}", file=sys.stderr)
            continue
        tasks += [(split, str(p)) for p in sorted(lab_dir.glob("*.json"))]

    print(f"  {len(tasks):,} pares label/imagem a processar com {workers} workers")

    img_rows: list[dict] = []
    bld_rows: list[dict] = []
    errors: list[str] = []
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rec, blds in pool.map(_scan_one, tasks, chunksize=64):
            if "__error__" in rec:
                errors.append(rec["__error__"])
            else:
                img_rows.append(rec)
                bld_rows.extend(blds)
            done += 1
            if done % 2000 == 0:
                print(f"    {done:,}/{len(tasks):,}  ({len(bld_rows):,} poligonos)", flush=True)

    if errors:
        print(f"  ! {len(errors)} arquivos com erro; primeiros: {errors[:3]}", file=sys.stderr)

    images = pd.DataFrame(img_rows)
    buildings = pd.DataFrame(bld_rows)

    for col in ("split", "disaster", "disaster_type", "phase", "sensor", "dtype", "crs"):
        if col in images:
            images[col] = images[col].astype("category")
    for col in ("split", "disaster", "disaster_type", "phase", "subtype"):
        if col in buildings:
            buildings[col] = buildings[col].astype("category")

    images["capture_date"] = pd.to_datetime(images["capture_date"], format="ISO8601", utc=True)
    return images, buildings


def sample_band_stats(images: pd.DataFrame, root: Path, n: int, workers: int, seed: int = 42):
    """Amostra estratificada por desastre para estatisticas de pixel."""
    ok = images[images["tif_exists"]]
    per = max(1, n // max(1, ok["disaster"].nunique()))
    sample = (
        ok.groupby("disaster", observed=True, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per), random_state=seed), include_groups=False)
        .reset_index(drop=True)
    )
    # `apply` com include_groups=False descarta a coluna do groupby; recupera pelo stem
    stems = sample["stem"].tolist()
    lookup = images.set_index("stem")["split"].to_dict()
    paths = [str(root / lookup[s] / "images" / f"{s}.tif") for s in stems]

    print(f"  lendo pixels de {len(paths)} imagens amostradas")
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, 8)) as pool:
        futs = {pool.submit(_band_stats_one, p): p for p in paths}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 100 == 0:
                print(f"    {i}/{len(paths)}", flush=True)
    return pd.DataFrame(rows)


def registration_offsets(buildings: pd.DataFrame) -> pd.DataFrame:
    """Casa predios pre<->pos pelo uid e mede o deslocamento do centroide.

    No xBD o mesmo predio guarda o MESMO uid nas duas datas, entao qualquer
    deslocamento do centroide em pixels e erro de co-registro (ou reanotacao).
    """
    cols = ["scene_id", "split", "uid", "cx", "cy", "area_px"]
    pre = buildings.loc[buildings["phase"] == "pre", cols]
    post = buildings.loc[buildings["phase"] == "post", cols]
    m = pre.merge(post, on=["scene_id", "uid"], suffixes=("_pre", "_post"))
    m["dx"] = m["cx_post"] - m["cx_pre"]
    m["dy"] = m["cy_post"] - m["cy_pre"]
    m["shift_px"] = np.hypot(m["dx"], m["dy"])
    m["area_ratio"] = m["area_px_post"] / m["area_px_pre"].replace(0, np.nan)
    return m[["scene_id", "split_pre", "uid", "dx", "dy", "shift_px", "area_ratio"]].rename(
        columns={"split_pre": "split"}
    )


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--root", type=Path, default=here / "geotiffs")
    ap.add_argument("--out", type=Path, default=here / "cache")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--sample-n", type=int, default=400, help="imagens lidas para estatisticas de pixel")
    ap.add_argument("--skip-band-stats", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("[1/4] varrendo labels + headers dos GeoTIFF...")
    images, buildings = scan_labels(args.root, args.workers)
    images.to_parquet(args.out / "images.parquet", index=False)
    buildings.to_parquet(args.out / "buildings.parquet", index=False)
    print(f"  -> {len(images):,} imagens, {len(buildings):,} poligonos")

    print("[2/4] casando pre<->pos para checar co-registro...")
    reg = registration_offsets(buildings)
    reg.to_parquet(args.out / "registration.parquet", index=False)
    print(f"  -> {len(reg):,} predios pareados")

    if args.skip_band_stats:
        print("[3/4] estatisticas de banda: puladas (--skip-band-stats)")
    else:
        print("[3/4] estatisticas de pixel em amostra...")
        bs = sample_band_stats(images, args.root, args.sample_n, args.workers)
        bs.to_parquet(args.out / "band_stats.parquet", index=False)
        print(f"  -> {len(bs)} imagens amostradas")

    print("[4/4] pronto. Arquivos em", args.out)
    for p in sorted(args.out.glob("*.parquet")):
        print(f"  {p.name:24s} {p.stat().st_size / 1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
