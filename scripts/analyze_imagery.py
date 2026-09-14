#!/usr/bin/env python3
"""
Analise radiometrica e de qualidade de imagem do xBD, lendo os pixels.

Complementa `build_index.py` (que so le cabecalhos). Aqui cada imagem amostrada e
lida inteira e medida em quatro eixos:

  radiometria  media/desvio/percentis por banda, clipping em 0 e 255, faixa
               dinamica util, entropia, correlacao entre bandas
  ruido        sigma estimado pelo metodo de Immerkaer (convolucao laplaciana),
               que separa ruido de conteudo sem precisar de imagem de referencia
  nitidez      variancia do laplaciano e energia de gradiente (Tenengrad)
  compressao   blocagem em grade 8x8 (assinatura de JPEG)

AVISO SOBRE "REFLECTANCIA": os metadados do xBD nao trazem `absCalFactor`,
`effectiveBandwidth`, `earthSunDistance` nem `ESUN`, que sao os coeficientes
necessarios para converter DN em radiancia e depois em reflectancia TOA. O que
este script mede e brilho RELATIVO em DN 8-bit, opcionalmente normalizado pelo
cosseno do angulo zenital solar (= sin da elevacao solar) para remover o efeito
de iluminacao. Isso e uma correcao geometrica de primeira ordem, nao calibracao.

Amostragem: pares pre/pos completos, estratificados por desastre, para permitir
medir tambem o descasamento radiometrico ENTRE as duas datas da mesma cena.

Uso:
    python scripts/analyze_imagery.py --pairs-per-disaster 40 --workers 8
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

BANDS = ("R", "G", "B")

# Mascara laplaciana de Immerkaer (1996): responde a ruido e quase nao a
# estruturas suaves, o que permite estimar sigma sem imagem de referencia.
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def noise_sigma(gray: np.ndarray) -> float:
    """Desvio-padrao do ruido estimado sem referencia (Immerkaer)."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return float("nan")
    conv = cv2.filter2D(gray, cv2.CV_32F, IMMERKAER, borderType=cv2.BORDER_REPLICATE)
    # recorta a borda, onde a replicacao inventa valores
    conv = conv[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * conv.size))


def sharpness(gray: np.ndarray) -> tuple[float, float]:
    """(variancia do laplaciano, energia de gradiente de Tenengrad)."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(lap.var()), float((gx * gx + gy * gy).mean())


def blockiness(gray: np.ndarray, block: int = 8) -> float:
    """Razao entre a descontinuidade nas bordas de bloco 8x8 e dentro do bloco.

    ~1.0 = sem grade perceptivel. Acima de ~1.15 sugere compressao em blocos
    (JPEG) na cadeia de producao da imagem.
    """
    d = np.abs(np.diff(gray.astype(np.float32), axis=1))
    if d.shape[1] < block * 2:
        return float("nan")
    cols = np.arange(d.shape[1])
    borda = d[:, (cols + 1) % block == 0].mean()
    dentro = d[:, (cols + 1) % block != 0].mean()
    return float(borda / dentro) if dentro > 0 else float("nan")


def entropy(gray_u8: np.ndarray) -> float:
    h = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def _metrics(arr: np.ndarray, prefix: str) -> dict:
    """arr: HxWx3 float32 em DN 0-255."""
    out: dict[str, float] = {}
    valid = arr.max(axis=2) > 0  # pixels fora da borda sem dado
    frac_valid = float(valid.mean())
    out[f"{prefix}_frac_valida"] = frac_valid

    for i, b in enumerate(BANDS):
        ch = arr[:, :, i]
        ch_v = ch[valid] if frac_valid > 0.01 else ch.ravel()
        out[f"{prefix}_media_{b}"] = float(ch_v.mean())
        out[f"{prefix}_dp_{b}"] = float(ch_v.std())
        p1, p50, p99 = np.percentile(ch_v, [1, 50, 99])
        out[f"{prefix}_p01_{b}"] = float(p1)
        out[f"{prefix}_p50_{b}"] = float(p50)
        out[f"{prefix}_p99_{b}"] = float(p99)
        out[f"{prefix}_faixa_util_{b}"] = float(p99 - p1)
        out[f"{prefix}_clip_alto_{b}"] = float((ch_v >= 255).mean())
        out[f"{prefix}_clip_baixo_{b}"] = float((ch_v <= 0).mean())

    # correlacao entre bandas: alta = pouca informacao cromatica independente
    flat = arr.reshape(-1, 3)[valid.ravel()] if frac_valid > 0.01 else arr.reshape(-1, 3)
    if len(flat) > 10:
        c = np.corrcoef(flat.T)
        out[f"{prefix}_corr_RG"] = float(c[0, 1])
        out[f"{prefix}_corr_RB"] = float(c[0, 2])
        out[f"{prefix}_corr_GB"] = float(c[1, 2])

    gray = arr.mean(axis=2)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    out[f"{prefix}_entropia"] = entropy(gray_u8)
    out[f"{prefix}_ruido_sigma"] = noise_sigma(gray)
    lapvar, tenen = sharpness(gray)
    out[f"{prefix}_lap_var"] = lapvar
    out[f"{prefix}_tenengrad"] = tenen
    out[f"{prefix}_blocagem"] = blockiness(gray_u8)
    # razao sinal/ruido aproximada, em dB
    sig = out[f"{prefix}_dp_R"] + out[f"{prefix}_dp_G"] + out[f"{prefix}_dp_B"]
    nz = out[f"{prefix}_ruido_sigma"]
    out[f"{prefix}_snr_db"] = float(20 * np.log10((sig / 3) / nz)) if nz and nz > 1e-6 else float("nan")
    return out


def _read(path: Path) -> np.ndarray | None:
    try:
        with rasterio.open(path) as src:
            a = src.read().astype(np.float32)
    except Exception:
        return None
    return np.transpose(a, (1, 2, 0))


def _process_pair(task: tuple[str, str, str, str, float, float, float, float]) -> dict | None:
    """Mede a imagem pre, a pos, e o descasamento entre elas."""
    split, scene, disaster, dtype_, sun_pre, sun_post, nadir_pre, nadir_post = task
    root = Path(os.environ["XBD_ROOT"])
    pre = _read(root / split / "images" / f"{scene}_pre_disaster.tif")
    post = _read(root / split / "images" / f"{scene}_post_disaster.tif")
    if pre is None or post is None:
        return None

    rec: dict = {"split": split, "scene_id": scene, "disaster": disaster, "disaster_type": dtype_,
                 "sun_elev_pre": sun_pre, "sun_elev_post": sun_post,
                 "nadir_pre": nadir_pre, "nadir_post": nadir_post}
    rec.update(_metrics(pre, "pre"))
    rec.update(_metrics(post, "post"))

    # --- descasamento radiometrico entre as duas datas ---
    for b in BANDS:
        rec[f"d_media_{b}"] = rec[f"post_media_{b}"] - rec[f"pre_media_{b}"]
        rec[f"d_ganho_{b}"] = (rec[f"post_dp_{b}"] / rec[f"pre_dp_{b}"]
                               if rec[f"pre_dp_{b}"] > 1e-6 else np.nan)
    rec["d_media_L"] = np.mean([rec[f"d_media_{b}"] for b in BANDS])
    rec["d_ruido"] = rec["post_ruido_sigma"] - rec["pre_ruido_sigma"]
    rec["d_nitidez_razao"] = (rec["post_lap_var"] / rec["pre_lap_var"]
                              if rec["pre_lap_var"] > 1e-6 else np.nan)

    # --- brilho normalizado pelo angulo solar ---
    # cos(zenite solar) = sin(elevacao solar). Corrige iluminacao, nao calibra.
    for fase, sun in (("pre", sun_pre), ("post", sun_post)):
        cosz = np.sin(np.radians(sun)) if sun and sun > 0 else np.nan
        rec[f"{fase}_cos_zenite"] = float(cosz)
        L = np.mean([rec[f"{fase}_media_{b}"] for b in BANDS])
        rec[f"{fase}_brilho"] = float(L)
        rec[f"{fase}_brilho_norm"] = float(L / cosz) if cosz and cosz > 0 else np.nan

    # distancia entre histogramas das duas datas (Bhattacharyya, 0=identico)
    g_pre = np.clip(pre.mean(2), 0, 255).astype(np.uint8)
    g_post = np.clip(post.mean(2), 0, 255).astype(np.uint8)
    h1 = cv2.calcHist([g_pre], [0], None, [64], [0, 256]).ravel()
    h2 = cv2.calcHist([g_post], [0], None, [64], [0, 256]).ravel()
    h1 /= h1.sum() + 1e-9
    h2 /= h2.sum() + 1e-9
    rec["d_hist_bhatt"] = float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32),
                                                cv2.HISTCMP_BHATTACHARYYA))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--root", type=Path, default=here / "geotiffs")
    ap.add_argument("--cache", type=Path, default=here / "cache")
    ap.add_argument("--pairs-per-disaster", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = pd.read_parquet(args.cache / "images.parquet")
    for c in images.select_dtypes("category").columns:
        images[c] = images[c].astype(str)

    piv = images.pivot_table(index=["split", "scene_id", "disaster", "disaster_type"],
                             columns="phase", values=["sun_elevation", "off_nadir_angle"],
                             aggfunc="first").dropna()
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    rng = np.random.default_rng(args.seed)
    idx = np.concatenate([
        rng.choice(g.index.to_numpy(), size=min(len(g), args.pairs_per_disaster), replace=False)
        for _, g in piv.groupby("disaster")
    ])
    sample = piv.loc[np.sort(idx)]

    tasks = [(r.split, r.scene_id, r.disaster, r.disaster_type,
              r.sun_elevation_pre, r.sun_elevation_post,
              r.off_nadir_angle_pre, r.off_nadir_angle_post)
             for r in sample.itertuples(index=False)]

    os.environ["XBD_ROOT"] = str(args.root)
    print(f"medindo {len(tasks)} pares pre/pos ({args.workers} workers)")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(_process_pair, tasks, chunksize=4), 1):
            if rec:
                rows.append(rec)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)

    df = pd.DataFrame(rows)
    out = args.cache / "image_quality.parquet"
    df.to_parquet(out, index=False)
    print(f"-> {len(df)} pares em {out}  ({out.stat().st_size/1e6:.1f} MB, {df.shape[1]} colunas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
