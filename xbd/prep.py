"""
Converte os GeoTIFF em chips PNG uint8 + mascaras rasterizadas.

Motivacao (ver secao 2 da EDA): os GeoTIFF sao `int16` sem compressao com
conteudo 8-bit, entao ocupam 139 GB para carregar ~26 GB de informacao. Ler
`int16` cru a cada epoca tambem desperdicia banda de disco, que e o gargalo real
em maquinas sem SSD rapido.

A conversao e **shardada e idempotente**: cada maquina roda `--shard i --of n` e
processa apenas a sua fatia; rodar de novo pula o que ja existe. Nao ha estado
compartilhado, entao as maquinas nao precisam se coordenar nem rodar ao mesmo
tempo.

Uma mascara por cena basta: a EDA mostrou que os poligonos pre e pos sao
geometricamente identicos (mesmos uids, deslocamento mediano de 1e-7 px), entao
a mascara de localizacao e simplesmente `mascara_dano > 0`.

Saida:
    data/images/{split}/{cena}_pre.png    RGB uint8
    data/images/{split}/{cena}_post.png   RGB uint8
    data/masks/{split}/{cena}.png         uint8, 0=fundo 1..4=dano 5=un-classified
    data/index.parquet                    catalogo para amostragem e splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.features import rasterize
from shapely import wkt as shapely_wkt

from .config import DESTROYED, MAJOR, MINOR, NO_DAMAGE, UNCLASSIFIED

SUBTYPE_PARA_ID = {
    "no-damage": NO_DAMAGE,
    "minor-damage": MINOR,
    "major-damage": MAJOR,
    "destroyed": DESTROYED,
    "un-classified": UNCLASSIFIED,
    "no-subtype": NO_DAMAGE,  # so aparece em labels pre, onde a mascara vira localizacao
}


def shard_de(cena: str, n: int) -> int:
    """Atribuicao estavel de cena -> shard. Nao depende de ordem nem de listagem."""
    return int(hashlib.md5(cena.encode()).hexdigest(), 16) % n


def _ler_uint8(caminho: Path) -> np.ndarray:
    with rasterio.open(caminho) as src:
        a = src.read()  # (3, H, W) int16 com conteudo 0-255
    return np.clip(np.transpose(a, (1, 2, 0)), 0, 255).astype(np.uint8)


def _mascara(label_json: Path, shape: tuple[int, int]) -> np.ndarray:
    doc = json.loads(label_json.read_text())
    formas = []
    for feat in doc.get("features", {}).get("xy", []):
        sub = feat.get("properties", {}).get("subtype", "no-subtype")
        try:
            geom = shapely_wkt.loads(feat["wkt"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)  # conserta os 80 poligonos auto-intersectantes
            if geom.is_empty:
                continue
        formas.append((geom, SUBTYPE_PARA_ID.get(sub, NO_DAMAGE)))
    if not formas:
        return np.zeros(shape, dtype=np.uint8)
    # ordena por area decrescente: predios pequenos sobrepostos nao somem sob os grandes
    formas.sort(key=lambda t: t[0].area, reverse=True)
    return rasterize(formas, out_shape=shape, fill=0, dtype=np.uint8)


def _processar(tarefa: tuple[str, str, str, str, int, bool]) -> dict | None:
    split, cena, raiz_s, saida_s, png_level, force = tarefa
    raiz, saida = Path(raiz_s), Path(saida_s)

    dir_img = saida / "images" / split
    dir_msk = saida / "masks" / split
    dir_img.mkdir(parents=True, exist_ok=True)
    dir_msk.mkdir(parents=True, exist_ok=True)

    p_pre = dir_img / f"{cena}_pre.png"
    p_post = dir_img / f"{cena}_post.png"
    p_msk = dir_msk / f"{cena}.png"

    pronto = all(p.exists() and p.stat().st_size > 0 for p in (p_pre, p_post, p_msk))
    if pronto and not force:
        try:
            msk = np.array(Image.open(p_msk))
            return _stats(split, cena, msk, p_pre, p_post, p_msk, pulado=True)
        except Exception:
            pass  # arquivo corrompido: refaz

    try:
        pre = _ler_uint8(raiz / split / "images" / f"{cena}_pre_disaster.tif")
        post = _ler_uint8(raiz / split / "images" / f"{cena}_post_disaster.tif")
        msk = _mascara(raiz / split / "labels" / f"{cena}_post_disaster.json", post.shape[:2])
    except Exception as exc:
        return {"split": split, "scene_id": cena, "erro": str(exc)[:200]}

    # escreve em tmp e renomeia: um kill no meio nao deixa PNG truncado para tras
    for arr, destino, modo in ((pre, p_pre, "RGB"), (post, p_post, "RGB"), (msk, p_msk, "L")):
        tmp = destino.with_suffix(".tmp.png")
        Image.fromarray(arr, mode=modo).save(tmp, format="PNG", compress_level=png_level)
        os.replace(tmp, destino)

    return _stats(split, cena, msk, p_pre, p_post, p_msk, pulado=False)


def _stats(split, cena, msk, p_pre, p_post, p_msk, pulado: bool) -> dict:
    cont = np.bincount(msk.ravel(), minlength=6)
    n_dano = int(cont[MINOR] + cont[MAJOR] + cont[DESTROYED])
    return {
        "split": split,
        "scene_id": cena,
        "disaster": cena.rsplit("_", 1)[0],
        "px_fundo": int(cont[0]),
        "px_no_damage": int(cont[NO_DAMAGE]),
        "px_minor": int(cont[MINOR]),
        "px_major": int(cont[MAJOR]),
        "px_destroyed": int(cont[DESTROYED]),
        "px_unclassified": int(cont[UNCLASSIFIED]),
        "px_edificacao": int(cont[1:].sum()),
        "tem_edificacao": bool(cont[1:].sum() > 0),
        "tem_dano": bool(n_dano > 0),
        "bytes": int(p_pre.stat().st_size + p_post.stat().st_size + p_msk.stat().st_size),
        "pulado": pulado,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    aqui = Path(__file__).resolve().parent.parent
    ap.add_argument("--root", type=Path, default=aqui / "geotiffs")
    ap.add_argument("--out", type=Path, default=aqui / "data")
    ap.add_argument("--cache", type=Path, default=aqui / "cache")
    ap.add_argument("--shard", type=int, default=0, help="indice desta maquina (0-based)")
    ap.add_argument("--of", type=int, default=1, help="numero total de maquinas")
    ap.add_argument("--splits", nargs="+", default=["tier1", "tier3", "hold", "test"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--png-level", type=int, default=6, help="0-9; 1 e ~3x mais rapido e ~8%% maior")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not 0 <= args.shard < args.of:
        raise SystemExit(f"--shard deve estar em [0, {args.of})")

    imagens = pd.read_parquet(args.cache / "images.parquet")
    cenas = (imagens[imagens.split.astype(str).isin(args.splits)]
             [["split", "scene_id"]].astype(str).drop_duplicates())
    minhas = [(s, c) for s, c in cenas.itertuples(index=False)
              if shard_de(c, args.of) == args.shard]

    print(f"shard {args.shard}/{args.of}: {len(minhas):,} cenas de {len(cenas):,}")
    tarefas = [(s, c, str(args.root), str(args.out), args.png_level, args.force) for s, c in minhas]

    linhas, erros = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(_processar, tarefas, chunksize=8), 1):
            if rec is None:
                continue
            (erros if "erro" in rec else linhas).append(rec)
            if i % 500 == 0:
                print(f"  {i:,}/{len(tarefas):,}", flush=True)

    df = pd.DataFrame(linhas)
    args.out.mkdir(parents=True, exist_ok=True)
    parcial = args.out / f"index_shard{args.shard:03d}of{args.of:03d}.parquet"
    df.to_parquet(parcial, index=False)

    novos = int((~df.pulado).sum()) if len(df) else 0
    gb = df.bytes.sum() / 1e9 if len(df) else 0
    print(f"-> {len(df):,} cenas ({novos:,} novas, {len(df)-novos:,} ja existiam), {gb:.1f} GB")
    print(f"-> indice parcial: {parcial.name}")
    if erros:
        print(f"!! {len(erros)} erros; primeiro: {erros[0]}")
    print("\nQuando todos os shards terminarem, junte os indices com:")
    print("   python -m xbd.prep_merge --out data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
