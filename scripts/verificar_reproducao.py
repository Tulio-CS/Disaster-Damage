#!/usr/bin/env python3
"""
Confere que a preparacao dos dados e reproduzivel a partir dos arquivos brutos.

Para cada cena do manifesto, refaz a partir dos GeoTIFF e do label JSON — com as
MESMAS funcoes de `xbd/prep.py` — os pixels da imagem pre, da pos e da mascara, e
compara o SHA-256 com o registrado em `manifestos/manifesto.csv`. Nao grava PNG
nenhum, entao nao precisa dos 33 GB de `data/`: so dos GeoTIFF extraidos.

Quem quiser conferir o dataset preparado do zero:

    sha256sum xview2_geotiff.tgz        # 5643e8c7b3ca7908...  (ver README)
    tar -xzf xview2_geotiff.tgz         # -> geotiffs/
    python scripts/verificar_reproducao.py --geotiffs geotiffs --workers 4

Saida: um resumo no terminal e em `manifestos/reproducao.json`. Codigo de saida 0
so se todas as cenas conferidas baterem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from xbd.prep import _ler_uint8, _mascara  # noqa: E402


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _conferir(tarefa: tuple[str, str, str, str, str, str]) -> dict:
    raiz, tier, cena, h_pre, h_post, h_msk = tarefa
    base = Path(raiz) / tier
    try:
        pre = _ler_uint8(base / "images" / f"{cena}_pre_disaster.tif")
        post = _ler_uint8(base / "images" / f"{cena}_post_disaster.tif")
        msk = _mascara(base / "labels" / f"{cena}_post_disaster.json", post.shape[:2])
    except Exception as exc:
        return {"cena": cena, "erro": f"{type(exc).__name__}: {exc}"[:200]}
    return {"cena": cena, "pre": _sha(pre) == h_pre, "post": _sha(post) == h_post, "mask": _sha(msk) == h_msk}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geotiffs", type=Path, default=RAIZ / "geotiffs")
    ap.add_argument("--manifesto", type=Path, default=RAIZ / "manifestos" / "manifesto.csv")
    ap.add_argument("--saida", type=Path, default=RAIZ / "manifestos" / "reproducao.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amostra", type=int, default=0, help="conferir so N cenas (sorteio fixo, seed 42)")
    args = ap.parse_args()

    m = pd.read_csv(args.manifesto, keep_default_na=False)
    if args.amostra:
        m = m.sample(min(args.amostra, len(m)), random_state=42)
    tarefas = [(str(args.geotiffs), r.tier, r.sample_id, r.sha256_pre, r.sha256_post, r.sha256_mask)
               for r in m.itertuples(index=False)]

    print(f"conferindo {len(tarefas):,} cenas de {args.geotiffs} contra {args.manifesto.name} ...")
    t0, res = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(_conferir, tarefas, chunksize=8), 1):
            res.append(r)
            if i % 1000 == 0:
                print(f"  {i:,}/{len(tarefas):,}", flush=True)

    erros = [r for r in res if "erro" in r]
    ok = [r for r in res if "erro" not in r]
    divergentes = [r["cena"] for r in ok if not (r["pre"] and r["post"] and r["mask"])]
    resumo = {
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "cenas_conferidas": len(res),
        "imagens_e_mascaras_iguais": int(sum(r["pre"] + r["post"] + r["mask"] for r in ok)),
        "imagens_e_mascaras_conferidas": 3 * len(ok),
        "cenas_divergentes": divergentes[:50],
        "n_divergentes": len(divergentes),
        "falhas_de_leitura": erros[:20],
        "n_falhas": len(erros),
        "segundos": round(time.time() - t0),
        "ambiente": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "rasterio": __import__("rasterio").__version__, "shapely": __import__("shapely").__version__,
                     "sistema": platform.platform()},
    }
    args.saida.write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in resumo.items() if k not in ("cenas_divergentes", "falhas_de_leitura")},
                     indent=2, ensure_ascii=False))
    print("OK: dataset preparado reproduzido bit a bit" if not divergentes and not erros
          else "!! divergencias: veja " + str(args.saida))
    return 0 if not divergentes and not erros else 1


if __name__ == "__main__":
    raise SystemExit(main())
