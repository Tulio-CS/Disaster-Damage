#!/usr/bin/env python3
"""
Demonstracao de ponta a ponta: imagem -> pre-processamento -> modelo -> resultado.

    # as 4 cenas de amostras/ (split `test` do xView2, nunca vistas no treino)
    python scripts/demo.py

    # qualquer par pre/pos, PNG/JPG ou o GeoTIFF original do xBD
    python scripts/demo.py --pre a_pre.png --post a_post.png [--mascara a_mask.png]

Sem `modelos/modelo_final.pt` no disco, o peso e baixado do release v1.0 do
GitHub. Roda em CPU: ~5 s e ~1,3 GB de RAM por cena de 1024x1024.

Saida em `saidas/demo/`: a figura (pre | pos | [verdade] | predicao), a mascara
predita em PNG (0 fundo, 1 intacta, 2 dano leve, 3 dano severo, 4 destruida) e
um resumo no terminal com a contagem de edificacoes por severidade.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from PIL import Image                                        # noqa: E402

from xbd.inferencia import (MODELO_PADRAO, NOMES, carregar_modelo, figura,  # noqa: E402
                            ler_imagem, ler_mascara, metricas, prever)

AMOSTRAS = RAIZ / "amostras"


def pares_de_amostras() -> list[tuple[str, Path, Path, Path | None]]:
    out = []
    for pre in sorted(AMOSTRAS.glob("*_pre.png")):
        cena = pre.name.removesuffix("_pre.png")
        mask = AMOSTRAS / f"{cena}_mask.png"
        out.append((cena, pre, AMOSTRAS / f"{cena}_post.png", mask if mask.exists() else None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", type=Path, help="imagem pre-desastre")
    ap.add_argument("--post", type=Path, help="imagem pos-desastre")
    ap.add_argument("--mascara", type=Path, default=None,
                    help="mascara verdadeira (opcional): calcula as metricas da cena")
    ap.add_argument("--modelo", type=Path, default=MODELO_PADRAO)
    ap.add_argument("--saida", type=Path, default=RAIZ / "saidas" / "demo")
    args = ap.parse_args()

    if bool(args.pre) != bool(args.post):
        ap.error("--pre e --post vao juntos")
    if args.pre:
        pares = [(args.post.stem.removesuffix("_post").removesuffix("_post_disaster"),
                  args.pre, args.post, args.mascara)]
    else:
        pares = pares_de_amostras()
        if not pares:
            ap.error(f"nenhum *_pre.png em {AMOSTRAS}; passe --pre e --post")

    modelo, cfg, meta = carregar_modelo(args.modelo)
    print(f"modelo   {meta.get('run_id', args.modelo.name)}  "
          f"({cfg.encoder}, fusao {cfg.fuse_mode}, norm {cfg.norm})")
    args.saida.mkdir(parents=True, exist_ok=True)

    for cena, f_pre, f_post, f_mask in pares:
        print(f"\n== {cena}")
        pre, post = ler_imagem(f_pre), ler_imagem(f_post)
        print(f"  1. imagem          pre {pre.shape} pos {post.shape} {pre.dtype}")
        print(f"  2. pre-processam.  /255, normalizacao '{cfg.norm}', padding ate multiplo de 32")
        t0 = time.time()
        res = prever(modelo, pre, post, cfg.norm)
        print(f"  3. modelo          logits (5, {pre.shape[0]}, {pre.shape[1]}) -> argmax "
              f"em {time.time() - t0:.1f}s")
        total = sum(res.edificacoes.values())
        print(f"  4. resultado       {total} edificacoes: "
              + ", ".join(f"{n} {k}" for k, n in res.edificacoes.items()))

        alvo = ler_mascara(f_mask) if f_mask else None
        if alvo is not None:
            m = metricas(res.classe, alvo)
            print(f"     vs. anotacao    F1 localizacao {m['f1_loc']:.3f}  "
                  f"severidade certa em {100 * m['acerto_severidade']:.0f}% dos pixels de "
                  f"edificacao encontrados")

        Image.fromarray(res.classe).save(args.saida / f"{cena}_predicao.png")
        fig = figura(pre, post, res, alvo, titulo=cena)
        fig.savefig(args.saida / f"{cena}.png", dpi=110)
        plt.close(fig)
        print(f"     -> {args.saida / (cena + '.png')}")

    print(f"\nlegenda da mascara: " + ", ".join(f"{i} {n}" for i, n in enumerate(NOMES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
