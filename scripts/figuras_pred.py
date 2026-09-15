#!/usr/bin/env python3
"""
Figura qualitativa: pre, pos, verdade e predicao, lado a lado.

    .venv/bin/python scripts/figuras_pred.py

Roda o melhor modelo coletado sobre algumas cenas de `hold` escolhidas por
composicao de dano, e salva `paper/figuras/exemplos.png`.

Usa recorte de 512 e um lote por vez de proposito: a maquina que gera as figuras
tem pouca RAM, e uma cena 1024x1024 em resnet34 nao cabe com folga. Para a
figura isso e irrelevante — o que se ilustra e o padrao de erro, nao a metrica.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

from xbd.config import Config
from xbd.data import XBDDataset
from xbd.models import build_model

SAIDA = RAIZ / "paper" / "figuras"
SAIDA.mkdir(parents=True, exist_ok=True)

# fundo transparente, depois verde -> amarelo -> laranja -> vermelho
CORES = ["#00000000", "#2e8b57", "#e6c229", "#e07b39", "#c0392b"]
NOMES = ["fundo", "intacta", "dano leve", "dano severo", "destruída"]
CMAP = ListedColormap(CORES)

LADO = 512          # recorte central usado na figura

# Perfis desejados, um por linha da figura. A escolha e feita sobre o RECORTE
# CENTRAL, e nao sobre a cena inteira: varias cenas do xBD tem as edificacoes
# nas bordas, e selecionar pela estatistica global produz paineis vazios.
PERFIS = [
    ("dano severo predominante", lambda f: f[3] > .35),
    ("majoritariamente intacta", lambda f: f[1] > .80),
    ("com dano leve presente",   lambda f: f[2] > .12),
    ("com destruicao",           lambda f: f[4] > .20),
]


def _composicao_central(mask_path: Path) -> np.ndarray | None:
    """Fracao de cada severidade ENTRE OS PIXELS DE EDIFICACAO, no recorte central.

    Normalizar pelo total de pixels nao serve: 97% da cena e fundo, entao
    nenhuma classe passaria de ~0,1 e todo criterio de proporcao falharia.
    Devolve None se o recorte nao tem edificacao suficiente para ilustrar nada.
    """
    from PIL import Image
    m = np.array(Image.open(mask_path))
    h, w = m.shape
    i, j = (h - LADO) // 2, (w - LADO) // 2
    cont = np.bincount(m[i:i + LADO, j:j + LADO].ravel(), minlength=6)[:5].astype(float)
    edif = cont[1:].sum()
    if edif < 8000:                    # < ~3% do recorte: painel ficaria vazio
        return None
    return np.concatenate([[0.0], cont[1:] / edif])


def escolher_cenas(index: pd.DataFrame, dados: Path) -> pd.DataFrame:
    cand = index[index.split == "hold"].copy()
    cand = cand[cand.px_edificacao > 60_000]
    escolhidas, usadas = [], set()
    for rotulo, criterio in PERFIS:
        for _, r in cand.iterrows():
            if r.scene_id in usadas:
                continue
            f = _composicao_central(dados / "masks" / r.split / f"{r.scene_id}.png")
            if f is None or not criterio(f):
                continue
            usadas.add(r.scene_id)
            escolhidas.append({**r.to_dict(), "_perfil": rotulo})
            break
    return pd.DataFrame(escolhidas)


def desnormalizar(t: torch.Tensor, cfg) -> np.ndarray:
    from xbd.data import IMAGENET_MEAN, IMAGENET_STD, XBD_MEAN, XBD_STD
    x = t.numpy().transpose(1, 2, 0)
    if cfg.norm == "imagenet":
        m, s = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    elif cfg.norm == "per_image":
        return np.clip((x - x.min()) / (x.ptp() + 1e-6), 0, 1)
    else:
        m, s = np.array(XBD_MEAN), np.array(XBD_STD)
    return np.clip(x * s + m, 0, 1)


def main() -> int:
    cks = sorted((RAIZ / "colheita").rglob("*abl-lr-encoder-igual*/melhor.pt"))
    if not cks:
        cks = sorted((RAIZ / "colheita").rglob("*ref-damage*/melhor.pt"))
    if not cks:
        print("!! nenhum checkpoint encontrado em colheita/")
        return 1
    ck = cks[0]
    print(f"modelo: {ck.parent.name}")

    bruto = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(bruto["config"])
    cfg.data_dir = str(RAIZ / "data")
    cfg_vis = Config.from_dict({**cfg.to_dict(), "crop": 512})

    modelo = build_model(cfg)
    estado = bruto["modelo"]
    if any(k.startswith("_orig_mod.") for k in estado):
        estado = {k.replace("_orig_mod.", "", 1): v for k, v in estado.items()}
    modelo.load_state_dict(estado)
    modelo.eval()

    index = pd.read_parquet(RAIZ / "data" / "index.parquet")
    sel = escolher_cenas(index, RAIZ / "data").reset_index(drop=True)
    if sel.empty:
        print("!! nenhuma cena atendeu aos perfis")
        return 1
    for _, r in sel.iterrows():
        print(f"  {r.scene_id:<34} {r._perfil}")

    ds = XBDDataset(sel, RAIZ / "data", cfg_vis, treino=False, norm=cfg.norm)

    n = len(ds)
    fig, axs = plt.subplots(n, 4, figsize=(11, 2.85 * n))
    axs = np.atleast_2d(axs)

    with torch.no_grad():
        for i in range(n):
            am = ds[i]
            pre, post, alvo = am["pre"], am["post"], am["mask"].numpy()
            logits = modelo(pre.unsqueeze(0), post.unsqueeze(0))
            pred = logits.argmax(1)[0].numpy()

            alvo_vis = np.where(alvo == 255, 0, alvo)
            acerto = (pred == alvo_vis)[alvo_vis > 0].mean() if (alvo_vis > 0).any() else float("nan")

            for j, (img, tit) in enumerate([
                    (desnormalizar(pre, cfg), "pré-desastre"),
                    (desnormalizar(post, cfg), "pós-desastre"),
                    (desnormalizar(post, cfg), "verdade"),
                    (desnormalizar(post, cfg), "predição")]):
                ax = axs[i, j]
                ax.imshow(img)
                if j == 2:
                    ax.imshow(alvo_vis, cmap=CMAP, vmin=0, vmax=4, alpha=.62,
                              interpolation="nearest")
                if j == 3:
                    ax.imshow(pred, cmap=CMAP, vmin=0, vmax=4, alpha=.62,
                              interpolation="nearest")
                if i == 0:
                    ax.set_title(tit if j < 3 else "predição", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            axs[i, 0].set_ylabel(f"{sel.scene_id[i].replace('_', chr(10))}\n"
                                 f"({sel._perfil[i]})", fontsize=7)
            axs[i, 3].set_xlabel(
                f"{100*acerto:.0f}% dos pixels de edificação corretos"
                if acerto == acerto else "sem edificação no recorte", fontsize=7.5)

    handles = [mpatches.Patch(color=CORES[k], label=NOMES[k]) for k in range(1, 5)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, -.005), fontsize=9)
    fig.tight_layout(rect=(0, .022, 1, 1))
    fig.savefig(SAIDA / "exemplos.png", dpi=155)
    print(f"  exemplos.png  ({n} cenas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
