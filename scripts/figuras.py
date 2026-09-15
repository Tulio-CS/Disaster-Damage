#!/usr/bin/env python3
"""
Gera as figuras do relatorio a partir dos dados colhidos.

    .venv/bin/python scripts/figuras.py

Saida em paper/figuras/*.pdf (vetorial, para o LaTeX). Le apenas
`colheita/avaliacao.json` e `colheita/*/results/history/*.jsonl` — nao executa
modelo. As figuras qualitativas (predicoes) vem de `scripts/figuras_pred.py`,
que precisa dos pesos e dos dados.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "paper" / "figuras"
SAIDA.mkdir(parents=True, exist_ok=True)

AZUL, VERMELHO, VERDE, CINZA = "#1f569e", "#b02a2a", "#22783c", "#6e6e6e"
LARANJA, ROXO = "#d2691e", "#7b3fa0"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

AVAL = json.loads((RAIZ / "colheita" / "avaliacao.json").read_text())
POR = {r["nome"]: r for r in AVAL}

# Ranking oficial do xView2 (top-9 da tabela publica do desafio).
LEADERBOARD = [
    (1, 0.81193, 0.86350, 0.92344, 0.64445, 0.78591, 0.86403),
    (2, 0.80761, 0.86318, 0.92556, 0.63891, 0.77302, 0.85903),
    (3, 0.80543, 0.86695, 0.92071, 0.63693, 0.76857, 0.84954),
    (4, 0.78709, 0.86285, 0.92642, 0.59691, 0.74422, 0.83173),
    (5, 0.77862, 0.86589, 0.89853, 0.59048, 0.73778, 0.80994),
    (6, 0.77525, 0.84614, 0.92152, 0.57997, 0.74545, 0.82018),
    (7, 0.75287, 0.85134, 0.88578, 0.55875, 0.70744, 0.77141),
    (8, 0.74146, 0.83646, 0.90645, 0.49347, 0.72245, 0.83723),
    (9, 0.73289, 0.85250, 0.88864, 0.47583, 0.71313, 0.80700),
]
METRICAS = ["F1 global", "Localização", "Intacta", "Dano leve", "Dano severo", "Destruída"]
CHAVES = ["score_xview2", "f1_loc", "f1_no-damage", "f1_minor-damage",
          "f1_major-damage", "f1_destroyed"]


def historico(nome: str) -> list[dict]:
    """Metricas por epoca do run cujo experimento se chama `nome`."""
    for f in (RAIZ / "colheita").rglob("results/history/*.jsonl"):
        if re.match(rf"^{re.escape(nome)}-(damage|loc)-f\d", f.name):
            return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return []


# ----------------------------------------------------------------- 1. curvas


def fig_curvas() -> None:
    alvos = [("ref", AZUL, "referência"),
             ("abl-lr-encoder-igual", VERDE, "lr do encoder $1{,}0\\times$"),
             ("abl-sem-pretrained", LARANJA, "sem pré-treino"),
             ("abl-sem-siamese", VERMELHO, "sem ramo siamês")]

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for nome, cor, rot in alvos:
        h = historico(nome)
        if not h:
            continue
        ep = [r["epoca"] for r in h]
        axs[0].plot(ep, [r.get("perda_treino") for r in h], color=cor, lw=1.4, label=rot)
        v = [(r["epoca"], r["perda_val"]) for r in h if r.get("perda_val") is not None]
        if v:
            axs[1].plot(*zip(*v), color=cor, lw=1.4, label=rot)
        s = [(r["epoca"], r["score_xview2"]) for r in h if r.get("score_xview2") is not None]
        if s:
            axs[2].plot(*zip(*s), color=cor, lw=1.4, label=rot)

    for ax, tit, yl in zip(axs, ["Perda de treino", "Perda de validação",
                                 "Escore xView2 (validação)"],
                           ["perda", "perda", "escore"]):
        ax.set_title(tit); ax.set_xlabel("época"); ax.set_ylabel(yl)
        ax.grid(alpha=.25, lw=.5)
    axs[2].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(SAIDA / "curvas.pdf")
    plt.close(fig)
    print("  curvas.pdf")


# --------------------------------------------------- 2. precisao vs recall


def fig_prec_recall() -> None:
    classes = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    rot = ["intacta", "dano leve", "dano severo", "destruída"]
    r = POR["ref"]
    p = [r[f"prec_{c}"] for c in classes]
    rc = [r[f"rec_{c}"] for c in classes]
    f1 = [r[f"f1_{c}"] for c in classes]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.2))

    x = np.arange(len(classes)); w = 0.26
    a1.bar(x - w, p, w, label="precisão", color=AZUL, alpha=.85)
    a1.bar(x, rc, w, label="recall", color=LARANJA, alpha=.85)
    a1.bar(x + w, f1, w, label="$F_1$", color=CINZA, alpha=.85)
    a1.set_xticks(x); a1.set_xticklabels(rot); a1.set_ylim(0, 1)
    a1.set_ylabel("valor"); a1.set_title("Referência, por classe de severidade")
    a1.legend(frameon=False, ncol=3, loc="upper center")
    a1.grid(axis="y", alpha=.25, lw=.5)
    a1.axhline(min(f1), color=VERMELHO, ls=":", lw=1)
    a1.annotate("média harmônica é\ndominada por este termo",
                xy=(1 + w, f1[1]), xytext=(1.62, 0.12), fontsize=7, color=VERMELHO,
                arrowprops=dict(arrowstyle="->", color=VERMELHO, lw=.8))

    # precisao x recall de minor-damage em todos os experimentos oficiais
    ofic = [z for z in AVAL if z["split_mode"] == "oficial" and z["nome"] != "base-loc-only"]
    px = [z["prec_minor-damage"] for z in ofic]
    py = [z["rec_minor-damage"] for z in ofic]
    a2.scatter(px, py, s=26, color=AZUL, alpha=.7, zorder=3)
    lim = [min(px + py) - .03, max(px + py) + .03]
    a2.plot(lim, lim, ls="--", color=CINZA, lw=1, zorder=1)
    a2.fill_between(lim, lim, lim[1], color=LARANJA, alpha=.07, zorder=0)
    a2.text(.97, .04, "recall > precisão\n(precisão é o gargalo)", transform=a2.transAxes,
            ha="right", va="bottom", fontsize=7, color=CINZA)
    ref_i = next(i for i, z in enumerate(ofic) if z["nome"] == "ref")
    a2.scatter([px[ref_i]], [py[ref_i]], s=70, facecolor="none",
               edgecolor=VERMELHO, lw=1.6, zorder=4)
    a2.annotate("referência", (px[ref_i], py[ref_i]), xytext=(6, -10),
                textcoords="offset points", fontsize=7, color=VERMELHO)
    a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("precisão"); a2.set_ylabel("recall")
    a2.set_title("\\texttt{minor-damage}: 18 experimentos".replace("\\texttt{", "").replace("}", ""))
    a2.grid(alpha=.25, lw=.5)

    fig.tight_layout()
    fig.savefig(SAIDA / "prec_recall.pdf")
    plt.close(fig)
    print("  prec_recall.pdf")


# ------------------------------------------------------- 3. matriz de confusao


def fig_confusao() -> None:
    cls = ["fundo", "intacta", "leve", "severo", "destruída"]
    m = np.array(POR["ref"]["matriz"], dtype=float).reshape(5, 5)
    pct = 100 * m / m.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{pct[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 55 else "#333333")
    ax.set_xticks(range(5)); ax.set_xticklabels(cls, rotation=30, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(cls)
    ax.set_xlabel("predito"); ax.set_ylabel("verdadeiro")
    ax.set_title("Matriz de confusão, % da classe verdadeira")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="%")
    fig.tight_layout()
    fig.savefig(SAIDA / "confusao.pdf")
    plt.close(fig)
    print("  confusao.pdf")


# ---------------------------------------------------------- 4. leaderboard


def fig_leaderboard() -> None:
    nosso = POR["abl-lr-encoder-igual"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    x = np.arange(len(METRICAS))
    topo = [max(l[i + 1] for l in LEADERBOARD) for i in range(6)]
    nono = [min(l[i + 1] for l in LEADERBOARD) for i in range(6)]
    meu = [nosso[k] for k in CHAVES]

    a1.bar(x - .27, topo, .27, label="1º do ranking", color=CINZA, alpha=.55)
    a1.bar(x, nono, .27, label="9º do ranking", color=CINZA, alpha=.28)
    a1.bar(x + .27, meu, .27, label="nosso melhor modelo", color=AZUL, alpha=.9)
    for i, (v, lo) in enumerate(zip(meu, nono)):
        if v >= lo:
            a1.annotate("dentro\ndo top-9", (i + .27, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=6.5,
                        color=VERDE, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(METRICAS, rotation=18, ha="right")
    a1.set_ylim(0, 1.05); a1.set_ylabel("$F_1$")
    a1.set_title("Contra o ranking oficial do xView2")
    a1.legend(frameon=False, ncol=1, loc="lower left", fontsize=7)
    a1.grid(axis="y", alpha=.25, lw=.5)

    # dispersao: localizacao x dano, ranking vs nos
    lb_loc = [l[2] for l in LEADERBOARD]
    lb_dano = [4 / sum(1 / l[i] for i in (4, 5, 6, 3)) for l in LEADERBOARD]
    a2.scatter(lb_loc, lb_dano, s=30, color=CINZA, alpha=.75, label="top-9 oficial")
    for l, xx, yy in zip(LEADERBOARD, lb_loc, lb_dano):
        a2.annotate(str(l[0]), (xx, yy), xytext=(4, -3), textcoords="offset points",
                    fontsize=6, color=CINZA)
    a2.scatter([nosso["f1_loc"]], [nosso["f1_dano_harmonico"]], s=90, marker="D",
               color=AZUL, zorder=4, label="nosso melhor")
    a2.scatter([POR["ref"]["f1_loc"]], [POR["ref"]["f1_dano_harmonico"]], s=60,
               marker="o", facecolor="none", edgecolor=AZUL, lw=1.4, zorder=4,
               label="nossa referência")
    a2.set_xlabel("$F_1$ localização"); a2.set_ylabel("$F_1$ dano (harmônica)")
    a2.set_title("A distância está no eixo vertical")
    a2.legend(frameon=False, loc="lower left", fontsize=7)
    a2.grid(alpha=.25, lw=.5)

    fig.tight_layout()
    fig.savefig(SAIDA / "leaderboard.pdf")
    plt.close(fig)
    print("  leaderboard.pdf")


# ------------------------------------------------------- 5. generalizacao


def fig_folds() -> None:
    folds = [POR[f"base-grouped-f{i}"] for i in range(5)]
    ref = POR["ref"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    x = np.arange(5)
    ax.bar(x - .19, [f["f1_loc"] for f in folds], .38, color=AZUL, alpha=.85,
           label="$F_1$ localização")
    ax.bar(x + .19, [f["f1_dano_harmonico"] for f in folds], .38, color=VERMELHO,
           alpha=.85, label="$F_1$ dano")
    ax.axhline(ref["f1_loc"], color=AZUL, ls="--", lw=1.1)
    ax.axhline(ref["f1_dano_harmonico"], color=VERMELHO, ls="--", lw=1.1)
    ax.text(4.55, ref["f1_loc"] + .012, "oficial", fontsize=6.5, color=AZUL, ha="right")
    ax.text(4.55, ref["f1_dano_harmonico"] + .012, "oficial", fontsize=6.5,
            color=VERMELHO, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f"fold {i}" for i in range(5)])
    ax.set_ylim(0, 1); ax.set_ylabel("$F_1$")
    ax.set_title("Validação agrupada por evento: a localização resiste, o dano não")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(SAIDA / "folds.pdf")
    plt.close(fig)
    print("  folds.pdf")


if __name__ == "__main__":
    print(f"gerando em {SAIDA}")
    fig_curvas()
    fig_prec_recall()
    fig_confusao()
    fig_leaderboard()
    fig_folds()
    print("pronto.")
