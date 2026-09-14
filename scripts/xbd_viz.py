"""Tema de graficos e renderizacao de cenas xBD para o notebook de EDA."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
from shapely import wkt as shapely_wkt

# ----------------------------------------------------------------- paleta (light)

# Escala de severidade = paleta de STATUS (good / warning / serious / critical).
# Nao e uma paleta categorica: os quatro tons nao se separam sozinhos sob
# deuteranopia (destroyed x no-damage: dE 4.1), entao a cor NUNCA carrega a
# identidade sozinha - nos graficos ha rotulo de eixo/legenda, e nos overlays
# sobre imagem ha hachura como codificacao secundaria.
DAMAGE_COLORS = {
    "no-damage": "#0ca30c",
    "minor-damage": "#fab219",
    "major-damage": "#ec835a",
    "destroyed": "#d03b3b",
    "un-classified": "#898781",
    "no-subtype": "#2a78d6",
}
DAMAGE_HATCH = {
    "no-damage": None,
    "minor-damage": "///",
    "major-damage": "xxx",
    "destroyed": "...",
    "un-classified": "\\\\\\",
    "no-subtype": None,
}
DAMAGE_ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]

# Paleta categorica validada (ordem fixa, nunca ciclada).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Rampa sequencial azul 100 -> 700 (magnitude continua).
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def use_theme() -> None:
    """Aplica o tema de grafico (marcas finas, grade recessiva, tinta neutra)."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "patch.linewidth": 0,
            "figure.dpi": 110,
        }
    )


def cat(n: int) -> list[str]:
    """n cores categoricas em ordem fixa. Acima de 8 slots, agrupe em 'Outros'."""
    if n > len(CATEGORICAL):
        raise ValueError(f"{n} series: acima de 8 use 'Outros' ou small multiples")
    return CATEGORICAL[:n]


def seq_cmap(name: str = "xbd_blue"):
    """Colormap sequencial de um unico matiz (azul), claro -> escuro."""
    return mpl.colors.LinearSegmentedColormap.from_list(name, SEQUENTIAL)


def ref_line(ax, value, label, color=INK_2, axis="x"):
    """Linha de referencia com rotulo FORA da area de plotagem (nunca colide com as marcas)."""
    if axis == "x":
        ax.axvline(value, color=color, lw=1.1, ls="--", zorder=3)
        ax.annotate(label, xy=(value, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color=color, ha="left", va="bottom", clip_on=False)
    else:
        ax.axhline(value, color=color, lw=1.1, ls="--", zorder=3)
        ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=color, ha="left", va="center", clip_on=False)


def bar_labels(ax, bars, fmt="{:,.0f}", pad=3, color=INK_2, orient=None):
    """Rotula barras diretamente (identidade sem depender da cor).

    `orient` vem do BarContainer. Nunca infira a orientacao comparando largura
    com altura: uma barra horizontal curta (valor perto de zero) seria lida como
    vertical e o rotulo sairia na linha errada, mostrando a altura da barra.
    """
    orient = orient or getattr(bars, "orientation", "vertical")
    for b in bars:
        w, h = b.get_width(), b.get_height()
        if orient == "vertical":
            ax.annotate(fmt.format(h), (b.get_x() + w / 2, h), (0, pad),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color=color)
        else:
            ax.annotate(fmt.format(w), (w, b.get_y() + h / 2), (pad, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=9, color=color)


# ----------------------------------------------------------------- leitura de cena


def read_chip(tif_path: str | Path, stretch: bool = True) -> np.ndarray:
    """Le um GeoTIFF xBD como HxWx3 uint8, com realce opcional de contraste."""
    with rasterio.open(tif_path) as src:
        arr = src.read().astype(np.float32)
    arr = np.transpose(arr, (1, 2, 0))
    if stretch:
        lo, hi = np.percentile(arr, [2, 98])
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def read_polys(json_path: str | Path) -> list[tuple[str, np.ndarray]]:
    """Devolve [(subtype, array Nx2 de coordenadas de pixel), ...]."""
    doc = json.loads(Path(json_path).read_text())
    out = []
    for feat in doc.get("features", {}).get("xy", []):
        subtype = feat.get("properties", {}).get("subtype", "no-subtype")
        geom = shapely_wkt.loads(feat["wkt"])
        if geom.is_empty:
            continue
        out.append((subtype, np.asarray(geom.exterior.coords)))
    return out


def scene_paths(stem_post: str, split: str, root: Path) -> dict[str, Path]:
    stem_pre = stem_post.replace("_post_disaster", "_pre_disaster")
    return {
        "img_pre": root / split / "images" / f"{stem_pre}.tif",
        "img_post": root / split / "images" / f"{stem_post}.tif",
        "lbl_pre": root / split / "labels" / f"{stem_pre}.json",
        "lbl_post": root / split / "labels" / f"{stem_post}.json",
    }


def draw_polys(ax, polys, alpha: float = 0.35, lw: float = 1.4, hatch: bool = True) -> None:
    for subtype, coords in polys:
        color = DAMAGE_COLORS.get(subtype, MUTED)
        ax.add_patch(
            MplPolygon(
                coords,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0,
                hatch=DAMAGE_HATCH.get(subtype) if hatch else None,
            )
        )
        # anel de contorno opaco por cima: separa poligonos vizinhos
        ax.add_patch(MplPolygon(coords, closed=True, fill=False, edgecolor=color, linewidth=lw))


def damage_legend(ax, classes=None, **kw):
    """Legenda com cor + hachura, sempre presente quando ha >= 2 classes."""
    classes = classes or DAMAGE_ORDER
    handles = [
        Patch(facecolor=DAMAGE_COLORS[c], edgecolor=DAMAGE_COLORS[c], alpha=0.55,
              hatch=DAMAGE_HATCH.get(c), label=c)
        for c in classes
    ]
    return ax.legend(handles=handles, **kw)


def show_scene(stem_post: str, split: str, root: Path, figsize=(13, 6.8),
               title: str | None = None, alpha: float = 0.35, stretch: bool = True):
    """Painel pre | pos com os poligonos de dano sobrepostos no pos."""
    p = scene_paths(stem_post, split, root)
    pre, post = read_chip(p["img_pre"], stretch), read_chip(p["img_post"], stretch)
    polys_pre, polys_post = read_polys(p["lbl_pre"]), read_polys(p["lbl_post"])

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, polys, tag in (
        (axes[0], pre, polys_pre, "PRE  ·  localizacao"),
        (axes[1], post, polys_post, "POS  ·  classificacao de dano"),
    ):
        ax.imshow(img)
        draw_polys(ax, polys, alpha=alpha)
        ax.set_title(tag, fontsize=10, color=INK_2)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    present = [c for c in DAMAGE_ORDER if any(s == c for s, _ in polys_post)]
    if len(present) >= 1:
        damage_legend(axes[1], present, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(title or f"{stem_post}  ({split})", x=0.008, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
