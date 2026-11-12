"""
Inferencia de ponta a ponta: imagem -> pre-processamento -> modelo -> resultado.

Usado por `scripts/demo.py` e pelo notebook 03. Nao depende de `data/` nem do
indice: recebe dois arquivos de imagem (pre e pos) e devolve a mascara de dano.

    imagem      PNG/JPG uint8, ou o GeoTIFF original do xBD (int16 com conteudo
                8-bit, convertido como em `xbd/prep.py`)
    pre-proc.   [0,1] -> normalizacao por canal com as constantes do treino ->
                CHW -> padding ate multiplo de 32 (o encoder reduz 32x)
    modelo      U-Net siamesa; logits (5, H, W) -> softmax -> argmax por pixel
    resultado   mascara 0..4, contagem de edificacoes por severidade (componentes
                conexos da mascara, classe pela maioria dos pixels) e figura
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import IGNORE_INDEX, UNCLASSIFIED, Config
from .data import IMAGENET_MEAN, IMAGENET_STD, XBD_MEAN, XBD_STD, XBD_TREINO_MEAN, XBD_TREINO_STD
from .metrics import Confusao
from .models import build_model

RAIZ = Path(__file__).resolve().parent.parent
MODELO_PADRAO = RAIZ / "modelos" / "modelo_final.pt"
URL_RELEASE = "https://github.com/Tulio-CS/Disaster-Damage/releases/download/v1.0"

# fundo transparente, depois verde -> amarelo -> laranja -> vermelho
CORES = ["#00000000", "#2e8b57", "#e6c229", "#e07b39", "#c0392b"]
NOMES = ["fundo", "intacta", "dano leve", "dano severo", "destruída"]

AREA_MIN_EDIFICACAO = 20   # px; componentes menores sao ruido de borda


# --------------------------------------------------------------------- modelo


def baixar(destino: Path) -> None:
    """Baixa do release v1.0. Com o repositorio privado, o download anonimo da 404:
    nesse caso usa o `gh` autenticado, se houver, ou explica como baixar a mao."""
    import shutil
    import subprocess
    url = f"{URL_RELEASE}/{destino.name}"
    print(f"baixando {url} ...")
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".parcial")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(destino)
        return
    except OSError:
        tmp.unlink(missing_ok=True)
    if shutil.which("gh"):
        r = subprocess.run(["gh", "release", "download", "v1.0", "--repo", "Tulio-CS/Disaster-Damage",
                            "--pattern", destino.name, "--dir", str(destino.parent)])
        if r.returncode == 0 and destino.exists():
            return
    raise SystemExit(f"nao consegui baixar {destino.name}. Baixe-o da pagina do release "
                     f"(https://github.com/Tulio-CS/Disaster-Damage/releases/tag/v1.0) "
                     f"e salve em {destino}")


def carregar_modelo(caminho: str | Path = MODELO_PADRAO,
                    dispositivo: str = "cpu") -> tuple[torch.nn.Module, Config, dict]:
    """Aceita tanto o `melhor.pt` da colheita quanto o arquivo enxuto de `modelos/`.

    Os dois tem as chaves `modelo` (state_dict) e `config`. `pretrained` e
    desligado ao construir: os pesos do ImageNet seriam baixados e logo
    sobrescritos pelo state_dict.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        baixar(caminho)
    bruto = torch.load(caminho, map_location="cpu", weights_only=False)
    cfg = Config.from_dict({"remover_duplicatas": False, **bruto["config"]})
    cfg_sem_download = Config.from_dict({**cfg.to_dict(), "pretrained": False})
    modelo = build_model(cfg_sem_download)
    estado = bruto["modelo"]
    if any(k.startswith("_orig_mod.") for k in estado):
        estado = {k.replace("_orig_mod.", "", 1): v for k, v in estado.items()}
    modelo.load_state_dict({k: v.float() for k, v in estado.items()})
    modelo.to(dispositivo).eval()
    meta = {k: v for k, v in bruto.items() if k not in ("modelo", "ema")}
    return modelo, cfg, meta


# --------------------------------------------------------------- pre-processamento


def ler_imagem(caminho: str | Path) -> np.ndarray:
    """RGB uint8 (H, W, 3). GeoTIFF passa pela mesma conversao de `xbd/prep.py`."""
    caminho = Path(caminho)
    if caminho.suffix.lower() in (".tif", ".tiff"):
        import rasterio   # so a leitura de GeoTIFF precisa dele
        with rasterio.open(caminho) as src:
            a = src.read()  # (3, H, W) int16 com conteudo 0-255
        return np.clip(np.transpose(a, (1, 2, 0)), 0, 255).astype(np.uint8)
    return np.array(Image.open(caminho).convert("RGB"))


def ler_mascara(caminho: str | Path) -> np.ndarray:
    """Mascara de `data/masks` (0..4, 5 = un-classified) -> 0..4 e 255 (ignorar)."""
    m = np.array(Image.open(caminho)).astype(np.uint8)
    m[m == UNCLASSIFIED] = IGNORE_INDEX
    return m


def constantes(norm: str) -> tuple[np.ndarray, np.ndarray] | None:
    tabela = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD),
              "dataset_treino": (XBD_TREINO_MEAN, XBD_TREINO_STD),
              "dataset": (XBD_MEAN, XBD_STD)}
    if norm not in tabela:
        return None   # per_image
    m, s = tabela[norm]
    return np.array(m, np.float32), np.array(s, np.float32)


def preprocessar(img: np.ndarray, norm: str) -> torch.Tensor:
    """Igual a `XBDDataset._normalizar` em validacao/teste: sem augmentacao."""
    x = img.astype(np.float32) / 255.0
    c = constantes(norm)
    if c is not None:
        x = (x - c[0]) / c[1]
    else:
        x = (x - x.mean(axis=(0, 1))) / (x.std(axis=(0, 1)) + 1e-6)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


# --------------------------------------------------------------------- predicao


@dataclass
class Resultado:
    classe: np.ndarray          # (H, W) uint8, 0..4
    prob: np.ndarray            # (5, H, W) float32
    edificacoes: dict[str, int]  # severidade -> numero de edificacoes
    pixels: dict[str, int]       # severidade -> pixels


@torch.no_grad()
def prever(modelo: torch.nn.Module, pre: np.ndarray, post: np.ndarray, norm: str,
           dispositivo: str = "cpu") -> Resultado:
    if pre.shape != post.shape:
        raise ValueError(f"pre {pre.shape} e pos {post.shape} precisam ter o mesmo tamanho")
    h, w = pre.shape[:2]
    a, b = preprocessar(pre, norm), preprocessar(post, norm)
    ph, pw = (-h) % 32, (-w) % 32           # o encoder reduz a resolucao 32x
    if ph or pw:
        a = F.pad(a, (0, pw, 0, ph), mode="reflect")
        b = F.pad(b, (0, pw, 0, ph), mode="reflect")
    logits = modelo(a.to(dispositivo), b.to(dispositivo))[..., :h, :w]
    prob = torch.softmax(logits.float(), 1)[0].cpu().numpy()
    classe = prob.argmax(0).astype(np.uint8)
    return Resultado(classe, prob, contar_edificacoes(classe),
                     {NOMES[c]: int((classe == c).sum()) for c in range(1, 5)})


def contar_edificacoes(classe: np.ndarray) -> dict[str, int]:
    """Componentes conexos da mascara de edificacao; cada um leva a classe da maioria."""
    import cv2
    n, rotulos = cv2.connectedComponents((classe > 0).astype(np.uint8), connectivity=8)
    contagem = {NOMES[c]: 0 for c in range(1, 5)}
    for k in range(1, n):
        px = classe[rotulos == k]
        if px.size < AREA_MIN_EDIFICACAO:
            continue
        contagem[NOMES[int(np.bincount(px, minlength=5)[1:].argmax()) + 1]] += 1
    return contagem


def metricas(classe: np.ndarray, alvo: np.ndarray) -> dict:
    """Mesmas metricas da avaliacao (`xbd.metrics.Confusao`), para uma cena.

    O `score_xview2` de uma cena isolada quase sempre vale ~0,3*F1_loc: a media
    harmonica zera quando alguma severidade nao aparece na cena. Para uma cena,
    `acerto_severidade` diz mais: entre os pixels de edificacao que o modelo achou,
    a fracao com a severidade certa.
    """
    c = Confusao(5)
    c.update(torch.from_numpy(classe.astype(np.int64)), torch.from_numpy(alvo.astype(np.int64)))
    m = c.resumo()
    achados = (alvo >= 1) & (alvo <= 4) & (classe >= 1)
    m["acerto_severidade"] = float((classe[achados] == alvo[achados]).mean()) if achados.any() else float("nan")
    return m


# ----------------------------------------------------------------------- figura


def figura(pre: np.ndarray, post: np.ndarray, res: Resultado,
           alvo: np.ndarray | None = None, titulo: str = ""):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(CORES)
    paineis = [(pre, None, "pré-desastre"), (post, None, "pós-desastre")]
    if alvo is not None:
        paineis.append((post, np.where(alvo == IGNORE_INDEX, 0, alvo), "verdade (anotação)"))
    paineis.append((post, res.classe, "predição do modelo"))

    fig, axs = plt.subplots(1, len(paineis), figsize=(4.2 * len(paineis), 4.6))
    for ax, (img, sobre, tit) in zip(axs, paineis):
        ax.imshow(img)
        if sobre is not None:
            ax.imshow(sobre, cmap=cmap, vmin=0, vmax=4, alpha=.6, interpolation="nearest")
        ax.set_title(tit, fontsize=11)
        ax.axis("off")
    handles = [mpatches.Patch(color=CORES[k], label=NOMES[k]) for k in range(1, 5)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    if titulo:
        fig.suptitle(titulo, fontsize=12)
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig
