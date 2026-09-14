"""
Dataset e augmentacao.

As augmentacoes fotometricas foram calibradas pela analise de imagem
(`scripts/analyze_imagery.py`), nao chutadas:

  brilho    32% dos pares pre/pos diferem em mais de 10 DN de brilho medio, e
            11% em mais de 25 DN -> jitter de brilho de ate +-30 DN
  nitidez   35% dos pares diferem em mais de 2x na variancia do laplaciano, e
            9,5% em mais de 4x -> blur/sharpen agressivo
  ruido     sigma medido entre 1,0 e 12,5 DN conforme o evento -> ruido gaussiano
            amostrado nessa faixa
  cor       as bandas R,G,B tem correlacao de 0,93-0,98, ou seja quase nao ha
            informacao cromatica independente -> jitter de matiz e desperdicio;
            o ganho esta em luminancia

O ponto critico: **as fotometricas sao aplicadas de forma INDEPENDENTE em pre e
pos**, porque e exatamente assim que o descasamento aparece no dataset real. As
geometricas sao aplicadas em CONJUNTO, porque as duas datas sao co-registradas e
compartilham a mesma mascara.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import IGNORE_INDEX, UNCLASSIFIED

# Medias/desvios medidos em amostra estratificada do proprio xBD (secao 8 da EDA).
# Diferem bastante das constantes do ImageNet: use estas.
XBD_MEAN = (0.3293, 0.3571, 0.2692)
XBD_STD = (0.1218, 0.1064, 0.1027)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------- augmentacao


class Aug:
    """Augmentacao em numpy uint8. Leve o bastante para rodar nos workers."""

    def __init__(self, crop: int = 512, treino: bool = True, forca: float = 1.0):
        self.crop = crop
        self.treino = treino
        self.f = forca

    # ---- geometricas: aplicadas juntas em pre, pos e mascara ----

    def _crop_conjunto(self, pre, post, msk, rng):
        h, w = msk.shape
        c = min(self.crop, h, w)
        if not self.treino:
            i, j = (h - c) // 2, (w - c) // 2
        else:
            # duas tentativas de cair sobre uma edificacao antes de aceitar qualquer lugar
            i, j = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
            for _ in range(2):
                if msk[i:i + c, j:j + c].any():
                    break
                i, j = rng.integers(0, h - c + 1), rng.integers(0, w - c + 1)
        return pre[i:i + c, j:j + c], post[i:i + c, j:j + c], msk[i:i + c, j:j + c]

    def _flip_rot(self, pre, post, msk, rng):
        if rng.random() < 0.5:
            pre, post, msk = pre[:, ::-1], post[:, ::-1], msk[:, ::-1]
        if rng.random() < 0.5:
            pre, post, msk = pre[::-1], post[::-1], msk[::-1]
        k = int(rng.integers(0, 4))
        if k:
            pre, post, msk = (np.rot90(pre, k), np.rot90(post, k), np.rot90(msk, k))
        return np.ascontiguousarray(pre), np.ascontiguousarray(post), np.ascontiguousarray(msk)

    # ---- fotometricas: aplicadas SEPARADAMENTE em cada data ----

    def _foto(self, img: np.ndarray, rng) -> np.ndarray:
        x = img.astype(np.float32)

        if rng.random() < 0.8:                      # brilho: faixa medida +-30 DN
            x += rng.uniform(-30, 30) * self.f
        if rng.random() < 0.8:                      # contraste em torno da media
            m = x.mean()
            x = (x - m) * rng.uniform(0.75, 1.35) + m
        if rng.random() < 0.3:                      # gama
            x = 255.0 * np.clip(x / 255.0, 0, 1) ** rng.uniform(0.7, 1.4)
        if rng.random() < 0.3:                      # ruido: sigma observado 1-12 DN
            x += rng.normal(0, rng.uniform(1.0, 8.0) * self.f, size=x.shape).astype(np.float32)
        if rng.random() < 0.3:                      # nitidez: pares diferem ate 4x
            x = self._blur_ou_sharpen(x, rng)

        return np.clip(x, 0, 255).astype(np.uint8)

    @staticmethod
    def _blur_ou_sharpen(x: np.ndarray, rng) -> np.ndarray:
        import cv2
        if rng.random() < 0.5:
            k = int(rng.choice([3, 5]))
            return cv2.GaussianBlur(x, (k, k), sigmaX=rng.uniform(0.5, 1.5))
        borrado = cv2.GaussianBlur(x, (5, 5), sigmaX=1.0)
        return cv2.addWeighted(x, 1.0 + rng.uniform(0.3, 1.0), borrado, -rng.uniform(0.3, 1.0), 0)

    # ---- ----

    def __call__(self, pre, post, msk, rng):
        pre, post, msk = self._crop_conjunto(pre, post, msk, rng)
        if self.treino:
            pre, post, msk = self._flip_rot(pre, post, msk, rng)
            pre = self._foto(pre, rng)
            post = self._foto(post, rng)
        return pre, post, msk


# ------------------------------------------------------------------------ dataset


class XBDDataset(Dataset):
    def __init__(self, index: pd.DataFrame, data_dir: str | Path, cfg,
                 treino: bool = True, norm: str = "dataset"):
        self.idx = index.reset_index(drop=True)
        self.dir = Path(data_dir)
        self.cfg = cfg
        self.treino = treino
        self.aug = Aug(crop=cfg.crop, treino=treino)
        self.task = cfg.task

        if norm == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        elif norm == "per_image":
            mean = std = None
        else:
            mean, std = XBD_MEAN, XBD_STD
        self.mean = np.array(mean, np.float32) if mean else None
        self.std = np.array(std, np.float32) if std else None

    def __len__(self) -> int:
        return len(self.idx)

    def _png(self, split: str, cena: str, sufixo: str) -> np.ndarray:
        sub = "masks" if sufixo == "" else "images"
        nome = f"{cena}.png" if sufixo == "" else f"{cena}_{sufixo}.png"
        return np.array(Image.open(self.dir / sub / split / nome))

    def _normalizar(self, img: np.ndarray) -> torch.Tensor:
        x = img.astype(np.float32) / 255.0
        if self.mean is not None:
            x = (x - self.mean) / self.std
        else:  # per_image: padroniza cada imagem por si
            x = (x - x.mean(axis=(0, 1))) / (x.std(axis=(0, 1)) + 1e-6)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def __getitem__(self, i: int):
        r = self.idx.iloc[i]
        split, cena = str(r.split), str(r.scene_id)

        pre = self._png(split, cena, "pre")
        post = self._png(split, cena, "post")
        msk = self._png(split, cena, "")

        # semente derivada do indice: augmentacao reprodutivel entre maquinas
        semente = (self.cfg.seed * 1_000_003 + i) % (2 ** 31)
        rng = np.random.default_rng(semente if not self.treino else None)
        pre, post, msk = self.aug(pre, post, msk, rng)

        msk = msk.astype(np.int64)
        msk[msk == UNCLASSIFIED] = IGNORE_INDEX      # fora de toda perda e metrica
        if self.task == "loc":
            msk = np.where(msk == IGNORE_INDEX, IGNORE_INDEX, (msk > 0).astype(np.int64))

        return {
            "pre": self._normalizar(pre),
            "post": self._normalizar(post),
            "mask": torch.from_numpy(msk),
            "scene_id": cena,
        }


# -------------------------------------------------------------------------- splits


def montar_splits(index: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (treino, validacao).

    `oficial`  reproduz a divisao do xView2: tier1 (+tier3) treina, hold valida.
               Cuidado: a EDA mostrou que >65% dos tiles de hold/test sao
               adjacentes a um tile de treino, entao esse numero e otimista.
    `grouped`  agrupa por DESASTRE: nenhum evento aparece nos dois lados. Mede
               generalizacao para um desastre novo, que e o caso de uso real.
    """
    idx = index.copy()
    idx["split"] = idx["split"].astype(str)

    if cfg.split_mode == "oficial":
        splits_treino = ["tier1", "tier3"] if cfg.usar_tier3 else ["tier1"]
        treino = idx[idx.split.isin(splits_treino)]
        val = idx[idx.split == "hold"]
    elif cfg.split_mode == "grouped":
        if not cfg.usar_tier3:
            idx = idx[idx.split != "tier3"]
        eventos = sorted(idx.disaster.unique())
        # atribuicao estavel de evento -> fold, independente da ordem de listagem
        fold_de = {d: i % cfg.n_folds for i, d in enumerate(eventos)}
        f = idx.disaster.map(fold_de)
        treino, val = idx[f != cfg.fold], idx[f == cfg.fold]
    else:
        raise ValueError(f"split_mode desconhecido: {cfg.split_mode}")

    if cfg.excluir_cenas_vazias:
        treino = treino[treino.tem_edificacao]

    return treino.reset_index(drop=True), val.reset_index(drop=True)


def build_loaders(index: pd.DataFrame, cfg, plano, norm: str | None = None):
    """`norm` vem da config; o parametro so existe para sobrescrita pontual da CLI."""
    norm = norm or cfg.norm
    treino_df, val_df = montar_splits(index, cfg)

    ds_tr = XBDDataset(treino_df, cfg.data_dir, cfg, treino=True, norm=norm)
    ds_va = XBDDataset(val_df, cfg.data_dir, cfg, treino=False, norm=norm)

    # 35% das cenas nao tem edificacao alguma e so 1 em 3 tem dano: sem reamostrar,
    # a maior parte dos passos nao ve nenhuma classe positiva.
    peso = np.ones(len(treino_df), dtype=np.float64)
    peso[treino_df.tem_dano.to_numpy()] *= cfg.oversample_dano
    peso[~treino_df.tem_edificacao.to_numpy()] *= 0.25
    sampler = WeightedRandomSampler(torch.as_tensor(peso), num_samples=len(treino_df),
                                    replacement=True)

    comum = dict(num_workers=plano.num_workers, pin_memory=plano.pin_memory,
                 persistent_workers=plano.persistent_workers and plano.num_workers > 0)
    if plano.num_workers > 0 and plano.prefetch_factor:
        comum["prefetch_factor"] = plano.prefetch_factor

    dl_tr = DataLoader(ds_tr, batch_size=plano.batch_size, sampler=sampler,
                       drop_last=True, **comum)
    dl_va = DataLoader(ds_va, batch_size=max(1, plano.batch_size), shuffle=False,
                       drop_last=False, **comum)
    return dl_tr, dl_va, treino_df, val_df


def contagens_de_classe(index: pd.DataFrame, n_classes: int) -> torch.Tensor:
    """Contagem de pixels por classe, para derivar os pesos da perda."""
    if n_classes == 2:
        return torch.tensor([index.px_fundo.sum(), index.px_edificacao.sum()], dtype=torch.float64)
    return torch.tensor([
        index.px_fundo.sum(), index.px_no_damage.sum(), index.px_minor.sum(),
        index.px_major.sum(), index.px_destroyed.sum(),
    ], dtype=torch.float64)
