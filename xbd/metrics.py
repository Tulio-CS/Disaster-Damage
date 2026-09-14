"""
Metrica oficial do xView2 e as diagnosticas que importam nesse dataset.

    score = 0.3 * F1_localizacao + 0.7 * F1_dano

onde `F1_dano` e a **media harmonica** dos F1 das 4 classes de severidade. A
media harmonica e o ponto central: ela e dominada pelo pior termo, entao zerar
`minor-damage` (a classe mais rara) derruba o score inteiro. Com media aritmetica
daria para ignorar as classes raras e ainda pontuar bem — e e exatamente isso que
o desbalanceamento de 10:1 do xBD incentiva.

A matriz de confusao e acumulada em inteiros no dispositivo, entao vale para
qualquer numero de lotes sem guardar predicoes.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import CLASS_NAMES, IGNORE_INDEX


class Confusao:
    """Matriz de confusao acumulada, n_classes x n_classes (linha=alvo, col=predito)."""

    def __init__(self, n_classes: int, device: str | torch.device = "cpu"):
        self.n = n_classes
        self.mat = torch.zeros(n_classes, n_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, alvo: torch.Tensor) -> None:
        valido = alvo != IGNORE_INDEX
        p = pred[valido].reshape(-1).long()
        a = alvo[valido].reshape(-1).long()
        if a.numel() == 0:
            return
        idx = a * self.n + p
        self.mat += torch.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def reset(self) -> None:
        self.mat.zero_()

    # ------------------------------------------------------------------

    def f1_por_classe(self) -> np.ndarray:
        m = self.mat.double().cpu().numpy()
        tp = np.diag(m)
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        denom = 2 * tp + fp + fn
        return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)

    def iou_por_classe(self) -> np.ndarray:
        m = self.mat.double().cpu().numpy()
        tp = np.diag(m)
        denom = m.sum(0) + m.sum(1) - tp
        return np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)

    def f1_localizacao(self) -> float:
        """F1 binario de 'existe edificacao', colapsando as classes 1..4."""
        m = self.mat.double().cpu().numpy()
        if self.n == 2:
            tp, fp, fn = m[1, 1], m[0, 1], m[1, 0]
        else:
            tp = m[1:, 1:].sum()
            fp = m[0, 1:].sum()
            fn = m[1:, 0].sum()
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else 0.0

    def f1_dano_harmonico(self) -> float:
        """Media harmonica dos F1 das 4 classes de severidade."""
        if self.n < 5:
            return float("nan")
        f1 = self.f1_por_classe()[1:5]
        if np.any(f1 <= 0):
            return 0.0  # media harmonica com um termo zero e zero, por definicao
        return float(len(f1) / np.sum(1.0 / f1))

    def score_xview2(self) -> float:
        if self.n < 5:
            return self.f1_localizacao()
        return 0.3 * self.f1_localizacao() + 0.7 * self.f1_dano_harmonico()

    def resumo(self) -> dict[str, float]:
        f1 = self.f1_por_classe()
        iou = self.iou_por_classe()
        out: dict[str, float] = {
            "f1_loc": self.f1_localizacao(),
            "score_xview2": self.score_xview2(),
        }
        if self.n >= 5:
            out["f1_dano_harmonico"] = self.f1_dano_harmonico()
            out["f1_dano_media"] = float(np.mean(f1[1:5]))
            for i, nome in enumerate(CLASS_NAMES[: self.n]):
                out[f"f1_{nome}"] = float(f1[i])
                out[f"iou_{nome}"] = float(iou[i])
        else:
            out["iou_edificacao"] = float(iou[1]) if len(iou) > 1 else 0.0
        return out

    def tabela(self) -> str:
        m = self.mat.cpu().numpy()
        nomes = CLASS_NAMES[: self.n] if self.n == 5 else ["fundo", "edificacao"]
        largura = max(len(n) for n in nomes) + 2
        linhas = [" " * largura + "".join(f"{n[:10]:>12s}" for n in nomes) + "      F1"]
        f1 = self.f1_por_classe()
        for i, nome in enumerate(nomes):
            linhas.append(f"{nome:<{largura}s}" + "".join(f"{v:>12,d}" for v in m[i])
                          + f"  {f1[i]:6.3f}")
        return "\n".join(linhas)
