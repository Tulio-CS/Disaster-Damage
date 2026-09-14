"""
Perdas para o xBD.

Duas pressoes vindas da EDA governam a escolha:

1. **97% dos pixels sao fundo.** Entropia cruzada pura converge para "preveja
   fundo". Dice olha sobreposicao em vez de acerto por pixel, entao nao tem esse
   otimo local; Focal reduz o peso dos pixels ja faceis. Usar os dois juntos e o
   que as solucoes vencedoras do xView2 fazem.

2. **Desbalanceamento de ~10:1 entre `no-damage` e as classes de dano.** Os pesos
   por classe saem do inverso da frequencia, amortecido por raiz quadrada: o
   inverso puro da peso ~40x a `destroyed` e desestabiliza o treino.

`un-classified` vira `IGNORE_INDEX` e nao entra em nenhuma das perdas — nao e um
nivel de severidade, e sim ausencia de julgamento do anotador.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IGNORE_INDEX


def pesos_por_frequencia(contagens: torch.Tensor, amortecimento: float = 0.5,
                         max_peso: float = 10.0) -> torch.Tensor:
    """Peso ~ (1/frequencia)^amortecimento, normalizado para media 1 e limitado."""
    freq = contagens.float().clamp(min=1.0)
    freq = freq / freq.sum()
    w = (1.0 / freq) ** amortecimento
    w = w / w.mean()
    return w.clamp(max=max_peso)


class FocalLoss(nn.Module):
    """Focal multiclasse sobre log-softmax, com ignore_index e pesos por classe."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, alvo, weight=self.weight,
                             ignore_index=self.ignore_index, reduction="none")
        # pt = probabilidade da classe correta; derivada de ce para evitar gather manual
        pt = torch.exp(-ce)
        perda = (1.0 - pt) ** self.gamma * ce
        valido = alvo != self.ignore_index
        return perda[valido].mean() if valido.any() else logits.sum() * 0.0


class DiceLoss(nn.Module):
    """Dice soft multiclasse (macro sobre as classes, fundo opcionalmente fora)."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, incluir_fundo: bool = False,
                 eps: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.incluir_fundo = incluir_fundo
        self.eps = eps

    def forward(self, logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
        n_cls = logits.shape[1]
        valido = (alvo != self.ignore_index)
        if not valido.any():
            return logits.sum() * 0.0

        prob = logits.softmax(dim=1)
        alvo_seguro = torch.where(valido, alvo, torch.zeros_like(alvo))
        onehot = F.one_hot(alvo_seguro, n_cls).permute(0, 3, 1, 2).float()

        m = valido.unsqueeze(1).float()
        prob, onehot = prob * m, onehot * m

        inicio = 0 if self.incluir_fundo else 1
        prob, onehot = prob[:, inicio:], onehot[:, inicio:]

        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        denom = prob.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)

        # so pontua classes presentes no lote; senao classes ausentes viram dice=1
        presente = onehot.sum(dims) > 0
        return (1 - dice[presente]).mean() if presente.any() else logits.sum() * 0.0


class PerdaXBD(nn.Module):
    """Combina dano (focal+dice em 5 classes) com localizacao (BCE+dice binario).

    A localizacao nao tem cabeca propria: sai de `1 - p(fundo)`, entao as duas
    perdas supervisionam o mesmo tensor de logits por caminhos diferentes.
    """

    def __init__(self, task: str = "damage", gamma: float = 2.0,
                 class_weights: torch.Tensor | None = None,
                 peso_loc: float = 1.0, peso_dmg: float = 1.0,
                 loss_dmg: str = "focal_dice", loss_loc: str = "bce_dice"):
        super().__init__()
        self.task = task
        self.peso_loc = peso_loc
        self.peso_dmg = peso_dmg
        self.loss_dmg = loss_dmg
        self.loss_loc = loss_loc
        self.focal = FocalLoss(gamma=gamma, weight=class_weights)
        self.dice = DiceLoss()
        self.register_buffer("_w_ce", class_weights)

    # ------------------------------------------------------------------ termos

    def _termo_dano(self, logits, alvo) -> tuple[torch.Tensor, dict]:
        """Compoe o termo de dano conforme `loss_dmg`.

        Existe para a ablacao "o Focal esta pagando o proprio custo?": com pesos
        por classe ja aplicados, parte do trabalho do Focal e redundante, entao
        `ce_dice` e a comparacao honesta — mesma estrutura, sem o termo de
        modulacao.
        """
        partes: dict[str, float] = {}
        termos = []

        if self.loss_dmg in ("focal_dice", "focal"):
            l = self.focal(logits, alvo)
            partes["focal"] = float(l.detach())
            termos.append(l)
        if self.loss_dmg == "ce_dice":
            l = F.cross_entropy(logits, alvo, weight=self._w_ce,
                                ignore_index=IGNORE_INDEX)
            partes["ce"] = float(l.detach())
            termos.append(l)
        if self.loss_dmg in ("focal_dice", "ce_dice", "dice"):
            l = self.dice(logits, alvo)
            partes["dice"] = float(l.detach())
            termos.append(l)

        return sum(termos), partes

    def _termo_loc(self, logits, alvo) -> tuple[torch.Tensor, dict]:
        """Localizacao derivada de `1 - p(fundo)`: sem cabeca propria."""
        if self.loss_loc == "nenhuma":
            return logits.sum() * 0.0, {}

        valido = alvo != IGNORE_INDEX
        p_edificacao = 1.0 - logits.softmax(dim=1)[:, 0]
        alvo_loc = (alvo > 0).float()

        l = F.binary_cross_entropy(
            p_edificacao.clamp(1e-6, 1 - 1e-6)[valido], alvo_loc[valido]
        ) if valido.any() else logits.sum() * 0.0

        if self.loss_loc == "bce_dice":
            inter = (p_edificacao * alvo_loc * valido).sum()
            denom = (p_edificacao * valido).sum() + (alvo_loc * valido).sum()
            l = l + (1 - (2 * inter + 1.0) / (denom + 1.0))

        return l, {"loc": float(l.detach())}

    # ------------------------------------------------------------------

    def forward(self, logits: torch.Tensor, alvo: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.task == "loc":
            # 2 classes: o termo de dano JA e a localizacao, nao ha o que derivar
            perda = self.focal(logits, alvo) + self.dice(logits, alvo)
            return perda, {"loc": float(perda.detach())}

        l_dmg, partes = self._termo_dano(logits, alvo)
        l_loc, partes_loc = self._termo_loc(logits, alvo)
        partes.update(partes_loc)

        total = self.peso_dmg * l_dmg + self.peso_loc * l_loc
        partes["total"] = float(total.detach())
        return total, partes


def build_loss(cfg, class_weights: torch.Tensor | None = None) -> PerdaXBD:
    return PerdaXBD(task=cfg.task, gamma=cfg.focal_gamma, class_weights=class_weights,
                    peso_loc=cfg.peso_loc, peso_dmg=cfg.peso_dmg,
                    loss_dmg=cfg.loss_dmg, loss_loc=cfg.loss_loc)
