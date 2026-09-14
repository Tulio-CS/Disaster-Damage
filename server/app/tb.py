"""
Traduz as metricas recebidas pela API em arquivos de evento do TensorBoard.

O worker nunca escreve tfevents. Se cada uma das 20 maquinas escrevesse o seu, o
TensorBoard so veria tudo junto atraves de um diretorio compartilhado — que e
justamente o que nao existe, porque as maquinas estao na rede do laboratorio e o
painel roda numa VPS. Aqui os escalares chegam por HTTP e viram tfevents num
unico volume, entao um so TensorBoard enxerga o laboratorio inteiro.

O nome do run no TensorBoard e `<grupo>/<nome>` (e nao o `run_id`, que carrega um
hash) para que o seletor de runs possa ser filtrado por regex: `^baseline` mostra
so as referencias, `^ablacao` so o estudo.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from tensorboardX import SummaryWriter

# metrica recebida -> tag no TensorBoard. O que nao estiver aqui e ignorado, o
# que evita poluir o painel com contadores internos (passos_otim, pulado, ...).
TAGS = {
    "score_xview2": "score/xview2",
    "f1_loc": "score/f1_localizacao",
    "f1_dano_harmonico": "score/f1_dano_harmonico",
    "f1_dano_media": "score/f1_dano_media",
    "perda_treino": "perda/treino",
    "perda_val": "perda/val",
    "img_por_s": "exec/img_por_s",
    "seg_treino": "exec/seg_treino",
    "seg_val": "exec/seg_val",
    "lr_final": "exec/lr",
}

# Campos da config que definem o experimento e valem como hiperparametro na aba
# HPARAMS — onde se compara ablacao contra referencia numa tabela so.
HPARAMS = [
    "nome", "task", "seed", "fold", "split_mode", "crop", "encoder", "pretrained",
    "siamese", "fuse_mode", "norm", "usar_tier3", "oversample_dano",
    "excluir_cenas_vazias", "loss_dmg", "loss_loc", "focal_gamma", "lr",
    "effective_batch", "epochs", "ema_decay", "lr_encoder_mult",
]

_SEGURO = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(s: str) -> str:
    return _SEGURO.sub("-", str(s)).strip("-") or "sem-nome"


class Escritor:
    """Cache de SummaryWriter por run, com acesso serializado."""

    def __init__(self, raiz: str | Path):
        self.raiz = Path(raiz)
        self.raiz.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, SummaryWriter] = {}
        self._lock = threading.Lock()

    def _writer(self, grupo: str, nome: str) -> SummaryWriter:
        chave = f"{_slug(grupo)}/{_slug(nome)}"
        with self._lock:
            w = self._writers.get(chave)
            if w is None:
                w = SummaryWriter(logdir=str(self.raiz / chave), flush_secs=10)
                self._writers[chave] = w
            return w

    def escalares(self, grupo: str, nome: str, epoca: int, metricas: dict[str, Any]) -> int:
        w = self._writer(grupo, nome)
        n = 0
        for chave, valor in metricas.items():
            tag = TAGS.get(chave)
            if tag is None:
                for prefixo, grupo in (("f1_", "f1"), ("iou_", "iou"),
                                       ("prec_", "precisao"), ("rec_", "recall")):
                    if chave.startswith(prefixo):
                        tag = f"{grupo}/{chave[len(prefixo):]}"
                        break
                else:
                    continue    # inclui `matriz`, que nao e escalar — fica so no
                                # banco, de onde qualquer metrica pode ser refeita
            try:
                v = float(valor)
            except (TypeError, ValueError):
                continue
            if v != v:      # NaN nao vira ponto; sujaria a escala do grafico
                continue
            w.add_scalar(tag, v, epoca)
            n += 1
        w.flush()
        return n

    def hparams(self, grupo: str, nome: str, config: dict, melhor: dict) -> None:
        """Registra a linha do experimento na aba HPARAMS, no fim do run."""
        hp = {}
        for k in HPARAMS:
            v = config.get(k)
            if isinstance(v, (list, tuple)):
                v = "x".join(str(i) for i in v)
            if v is None:
                continue
            hp[k] = v if isinstance(v, (int, float, bool, str)) else str(v)

        met = {}
        for k in ("score_xview2", "f1_loc", "f1_dano_harmonico", "f1_dano_media"):
            try:
                v = float(melhor[k])
            except (KeyError, TypeError, ValueError):
                continue
            if v == v:
                met[f"melhor/{k}"] = v
        if not met:
            return

        w = self._writer(grupo, nome)
        try:
            w.add_hparams(hp, met, name=".", global_step=0)
        except Exception:
            pass    # a aba HPARAMS e um extra; nunca derruba a ingestao
        w.flush()

    def fechar(self) -> None:
        with self._lock:
            for w in self._writers.values():
                try:
                    w.close()
                except Exception:
                    pass
            self._writers.clear()
