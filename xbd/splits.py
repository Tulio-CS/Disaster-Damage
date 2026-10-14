"""
Divisao treino/validacao, sem depender de torch: o treino (`xbd.data`), o
manifesto (`scripts/gerar_manifesto.py`) e o notebook 02 usam este mesmo codigo.

Dois modos (ver METODOLOGIA.md §6):

  `oficial`  reproduz a divisao do xView2: tier1 (+tier3) treina, hold valida.
             Cuidado: a EDA mostrou que, em 7 dos 10 eventos avaliados, >65% dos
             tiles de hold/test sao adjacentes a um tile de treino, entao esse
             numero e otimista.
  `grouped`  agrupa por DESASTRE: nenhum evento aparece nos dois lados. Mede
             generalizacao para um desastre novo, que e o caso de uso real.

Duplicatas (auditoria de 13/10/2026, `manifestos/auditoria.json`): `socal-fire` e
`woolsey-fire` sao os MESMOS tiles — mesma captura pre de 23/10/2018, pos de 14/11
e 18/11 em satelites diferentes. Sao 725 pares de cenas com a imagem pre
identica; 143 ligam uma cena de `hold` e 156 uma de `test` a uma copia em
`tier3`. Com `cfg.remover_duplicatas` (padrao):

  oficial  a copia de treino de cada cena de hold/test sai do treino (299 cenas)
  grouped  os dois eventos formam um grupo so e caem no mesmo fold

Os pares vem de `manifestos/duplicatas.csv`, gerado por `scripts/gerar_manifesto.py`
a partir do SHA-256 dos pixels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TREINO = ("tier1", "tier3")
AVALIACAO = ("hold", "test")
ARQ_DUPLICATAS = Path(__file__).resolve().parent.parent / "manifestos" / "duplicatas.csv"

# Eventos que cobrem os mesmos tiles: precisam ficar do mesmo lado de qualquer divisao.
MESMO_LUGAR = {"socal-fire": "socal-fire+woolsey-fire",
               "woolsey-fire": "socal-fire+woolsey-fire"}


def grupo_de(disaster: str, remover_duplicatas: bool = True) -> str:
    """Grupo do split agrupado: o evento, unindo os que cobrem os mesmos tiles."""
    return MESMO_LUGAR.get(disaster, disaster) if remover_duplicatas else disaster


def folds(grupos, n_folds: int) -> dict[str, int]:
    """Atribuicao estavel grupo -> fold, independente da ordem de listagem."""
    return {g: i % n_folds for i, g in enumerate(sorted(set(grupos)))}


def copias_de_avaliacao(index: pd.DataFrame, arquivo: Path = ARQ_DUPLICATAS) -> set[str]:
    """Cenas de tier1/tier3 que tem copia identica em hold ou test."""
    if not arquivo.exists():
        raise FileNotFoundError(f"{arquivo} nao existe: rode scripts/gerar_manifesto.py "
                                "ou use `remover_duplicatas: false`")
    pares = pd.read_csv(arquivo, usecols=["a", "b"])
    tier = dict(zip(index.scene_id.astype(str), index.split.astype(str)))
    fora = set()
    for a, b in zip(pares.a, pares.b):
        ta, tb = tier.get(a), tier.get(b)
        if ta in AVALIACAO and tb in TREINO:
            fora.add(b)
        elif tb in AVALIACAO and ta in TREINO:
            fora.add(a)
    return fora


def montar_splits(index: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (treino, validacao)."""
    idx = index.copy()
    idx["split"] = idx["split"].astype(str)

    if cfg.split_mode == "oficial":
        splits_treino = list(TREINO) if cfg.usar_tier3 else ["tier1"]
        treino = idx[idx.split.isin(splits_treino)]
        if cfg.remover_duplicatas:
            treino = treino[~treino.scene_id.isin(copias_de_avaliacao(idx))]
        val = idx[idx.split == "hold"]
    elif cfg.split_mode == "grouped":
        if not cfg.usar_tier3:
            idx = idx[idx.split != "tier3"]
        grupo = idx.disaster.map(lambda d: grupo_de(d, cfg.remover_duplicatas))
        f = grupo.map(folds(grupo, cfg.n_folds))
        treino, val = idx[f != cfg.fold], idx[f == cfg.fold]
    else:
        raise ValueError(f"split_mode desconhecido: {cfg.split_mode}")

    if cfg.excluir_cenas_vazias:
        treino = treino[treino.tem_edificacao]

    return treino.reset_index(drop=True), val.reset_index(drop=True)
