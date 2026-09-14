"""
Leitura e resolucao da matriz de experimentos.

Fica separado de `main.py` e depende so de PyYAML para que o script de conferencia
(`scripts/validar_matriz.py`) possa carregar exatamente o mesmo codigo sem
precisar de FastAPI instalado. Duas implementacoes do merge seria a forma mais
facil de a conferencia local aprovar uma matriz que o servidor le diferente.
"""

from __future__ import annotations

from pathlib import Path

import yaml

GRUPOS = ("referencia", "baseline", "ablacao")
PRIORIDADE_PADRAO = {"referencia": 0, "baseline": 10, "ablacao": 50}


def carregar_matriz(caminho: str | Path) -> list[dict]:
    """Resolve a heranca preset -> comum -> overrides e devolve os slots.

    O merge acontece no servidor, e nao no worker, para que o painel mostre a
    config efetiva de cada slot antes de qualquer maquina pegar o trabalho — e
    para que um erro de digitacao apareca no deploy, nao as 9h no laboratorio.
    """
    caminho = Path(caminho)
    doc = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    preset_padrao = doc.get("preset", "base")
    comum = doc.get("comum") or {}
    if not isinstance(comum, dict):
        raise ValueError("'comum' deve ser um mapa")

    itens = doc.get("experimentos") or []
    if not itens:
        raise ValueError(f"{caminho} nao tem experimentos")

    saida, vistos = [], set()
    for i, e in enumerate(itens):
        nome = e.get("nome")
        if not nome:
            raise ValueError(f"experimento #{i + 1} sem 'nome'")
        if nome in vistos:
            raise ValueError(f"nome de experimento duplicado: {nome!r}")
        vistos.add(nome)

        grupo = e.get("grupo")
        if grupo not in GRUPOS:
            raise ValueError(f"{nome}: grupo={grupo!r}; use um de {list(GRUPOS)}")

        overrides = e.get("config") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"{nome}: 'config' deve ser um mapa")

        cfg = {**comum, **overrides}
        cfg["nome"] = nome
        cfg.setdefault("preset", e.get("preset", preset_padrao))

        saida.append({
            "nome": nome,
            "slot": e.get("slot", i + 1),
            "grupo": grupo,
            "eixo": e.get("eixo"),
            "hipotese": (e.get("hipotese") or "").strip() or None,
            "prioridade": e.get("prioridade", PRIORIDADE_PADRAO[grupo]),
            "config": cfg,
            "overrides": overrides,     # so o que difere de `comum`; usado na conferencia
        })

    if sum(1 for s in saida if s["grupo"] == "referencia") != 1:
        raise ValueError("a matriz precisa de exatamente um experimento com grupo=referencia")
    return saida
