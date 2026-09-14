#!/usr/bin/env python3
"""
Confere a matriz de experimentos ANTES de ir para o laboratorio.

    .venv/bin/python scripts/validar_matriz.py

Constroi de verdade a config de cada slot com `from_preset`, exatamente como o
agente fara na maquina, e verifica o que so daria erro tarde demais:

  - campo com nome ou valor invalido (`fuse_mode: concat`)
  - dois slots com `run_id` identico — seriam o MESMO experimento com dois nomes,
    escrevendo no mesmo checkpoint e sobrescrevendo um ao outro
  - ablacao que nao muda nada em relacao a referencia (override sem efeito)
  - ablacao que muda mais de um fator, o que torna o delta nao atribuivel

Sai com codigo 1 se algo estiver errado, para poder entrar num pre-commit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from xbd.config import Config, from_preset  # noqa: E402

_spec = importlib.util.spec_from_file_location("matriz", RAIZ / "server" / "app" / "matriz.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)               # type: ignore[union-attr]
carregar_matriz = _mod.carregar_matriz

# Campos de execucao: variam por maquina de proposito e nao contam como fator.
IGNORAR = set(Config._NAO_HASH) | {"nome"}


def diferencas(a: Config, b: Config) -> dict[str, tuple]:
    da, db = a.to_dict(), b.to_dict()
    return {k: (da[k], db[k]) for k in da
            if k not in IGNORAR and da[k] != db[k]}


def main() -> int:
    caminho = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "experiments" / "matrix.yaml"
    print(f"matriz: {caminho}\n")

    try:
        slots = carregar_matriz(caminho)
    except Exception as exc:
        print(f"ERRO ao ler a matriz: {exc}")
        return 1

    problemas: list[str] = []
    configs: dict[str, Config] = {}

    for s in slots:
        overrides = dict(s["config"])
        preset = overrides.pop("preset")
        try:
            configs[s["nome"]] = from_preset(preset, **overrides)
        except Exception as exc:
            problemas.append(f"{s['nome']}: {exc}")

    if problemas:
        print("Configs invalidas:")
        for p in problemas:
            print(f"  x {p}")
        return 1

    ref_slot = next(s for s in slots if s["grupo"] == "referencia")
    ref = configs[ref_slot["nome"]]

    # run_id duplicado: dois nomes, um experimento so. O segundo a rodar retomaria
    # o checkpoint do primeiro achando que e o seu.
    por_id: dict[str, list[str]] = {}
    for nome, cfg in configs.items():
        por_id.setdefault(cfg.run_id, []).append(nome)
    for rid, nomes in por_id.items():
        if len(nomes) > 1:
            problemas.append(f"run_id {rid} compartilhado por: {', '.join(nomes)}")

    larg = max(len(s["nome"]) for s in slots) + 1
    print(f"{'#':>3} {'slot':<{larg}} {'grupo':<11} {'eixo':<15} fator alterado vs referencia")
    print("-" * (44 + larg + 34))

    for i, s in enumerate(slots, 1):
        cfg = configs[s["nome"]]
        d = diferencas(ref, cfg)
        if s["grupo"] == "referencia":
            desc = "— (referencia)"
        elif not d:
            desc = "NADA MUDA"
            problemas.append(f"{s['nome']}: identico a referencia; o slot seria desperdicado")
        else:
            desc = ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in sorted(d.items()))

        marca = " "
        if s["grupo"] == "ablacao" and len(d) > 1:
            marca = "!"
            problemas.append(
                f"{s['nome']}: muda {len(d)} fatores ({', '.join(sorted(d))}); "
                "o delta nao e atribuivel a um deles")

        print(f"{i:>3}{marca}{s['nome']:<{larg}} {s['grupo']:<11} "
              f"{(s['eixo'] or '—'):<15} {desc[:80]}")

    grupos: dict[str, int] = {}
    for s in slots:
        grupos[s["grupo"]] = grupos.get(s["grupo"], 0) + 1
    principais = sum(1 for s in slots if s["prioridade"] < 90)
    reserva = len(slots) - principais

    print(f"\n{len(slots)} slots: " + ", ".join(f"{n} {g}" for g, n in sorted(grupos.items())))
    print(f"{principais} principais + {reserva} de reserva (prioridade 90, so se sobrar maquina)")

    replicas = [s["nome"] for s in slots if s["eixo"] == "ruido"]
    if len(replicas) < 2:
        print("\nAVISO: menos de 2 replicas de semente (eixo: ruido). Sem elas nao da")
        print("       para saber se um delta pequeno e efeito real ou variacao da semente.")
    else:
        print(f"piso de ruido: referencia + {len(replicas)} replicas ({', '.join(replicas)})")

    if problemas:
        print("\nProblemas:")
        for p in dict.fromkeys(problemas):
            print(f"  x {p}")
        return 1

    print("\nOK — matriz consistente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
