#!/usr/bin/env python3
"""
Recolhe para o HD TUDO que esta maquina produziu. Rode no fim, em cada uma.

    Windows:  clique duplo em COLETAR.bat, na raiz do HD
              ou:  python E:\\Disaster-Damage\\scripts\\coletar.py

    Linux:    python3 /caminho/do/HD/Disaster-Damage/scripts/coletar.py

Destino: `colheita/<HOSTNAME>/` na RAIZ do HD — fora de `Disaster-Damage/`,
para nao se misturar com o codigo nem ser apagado por um `preparar_hd.sh`
posterior (que usa `rsync --delete` naquela pasta).

Roda a partir do HD, e nao da copia instalada na maquina: funciona igual numa
maquina instalada cedo ou tarde, sem depender da versao que ficou nela.

E idempotente e retomavel: rodar de novo so copia o que falta ou mudou de
tamanho. Escrita atomica, entao o HD arrancado no meio nao deixa arquivo
truncado se passando por bom.

O que vem:
    checkpoints/<run_id>/melhor.pt    pesos do melhor score  (~102 MB)
    checkpoints/<run_id>/ultimo.pt    + otimizador e RNG     (~306 MB)
    results/runs/*.json               manifesto de cada run
    results/history/*.jsonl           metricas de cada epoca
    results/logs/*.log                saida completa do treino
    results/configs/*.yaml            config efetiva de cada experimento
    results/spool/*.jsonl             metricas que NAO chegaram ao painel
    agente.out                        log do agente
    resumo.json                       indice legivel do que veio

`--so-melhor` deixa o ultimo.pt para tras, se o espaco apertar.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

WINDOWS = os.name == "nt"
AQUI = Path(__file__).resolve().parent
HD = AQUI.parent.parent          # .../HD/Disaster-Damage/scripts -> .../HD


def fim(codigo: int) -> "NoReturn":            # type: ignore[valid-type]
    if WINDOWS:
        input("\nEnter para fechar...")
    raise SystemExit(codigo)


def copiar(origem: Path, destino: Path) -> tuple[str, float]:
    """Devolve ('copiado'|'igual', MB). Atomico: grava .parcial e renomeia."""
    tam = origem.stat().st_size
    if destino.exists() and destino.stat().st_size == tam:
        return "igual", tam / 1e6
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".parcial")
    shutil.copy2(origem, tmp)
    os.replace(tmp, destino)
    return "copiado", tam / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origem", type=Path,
                    default=Path(os.environ.get("USERPROFILE") or Path.home())
                    / "Disaster-Damage")
    ap.add_argument("--destino", type=Path, default=HD / "colheita",
                    help="padrao: <raiz do HD>/colheita")
    ap.add_argument("--so-melhor", action="store_true",
                    help="nao leva ultimo.pt (~306 MB por run)")
    args = ap.parse_args()

    maquina = socket.gethostname()
    origem: Path = args.origem
    destino: Path = args.destino / maquina

    print("=" * 72)
    print(f"  {maquina}")
    print(f"  de:   {origem}")
    print(f"  para: {destino}")
    print("=" * 72)

    if not origem.is_dir():
        print(f"\n!! {origem} nao existe — esta maquina foi instalada?")
        return 1

    # ---- quanto vai vir, e cabe? ----
    arquivos: list[tuple[Path, Path]] = []

    for sub in ("runs", "history", "logs", "configs", "spool"):
        d = origem / "results" / sub
        if d.is_dir():
            arquivos += [(f, destino / "results" / sub / f.name)
                         for f in sorted(d.iterdir()) if f.is_file()]

    ck = origem / "checkpoints"
    if ck.is_dir():
        alvos = ["melhor.pt"] if args.so_melhor else ["melhor.pt", "ultimo.pt"]
        for run_dir in sorted(p for p in ck.iterdir() if p.is_dir()):
            arquivos += [(run_dir / n, destino / "checkpoints" / run_dir.name / n)
                         for n in alvos if (run_dir / n).is_file()]

    for extra in ("agente.out",):
        if (origem / extra).is_file():
            arquivos.append((origem / extra, destino / extra))

    if not arquivos:
        print("\n!! nada para coletar. Esta maquina chegou a treinar?")
        return 1

    precisa = sum(o.stat().st_size for o, _ in arquivos) / 1e9
    livre = shutil.disk_usage(HD).free / 1e9
    print(f"\n{len(arquivos)} arquivo(s), {precisa:.2f} GB no total")
    print(f"livre no HD: {livre:.0f} GB")
    if livre < precisa * 1.1:
        print("\n!! espaco insuficiente no HD. Use --so-melhor, ou libere espaco.")
        return 1

    # ---- copia ----
    print()
    copiados = iguais = 0
    mb_total = 0.0
    t0 = time.time()
    for i, (o, d) in enumerate(arquivos, 1):
        try:
            estado, mb = copiar(o, d)
        except OSError as exc:
            print(f"  FALHA   {o.name}: {exc}")
            continue
        if estado == "copiado":
            copiados += 1
            mb_total += mb
        else:
            iguais += 1
        if mb > 50 or i % 20 == 0 or i == len(arquivos):
            print(f"  [{i:3d}/{len(arquivos)}] {estado:8s} {mb:7.1f} MB  "
                  f"{d.relative_to(destino)}")

    # ---- indice legivel ----
    resumo, sem_subir = [], 0
    dir_runs = destino / "results" / "runs"
    if dir_runs.is_dir():
        for f in sorted(dir_runs.glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            resumo.append({
                "nome": d.get("config", {}).get("nome"),
                "run_id": d.get("run_id"),
                "status": d.get("status"),
                "epocas": d.get("epocas_concluidas"),
                "epochs_total": d.get("config", {}).get("epochs"),
                "score": d.get("melhor", {}).get("score_xview2"),
                "f1_loc": d.get("melhor", {}).get("f1_loc"),
                "f1_dano": d.get("melhor", {}).get("f1_dano_harmonico"),
                "maquina": maquina,
                "git_sha": d.get("git_sha"),
            })
    dir_spool = destino / "results" / "spool"
    if dir_spool.is_dir():
        sem_subir = sum(1 for f in dir_spool.glob("*.jsonl") if f.stat().st_size > 0)

    if resumo:
        (destino / "resumo.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- relatorio ----
    print()
    print("=" * 72)
    print(f"  {copiados} copiado(s), {iguais} ja estava(m) la, "
          f"{mb_total/1000:.2f} GB em {(time.time()-t0)/60:.1f} min")
    if resumo:
        print()
        print(f"  {'experimento':<26}{'status':<13}{'epocas':<9}{'score'}")
        print("  " + "-" * 60)
        for r in sorted(resumo, key=lambda x: x["nome"] or ""):
            sc = f"{r['score']:.4f}" if isinstance(r["score"], (int, float)) else "—"
            ep = f"{r['epocas'] or 0}/{r['epochs_total'] or '?'}"
            print(f"  {(r['nome'] or '?'):<26}{(r['status'] or '?'):<13}{ep:<9}{sc}")
    if sem_subir:
        print()
        print(f"  ATENCAO: {sem_subir} arquivo(s) de spool com conteudo.")
        print("  Sao metricas que NAO chegaram ao painel — a rede caiu antes. Elas")
        print("  vieram no pacote (results/spool/) e podem ser reenviadas depois.")
    print()
    print(f"  Tudo em: {destino}")
    print("  Desmonte o HD pelo icone da bandeja antes de tirar.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    fim(main())
