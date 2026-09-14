#!/usr/bin/env python3
"""
Recolhe para o HD o que esta maquina produziu. Rode no fim do dia, em cada uma.

    Windows:  python E:\\Disaster-Damage\\scripts\\coletar.py
              (ou clique duplo em COLETAR.bat, na raiz do HD)

    Linux:    python3 /caminho/do/HD/Disaster-Damage/scripts/coletar.py

Roda a partir do HD de proposito, e nao da copia instalada na maquina: assim
funciona igual numa maquina instalada cedo ou tarde, sem depender da versao do
codigo que ficou nela.

Leva os pesos do melhor checkpoint, os manifestos, o historico por epoca e os
logs. NAO leva `ultimo.pt` (3x maior, e so serve para retomar naquela mesma
maquina) — use --com-ultimo se quiser continuar o treino em outro lugar.

E idempotente: rodar de novo so copia o que mudou.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from pathlib import Path

WINDOWS = os.name == "nt"
AQUI = Path(__file__).resolve().parent
HD = AQUI.parent.parent                       # .../HD/Disaster-Damage/scripts -> .../HD


def mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def copiar(origem: Path, destino: Path) -> tuple[bool, float]:
    """Copia se faltar ou se o tamanho diferir. Devolve (copiou, MB)."""
    tam = mb(origem)
    if destino.exists() and destino.stat().st_size == origem.stat().st_size:
        return False, tam
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    shutil.copy2(origem, tmp)
    os.replace(tmp, destino)                  # atomico: HD arrancado nao deixa
    return True, tam                          # arquivo truncado para tras


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origem", type=Path,
                    default=Path(os.environ.get("USERPROFILE") or Path.home()) / "Disaster-Damage")
    ap.add_argument("--destino", type=Path, default=HD / "colheita")
    ap.add_argument("--com-ultimo", action="store_true",
                    help="leva tambem ultimo.pt (~306 MB por run) para retomar alhures")
    args = ap.parse_args()

    maquina = socket.gethostname()
    origem, destino = args.origem, args.destino / maquina

    print("=" * 70)
    print(f"  {maquina}")
    print(f"  de:   {origem}")
    print(f"  para: {destino}")
    print("=" * 70)

    if not origem.is_dir():
        print(f"\n!! {origem} nao existe. Esta maquina nao foi instalada?")
        if WINDOWS:
            input("\nEnter para fechar...")
        return 1

    livre = shutil.disk_usage(HD).free / 1e9
    print(f"\nespaco livre no HD: {livre:.0f} GB")

    total_mb = novos = 0

    # ---- manifestos, historico e logs: pequenos, levam tudo ----
    print("\n[1/2] resultados e logs...")
    for sub in ("runs", "history", "logs", "configs"):
        d = origem / "results" / sub
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file():
                fez, m = copiar(f, destino / "results" / sub / f.name)
                total_mb += m if fez else 0
                novos += fez
    print(f"      {novos} arquivo(s) novo(s)")

    # ---- checkpoints ----
    print("\n[2/2] checkpoints...")
    ck = origem / "checkpoints"
    if not ck.is_dir():
        print("      (nenhum)")
    else:
        for run_dir in sorted(p for p in ck.iterdir() if p.is_dir()):
            alvos = ["melhor.pt"] + (["ultimo.pt"] if args.com_ultimo else [])
            for nome in alvos:
                f = run_dir / nome
                if not f.is_file():
                    continue
                fez, m = copiar(f, destino / "checkpoints" / run_dir.name / nome)
                total_mb += m if fez else 0
                novos += fez
                print(f"      {'copiado ' if fez else 'ja tinha'}  {run_dir.name}/{nome}"
                      f"  {m:.0f} MB")

    # ---- indice legivel do que veio ----
    resumo = []
    for f in sorted((destino / "results" / "runs").glob("*.json")) \
            if (destino / "results" / "runs").is_dir() else []:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        resumo.append({
            "run_id": d.get("run_id"), "nome": d.get("config", {}).get("nome"),
            "status": d.get("status"), "epocas": d.get("epocas_concluidas"),
            "score": d.get("melhor", {}).get("score_xview2"), "maquina": maquina,
        })
    if resumo:
        (destino / "resumo.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"  {novos} arquivo(s) novo(s), {total_mb/1000:.2f} GB copiados")
    if resumo:
        print()
        print(f"  {'experimento':<26}{'status':<13}{'ep':<5}score")
        for r in sorted(resumo, key=lambda x: x["nome"] or ""):
            sc = f"{r['score']:.4f}" if isinstance(r["score"], (int, float)) else "—"
            print(f"  {(r['nome'] or '?'):<26}{(r['status'] or '?'):<13}"
                  f"{str(r['epocas'] or 0):<5}{sc}")
    print()
    print("  Desmonte o HD pelo icone da bandeja antes de tirar.")
    print("=" * 70)
    if WINDOWS:
        input("\nEnter para fechar...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
