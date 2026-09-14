#!/usr/bin/env python3
"""
Religa o agente desta maquina. Use depois de um reinicio ou de matar o processo.

    Windows:  clique duplo em RELIGAR.bat, na raiz do HD
    Linux:    python3 /caminho/do/HD/Disaster-Damage/scripts/religar.py

Roda a partir do HD, e nao da copia instalada: funciona igual em qualquer
maquina, sem depender da versao que ficou nela.

Nao reinstala nada — os dados e o venv da maquina continuam validos. Retoma o
experimento do checkpoint local, da epoca exata em que parou.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

WINDOWS = os.name == "nt"
AQUI = Path(__file__).resolve().parent
HD = AQUI.parent.parent
DESTINO = Path(os.environ.get("USERPROFILE") or Path.home()) / "Disaster-Damage"


def fim(codigo: int) -> "NoReturn":            # type: ignore[valid-type]
    if WINDOWS:
        input("\nEnter para fechar...")
    raise SystemExit(codigo)


def main() -> int:
    print("=" * 70)
    print(f"  {socket.gethostname()}  —  religando o agente")
    print("=" * 70)

    pyv = DESTINO / (".venv/Scripts/python.exe" if WINDOWS else ".venv/bin/python")
    if not pyv.is_file():
        print(f"\n!! {pyv} nao existe.")
        print("   Esta maquina nunca foi instalada — use INSTALAR.bat.")
        return 1

    # ---- credenciais do HD ----
    env = {}
    arq = HD / "lab.env"
    if arq.is_file():
        for linha in arq.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if linha and not linha.startswith("#") and "=" in linha:
                k, v = linha.split("=", 1)
                env[k.strip()] = v.strip()
    url = os.environ.get("LAB_URL") or env.get("LAB_URL", "")
    token = os.environ.get("LAB_TOKEN") or env.get("LAB_TOKEN", "")
    if not url or not token or token == "COLE_O_TOKEN_AQUI":
        print(f"\n!! LAB_URL/LAB_TOKEN ausentes. Edite {arq}.")
        return 1

    # ---- ja esta rodando? ----
    try:
        if WINDOWS:
            saida = subprocess.run(["wmic", "process", "where",
                                    "name='python.exe'", "get", "CommandLine"],
                                   capture_output=True, text=True, timeout=20).stdout
        else:
            saida = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                                   text=True, timeout=20).stdout
        if "xbd.agent" in saida or "rodar_agente" in saida:
            print("\n  Ja existe um agente rodando nesta maquina. Nada a fazer.")
            print(f"  Acompanhe:  {DESTINO / 'agente.out'}")
            return 0
    except (OSError, subprocess.SubprocessError):
        pass                                   # sem deteccao confiavel; segue e sobe

    # ---- sobe ----
    saida_log = DESTINO / "agente.out"
    ambiente = {**os.environ, "LAB_URL": url, "LAB_TOKEN": token,
                "LAB_WORKER": socket.gethostname(), "PYTHONUNBUFFERED": "1"}
    for k, v in env.items():                   # LAB_NO_COMPILE e afins
        ambiente.setdefault(k, v)

    flags = ({"creationflags": subprocess.DETACHED_PROCESS
              | subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS
             else {"start_new_session": True})

    with open(saida_log, "ab") as fh:
        subprocess.Popen([str(pyv), str(DESTINO / "scripts" / "rodar_agente.py")],
                         cwd=str(DESTINO), env=ambiente, stdout=fh,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **flags)
    time.sleep(8)

    print(f"\n  agente disparado")
    print(f"  acompanhe:  Get-Content -Wait -Tail 30 {saida_log}")
    print(f"  painel:     {url}")
    print("\n  O experimento retoma do checkpoint local, da epoca em que parou.")
    return 0


if __name__ == "__main__":
    fim(main())
