#!/usr/bin/env python3
"""
Mantem o agente desta maquina rodando, reiniciando se ele cair. Windows ou Linux.

    python scripts/rodar_agente.py

Normalmente e disparado por `scripts/instalar.py`, mas serve para religar uma
maquina que foi desligada:

    Windows:  cd %USERPROFILE%\\Disaster-Damage
              .venv\\Scripts\\python.exe scripts\\rodar_agente.py

    Linux:    cd ~/Disaster-Damage && .venv/bin/python scripts/rodar_agente.py

O laco cobre falhas do PROPRIO agente (rede caiu durante um claim, disco cheio).
Falha do treino nao chega aqui: o agente ja isola o treino num subprocesso,
reporta o erro e segue para o proximo experimento.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
WINDOWS = os.name == "nt"
ESPERA_SEG = 30

parar = False
filho: subprocess.Popen | None = None


def _sinal(signum, _frame):
    global parar
    parar = True
    print(f"\n[worker] sinal {signum}: encerrando...", flush=True)
    if filho and filho.poll() is None:
        # Repassa para o agente, que repassa para o treino, que salva o
        # checkpoint e marca 'interrompido'. O slot volta para a fila e retoma
        # depois, nesta mesma maquina, da epoca exata em que parou.
        try:
            filho.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT)
        except (OSError, ValueError):
            filho.terminate()


def main() -> int:
    global filho

    for sig in ([signal.SIGINT] if WINDOWS else [signal.SIGINT, signal.SIGTERM]):
        signal.signal(sig, _sinal)

    if not os.environ.get("LAB_URL") or not os.environ.get("LAB_TOKEN"):
        print("! LAB_URL e LAB_TOKEN precisam estar no ambiente")
        return 1

    os.environ.setdefault("LAB_WORKER", os.environ.get("COMPUTERNAME") or "")
    print(f"[worker] {os.environ.get('LAB_WORKER') or 'esta maquina'} "
          f"-> {os.environ['LAB_URL']}", flush=True)

    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {}

    while not parar:
        filho = subprocess.Popen([sys.executable, "-m", "xbd.agent"],
                                 cwd=str(RAIZ), **flags)
        codigo = filho.wait()
        filho = None

        if parar:
            break
        if codigo == 0:
            print("[worker] agente saiu normalmente (fila vazia); encerrando.", flush=True)
            break
        print(f"[worker] agente caiu (codigo {codigo}); reiniciando em {ESPERA_SEG}s",
              flush=True)
        for _ in range(ESPERA_SEG):
            if parar:
                break
            time.sleep(1)

    print("[worker] fim.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
