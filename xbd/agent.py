"""
Agente da maquina de laboratorio: pede trabalho ao painel e executa, em laco.

    python -m xbd.agent --url https://lab.exemplo.com --token XXXX

Por que uma fila puxada e nao uma lista fixa de "maquina 7 roda o experimento 7":
com 20 maquinas, alguma vai estar ocupada, desligada ou com o disco cheio. Numa
atribuicao fixa, o experimento dessa maquina simplesmente nao acontece e so se
descobre no fim do dia. Puxando da fila, quem termina primeiro pega o proximo, e
um slot que morreu volta para a fila sozinho depois do lease vencer.

O treino roda como SUBPROCESSO, de proposito: OOM da GPU, `torch.compile`
explodindo ou o driver da NVIDIA travando matam o processo inteiro. Isolando, o
agente sobrevive, reporta o erro com as ultimas linhas do log e segue para o
proximo experimento em vez de deixar a maquina parada.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from .config import from_preset
from .reporter import Reporter, claim, worker_id

RAIZ = Path(__file__).resolve().parent.parent
WINDOWS = os.name == "nt"

# Encerrar o treino sem perder a epoca em curso depende do sinal certo, e ele
# difere por sistema. No Unix, SIGINT vira KeyboardInterrupt no filho. No
# Windows nao existe "enviar SIGINT para um processo": a unica forma de
# interromper um filho especifico e CTRL_BREAK_EVENT, e ele so alcanca o filho
# se este tiver sido criado com CREATE_NEW_PROCESS_GROUP. Sem o par
# (flag na criacao, evento no envio) so restaria matar o processo, o que
# descartaria as horas de treino desde o ultimo checkpoint.
FLAGS_CRIACAO = subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0
SINAL_PARADA = signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT


def _descrever_gpu() -> tuple[str | None, float | None]:
    try:
        from .hardware import probe
        hw = probe()
        if not hw.gpus:
            return None, None
        g = hw.gpus[0]
        sufixo = f" x{len(hw.gpus)}" if len(hw.gpus) > 1 else ""
        return f"{g.name} {g.vram_total_gb:.0f}GB{sufixo}", g.vram_total_gb
    except Exception:
        return None, None


def montar_config(spec: dict):
    """Resolve a config do experimento a partir do preset + overrides da matriz."""
    overrides = dict(spec["config"])
    preset = overrides.pop("preset", "base")
    return from_preset(preset, **overrides)      # from_preset valida os campos


class Agente:
    def __init__(self, args):
        self.args = args
        self.wid = args.worker or worker_id()
        self.gpu, self.vram = _descrever_gpu()
        self.reporter = Reporter(args.url, args.token, Path(args.results_dir) / "spool",
                                 nome_spool=f"agente-{self.wid}", verboso=args.verboso)
        self.parando = False
        self.proc: subprocess.Popen | None = None

        # SIGTERM existe no Windows mas nunca e entregue; registrar so no Unix.
        sinais = [signal.SIGINT] if WINDOWS else [signal.SIGINT, signal.SIGTERM]
        for sig in sinais:
            signal.signal(sig, self._sinal)

    def _sinal(self, signum, _frame) -> None:
        # Encerramento do dia: repassa o sinal para o treino, que salva o
        # checkpoint e marca 'interrompido'. O experimento volta para a fila e
        # retoma amanha do ponto exato em que parou.
        print(f"\n[agente] sinal {signum}: encerrando apos o run atual", flush=True)
        self.parando = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(SINAL_PARADA)
            except (OSError, ValueError) as exc:
                print(f"[agente] nao consegui interromper o treino ({exc}); encerrando", flush=True)
                self.proc.terminate()

    # ------------------------------------------------------------------ execucao

    def _pulsar(self, run_id: str, parar: threading.Event) -> None:
        while not parar.wait(30):
            self.reporter.enviar("heartbeat", run_id=run_id, worker_id=self.wid,
                                 hostname=socket.gethostname())

    def rodar_experimento(self, spec: dict) -> str:
        cfg = montar_config(spec)
        run_id = cfg.run_id

        dir_cfg = Path(self.args.results_dir) / "configs"
        dir_cfg.mkdir(parents=True, exist_ok=True)
        caminho_cfg = dir_cfg / f"{spec['nome']}.yaml"
        cfg.save(caminho_cfg)

        print("=" * 70, flush=True)
        print(f"[agente] {spec['nome']}  ({spec['grupo']}"
              + (f" / {spec['eixo']}" if spec.get("eixo") else "") + ")", flush=True)
        if spec.get("hipotese"):
            print(f"[agente] hipotese: {spec['hipotese']}", flush=True)
        print(f"[agente] run_id {run_id}"
              + ("  [retomando]" if spec.get("retomando") else ""), flush=True)
        print("=" * 70, flush=True)

        cmd = [sys.executable, "-m", "xbd.train", "--config", str(caminho_cfg),
               "--data-dir", str(self.args.data_dir),
               "--results-dir", str(self.args.results_dir),
               "--ckpt-dir", str(self.args.ckpt_dir)]
        if self.args.autotune:
            cmd.append("--autotune")

        env = {**os.environ,
               "LAB_URL": self.args.url, "LAB_TOKEN": self.args.token,
               "LAB_NOME": spec["nome"], "LAB_GRUPO": spec["grupo"],
               "LAB_EIXO": spec.get("eixo") or "", "LAB_WORKER": self.wid,
               "PYTHONUNBUFFERED": "1"}

        dir_log = Path(self.args.results_dir) / "logs"
        dir_log.mkdir(parents=True, exist_ok=True)
        log = dir_log / f"{spec['nome']}.log"

        cauda: deque[str] = deque(maxlen=120)
        parar_pulso = threading.Event()
        pulso = threading.Thread(target=self._pulsar, args=(run_id, parar_pulso), daemon=True)
        pulso.start()

        try:
            self.proc = subprocess.Popen(cmd, cwd=RAIZ, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         errors="replace", creationflags=FLAGS_CRIACAO)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"\n{'=' * 60}\n{time.strftime('%Y-%m-%d %H:%M:%S')}  {run_id}\n")
                for linha in self.proc.stdout:          # type: ignore[union-attr]
                    sys.stdout.write(linha)
                    fh.write(linha)
                    cauda.append(linha.rstrip())
            codigo = self.proc.wait()
        finally:
            parar_pulso.set()
            self.proc = None

        # 130 = o treino tratou o KeyboardInterrupt e saiu limpo. 0xC000013A e o
        # STATUS_CONTROL_C_EXIT do Windows: o Ctrl+Break derrubou o processo antes
        # de o Python tratar. Nos dois casos o checkpoint da ultima epoca esta em
        # disco, entao o desfecho e 'interrompido' e o slot volta para a fila.
        status = {0: "concluido", 130: "interrompido",
                  3221225786: "interrompido"}.get(codigo, "erro")
        # O treino ja reportou o proprio desfecho; este evento e a rede de
        # seguranca para quando ele nao conseguiu (OOM killer, kill -9, driver).
        self.reporter.enviar(
            "fim", run_id=run_id, nome=spec["nome"], grupo=spec["grupo"],
            eixo=spec.get("eixo"), status=status,
            erro=None if status == "concluido" else "\n".join(cauda)[-2000:])
        print(f"[agente] {spec['nome']}: {status} (codigo {codigo}); log em {log}", flush=True)
        return status

    # ---------------------------------------------------------------------- laco

    def laco(self) -> int:
        print(f"[agente] worker {self.wid}  gpu {self.gpu or 'nenhuma'}  -> {self.args.url}",
              flush=True)
        ocioso = 0
        while not self.parando:
            try:
                spec = claim(self.args.url, self.args.token, {
                    "worker_id": self.wid, "hostname": socket.gethostname(),
                    "gpu": self.gpu, "vram_gb": self.vram,
                })
            except Exception as exc:
                print(f"[agente] painel inacessivel ({exc}); nova tentativa em 30s", flush=True)
                time.sleep(30)
                continue

            if spec is None:
                if self.args.sair_vazio:
                    print("[agente] fila vazia; saindo (--sair-vazio)", flush=True)
                    break
                ocioso += 1
                if ocioso == 1:
                    print(f"[agente] fila vazia; verificando a cada {self.args.intervalo}s",
                          flush=True)
                time.sleep(self.args.intervalo)
                continue

            ocioso = 0
            try:
                self.rodar_experimento(spec)
            except Exception as exc:
                print(f"[agente] falha ao executar {spec.get('nome')}: {exc}", flush=True)
                time.sleep(10)
            if self.args.uma_vez:
                break

        self.reporter.parar(timeout=30)
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("LAB_URL", ""),
                    help="URL do painel (ou LAB_URL)")
    ap.add_argument("--token", default=os.environ.get("LAB_TOKEN", ""),
                    help="token do painel (ou LAB_TOKEN)")
    ap.add_argument("--worker", default=os.environ.get("LAB_WORKER"),
                    help="identidade desta maquina (padrao: hostname)")
    ap.add_argument("--data-dir", type=Path, default=RAIZ / "data")
    ap.add_argument("--results-dir", type=Path, default=RAIZ / "results")
    ap.add_argument("--ckpt-dir", type=Path, default=RAIZ / "checkpoints")
    ap.add_argument("--intervalo", type=float, default=60, help="espera com a fila vazia")
    ap.add_argument("--uma-vez", action="store_true", help="roda um experimento e sai")
    ap.add_argument("--sair-vazio", action="store_true", help="sai quando a fila esvaziar")
    ap.add_argument("--autotune", action="store_true", help="repassa --autotune ao treino")
    ap.add_argument("--verboso", action="store_true")
    args = ap.parse_args(argv)

    if not args.url or not args.token:
        ap.error("informe --url e --token (ou LAB_URL e LAB_TOKEN)")
    args.url = args.url.rstrip("/")

    return Agente(args).laco()


if __name__ == "__main__":
    raise SystemExit(main())
