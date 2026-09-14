"""
Registro de runs pensado para juntar resultados de varias maquinas sem conflito.

Regra central: **um arquivo por run, nunca um arquivo compartilhado.** Um
`results.csv` unico garante conflito de merge no git toda vez que duas maquinas
escrevem. Com `results/runs/<run_id>.json`, cada maquina so cria arquivos novos,
entao `git pull --rebase` sempre resolve sozinho.

O historico por epoca vai em JSONL append-only ao lado, pelo mesmo motivo: linhas
sao acrescentadas, nunca reescritas, e um `cat` de varios arquivos e um merge
valido.

Quando ha um painel configurado, o mesmo `Run` espelha cada evento para ele
atraves de um `Reporter` (ver `xbd/reporter.py`). O arquivo local continua sendo
a fonte da verdade: o envio e assincrono, tolera a rede cair e nunca interrompe o
treino. Sem `LAB_URL`/`LAB_TOKEN` no ambiente, nada muda.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path


def git_sha(curto: bool = True) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short" if curto else "HEAD", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def git_sujo() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, timeout=5, check=True)
        return bool(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return False


class Run:
    """Escreve o manifesto e o historico de um run, e opcionalmente espelha no painel."""

    def __init__(self, cfg, hardware, plano, results_dir: str | Path | None = None,
                 reporter=None):
        self.cfg = cfg
        self.run_id = cfg.run_id
        self.reporter = reporter
        base = Path(results_dir or cfg.results_dir)
        self.dir_runs = base / "runs"
        self.dir_hist = base / "history"
        self.dir_runs.mkdir(parents=True, exist_ok=True)
        self.dir_hist.mkdir(parents=True, exist_ok=True)

        self.path_manifesto = self.dir_runs / f"{self.run_id}.json"
        self.path_hist = self.dir_hist / f"{self.run_id}.jsonl"

        self.manifesto = {
            "run_id": self.run_id,
            "config_hash": cfg.config_hash(),
            "config": cfg.to_dict(),
            "hardware": {
                "hostname": hardware.hostname,
                "cpu": hardware.cpu_model,
                "cpu_logical": hardware.cpu_logical,
                "ram_gb": round(hardware.ram_total_gb, 1),
                "gpus": [f"{g.name} {g.vram_total_gb:.0f}GB" for g in hardware.gpus],
                "torch": hardware.torch_version,
                "cuda": hardware.cuda_version,
            },
            "plano": plano.to_dict(),
            "git_sha": git_sha(),
            "git_sujo": git_sujo(),
            "python": platform.python_version(),
            "iniciado_em": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "rodando",
            "epocas_concluidas": 0,
            "melhor": {},
            "pid": os.getpid(),
            "fqdn": socket.getfqdn(),
        }
        self._salvar()

        # Identidade do slot na matriz: o worker passa pelo ambiente para que o
        # painel saiba a qual experimento este run pertence. Sem isso o run
        # aparece, mas orfao de grupo/eixo.
        self._rotulo = {
            "run_id": self.run_id,
            "nome": os.environ.get("LAB_NOME") or cfg.nome,
            "grupo": os.environ.get("LAB_GRUPO") or "avulso",
            "eixo": os.environ.get("LAB_EIXO") or None,
        }
        self._remoto("inicio", worker=os.environ.get("LAB_WORKER") or hardware.hostname,
                     hostname=hardware.hostname, config=cfg.to_dict(),
                     hardware=self.manifesto["hardware"], plano=plano.to_dict(),
                     epochs_total=cfg.epochs)

    def _remoto(self, tipo: str, **campos) -> None:
        if self.reporter is not None:
            self.reporter.enviar(tipo, **self._rotulo, **campos)

    def _salvar(self) -> None:
        tmp = self.path_manifesto.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifesto, indent=2, ensure_ascii=False, default=str))
        os.replace(tmp, self.path_manifesto)

    def log_epoca(self, epoca: int, metricas: dict) -> None:
        linha = {"run_id": self.run_id, "epoca": epoca,
                 "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **metricas}
        with open(self.path_hist, "a") as fh:      # append-only: merge por concatenacao
            fh.write(json.dumps(linha, ensure_ascii=False, default=str) + "\n")
        self.manifesto["epocas_concluidas"] = epoca + 1
        self._salvar()
        self._remoto("epoca", epoca=epoca, metricas=metricas)

    def registrar_melhor(self, metricas: dict, ckpt: str | Path | None = None) -> None:
        self.manifesto["melhor"] = {**metricas, "checkpoint": str(ckpt) if ckpt else None}
        self._salvar()
        self._remoto("melhor", metricas=metricas)

    def finalizar(self, status: str = "concluido", erro: str | None = None) -> None:
        self.manifesto["status"] = status
        self.manifesto["terminado_em"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if erro:
            self.manifesto["erro"] = erro[:2000]
        self._salvar()
        self._remoto("fim", status=status, erro=erro[:2000] if erro else None,
                     config=self.cfg.to_dict(), melhor=self.manifesto.get("melhor") or {})
