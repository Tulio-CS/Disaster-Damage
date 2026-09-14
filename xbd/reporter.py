"""
Cliente HTTP do painel, com fila em disco.

Premissa de projeto: **a rede nunca pode derrubar um treino**. O painel roda numa
VPS, fora da rede do laboratorio; DNS cair, wifi oscilar ou a VPS reiniciar sao
eventos esperados num dia de 8 horas, e nenhum deles pode custar 6 horas de GPU.

Por isso todo evento e primeiro gravado num spool JSONL local e so depois enviado
por um thread separado. `enviar()` faz uma escrita em arquivo e retorna; nunca
faz I/O de rede na thread do treino, e nunca levanta excecao.

Drenagem sem perda nem duplicata: o spool e RENOMEADO antes do POST, entao
escritas novas caem num arquivo limpo enquanto o lote antigo esta em voo. Se o
POST falhar, o lote volta para a frente da fila; se o processo morrer no meio, o
arquivo em voo continua no disco e e recuperado no proximo inicio. O servidor
trata reentrega como caso normal, entao um lote enviado duas vezes e inofensivo.

So usa a biblioteca padrao: a maquina do laboratorio nao precisa de `requests`.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TIMEOUT = 20
LOTE_MAX = 200


def _post(url: str, token: str, corpo: dict, timeout: int = TIMEOUT) -> tuple[int, bytes]:
    dados = json.dumps(corpo, ensure_ascii=False, default=str).encode()
    req = urllib.request.Request(url, data=dados, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def claim(base_url: str, token: str, payload: dict, timeout: int = TIMEOUT) -> dict | None:
    """Pede um experimento. `None` quando a fila acabou (204)."""
    status, corpo = _post(base_url.rstrip("/") + "/api/claim", token, payload, timeout)
    if status == 204 or not corpo:
        return None
    return json.loads(corpo)


class Reporter:
    def __init__(self, base_url: str, token: str, spool_dir: str | Path,
                 nome_spool: str = "spool", intervalo: float = 5.0,
                 verboso: bool = False):
        self.url = base_url.rstrip("/") + "/api/eventos"
        self.token = token
        self.intervalo = intervalo
        self.verboso = verboso

        self.dir = Path(spool_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spool = self.dir / f"{nome_spool}.jsonl"
        self.em_voo = self.dir / f"{nome_spool}.em-voo.jsonl"

        self._lock = threading.Lock()
        self._acordar = threading.Event()
        self._parar = threading.Event()
        self.pendentes = 0
        self.ultimo_erro: str | None = None
        self.enviados = 0

        self._recuperar()
        self._thread = threading.Thread(target=self._laco, name="reporter", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ escrita

    def enviar(self, tipo: str, **campos: Any) -> None:
        """Enfileira um evento. Nao bloqueia e nao levanta."""
        ev = {"tipo": tipo, "ts": time.time(), **campos}
        try:
            with self._lock:
                with open(self.spool, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
                self.pendentes += 1
        except Exception as exc:                       # disco cheio, permissao...
            self.ultimo_erro = f"spool: {exc}"
            return
        self._acordar.set()

    # ---------------------------------------------------------------- drenagem

    def _recuperar(self) -> None:
        """Lote que ficou em voo quando o processo morreu volta para a frente da fila."""
        if not self.em_voo.exists():
            return
        try:
            self._devolver()
        except Exception as exc:
            self.ultimo_erro = f"recuperar: {exc}"

    def _devolver(self) -> None:
        """Concatena o lote em voo na FRENTE do spool, preservando a ordem."""
        with self._lock:
            resto = self.spool.read_text(encoding="utf-8") if self.spool.exists() else ""
            self.spool.write_text(self.em_voo.read_text(encoding="utf-8") + resto,
                                  encoding="utf-8")
            self.em_voo.unlink(missing_ok=True)

    def _drenar_uma_vez(self) -> bool:
        """Devolve True se havia algo para enviar e o envio funcionou."""
        with self._lock:
            if not self.spool.exists() or self.spool.stat().st_size == 0:
                return False
            os.replace(self.spool, self.em_voo)        # atomico: escritas novas
                                                       # ja caem num spool limpo

        linhas = [l for l in self.em_voo.read_text(encoding="utf-8").splitlines() if l.strip()]
        eventos = []
        for l in linhas:
            try:
                eventos.append(json.loads(l))
            except json.JSONDecodeError:
                pass                                    # linha truncada por kill -9

        try:
            for i in range(0, len(eventos), LOTE_MAX):
                status, _ = _post(self.url, self.token, {"eventos": eventos[i:i + LOTE_MAX]})
                if status >= 400:
                    raise urllib.error.HTTPError(self.url, status, "recusado", None, None)
        except Exception as exc:
            self.ultimo_erro = str(exc)[:200]
            self._devolver()
            return False

        self.em_voo.unlink(missing_ok=True)
        with self._lock:
            self.pendentes = max(0, self.pendentes - len(eventos))
        self.enviados += len(eventos)
        self.ultimo_erro = None
        if self.verboso:
            print(f"[reporter] {len(eventos)} evento(s) enviados", flush=True)
        return True

    def _laco(self) -> None:
        espera = self.intervalo
        while not self._parar.is_set():
            self._acordar.wait(timeout=espera)
            self._acordar.clear()
            try:
                ok = self._drenar_uma_vez()
            except Exception as exc:                    # nunca deixa o thread morrer
                self.ultimo_erro = str(exc)[:200]
                ok = False
            # recuo exponencial ate 2 min: com a VPS fora do ar, 20 maquinas
            # tentando a cada 5 s so gastariam CPU e encheriam log
            espera = self.intervalo if ok else min(120.0, max(self.intervalo, espera * 1.8))

    # ------------------------------------------------------------------ fim

    def parar(self, timeout: float = 30.0) -> bool:
        """Tenta esvaziar o spool antes de sair. True se conseguiu esvaziar tudo."""
        limite = time.time() + timeout
        while time.time() < limite:
            try:
                if not self._drenar_uma_vez():
                    break
            except Exception:
                break
            time.sleep(0.2)
        self._parar.set()
        self._acordar.set()
        vazio = not self.spool.exists() or self.spool.stat().st_size == 0
        if not vazio:
            print(f"[reporter] {self.pendentes} evento(s) ficaram em {self.spool}; "
                  f"sobem no proximo run desta maquina", flush=True)
        return vazio


# ---------------------------------------------------------------------- fabrica


def do_ambiente(spool_dir: str | Path, nome_spool: str = "spool") -> Reporter | None:
    """Constroi o Reporter a partir de LAB_URL/LAB_TOKEN. `None` se nao configurado.

    Deixar o treino rodar sem painel e proposital: `python -m xbd.train` numa
    maquina solta continua funcionando exatamente como antes.
    """
    url, token = os.environ.get("LAB_URL", "").strip(), os.environ.get("LAB_TOKEN", "").strip()
    if not url or not token:
        return None
    return Reporter(url, token, spool_dir, nome_spool=nome_spool,
                    verboso=os.environ.get("LAB_VERBOSO") == "1")


def worker_id() -> str:
    return os.environ.get("LAB_WORKER", "").strip() or socket.gethostname()
