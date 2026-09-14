"""
Estado do laboratorio em SQLite.

Por que SQLite e nao arquivos soltos como em `xbd/runs.py`: la o objetivo era
evitar conflito de merge no git entre maquinas que nao se falam. Aqui e o
oposto — existe um unico escritor (este processo) e o que se precisa e de uma
**fila com posse exclusiva**: 20 maquinas pedindo trabalho ao mesmo tempo nao
podem receber o mesmo experimento. `BEGIN IMMEDIATE` resolve isso em uma linha;
com arquivos seria preciso inventar lock.

Todo evento vindo do worker e idempotente por chave natural ((run_id, epoca),
run_id, nome do experimento). O worker reenvia o que ficou no spool quando a
rede volta, entao reentrega e o caso normal, nao a excecao.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

ESQUEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS experimentos (
    nome          TEXT PRIMARY KEY,
    slot          INTEGER,
    grupo         TEXT NOT NULL,          -- referencia | baseline | ablacao
    eixo          TEXT,                   -- fator alterado (agrupa a analise)
    hipotese      TEXT,
    prioridade    INTEGER DEFAULT 100,
    config        TEXT NOT NULL,          -- JSON dos overrides sobre o preset
    status        TEXT DEFAULT 'pendente',-- pendente|rodando|concluido|erro
    run_id        TEXT,
    worker        TEXT,
    atribuido_em  REAL,
    tentativas    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    nome          TEXT,
    grupo         TEXT,
    eixo          TEXT,
    worker        TEXT,
    hostname      TEXT,
    config        TEXT,
    hardware      TEXT,
    plano         TEXT,
    status        TEXT DEFAULT 'rodando',
    epocas        INTEGER DEFAULT 0,
    epochs_total  INTEGER,
    melhor        TEXT,
    erro          TEXT,
    iniciado_em   REAL,
    atualizado_em REAL,
    terminado_em  REAL
);

CREATE TABLE IF NOT EXISTS epocas (
    run_id   TEXT NOT NULL,
    epoca    INTEGER NOT NULL,
    t        REAL,
    metricas TEXT,
    PRIMARY KEY (run_id, epoca)
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    hostname  TEXT,
    gpu       TEXT,
    info      TEXT,
    run_id    TEXT,
    status    TEXT,
    visto_em  REAL
);

CREATE INDEX IF NOT EXISTS idx_epocas_run ON epocas(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


def _dj(s: str | None, padrao: Any = None) -> Any:
    if not s:
        return padrao
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return padrao


class Store:
    def __init__(self, caminho: str | Path):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.con = sqlite3.connect(self.caminho, check_same_thread=False, timeout=10)
        self.con.row_factory = sqlite3.Row
        with self._lock:
            self.con.executescript(ESQUEMA)
            self.con.commit()

    # ------------------------------------------------------------------ matriz

    def sincronizar_matriz(self, experimentos: list[dict]) -> dict:
        """Insere experimentos novos sem tocar no progresso dos que ja existem.

        Reimportar a matriz e uma operacao de rotina (mudou uma hipotese no meio
        do dia, acrescentou um slot), entao ela nunca pode zerar um experimento
        ja rodando. So a config de quem esta `pendente` e atualizada.
        """
        novos = atualizados = 0
        with self._lock, self.con:
            for e in experimentos:
                atual = self.con.execute(
                    "SELECT status FROM experimentos WHERE nome=?", (e["nome"],)).fetchone()
                if atual is None:
                    self.con.execute(
                        "INSERT INTO experimentos (nome, slot, grupo, eixo, hipotese, "
                        "prioridade, config) VALUES (?,?,?,?,?,?,?)",
                        (e["nome"], e.get("slot"), e["grupo"], e.get("eixo"),
                         e.get("hipotese"), e.get("prioridade", 100), _j(e["config"])))
                    novos += 1
                elif atual["status"] == "pendente":
                    self.con.execute(
                        "UPDATE experimentos SET slot=?, grupo=?, eixo=?, hipotese=?, "
                        "prioridade=?, config=? WHERE nome=?",
                        (e.get("slot"), e["grupo"], e.get("eixo"), e.get("hipotese"),
                         e.get("prioridade", 100), _j(e["config"]), e["nome"]))
                    atualizados += 1
        return {"novos": novos, "atualizados": atualizados, "total": len(experimentos)}

    # ------------------------------------------------------------------- fila

    def reivindicar(self, worker_id: str, hostname: str, vram_gb: float | None) -> dict | None:
        """Entrega um experimento ao worker, em posse exclusiva.

        Prioridade de escolha:
          1. Um experimento que JA e deste worker e nao terminou. Os checkpoints
             ficam no disco local da maquina, entao devolver o mesmo trabalho
             para a mesma maquina permite retomar de onde parou; entregar para
             outra jogaria fora todas as epocas ja treinadas.
          2. Menor `prioridade`, depois menor `slot`.
        """
        agora = time.time()
        with self._lock:
            self.con.execute("BEGIN IMMEDIATE")
            try:
                linha = self.con.execute(
                    "SELECT * FROM experimentos WHERE worker=? AND status='rodando' "
                    "ORDER BY prioridade, slot LIMIT 1", (worker_id,)).fetchone()
                if linha is None:
                    linha = self.con.execute(
                        "SELECT * FROM experimentos WHERE status='pendente' "
                        "ORDER BY prioridade, slot LIMIT 1").fetchone()
                if linha is None:
                    self.con.commit()
                    return None

                retomando = linha["worker"] == worker_id and linha["run_id"] is not None
                self.con.execute(
                    "UPDATE experimentos SET status='rodando', worker=?, atribuido_em=?, "
                    "tentativas=tentativas+? WHERE nome=?",
                    (worker_id, agora, 0 if retomando else 1, linha["nome"]))
                self.con.commit()
            except Exception:
                self.con.rollback()
                raise

        return {
            "nome": linha["nome"],
            "slot": linha["slot"],
            "grupo": linha["grupo"],
            "eixo": linha["eixo"],
            "hipotese": linha["hipotese"],
            "config": _dj(linha["config"], {}),
            "retomando": bool(retomando),
            "tentativas": linha["tentativas"] + (0 if retomando else 1),
        }

    def liberar(self, nome: str, status: str = "pendente") -> None:
        """Devolve o experimento para a fila (ou marca o desfecho final)."""
        with self._lock, self.con:
            if status == "pendente":
                self.con.execute(
                    "UPDATE experimentos SET status='pendente', worker=NULL, run_id=NULL "
                    "WHERE nome=?", (nome,))
            else:
                self.con.execute("UPDATE experimentos SET status=? WHERE nome=?", (status, nome))

    # ----------------------------------------------------------------- eventos

    def registrar_worker(self, worker_id: str, hostname: str, gpu: str | None,
                         info: dict | None, run_id: str | None, status: str) -> None:
        with self._lock, self.con:
            self.con.execute(
                "INSERT INTO workers (worker_id, hostname, gpu, info, run_id, status, visto_em) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(worker_id) DO UPDATE SET "
                "hostname=excluded.hostname, gpu=COALESCE(excluded.gpu, workers.gpu), "
                "info=COALESCE(excluded.info, workers.info), "
                "run_id=excluded.run_id, status=excluded.status, visto_em=excluded.visto_em",
                (worker_id, hostname, gpu, _j(info) if info else None, run_id, status, time.time()))

    def inicio_run(self, ev: dict) -> None:
        agora = time.time()
        with self._lock, self.con:
            self.con.execute(
                "INSERT INTO runs (run_id, nome, grupo, eixo, worker, hostname, config, "
                "hardware, plano, status, epochs_total, iniciado_em, atualizado_em) "
                "VALUES (?,?,?,?,?,?,?,?,?,'rodando',?,?,?) "
                "ON CONFLICT(run_id) DO UPDATE SET status='rodando', worker=excluded.worker, "
                "hostname=excluded.hostname, hardware=excluded.hardware, plano=excluded.plano, "
                "atualizado_em=excluded.atualizado_em, erro=NULL",
                (ev["run_id"], ev.get("nome"), ev.get("grupo"), ev.get("eixo"),
                 ev.get("worker"), ev.get("hostname"), _j(ev.get("config")),
                 _j(ev.get("hardware")), _j(ev.get("plano")), ev.get("epochs_total"),
                 agora, agora))
            if ev.get("nome"):
                self.con.execute(
                    "UPDATE experimentos SET run_id=?, status='rodando' WHERE nome=?",
                    (ev["run_id"], ev["nome"]))

    def epoca(self, ev: dict) -> bool:
        """Grava uma epoca. Devolve True se a linha e NOVA.

        O retorno importa: o TensorBoard so pode receber o escalar se a epoca for
        inedita, senao um reenvio do spool desenharia o mesmo ponto duas vezes na
        curva.
        """
        with self._lock, self.con:
            cur = self.con.execute(
                "INSERT OR IGNORE INTO epocas (run_id, epoca, t, metricas) VALUES (?,?,?,?)",
                (ev["run_id"], int(ev["epoca"]), ev.get("ts", time.time()),
                 _j(ev.get("metricas", {}))))
            nova = cur.rowcount > 0
            self.con.execute(
                "UPDATE runs SET epocas=(SELECT COUNT(*) FROM epocas WHERE run_id=?), "
                "atualizado_em=? WHERE run_id=?",
                (ev["run_id"], time.time(), ev["run_id"]))
        return nova

    def melhor(self, ev: dict) -> None:
        with self._lock, self.con:
            self.con.execute(
                "UPDATE runs SET melhor=?, atualizado_em=? WHERE run_id=?",
                (_j(ev.get("metricas", {})), time.time(), ev["run_id"]))

    def heartbeat(self, run_id: str) -> None:
        with self._lock, self.con:
            self.con.execute("UPDATE runs SET atualizado_em=? WHERE run_id=?",
                             (time.time(), run_id))

    def fim_run(self, ev: dict) -> None:
        agora = time.time()
        status = ev.get("status", "concluido")
        with self._lock, self.con:
            self.con.execute(
                "UPDATE runs SET status=?, erro=?, terminado_em=?, atualizado_em=? "
                "WHERE run_id=?", (status, ev.get("erro"), agora, agora, ev["run_id"]))
            linha = self.con.execute("SELECT nome FROM runs WHERE run_id=?",
                                     (ev["run_id"],)).fetchone()
            if linha and linha["nome"]:
                # 'interrompido' volta para a fila: a maquina foi desligada no fim
                # do dia e o treino retoma do checkpoint amanha. 'erro' fica parado
                # para inspecao manual — reenfileirar so repetiria a mesma falha.
                novo = {"concluido": "concluido", "erro": "erro"}.get(status, "pendente")
                self.liberar(linha["nome"], novo)

    def marcar_perdidos(self, limite_seg: float) -> int:
        """Runs sem heartbeat viram 'perdido' e o experimento volta para a fila."""
        corte = time.time() - limite_seg
        with self._lock, self.con:
            perdidos = self.con.execute(
                "SELECT run_id, nome FROM runs WHERE status='rodando' AND atualizado_em < ?",
                (corte,)).fetchall()
            for r in perdidos:
                self.con.execute("UPDATE runs SET status='perdido' WHERE run_id=?", (r["run_id"],))
                if r["nome"]:
                    self.con.execute(
                        "UPDATE experimentos SET status='pendente' WHERE nome=? AND run_id=?",
                        (r["nome"], r["run_id"]))
        return len(perdidos)

    # ------------------------------------------------------------------ leitura

    def estado(self, limite_seg: float) -> dict:
        agora = time.time()
        with self._lock:
            exps = self.con.execute(
                "SELECT * FROM experimentos ORDER BY prioridade, slot").fetchall()
            runs = self.con.execute("SELECT * FROM runs ORDER BY iniciado_em").fetchall()
            wks = self.con.execute("SELECT * FROM workers ORDER BY hostname").fetchall()
            ultimas = self.con.execute(
                "SELECT e.run_id, e.epoca, e.metricas FROM epocas e JOIN "
                "(SELECT run_id, MAX(epoca) m FROM epocas GROUP BY run_id) x "
                "ON e.run_id=x.run_id AND e.epoca=x.m").fetchall()

        ultima_de = {r["run_id"]: {"epoca": r["epoca"], **_dj(r["metricas"], {})}
                     for r in ultimas}

        runs_out = []
        for r in runs:
            d = dict(r)
            for k in ("config", "hardware", "plano", "melhor"):
                d[k] = _dj(d[k], {})
            d["ultima"] = ultima_de.get(r["run_id"], {})
            d["idade_s"] = round(agora - (r["atualizado_em"] or agora))
            d["vivo"] = d["idade_s"] < limite_seg and r["status"] == "rodando"
            runs_out.append(d)

        return {
            "agora": agora,
            "limite_seg": limite_seg,
            "experimentos": [{**dict(e), "config": _dj(e["config"], {})} for e in exps],
            "runs": runs_out,
            "workers": [{**dict(w), "info": _dj(w["info"], {}),
                         "idade_s": round(agora - (w["visto_em"] or agora))} for w in wks],
        }

    def historico(self, run_id: str) -> list[dict]:
        with self._lock:
            linhas = self.con.execute(
                "SELECT epoca, t, metricas FROM epocas WHERE run_id=? ORDER BY epoca",
                (run_id,)).fetchall()
        return [{"epoca": l["epoca"], "t": l["t"], **_dj(l["metricas"], {})} for l in linhas]
