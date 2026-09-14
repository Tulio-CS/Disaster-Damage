"""
Painel do laboratorio: fila de experimentos, ingestao de progresso e TensorBoard.

Roda na VPS (Coolify), fora da rede do laboratorio. As 20 maquinas so precisam de
saida HTTPS: elas puxam trabalho e empurram metricas, nunca recebem conexao. Isso
evita ter que expor porta em maquina de laboratorio ou montar VPN.

Um unico processo serve tudo em uma porta so, porque no Coolify um dominio aponta
para um container:

    /              painel das 20 maquinas
    /api/...       fila e ingestao (Bearer token)
    /tb/...        TensorBoard, atras da mesma autenticacao

O TensorBoard sobe como subprocesso preso a 127.0.0.1 e so e alcancavel atraves
deste proxy — ele nao tem autenticacao propria e ficaria aberto na internet se
recebesse um dominio direto.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from .db import Store
from .matriz import carregar_matriz
from .tb import Escritor

AQUI = Path(__file__).resolve().parent

# ------------------------------------------------------------------ configuracao

TOKEN = os.environ.get("LAB_TOKEN", "").strip()
DATA = Path(os.environ.get("LAB_DATA", "/data"))
PORTA_TB = int(os.environ.get("LAB_PORTA_TB", "6006"))
LEASE_SEG = float(os.environ.get("LAB_LEASE_SEG", "1200"))    # 20 min sem heartbeat
TB_ATIVO = os.environ.get("LAB_TB", "1") not in ("0", "false", "no")

if not TOKEN:
    raise SystemExit(
        "LAB_TOKEN nao definido.\n"
        "Este servico fica exposto na internet pelo Coolify; sem token a fila de\n"
        "experimentos e os resultados ficariam abertos. Gere um com:\n"
        "  python -c \"import secrets; print(secrets.token_urlsafe(32))\"")

store = Store(DATA / "lab.db")
escritor = Escritor(DATA / "tb")


def _achar_matriz() -> Path:
    """Primeira que existir. A do volume vence a da imagem: assim da para trocar a
    matriz por `POST /api/matriz` sem rebuild, e a troca sobrevive ao redeploy."""
    candidatas = [
        Path(os.environ["LAB_MATRIZ"]) if os.environ.get("LAB_MATRIZ") else None,
        DATA / "matrix.yaml",
        Path("/app/matrix.yaml"),
        AQUI.parent.parent / "experiments" / "matrix.yaml",
    ]
    for c in candidatas:
        if c and c.exists():
            return c
    return DATA / "matrix.yaml"


MATRIZ = _achar_matriz()


# ------------------------------------------------------------------ autenticacao


def autorizar(request: Request) -> None:
    cab = request.headers.get("authorization", "")
    fornecido = ""
    if cab.lower().startswith("bearer "):
        fornecido = cab[7:].strip()
    else:
        fornecido = request.cookies.get("lab_token", "") or request.query_params.get("t", "")
    if not secrets.compare_digest(fornecido, TOKEN):
        raise HTTPException(401, "token invalido ou ausente")


Auth = Depends(autorizar)


# ------------------------------------------------------------------ ciclo de vida


_tb_proc: subprocess.Popen | None = None


async def _vigia() -> None:
    """Marca como perdido quem parou de dar sinal e devolve o slot para a fila."""
    while True:
        try:
            await asyncio.sleep(60)
            n = store.marcar_perdidos(LEASE_SEG)
            if n:
                print(f"[vigia] {n} run(s) sem heartbeat ha {LEASE_SEG:.0f}s -> reenfileirado(s)",
                      flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[vigia] {exc}", flush=True)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _tb_proc

    if MATRIZ.exists():
        try:
            r = store.sincronizar_matriz(carregar_matriz(MATRIZ))
            print(f"[matriz] {r['total']} experimentos ({r['novos']} novos, "
                  f"{r['atualizados']} atualizados)", flush=True)
        except Exception as exc:
            print(f"[matriz] FALHA ao carregar {MATRIZ}: {exc}", flush=True)
    else:
        print(f"[matriz] {MATRIZ} nao existe; use POST /api/matriz", flush=True)

    if TB_ATIVO:
        (DATA / "tb").mkdir(parents=True, exist_ok=True)
        # A saida vai para arquivo, e nao para DEVNULL: quando o TensorBoard morre
        # no boot (falta uma dependencia dele, porta ocupada), o unico sintoma
        # visivel e um 502 no proxy — sem este log nao ha como saber por que.
        log_tb = open(DATA / "tensorboard.log", "ab", buffering=0)
        _tb_proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", str(DATA / "tb"), "--host", "127.0.0.1",
             "--port", str(PORTA_TB), "--path_prefix", "/tb",
             "--reload_interval", "15"],
            stdout=log_tb, stderr=subprocess.STDOUT)
        await asyncio.sleep(3)
        if _tb_proc.poll() is not None:
            cauda = (DATA / "tensorboard.log").read_bytes()[-800:].decode(errors="replace")
            print(f"[tb] FALHOU ao subir (codigo {_tb_proc.returncode}):\n{cauda}", flush=True)
            _tb_proc = None
        else:
            print(f"[tb] 127.0.0.1:{PORTA_TB} (pid {_tb_proc.pid})", flush=True)

    app.state.http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORTA_TB}", timeout=30)
    tarefa = asyncio.create_task(_vigia())
    try:
        yield
    finally:
        tarefa.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tarefa
        await app.state.http.aclose()
        escritor.fechar()
        if _tb_proc:
            _tb_proc.terminate()


app = FastAPI(title="xBD lab", lifespan=lifespan, docs_url=None, redoc_url=None)


# ------------------------------------------------------------------------- fila


@app.post("/api/claim", dependencies=[Auth])
async def claim(payload: dict) -> Response:
    """Worker pede o proximo experimento. 204 quando a fila acabou."""
    wid = payload.get("worker_id")
    if not wid:
        raise HTTPException(400, "worker_id obrigatorio")
    host = payload.get("hostname", wid)

    exp = store.reivindicar(wid, host, payload.get("vram_gb"))
    store.registrar_worker(wid, host, payload.get("gpu"), payload.get("info"),
                           None, "ocioso" if exp is None else "atribuido")
    if exp is None:
        return Response(status_code=204)
    return JSONResponse(exp)


@app.post("/api/eventos", dependencies=[Auth])
async def eventos(payload: dict) -> dict:
    """Ingestao de progresso, em lote e idempotente.

    O worker sempre grava num spool em disco e um thread drena o spool para ca.
    Se a VPS ficar fora do ar, o treino continua e os eventos sobem depois — por
    isso reentrega e o caso comum, e todo tipo de evento aqui e idempotente.
    """
    lista = payload.get("eventos") or []
    if not isinstance(lista, list):
        raise HTTPException(400, "'eventos' deve ser uma lista")

    aceitos, ignorados = 0, 0
    for ev in lista:
        tipo, run_id = ev.get("tipo"), ev.get("run_id")
        if not tipo or not run_id:
            ignorados += 1
            continue
        try:
            if tipo == "inicio":
                store.inicio_run(ev)
            elif tipo == "epoca":
                nova = store.epoca(ev)
                if nova:
                    escritor.escalares(ev.get("grupo") or "sem-grupo",
                                       ev.get("nome") or run_id,
                                       int(ev["epoca"]), ev.get("metricas") or {})
            elif tipo == "melhor":
                store.melhor(ev)
            elif tipo == "heartbeat":
                store.heartbeat(run_id)
                if ev.get("worker_id"):
                    store.registrar_worker(ev["worker_id"], ev.get("hostname", ""),
                                           None, None, run_id, "treinando")
            elif tipo == "fim":
                store.fim_run(ev)
                if ev.get("status") == "concluido":
                    escritor.hparams(ev.get("grupo") or "sem-grupo",
                                     ev.get("nome") or run_id,
                                     ev.get("config") or {}, ev.get("melhor") or {})
            else:
                ignorados += 1
                continue
            aceitos += 1
        except Exception as exc:
            print(f"[eventos] {tipo} {run_id}: {exc}", flush=True)
            ignorados += 1

    return {"aceitos": aceitos, "ignorados": ignorados}


@app.post("/api/matriz", dependencies=[Auth])
async def recarregar_matriz(payload: dict | None = None) -> dict:
    """Recarrega a matriz do disco, ou aceita uma nova no corpo (chave 'yaml')."""
    if payload and payload.get("yaml"):
        caminho = DATA / "matrix.yaml"
        caminho.write_text(payload["yaml"], encoding="utf-8")
    else:
        caminho = _achar_matriz()
    try:
        return store.sincronizar_matriz(carregar_matriz(caminho))
    except Exception as exc:
        raise HTTPException(400, f"matriz invalida: {exc}")


@app.post("/api/experimentos/{nome}/liberar", dependencies=[Auth])
async def liberar(nome: str) -> dict:
    """Devolve um experimento travado para a fila (para retomar um slot que morreu)."""
    store.liberar(nome, "pendente")
    return {"ok": True, "nome": nome}


# ----------------------------------------------------------------------- leitura


@app.get("/api/estado", dependencies=[Auth])
async def estado() -> dict:
    return store.estado(LEASE_SEG)


@app.get("/api/runs/{run_id}/historico", dependencies=[Auth])
async def historico(run_id: str) -> list[dict]:
    return store.historico(run_id)


@app.get("/api/resultados.csv", dependencies=[Auth])
async def resultados_csv() -> PlainTextResponse:
    """Tabela final para a analise: uma linha por run, melhor metrica de cada."""
    import csv
    import io

    st = store.estado(LEASE_SEG)
    campos = ["nome", "grupo", "eixo", "run_id", "status", "hostname", "epocas",
              "epochs_total", "score_xview2", "f1_loc", "f1_dano_harmonico",
              "f1_dano_media", "seed", "fold", "encoder", "crop", "fuse_mode",
              "norm", "siamese", "usar_tier3", "oversample_dano", "loss_dmg",
              "pretrained", "ema_decay"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=campos, extrasaction="ignore")
    w.writeheader()
    for r in st["runs"]:
        cfg, melhor = r.get("config") or {}, r.get("melhor") or {}
        w.writerow({**cfg, **melhor,
                    "nome": r["nome"], "grupo": r["grupo"], "eixo": r["eixo"],
                    "run_id": r["run_id"], "status": r["status"],
                    "hostname": r["hostname"], "epocas": r["epocas"],
                    "epochs_total": r["epochs_total"]})
    return PlainTextResponse(buf.getvalue(), media_type="text/csv")


@app.get("/api/saude")
async def saude() -> dict:
    return {"ok": True, "t": time.time()}


# ------------------------------------------------------------------------ painel


@app.get("/entrar")
async def entrar(request: Request, t: str = "") -> Response:
    """Troca o token por um cookie, para o navegador nao carregar ?t= em cada link."""
    if not secrets.compare_digest(t, TOKEN):
        return HTMLResponse("<h1>401</h1><p>token invalido</p>", status_code=401)
    # Atras do proxy do Coolify o request chega em http; quem sabe se a ponta e
    # https e o cabecalho encaminhado. Marcar Secure sem https quebraria o teste local.
    https = (request.headers.get("x-forwarded-proto", request.url.scheme) == "https")
    r = Response(status_code=302, headers={"Location": "/"})
    r.set_cookie("lab_token", TOKEN, httponly=True, samesite="lax",
                 secure=https, max_age=60 * 60 * 24 * 30)
    return r


@app.get("/", response_class=HTMLResponse)
async def painel(request: Request) -> Response:
    try:
        autorizar(request)
    except HTTPException:
        return HTMLResponse(
            "<body style='font:16px system-ui;padding:3rem;max-width:34rem;margin:auto'>"
            "<h1>xBD lab</h1><p>Acesse com <code>/entrar?t=SEU_TOKEN</code>.</p></body>",
            status_code=401)
    return FileResponse(AQUI / "static" / "dashboard.html")


# ------------------------------------------------------------- proxy TensorBoard

_SEM_REPASSE = {"content-encoding", "content-length", "transfer-encoding", "connection"}


@app.api_route("/tb", methods=["GET"], dependencies=[Auth])
@app.api_route("/tb/{caminho:path}", methods=["GET", "POST"], dependencies=[Auth])
async def proxy_tb(request: Request, caminho: str = "") -> Response:
    if not TB_ATIVO:
        return PlainTextResponse("TensorBoard desativado (LAB_TB=0)", status_code=503)
    url = f"/tb/{caminho}"
    try:
        r = await request.app.state.http.request(
            request.method, url, params=request.query_params,
            content=await request.body(),
            headers={k: v for k, v in request.headers.items()
                     if k.lower() in ("content-type", "accept", "accept-encoding")})
    except httpx.HTTPError as exc:
        cauda = ""
        log = DATA / "tensorboard.log"
        if _tb_proc is None and log.exists():
            cauda = "\n\n" + log.read_bytes()[-800:].decode(errors="replace")
        return PlainTextResponse(
            f"TensorBoard indisponivel: {exc}{cauda}", status_code=502)
    return Response(
        content=r.content, status_code=r.status_code,
        headers={k: v for k, v in r.headers.items() if k.lower() not in _SEM_REPASSE},
        media_type=r.headers.get("content-type"))
