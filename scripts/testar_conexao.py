#!/usr/bin/env python3
"""
Verifica que ESTA maquina consegue falar com o painel. Windows ou Linux.

    Windows:  python E:\\Disaster-Damage\\scripts\\testar_conexao.py
    Linux:    python3 scripts/testar_conexao.py

Le LAB_URL/LAB_TOKEN do ambiente ou do lab.env na raiz do HD. Leva segundos e
nao consome slot da fila.

Usa `urllib.request` de proposito, e nao o navegador nem `curl`: e a mesma pilha
que `xbd/reporter.py` usa em producao, e as tres diferem no que importa aqui.
Navegador e curl carregam os proprios conjuntos de CA; o Python usa o do
OpenSSL. Numa rede com proxy de inspecao TLS — comum em instituicao — os dois
primeiros passam e o agente falha com CERTIFICATE_VERIFY_FAILED. Testar pelo
navegador daria um falso OK, e o erro so apareceria quando a primeira epoca
tentasse subir.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

WINDOWS = os.name == "nt"
falhas: list[tuple[str, Exception]] = []


def credenciais() -> tuple[str, str]:
    url = os.environ.get("LAB_URL", "")
    token = os.environ.get("LAB_TOKEN", "")
    if url and token:
        return url.rstrip("/"), token
    # lab.env fica na raiz do HD: .../HD/Disaster-Damage/scripts/este.py
    for cand in (Path(__file__).resolve().parent.parent.parent / "lab.env",
                 Path(__file__).resolve().parent.parent / "lab.env"):
        if cand.is_file():
            for linha in cand.read_text(encoding="utf-8").splitlines():
                linha = linha.strip()
                if linha and not linha.startswith("#") and "=" in linha:
                    k, v = linha.split("=", 1)
                    if k.strip() == "LAB_URL" and not url:
                        url = v.strip().rstrip("/")
                    elif k.strip() == "LAB_TOKEN" and not token:
                        token = v.strip()
            break
    return url, token


def passo(nome: str, fn) -> None:
    t0 = time.time()
    try:
        detalhe = fn()
        print(f"  ok    {nome:<26} {detalhe}  ({(time.time()-t0)*1000:.0f} ms)")
    except Exception as exc:
        print(f"  FALHA {nome:<26} {type(exc).__name__}: {exc}")
        falhas.append((nome, exc))


def main() -> int:
    url, token = credenciais()
    if not url or not token or token == "COLE_O_TOKEN_AQUI":
        print("! LAB_URL/LAB_TOKEN ausentes. Defina no ambiente ou no lab.env do HD.")
        return 2

    host = urlparse(url).hostname or ""
    print(f"painel: {url}")
    print(f"python: {sys.version.split()[0]}   openssl: {ssl.OPENSSL_VERSION}")
    print(f"maquina: {socket.gethostname()}   sistema: {'Windows' if WINDOWS else 'Linux'}")
    print()

    passo("DNS", lambda: socket.gethostbyname(host))

    def _tls():
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as s:
            with ctx.wrap_socket(s, server_hostname=host) as ss:
                emissor = dict(x[0] for x in ss.getpeercert()["issuer"])
                return f"{ss.version()}  emissor={emissor.get('organizationName', '?')}"

    passo("TLS + certificado", _tls)

    def _saude():
        with urllib.request.urlopen(f"{url}/api/saude", timeout=15) as r:
            return f"HTTP {r.status}  {json.load(r)}"

    passo("GET /api/saude", _saude)

    def _auth():
        req = urllib.request.Request(f"{url}/api/estado",
                                     headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.load(r)
        exps = d.get("experimentos", [])
        pend = sum(1 for e in exps if e["status"] == "pendente")
        return f"HTTP {r.status}  {len(exps)} experimentos ({pend} pendentes)"

    passo("GET /api/estado (token)", _auth)

    print()
    if not falhas:
        print("Tudo certo. Esta maquina consegue puxar experimentos e reportar metricas.")
        return 0

    print("Problemas encontrados:")
    for nome, exc in falhas:
        print(f"  x {nome}: {exc}")
    print()

    texto = " ".join(str(e) for _, e in falhas)
    if "CERTIFICATE_VERIFY_FAILED" in texto:
        print("Certificado rejeitado. Tipico de rede com proxy de inspecao TLS: o")
        print("navegador confia porque a CA corporativa esta instalada para ele, e o")
        print("Python nao. Peca o certificado da CA ao administrador e aponte:")
        print(r"    Windows:  set SSL_CERT_FILE=C:\caminho\ca-corporativa.pem")
        print("    Linux:    export SSL_CERT_FILE=/caminho/ca-corporativa.pem")
    elif "401" in texto or "403" in texto:
        print("Rede OK, token recusado. O LAB_TOKEN precisa ser identico ao que esta")
        print("nas variaveis de ambiente do Coolify.")
    elif "gaierror" in texto or "Name or service not known" in texto \
            or "getaddrinfo failed" in texto:
        print("DNS nao resolve. Se a rede exige proxy, defina antes de rodar:")
        print(r"    Windows:  set https_proxy=http://proxy.local:3128")
        print("    Linux:    export https_proxy=http://proxy.local:3128")
        print("(urllib respeita https_proxy sozinho, entao o agente tambem vai usar)")
    else:
        print("Sem saida para a internet, ou bloqueio por categoria/dominio.")
        print("Teste se a rede exige proxy (https_proxy, como acima).")
    return 1


if __name__ == "__main__":
    codigo = main()
    if WINDOWS:
        input("\nPressione Enter para fechar...")
    raise SystemExit(codigo)
