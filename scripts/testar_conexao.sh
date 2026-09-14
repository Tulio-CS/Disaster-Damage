#!/usr/bin/env bash
# Verifica, de uma maquina do laboratorio, que ela consegue falar com o painel.
#
#     export LAB_URL=https://experimento.tremdigital.com.br
#     export LAB_TOKEN=xxxx
#     bash scripts/testar_conexao.sh
#
# Rode ANTES de instalar as 20. Leva segundos e nao consome slot da fila.
#
# Usa `urllib.request` de proposito, e nao `curl`: e a mesma pilha que
# `xbd/reporter.py` usa em producao. As duas diferem no que importa aqui —
# `curl` tem o proprio bundle de CA e o Python usa o do OpenSSL, entao numa rede
# com proxy de inspecao TLS o `curl` passa e o agente falha com
# CERTIFICATE_VERIFY_FAILED. Testar com curl daria um falso OK.

set -uo pipefail
cd "$(dirname "$0")/.."

: "${LAB_URL:?defina LAB_URL}"
: "${LAB_TOKEN:?defina LAB_TOKEN}"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

"$PY" - "$LAB_URL" "$LAB_TOKEN" <<'PY'
import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

url, token = sys.argv[1].rstrip("/"), sys.argv[2]
host = urllib.parse.urlparse(url).hostname
falhas = []


def passo(nome, fn):
    t0 = time.time()
    try:
        detalhe = fn()
        print(f"  ok    {nome:<26} {detalhe}  ({(time.time()-t0)*1000:.0f} ms)")
    except Exception as exc:
        print(f"  FALHA {nome:<26} {type(exc).__name__}: {exc}")
        falhas.append((nome, exc))


print(f"painel: {url}")
print(f"python: {sys.version.split()[0]}   openssl: {ssl.OPENSSL_VERSION}")
print()

passo("DNS", lambda: socket.gethostbyname(host))


def _tls():
    ctx = ssl.create_default_context()
    with socket.create_connection((host, 443), timeout=10) as s:
        with ctx.wrap_socket(s, server_hostname=host) as ss:
            cert = ss.getpeercert()
            emissor = dict(x[0] for x in cert["issuer"]).get("organizationName", "?")
            return f"{ss.version()}  emissor={emissor}"


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
    n = len(d.get("experimentos", []))
    pend = sum(1 for e in d["experimentos"] if e["status"] == "pendente")
    return f"HTTP {r.status}  {n} experimentos ({pend} pendentes)"


passo("GET /api/estado (token)", _auth)

print()
if not falhas:
    print("Tudo certo. Esta maquina consegue puxar experimentos e reportar metricas.")
    sys.exit(0)

print("Problemas encontrados:")
for nome, exc in falhas:
    print(f"  x {nome}: {exc}")

texto = " ".join(str(e) for _, e in falhas)
print()
if "CERTIFICATE_VERIFY_FAILED" in texto:
    print("Certificado rejeitado. Tipico de rede com proxy de inspecao TLS: o")
    print("navegador confia porque a CA corporativa esta no sistema dele, e o")
    print("Python nao. Peca ao administrador o certificado da CA e aponte:")
    print("    export SSL_CERT_FILE=/caminho/ca-corporativa.pem")
    print("    export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE")
    print("(exporte junto com LAB_URL/LAB_TOKEN antes do lab_worker.sh)")
elif "401" in texto or "403" in texto:
    print("Rede OK, token recusado. Confira LAB_TOKEN — deve ser identico ao")
    print("LAB_TOKEN configurado nas variaveis de ambiente do Coolify.")
elif "Name or service not known" in texto or "gaierror" in texto:
    print("DNS nao resolve. Se a rede exige proxy, exporte antes de rodar:")
    print("    export https_proxy=http://proxy.local:3128")
    print("(urllib respeita https_proxy sozinho, entao o agente tambem vai usar)")
else:
    print("Sem saida para a internet, ou bloqueio por categoria/dominio. Teste")
    print("se a rede exige proxy:  export https_proxy=http://proxy.local:3128")
sys.exit(1)
PY
