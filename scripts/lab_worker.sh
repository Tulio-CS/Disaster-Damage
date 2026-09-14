#!/usr/bin/env bash
# Roda o agente desta maquina, com reinicio automatico.
#
#     export LAB_URL=https://seu-painel   LAB_TOKEN=xxxx
#     bash scripts/lab_worker.sh
#
# Deixar rodando depois de fechar a sessao (o caso normal no laboratorio):
#     setsid nohup bash scripts/lab_worker.sh > agente.out 2>&1 &
#     tail -f agente.out
#
# O laco de reinicio existe para falhas do PROPRIO agente (rede caiu no meio de
# um claim, disco encheu). Falha do treino nao chega aqui: o agente ja isola o
# treino num subprocesso, reporta o erro e pega o proximo experimento sozinho.

set -uo pipefail
cd "$(dirname "$0")/.."

: "${LAB_URL:?defina LAB_URL — ex.: export LAB_URL=https://lab.exemplo.com}"
: "${LAB_TOKEN:?defina LAB_TOKEN}"
export LAB_WORKER="${LAB_WORKER:-$(hostname)}"

[ -x .venv/bin/python ] || { echo "! .venv nao existe; rode antes: bash scripts/lab_setup.sh"; exit 1; }

# Encerramento limpo: Ctrl-C (ou o logoff no fim do dia) chega no agente, que
# repassa para o treino. O treino salva o checkpoint, marca 'interrompido', e o
# slot volta para a fila para retomar depois — desta mesma maquina, que e onde
# o checkpoint esta.
parar() { echo; echo "[worker] encerrando..."; SAIR=1; kill -INT "$FILHO" 2>/dev/null; }
trap parar INT TERM
SAIR=0

echo "[worker] $LAB_WORKER -> $LAB_URL"
while [ "$SAIR" -eq 0 ]; do
    .venv/bin/python -m xbd.agent "$@" &
    FILHO=$!
    wait "$FILHO"
    CODIGO=$?

    [ "$SAIR" -eq 1 ] && break
    if [ "$CODIGO" -eq 0 ]; then
        echo "[worker] agente saiu normalmente (fila vazia?); encerrando."
        break
    fi
    echo "[worker] agente caiu (codigo $CODIGO); reiniciando em 30s"
    sleep 30
done
echo "[worker] fim."
