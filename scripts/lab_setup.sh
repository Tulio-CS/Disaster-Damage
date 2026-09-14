#!/usr/bin/env bash
# Prepara UMA maquina do laboratorio. Nao precisa de root nem de Docker.
#
#     bash scripts/lab_setup.sh
#
# Cria o venv, instala a roda do torch compativel com o driver instalado e
# confere GPU e dados. Rodar de novo e barato: pip pula o que ja esta la.
#
# A versao do CUDA e detectada do driver porque num laboratorio as maquinas
# raramente estao todas iguais, e instalar cu124 onde o driver so suporta 12.1
# resulta num `torch.cuda.is_available() == False` silencioso: o treino cai para
# CPU, roda 50x mais devagar e ninguem percebe ate o fim do dia.

set -euo pipefail
cd "$(dirname "$0")/.."
RAIZ="$PWD"

echo "=== maquina: $(hostname) ==="

# ------------------------------------------------------------------ driver/CUDA

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    CUDA_DRIVER=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)
else
    echo "! nvidia-smi nao encontrado — esta maquina nao tem GPU utilizavel."
    echo "  O treino cairia para CPU. Confirme se e essa a intencao (Ctrl-C para abortar)."
    sleep 5
    CUDA_DRIVER=""
fi

: "${CUDA_TAG:=}"
if [ -z "$CUDA_TAG" ]; then
    case "$CUDA_DRIVER" in
        "")           CUDA_TAG="cpu" ;;
        12.[4-9]|13*) CUDA_TAG="cu124" ;;
        12.[1-3])     CUDA_TAG="cu121" ;;
        11.[8-9]|12.0) CUDA_TAG="cu118" ;;
        *)            CUDA_TAG="cu121" ;;
    esac
fi
echo "driver CUDA: ${CUDA_DRIVER:-nenhum} -> roda do torch: $CUDA_TAG"

# --------------------------------------------------------------------- ambiente

PY=${PY:-python3}
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' || {
    echo "! python >= 3.10 necessario; encontrado $($PY -V)"; exit 1; }

[ -d .venv ] || "$PY" -m venv .venv
PIP=".venv/bin/pip"
$PIP install --quiet --upgrade pip wheel

echo "--- torch ($CUDA_TAG) ---"
if [ "$CUDA_TAG" = "cpu" ]; then
    $PIP install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    $PIP install --quiet torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

echo "--- demais dependencias ---"
$PIP install --quiet -r requirements-treino.txt

# ------------------------------------------------------------------ conferencia

echo "--- conferencia ---"
.venv/bin/python - <<'PY'
import sys
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name}  {p.total_memory/1e9:.0f} GB  sm_{p.major}{p.minor}")
else:
    print("  ! sem GPU visivel para o torch — o treino rodaria em CPU.")
import timm, cv2, pandas, numpy   # noqa: F401  (falha aqui e melhor que falhar na epoca 1)
print("imports de treino OK")
PY

.venv/bin/python -c "
from pathlib import Path
import sys
d = Path('data')
idx = d/'index.parquet'
if not idx.exists():
    print('  ! data/index.parquet ausente — rode scripts/distribuir_dados.sh a partir da maquina que tem os dados')
    sys.exit(0)
import pandas as pd
df = pd.read_parquet(idx)
n_img = sum(1 for _ in (d/'images').rglob('*.png')) if (d/'images').exists() else 0
print(f'  indice: {len(df):,} cenas   PNG em disco: {n_img:,} (esperado ~{2*len(df):,})')
if n_img < 2*len(df):
    print('  ! faltam imagens; a copia dos dados nao terminou')
"

echo
echo "=== pronto. Para comecar a treinar:"
echo "    export LAB_URL=https://SEU-PAINEL"
echo "    export LAB_TOKEN=SEU_TOKEN"
echo "    bash scripts/lab_worker.sh"
