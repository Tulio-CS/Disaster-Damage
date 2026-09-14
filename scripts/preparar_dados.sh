#!/usr/bin/env bash
# Converte os GeoTIFF em PNG + mascaras e monta o indice. Roda UMA vez, na
# maquina que tem os 139 GB extraidos — nao nas maquinas do laboratorio.
#
#     bash scripts/preparar_dados.sh
#
# Saida em data/: ~30 GB, que e o que vai ser copiado para as 20 maquinas.
# E idempotente: interromper e rodar de novo continua de onde parou.

set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-$(( $(nproc) - 2 ))}
[ "$WORKERS" -lt 1 ] && WORKERS=1
PY=${PY:-.venv/bin/python}

[ -x "$PY" ] || { echo "! $PY nao existe"; exit 1; }
[ -d geotiffs ] || { echo "! geotiffs/ nao existe — extraia xview2_geotiff.tgz antes"; exit 1; }
[ -f cache/images.parquet ] || {
    echo "cache/images.parquet ausente; construindo o indice primeiro..."
    "$PY" scripts/build_index.py --workers "$WORKERS"; }

echo "=== convertendo com $WORKERS processos ==="
echo "    (compress_level 1: ~3x mais rapido que o padrao e so ~8% maior em disco;"
echo "     com 22 mil imagens, essa troca vale mais que o espaco)"
time "$PY" -m xbd.prep --shard 0 --of 1 --workers "$WORKERS" --png-level 1

"$PY" -m xbd.prep_merge --out data

echo
du -sh data
"$PY" - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_parquet("data/index.parquet")
print(f"{len(df):,} cenas   {df.tem_dano.sum():,} com dano   "
      f"{df.groupby('split').size().to_dict()}")
n = sum(1 for _ in Path("data/images").rglob("*.png"))
print(f"{n:,} PNG de imagem (esperado {2*len(df):,})")
PY

echo
echo "=== pronto. Proximo passo: bash scripts/distribuir_dados.sh"
