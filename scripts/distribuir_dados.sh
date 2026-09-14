#!/usr/bin/env bash
# Copia codigo + dados para as maquinas do laboratorio, em paralelo.
#
#     bash scripts/distribuir_dados.sh                 # todas de experiments/maquinas.txt
#     bash scripts/distribuir_dados.sh pc07 pc11       # so estas
#     PARALELO=8 bash scripts/distribuir_dados.sh      # mais copias simultaneas
#     ORIGEM=pc01 bash scripts/distribuir_dados.sh ... # copia A PARTIR de outra maquina
#
# ---------------------------------------------------------------------------
# Sobre o tempo: o gargalo e a placa de rede da ORIGEM, nao das destinatarias.
# 30 GB x 20 maquinas = 600 GB; num link de 1 Gb/s isso e ~1h20 no melhor caso,
# e aumentar PARALELO nao muda esse total — so reparte a mesma banda.
#
# Se 1h20 nao couber no dia, distribua em duas ondas:
#
#     bash scripts/distribuir_dados.sh pc01 pc02 pc03 pc04      # ~15 min
#     ORIGEM=pc01 bash scripts/distribuir_dados.sh pc05 ... pc08 &
#     ORIGEM=pc02 bash scripts/distribuir_dados.sh pc09 ... pc12 &
#
# Cada origem serve 4 destinos com a propria banda, e o total cai para ~1/4.
# rsync e retomavel: uma copia interrompida continua de onde parou.

set -uo pipefail
cd "$(dirname "$0")/.."
RAIZ="$PWD"

LISTA=${LISTA:-experiments/maquinas.txt}
PARALELO=${PARALELO:-4}
DESTINO=${DESTINO:-Disaster-Damage}
USUARIO=${USUARIO:-$USER}
ORIGEM=${ORIGEM:-}

if [ $# -gt 0 ]; then
    MAQUINAS=("$@")
else
    [ -f "$LISTA" ] || { echo "! $LISTA nao existe. Crie com um host por linha."; exit 1; }
    mapfile -t MAQUINAS < <(grep -vE '^\s*(#|$)' "$LISTA")
fi
[ ${#MAQUINAS[@]} -gt 0 ] || { echo "! nenhuma maquina informada"; exit 1; }

# --exclude e o que separa 30 GB de 200 GB. geotiffs/ e o tarball nao vao: as
# maquinas do laboratorio treinam a partir dos PNG. .venv tampouco — cada maquina
# monta o seu com a roda do CUDA que o driver dela suporta.
EXCLUIR=(--exclude '.git/' --exclude '.venv/' --exclude 'geotiffs/' --exclude '*.tgz'
         --exclude 'cache/' --exclude '__pycache__/' --exclude '.ipynb_checkpoints/'
         --exclude 'checkpoints/' --exclude 'results/' --exclude 'data/index_shard*')

copiar() {
    local m="$1"
    local log="/tmp/distribuir-$m.log"
    {
        echo "=== $m  $(date +%H:%M:%S)"
        if [ -n "$ORIGEM" ]; then
            # Fanout em arvore: a origem ja tem os dados e serve com a banda dela.
            ssh -o BatchMode=yes "$USUARIO@$ORIGEM" \
                "rsync -a --info=progress2 --partial ${EXCLUIR[*]} \
                 '$DESTINO/' '$USUARIO@$m:$DESTINO/'"
        else
            ssh -o BatchMode=yes "$USUARIO@$m" "mkdir -p '$DESTINO'" &&
            rsync -a --info=progress2 --partial "${EXCLUIR[@]}" \
                  "$RAIZ/" "$USUARIO@$m:$DESTINO/"
        fi
    } > "$log" 2>&1
    if [ $? -eq 0 ]; then
        echo "  ok   $m"
    else
        echo "  FALHOU $m  (veja $log)"
        return 1
    fi
}

echo "=== ${#MAQUINAS[@]} maquina(s), $PARALELO em paralelo${ORIGEM:+, a partir de $ORIGEM}"
echo "=== destino: ~/$DESTINO"
inicio=$(date +%s)

for m in "${MAQUINAS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALELO" ]; do wait -n; done
    copiar "$m" &
done
wait

echo
echo "=== $(( ($(date +%s) - inicio) / 60 )) min"
echo
echo "Confira uma maquina antes de disparar as 20:"
echo "    ssh $USUARIO@${MAQUINAS[0]} 'cd $DESTINO && du -sh data && ls data/images/*/ | wc -l'"
echo
echo "Depois, em cada maquina:"
echo "    cd $DESTINO && bash scripts/lab_setup.sh"
