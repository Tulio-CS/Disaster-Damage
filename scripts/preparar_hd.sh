#!/usr/bin/env bash
# Monta o HD externo que vai rodar as maquinas do laboratorio.
#
#     bash scripts/preparar_hd.sh /media/tulio/MEU_HD
#
# Depois disto, em cada maquina basta plugar o HD e rodar UM comando
# (o script imprime qual no final).
#
# ---------------------------------------------------------------------------
# Por que os dados viram um .tar unico em vez de 22.068 arquivos soltos
#
# Copiar 22 mil arquivos pequenos de um HD USB e dominado por latencia de
# acesso, nao por banda: costuma cair para 20-40 MB/s. Um arquivo unico de
# 33 GB e leitura sequencial pura, ~100-120 MB/s em USB 3.0. A diferenca e de
# ~25 min para ~6 min POR MAQUINA — com 10 maquinas, algumas horas de espera.
#
# Sem compressao (`-cf`, nao `-czf`): PNG ja esta comprimido, entao gzip so
# gastaria CPU nas duas pontas para economizar quase nada.

set -euo pipefail
cd "$(dirname "$0")/.."
RAIZ="$PWD"

HD="${1:-}"
[ -n "$HD" ] || { echo "uso: bash scripts/preparar_hd.sh /caminho/do/HD"; exit 1; }
[ -d "$HD" ] || { echo "! $HD nao existe ou nao esta montado"; exit 1; }
[ -w "$HD" ] || { echo "! sem permissao de escrita em $HD"; exit 1; }

[ -f data/index.parquet ] || { echo "! data/index.parquet ausente; rode antes scripts/preparar_dados.sh"; exit 1; }

PRECISA_GB=$(( $(du -sB1G data | cut -f1) + 3 ))
LIVRE_GB=$(df -BG --output=avail "$HD" | tail -1 | tr -dc '0-9')
echo "=== HD: $HD  ($LIVRE_GB GB livres, precisa de ~$PRECISA_GB GB)"
[ "$LIVRE_GB" -ge "$PRECISA_GB" ] || { echo "! espaco insuficiente"; exit 1; }

SIS=$(df -T "$HD" | tail -1 | awk '{print $2}')
echo "=== sistema de arquivos: $SIS"
case "$SIS" in
    ext4|ext3|xfs|btrfs) ;;
    exfat|fuseblk)  echo "    (exFAT funciona; ext4 seria mais rapido para o .tar)" ;;
    vfat|msdos)     echo "! FAT32 nao aceita arquivo maior que 4 GB — reformate para exFAT ou ext4"; exit 1 ;;
    *)              echo "    (nao reconhecido; seguindo)" ;;
esac

# ---------------------------------------------------------------- codigo

echo
echo "=== copiando o codigo ==="
mkdir -p "$HD/Disaster-Damage"
rsync -a --delete \
      --exclude '.git/' --exclude '.venv/' --exclude 'geotiffs/' --exclude '*.tgz' \
      --exclude 'cache/' --exclude '__pycache__/' --exclude '.ipynb_checkpoints/' \
      --exclude 'checkpoints/' --exclude 'results/' --exclude 'data/' \
      "$RAIZ/" "$HD/Disaster-Damage/"
echo "    ok"

# ----------------------------------------------------------------- dados

TAR="$HD/xbd-data.tar"
if [ -f "$TAR" ]; then
    echo
    echo "=== $TAR ja existe ($(du -h "$TAR" | cut -f1)); refazer? [s/N]"
    read -r resp
    [ "$resp" = "s" ] || [ "$resp" = "S" ] || SKIP_TAR=1
fi

if [ -z "${SKIP_TAR:-}" ]; then
    echo
    echo "=== empacotando data/ (33 GB, ~10 min) ==="
    tar -cf "$TAR.parcial" -C "$RAIZ" \
        data/index.parquet data/images data/masks \
        --checkpoint=2000 --checkpoint-action=dot
    echo
    mv "$TAR.parcial" "$TAR"
    echo "    $(du -h "$TAR" | cut -f1)"
fi

# ------------------------------------------------------------- credenciais

ENV="$HD/lab.env"
if [ ! -f "$ENV" ]; then
    cat > "$ENV" <<'EOF'
# Preencha o token e salve. Cada maquina le este arquivo do HD, entao voce
# digita uma vez so em vez de dez.
LAB_URL=https://experimento.tremdigital.com.br
LAB_TOKEN=COLE_O_TOKEN_AQUI
EOF
    echo
    echo "=== criado $ENV — EDITE e cole o LAB_TOKEN antes de sair"
else
    echo
    echo "=== $ENV ja existe (mantido)"
fi

# --------------------------------------------------------------- instrucao

cat > "$HD/INSTALAR.txt" <<EOF
Em cada maquina do laboratorio:

  1. Plugue este HD.
  2. Descubra onde ele montou:   lsblk -o NAME,MOUNTPOINT | grep -v '^loop'
  3. Rode (ajustando o caminho do HD):

       bash /caminho/do/HD/Disaster-Damage/scripts/instalar_do_hd.sh

O script copia o codigo e os dados para o disco local, instala o torch
compativel com o driver da maquina, testa a conexao com o painel e deixa o
agente rodando. Ao final ele imprime o HOSTNAME e o IP da maquina — anote,
e com isso da para acompanhar por ssh depois.

Leva ~15 min por maquina, quase tudo copia de disco.
EOF

echo
echo "======================================================================"
echo "HD pronto: $HD"
du -sh "$HD/Disaster-Damage" "$TAR" 2>/dev/null | sed 's/^/    /'
echo
echo "FALTA: editar $ENV e colar o LAB_TOKEN."
echo
echo "Em cada maquina do laboratorio, rode:"
echo "    bash $HD/Disaster-Damage/scripts/instalar_do_hd.sh"
echo "(o caminho do HD muda de maquina para maquina; veja com lsblk)"
echo "======================================================================"
