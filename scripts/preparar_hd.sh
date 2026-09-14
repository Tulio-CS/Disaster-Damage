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
    exfat|ntfs|ntfs3)
        echo "    ok — legivel no Windows, sem limite de tamanho de arquivo" ;;
    fuseblk)
        # ntfs-3g e exfat-fuse se apresentam os dois como 'fuseblk'.
        echo "    ok — legivel no Windows (montado via FUSE)"
        echo "    (o driver de kernel seria mais rapido para escrever 33 GB;"
        echo "     veja o runbook se a copia estiver lenta)" ;;
    ext4|ext3|xfs|btrfs)
        DEV=$(findmnt -no SOURCE "$HD" 2>/dev/null || echo /dev/sdX1)
        echo "! $SIS NAO e legivel pelo Windows sem driver de terceiros."
        echo "  As maquinas do laboratorio rodam Windows. Reformate:"
        echo "      sudo umount $HD"
        echo "      sudo mkfs.ntfs -f -L XBD-LAB $DEV     # -f = formato rapido"
        echo "  (ou mkfs.exfat -n XBD-LAB $DEV, se preferir exFAT)"
        exit 1 ;;
    vfat|msdos)
        echo "! FAT32 nao aceita arquivo maior que 4 GB, e o xbd-data.tar tem 33 GB."
        echo "  Reformate para exFAT (veja o runbook)."
        exit 1 ;;
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

# Atalho do Windows: clique duplo em vez de digitar caminho no terminal. O
# %~dp0 resolve para a pasta do .bat, entao funciona em qualquer letra de
# unidade — o HD pode montar como E:, F: ou G: dependendo da maquina.
bat() {   # bat <arquivo> <script.py> <comentario>
    printf '%s\r\n' \
      '@echo off' \
      "REM $3" \
      'cd /d "%~dp0"' \
      "where py >nul 2>&1 && (py \"%~dp0Disaster-Damage\\scripts\\$2\") || (python \"%~dp0Disaster-Damage\\scripts\\$2\")" \
      'if errorlevel 1 pause' \
      > "$HD/$1"
}
bat INSTALAR.bat instalar.py      "Instala e dispara esta maquina. Clique duplo."
bat COLETAR.bat  coletar.py       "Recolhe modelos e resultados desta maquina para o HD."
bat RELIGAR.bat  religar.py       "Religa o agente (apos reinicio da maquina)."

cat > "$HD/INSTALAR.txt" <<'EOF'
Em cada maquina do laboratorio (Windows):

  1. Plugue este HD.
  2. Abra o HD no Explorador de Arquivos.
  3. Clique duas vezes em INSTALAR.bat

     Se preferir o terminal (PowerShell ou cmd), com E: sendo a letra do HD:
         python E:\Disaster-Damage\scripts\instalar.py

Em Linux, o mesmo script:
     python3 /caminho/do/HD/Disaster-Damage/scripts/instalar.py

O script copia o codigo e os dados para a pasta do usuario, monta o ambiente
com a versao do torch compativel com o driver daquela maquina, testa a conexao
com o painel e deixa o agente rodando. NAO precisa de administrador.

Ao final ele imprime o HOSTNAME e o IP da maquina — ANOTE os dois.

Leva ~15 min por maquina, quase tudo copia de disco.

Se der erro de "python nao reconhecido", a maquina nao tem Python no PATH.
Tente `py` no lugar de `python`, ou instale pelo Microsoft Store (nao exige
administrador).
EOF

echo
echo "======================================================================"
echo "HD pronto: $HD"
du -sh "$HD/Disaster-Damage" "$TAR" 2>/dev/null | sed 's/^/    /'
echo
echo "FALTA: editar $ENV e colar o LAB_TOKEN."
echo
echo "Em cada maquina do laboratorio, rode:"
echo "    (Windows) clique duplo em INSTALAR.bat, na raiz do HD"
echo "ou:  python E:\\Disaster-Damage\\scripts\\instalar.py"
echo "======================================================================"
