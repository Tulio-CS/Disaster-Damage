#!/usr/bin/env bash
# Confere se o HD preparado esta integro e completo.
#
#     bash scripts/conferir_hd.sh /media/tulio/XBD-LAB
#
# Rode antes de sair para o laboratorio, e de novo se desconfiar do HD.
# Leva ~2 min (le o indice do tar, nao extrai).
#
# Distingue "arquivo incompleto" de "nao consegui ler" de proposito. Contar
# membros sem checar o codigo de saida do tar faz qualquer falha de leitura —
# HD desplugado, permissao, I/O error — virar "0 cenas", que se parece com
# corrupcao e manda refazer um trabalho de 30 minutos sem motivo.

set -uo pipefail

HD="${1:-}"
[ -n "$HD" ] || { echo "uso: bash scripts/conferir_hd.sh /caminho/do/HD"; exit 1; }

CENAS_ESPERADAS=11034
problemas=0

aviso() { echo "  x $1"; problemas=$((problemas + 1)); }

echo "=== HD: $HD"

if ! mountpoint -q "$HD" 2>/dev/null && [ ! -d "$HD" ]; then
    echo "  x nao esta montado ou nao existe."
    echo "    Plugue o HD e confira o caminho com: lsblk -o NAME,LABEL,MOUNTPOINT"
    exit 2
fi

# ---- estrutura ----
echo
echo "--- estrutura ---"
for item in "Disaster-Damage" "xbd-data.tar" "lab.env" "INSTALAR.bat" "INSTALAR.txt"; do
    if [ -e "$HD/$item" ]; then
        echo "  ok  $item"
    else
        aviso "$item AUSENTE"
    fi
done

for s in xbd/train.py xbd/agent.py scripts/instalar.py scripts/rodar_agente.py \
         experiments/matrix.yaml requirements-treino.txt; do
    [ -f "$HD/Disaster-Damage/$s" ] || aviso "Disaster-Damage/$s AUSENTE"
done

# ---- token ----
echo
echo "--- credenciais ---"
if [ -f "$HD/lab.env" ]; then
    if grep -q "COLE_O_TOKEN_AQUI" "$HD/lab.env"; then
        aviso "lab.env ainda tem o token de exemplo — EDITE antes de sair"
    else
        url=$(grep -E '^LAB_URL=' "$HD/lab.env" | cut -d= -f2-)
        tok=$(grep -E '^LAB_TOKEN=' "$HD/lab.env" | cut -d= -f2-)
        [ -n "$url" ] || aviso "LAB_URL vazio em lab.env"
        [ -n "$tok" ] || aviso "LAB_TOKEN vazio em lab.env"
        [ -n "$url" ] && echo "  ok  LAB_URL=$url"
        [ -n "$tok" ] && echo "  ok  LAB_TOKEN definido (${#tok} caracteres)"
    fi
fi

# ---- tar ----
echo
echo "--- dados (lendo o indice do tar, ~2 min) ---"
TAR="$HD/xbd-data.tar"
if [ ! -f "$TAR" ]; then
    aviso "xbd-data.tar AUSENTE"
else
    GB=$(du -BG "$TAR" | cut -f1 | tr -dc '0-9')
    echo "  ..  tamanho: ${GB} GB"
    [ "$GB" -ge 30 ] || aviso "tar tem so ${GB} GB; esperado ~33"

    LISTA=$(mktemp)
    if tar -tf "$TAR" > "$LISTA" 2>/dev/null; then
        n_msk=$(grep -c '^data/masks/.*\.png$' "$LISTA")
        n_img=$(grep -c '^data/images/.*\.png$' "$LISTA")
        n_idx=$(grep -c '^data/index\.parquet$' "$LISTA")
        echo "  ..  mascaras: $n_msk   imagens: $n_img   indice: $n_idx"
        [ "$n_msk" -eq "$CENAS_ESPERADAS" ] || aviso "mascaras: $n_msk, esperado $CENAS_ESPERADAS"
        [ "$n_img" -eq $((CENAS_ESPERADAS * 2)) ] || aviso "imagens: $n_img, esperado $((CENAS_ESPERADAS * 2))"
        [ "$n_idx" -eq 1 ] || aviso "data/index.parquet ausente do tar"
        [ "$n_msk" -eq "$CENAS_ESPERADAS" ] && echo "  ok  tar completo e legivel"
    else
        # A distincao que importa: o tar existe mas nao pode ser lido.
        aviso "NAO CONSEGUI LER o tar (HD desplugado no meio? erro de I/O?)"
        echo "     Isso NAO significa que os dados estao corrompidos —"
        echo "     significa que a leitura falhou. Replugue e rode de novo."
    fi
    rm -f "$LISTA"
fi

echo
if [ "$problemas" -eq 0 ]; then
    echo "======================================================"
    echo "HD OK. Pode levar para o laboratorio."
    echo "Lembre de desmontar antes de tirar:"
    echo "    sync && udisksctl unmount -b \$(findmnt -no SOURCE $HD)"
    echo "======================================================"
    exit 0
fi
echo "======================================================"
echo "$problemas problema(s). Resolva antes de sair."
echo "======================================================"
exit 1
