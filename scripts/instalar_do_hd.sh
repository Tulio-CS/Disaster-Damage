#!/usr/bin/env bash
# Instala e dispara ESTA maquina, a partir do HD externo. Comando unico.
#
#     bash /caminho/do/HD/Disaster-Damage/scripts/instalar_do_hd.sh
#
# Faz, em ordem: copia o codigo, extrai os dados, monta o venv com a roda do
# torch compativel com o driver desta maquina, testa a conexao com o painel e
# deixa o agente rodando em segundo plano. Ao final imprime o hostname e o IP.
#
# Pode ser interrompido e rodado de novo: cada etapa detecta o que ja existe.

set -uo pipefail

AQUI="$(cd "$(dirname "$0")" && pwd)"
HD="$(dirname "$(dirname "$AQUI")")"      # .../HD/Disaster-Damage/scripts -> .../HD
DESTINO="${DESTINO:-$HOME/Disaster-Damage}"

echo "======================================================================"
echo "  $(hostname)  —  instalacao a partir do HD"
echo "  HD:      $HD"
echo "  destino: $DESTINO"
echo "======================================================================"

falhar() { echo; echo "!! $1"; echo; exit 1; }

[ -d "$HD/Disaster-Damage" ] || falhar "nao achei $HD/Disaster-Damage — o HD montou em outro lugar? veja com: lsblk -o NAME,MOUNTPOINT"
[ -f "$HD/xbd-data.tar" ]    || falhar "nao achei $HD/xbd-data.tar — o HD nao foi preparado com scripts/preparar_hd.sh"

# ------------------------------------------------------------- credenciais

if [ -f "$HD/lab.env" ]; then
    set -a; . "$HD/lab.env"; set +a
fi
: "${LAB_URL:=}"; : "${LAB_TOKEN:=}"
if [ -z "$LAB_URL" ] || [ -z "$LAB_TOKEN" ] || [ "$LAB_TOKEN" = "COLE_O_TOKEN_AQUI" ]; then
    falhar "LAB_URL/LAB_TOKEN nao definidos. Edite $HD/lab.env e cole o token."
fi
export LAB_URL LAB_TOKEN
export LAB_WORKER="${LAB_WORKER:-$(hostname)}"

# ------------------------------------------------------------------ espaco

LIVRE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo
echo "[1/6] espaco livre em $HOME: ${LIVRE_GB}G"
[ "${LIVRE_GB:-0}" -ge 45 ] || falhar "precisa de ~45 GB livres (33 de dados + venv + checkpoints); ha ${LIVRE_GB}G"

# ------------------------------------------------------------------ codigo

echo "[2/6] copiando o codigo..."
mkdir -p "$DESTINO"
rsync -a --exclude 'data/' "$HD/Disaster-Damage/" "$DESTINO/" || falhar "falha ao copiar o codigo"
cd "$DESTINO"
echo "      ok"

# ------------------------------------------------------------------- dados

echo "[3/6] extraindo os dados (33 GB, ~6 min)..."
ESPERADO=11034
ATUAL=$(find data/masks -name '*.png' 2>/dev/null | wc -l)
if [ "$ATUAL" -ge "$ESPERADO" ] && [ -f data/index.parquet ]; then
    echo "      ja extraidos ($ATUAL cenas); pulando"
else
    tar -xf "$HD/xbd-data.tar" -C "$DESTINO" --checkpoint=4000 --checkpoint-action=dot \
        || falhar "falha ao extrair xbd-data.tar"
    echo
    N=$(find data/masks -name '*.png' | wc -l)
    [ "$N" -ge "$ESPERADO" ] || falhar "extraiu so $N de $ESPERADO cenas — HD com defeito ou tar incompleto"
    echo "      $N cenas"
fi

# ---------------------------------------------------------------- ambiente

echo "[4/6] instalando o ambiente (torch conforme o driver; ~5 min)..."
bash scripts/lab_setup.sh 2>&1 | sed 's/^/      /' || falhar "lab_setup.sh falhou (veja o erro acima)"

# ----------------------------------------------------------------- conexao

echo "[5/6] testando a conexao com o painel..."
if ! bash scripts/testar_conexao.sh 2>&1 | sed 's/^/      /'; then
    echo
    echo "!! esta maquina nao alcanca o painel. O treino AINDA ASSIM funciona:"
    echo "   as metricas ficam no spool local e sobem quando a rede voltar."
    echo "   Siga adiante se for o caso, ou resolva a rede antes."
    echo
    echo "   Continuar assim mesmo? [s/N]"
    read -r resp
    [ "$resp" = "s" ] || [ "$resp" = "S" ] || exit 1
fi

# ------------------------------------------------------------------ agente

echo "[6/6] disparando o agente..."
if pgrep -f "xbd.agent" >/dev/null 2>&1; then
    echo "      ja havia um agente rodando; mantido"
else
    setsid nohup bash scripts/lab_worker.sh > "$DESTINO/agente.out" 2>&1 &
    sleep 8
    pgrep -f "xbd.agent" >/dev/null 2>&1 \
        && echo "      agente no ar" \
        || { echo "      !! o agente nao subiu; ultimas linhas:"; tail -15 "$DESTINO/agente.out"; }
fi

# ------------------------------------------------------------------- resumo

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

echo
echo "======================================================================"
echo "  PRONTO"
echo
printf "  hostname : %s\n" "$(hostname)"
printf "  IP       : %s        <-- ANOTE\n" "${IP:-nao detectado}"
printf "  GPU      : %s\n" "${GPU:-nenhuma}"
printf "  usuario  : %s\n" "$USER"
echo
echo "  acompanhar aqui:   tail -f $DESTINO/agente.out"
echo "  acompanhar remoto: ssh $USER@${IP:-IP} 'tail -f $DESTINO/agente.out'"
echo "  painel:            $LAB_URL"
echo
echo "  Pode desplugar o HD e ir para a proxima maquina."
echo "======================================================================"
