#!/usr/bin/env bash
# Vistoria as maquinas do laboratorio ANTES de copiar 33 GB para cada uma.
#
#     bash scripts/checar_maquinas.sh              # todas de experiments/maquinas.txt
#     bash scripts/checar_maquinas.sh pc07 pc11    # so estas
#
# Confere, em cada uma: SSH sem senha, GPU e driver, python3 com o modulo venv,
# espaco em disco e rsync.
#
# Existe porque cada um desses problemas custa caro se descoberto tarde. Faltar
# `python3-venv` exige root, que voce pode nao ter as 9h de um sabado; disco
# cheio interrompe a copia no meio; e um driver antigo faz o lab_setup.sh
# instalar a roda errada do torch, o que vira um treino em CPU silencioso.
# Todos aparecem aqui em 10 segundos.

set -uo pipefail
cd "$(dirname "$0")/.."

LISTA=${LISTA:-experiments/maquinas.txt}
USUARIO=${USUARIO:-$USER}
DESTINO=${DESTINO:-Disaster-Damage}
MIN_GB=${MIN_GB:-45}       # 33 GB de dados + venv + checkpoints do run

if [ $# -gt 0 ]; then
    MAQUINAS=("$@")
else
    [ -f "$LISTA" ] || { echo "! $LISTA nao existe"; exit 1; }
    mapfile -t MAQUINAS < <(grep -vE '^\s*(#|$)' "$LISTA")
fi
[ ${#MAQUINAS[@]} -gt 0 ] || { echo "! nenhuma maquina informada"; exit 1; }

# Roda na maquina remota. Uma linha de campos separados por '|', para o lado de
# ca formatar. Nada aqui depende do repositorio ja estar copiado.
SONDA='
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d "\n")
  [ -z "$gpu" ] && gpu="SEM GPU"
  cuda=$(nvidia-smi 2>/dev/null | sed -n "s/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p" | head -1)
  [ -z "$cuda" ] && cuda="-"
  py=$(python3 -V 2>&1 | awk "{print \$2}")
  [ -z "$py" ] && py="AUSENTE"
  if python3 -c "import venv" 2>/dev/null; then venv="ok"; else venv="FALTA"; fi
  gb=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc "0-9")
  [ -z "$gb" ] && gb=0
  if command -v rsync >/dev/null 2>&1; then rs="ok"; else rs="FALTA"; fi
  nucleos=$(nproc 2>/dev/null || echo "?")
  echo "$(hostname)|$gpu|$cuda|$py|$venv|$gb|$rs|$nucleos"
'

printf "%-12s %-26s %-6s %-8s %-6s %-7s %-6s %s\n" \
       HOST GPU CUDA PYTHON VENV DISCO RSYNC NUCLEOS
printf '%.0s-' {1..92}; echo

problemas=()
ok=0

for m in "${MAQUINAS[@]}"; do
    linha=$(ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
                "$USUARIO@$m" "$SONDA" 2>/dev/null)
    if [ -z "$linha" ]; then
        printf "%-12s %s\n" "$m" "SEM ACESSO SSH (chave nao instalada, maquina desligada ou nome errado)"
        problemas+=("$m: sem acesso ssh")
        continue
    fi

    IFS='|' read -r host gpu cuda py venv gb rs nucleos <<< "$linha"
    printf "%-12s %-26s %-6s %-8s %-6s %-7s %-6s %s\n" \
           "$m" "${gpu:0:26}" "$cuda" "$py" "$venv" "${gb}G" "$rs" "$nucleos"

    falhou=0
    [ "$gpu" = "SEM GPU" ]   && { problemas+=("$m: nvidia-smi nao responde"); falhou=1; }
    [ "$venv" = "FALTA" ]    && { problemas+=("$m: falta python3-venv (precisa de root)"); falhou=1; }
    [ "$py" = "AUSENTE" ]    && { problemas+=("$m: sem python3"); falhou=1; }
    [ "$rs" = "FALTA" ]      && { problemas+=("$m: falta rsync"); falhou=1; }
    [ "${gb:-0}" -lt "$MIN_GB" ] && { problemas+=("$m: so ${gb}G livres, precisa de ${MIN_GB}G"); falhou=1; }
    case "$py" in
        3.1[0-9]*|3.[2-9]*) ;;
        AUSENTE) ;;
        *) problemas+=("$m: python $py e antigo demais (precisa >= 3.10)"); falhou=1 ;;
    esac
    [ "$falhou" -eq 0 ] && ok=$((ok + 1))
done

echo
echo "$ok de ${#MAQUINAS[@]} maquina(s) prontas."

if [ ${#problemas[@]} -gt 0 ]; then
    echo
    echo "Pendencias:"
    printf '  x %s\n' "${problemas[@]}"
    echo
    echo "Como resolver:"
    echo "  sem acesso ssh   -> ssh-copy-id $USUARIO@MAQUINA   (uma vez por maquina)"
    echo "  falta python3-venv -> sudo apt install python3-venv"
    echo "                        ou, sem root: pip install --user virtualenv"
    echo "  falta rsync      -> sudo apt install rsync"
    echo "  nvidia-smi mudo  -> driver nao carregado; um reboot costuma resolver"
    echo "  disco cheio      -> DESTINO=/outro/caminho bash scripts/distribuir_dados.sh"
    exit 1
fi

echo "Pode seguir para: bash scripts/distribuir_dados.sh"
