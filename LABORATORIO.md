# Runbook do dia de laboratório

Guia operacional: preparar os dados, subir o painel no Coolify, distribuir para as
20 máquinas e disparar os experimentos.

---

## Ordem do dia

| # | Etapa | Onde | Tempo | Pode ser paralelo? |
|---|---|---|---|---|
| 1 | Preparar os dados (GeoTIFF → PNG) | sua máquina | ~25 min | — |
| 2 | Subir o painel no Coolify | navegador | ~10 min | sim, durante a 1 |
| 3 | Ensaio geral (matriz smoke) | 1 máquina | ~5 min | depois da 2 |
| 4 | Distribuir código + dados | rede do lab | 30-95 min | — |
| 5 | Instalar e disparar em cada máquina | 20 máquinas | ~10 min cada | sim, todas juntas |

As etapas 1 e 2 são independentes: comece a preparação dos dados e configure o
Coolify enquanto ela roda.

---

## 1. Preparar os dados (na sua máquina)

**Isto ainda não foi feito.** Hoje existem 268 das 11.034 cenas em `data/`.
O que vai para as máquinas do laboratório **não** é o `xview2_geotiff.tgz` (52 GB,
GeoTIFF `int16` cru) — é o `data/` derivado: PNG `uint8` + máscaras rasterizadas,
que é o que o treino lê.

```bash
bash scripts/preparar_dados.sh
```

Converte as 22.068 imagens usando todos os núcleos menos dois. Medido nesta
máquina (16 núcleos): **~25 min**, resultando em **~32 GB** em `data/`.

É idempotente — interromper e rodar de novo continua de onde parou. Ao terminar,
confira:

```bash
du -sh data && ls data/images/*/ | wc -l
```

Esperado: ~32 GB e 22.068 PNG de imagem.

> **Plano B se o dia ficar apertado.** Preparar só `tier1` + `hold` corta para
> 7.464 cenas (~21 GB, ~17 min) e reduz a cópia na mesma proporção:
>
> ```bash
> .venv/bin/python -m xbd.prep --shard 0 --of 1 --workers 14 --png-level 1 \
>     --splits tier1 hold
> .venv/bin/python -m xbd.prep_merge --out data
> ```
>
> Nesse caso ponha `usar_tier3: false` no bloco `comum:` de
> `experiments/matrix.yaml` e **remova o slot `abl-sem-tier3`** — sem tier3 na
> referência, essa ablação não muda nada e o `validar_matriz.py` vai acusar.

---

## 2. Subir o painel no Coolify

O repositório é **privado**, então o Coolify precisa de acesso autenticado.

### 2.1 Conectar o GitHub

Coolify → **Sources** → **+ Add** → **GitHub App** → siga o fluxo de instalação e
dê acesso ao repositório `Tulio-CS/Disaster-Damage`.

(Alternativa: tornar o repositório público e usar *Public Repository*. Mais
rápido, mas o `matrix.yaml` e o código ficam abertos — o token nunca fica no
repositório, então não há vazamento de credencial, só de código.)

### 2.2 Criar o recurso

Coolify → **Projects** → **+ New** → **Private Repository (GitHub App)**

| Campo | Valor |
|---|---|
| Repository | `Tulio-CS/Disaster-Damage` |
| Branch | `main` |
| Build Pack | **Dockerfile** |
| Base Directory | `/` |
| Dockerfile Location | `/server/Dockerfile` |
| Ports Exposes | `8000` |

O **Base Directory tem que ser `/`** (a raiz). O `server/Dockerfile` faz
`COPY experiments/matrix.yaml`, que está fora de `server/`; apontar a base para
`/server` faz o build falhar no COPY.

### 2.3 Variáveis de ambiente

Gere o token primeiro:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Em **Environment Variables**, adicione (todas como *runtime*, não *build*):

| Nome | Valor |
|---|---|
| `LAB_TOKEN` | o token gerado acima |
| `LAB_LEASE_SEG` | `2400` |

`LAB_TOKEN` é obrigatório — o servidor se recusa a subir sem ele, porque o painel
fica exposto na internet e a fila e os resultados ficariam abertos.

`LAB_LEASE_SEG` é quanto tempo sem sinal de vida até o painel considerar um run
perdido e devolver o slot para a fila. 2400 s (40 min) é folgado de propósito:
com `crop: 768` numa GPU modesta uma época passa de 20 min, e um lease curto
reenfileiraria um treino que está apenas demorando.

### 2.4 Volume persistente

**Storages** → **+ Add** → **Volume Mount**

| Campo | Valor |
|---|---|
| Name | `lab-data` |
| Destination Path | `/data` |

Este volume é o único estado do sistema: o SQLite com a fila e o progresso, e os
tfevents do TensorBoard. Sem ele, um redeploy zera os resultados do dia.

### 2.5 Domínio e deploy

Em **Domains**, ponha o FQDN (ex.: `https://lab.seudominio.com`). O Coolify
provisiona o HTTPS via Let's Encrypt. Depois **Deploy**.

### 2.6 Conferir

```bash
curl https://lab.seudominio.com/api/saude
# {"ok":true,"t":...}
```

Nos logs do deploy, procure as três linhas de inicialização:

```
[matriz] 24 experimentos (24 novos, 0 atualizados)
[tb] 127.0.0.1:6006 (pid ...)
INFO:     Application startup complete.
```

Se aparecer `[tb] FALHOU ao subir`, o traceback do TensorBoard vem junto na
mensagem — ele roda como subprocesso e o log dele fica em `/data/tensorboard.log`.

Abra o painel no navegador:

```
https://lab.seudominio.com/entrar?t=SEU_TOKEN
```

Isso troca o token por um cookie; daí em diante `/` e `/tb/` funcionam sem
carregar o token na URL.

---

## 3. Como as máquinas falam com o painel

**Só saída HTTPS.** Nenhuma máquina do laboratório recebe conexão, não é preciso
abrir porta, redirecionar NAT nem montar VPN. Se o navegador do laboratório
alcança a internet, o agente também alcança.

```
  máquina do lab                         VPS (Coolify)
 ┌────────────────┐                    ┌──────────────────────┐
 │ xbd.agent      │ ──POST /api/claim──▶│ fila (SQLite)        │
 │                │ ◀──experimento──── │                      │
 │   └ xbd.train  │                    │                      │
 │       └ spool  │ ──POST /api/eventos▶│ métricas → tfevents  │
 └────────────────┘    (a cada 5 s)    │        └ TensorBoard │
      Bearer token                     └──────────────────────┘
                                          você → /  e  /tb/
```

Três propriedades que importam num laboratório:

- **Posse exclusiva da fila.** As 20 máquinas puxam trabalho em vez de receber
  atribuição fixa. Quem termina primeiro pega o próximo slot. Um `BEGIN IMMEDIATE`
  no SQLite garante que duas máquinas nunca recebem o mesmo experimento.
- **A rede não derruba o treino.** Cada métrica é gravada primeiro num spool em
  disco (`results/spool/`) e enviada por uma thread separada. Se a VPS cair, o
  treino continua e os eventos sobem quando a rede voltar.
- **Retomada na mesma máquina.** Os checkpoints são locais. Se o agente reiniciar,
  o painel devolve o mesmo experimento para a mesma máquina, que retoma da época
  em que parou em vez de recomeçar do zero.

---

## 4. Ensaio geral (antes de comprometer 20 GPUs)

Rode a matriz `smoke` numa máquina só — 3 experimentos de 2 épocas, minutos, roda
até em CPU. Prova o caminho inteiro antes do investimento real.

No Coolify, mude temporariamente a variável `LAB_MATRIZ` para
`/app/experiments/matrix-smoke.yaml`... **ou**, mais simples, sem tocar no deploy:

```bash
curl -X POST https://lab.seudominio.com/api/matriz \
  -H "Authorization: Bearer $LAB_TOKEN" -H 'Content-Type: application/json' \
  --data "$(python3 -c '
import json,sys; print(json.dumps({"yaml": open("experiments/matrix-smoke.yaml").read()}))')"
```

Depois, na sua máquina mesmo:

```bash
export LAB_URL=https://lab.seudominio.com LAB_TOKEN=xxxx
.venv/bin/python -m xbd.agent --sair-vazio
```

Confira no painel que os 3 slots ficaram verdes e que aparecem curvas em `/tb/`.
Depois recarregue a matriz real:

```bash
curl -X POST https://lab.seudominio.com/api/matriz \
  -H "Authorization: Bearer $LAB_TOKEN" -H 'Content-Type: application/json' \
  --data "$(python3 -c '
import json; print(json.dumps({"yaml": open("experiments/matrix.yaml").read()}))')"
```

---

## 5. Distribuir para as 20 máquinas

### 5.1 O que vai

| Vai | Não vai |
|---|---|
| `xbd/` — pacote de treino | `geotiffs/` (131 GB) |
| `scripts/` | `xview2_geotiff.tgz` (52 GB) |
| `experiments/` | `.venv/` (cada máquina monta o seu) |
| `requirements-treino.txt` | `cache/`, `checkpoints/`, `results/` |
| `data/images/`, `data/masks/`, `data/index.parquet` (~32 GB) | |

O `scripts/distribuir_dados.sh` já aplica exatamente esses `--exclude`.

### 5.2 SSH sem senha

Confira em **todas** antes de disparar — uma máquina que pede senha trava a
cópia paralela:

```bash
ssh-keygen -t ed25519
for m in $(grep -vE '^\s*(#|$)' experiments/maquinas.txt); do ssh-copy-id "$m"; done
for m in $(grep -vE '^\s*(#|$)' experiments/maquinas.txt); do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$m" hostname || echo "FALHOU: $m"
done
```

Ajuste `experiments/maquinas.txt` para os nomes reais do laboratório.

### 5.3 Copiar

```bash
bash scripts/distribuir_dados.sh
```

**Sobre o tempo:** o gargalo é a placa de rede da *origem*, não das destinatárias.
32 GB × 20 = 640 GB; num link de 1 Gb/s são ~95 min, e aumentar `PARALELO` não
muda esse total — só reparte a mesma banda.

Se 95 min não couber no dia, distribua em árvore:

```bash
# onda 1: ~15 min
bash scripts/distribuir_dados.sh pc01 pc02 pc03 pc04

# onda 2: cada origem serve 4 destinos com a própria banda, ~20 min no total
ORIGEM=pc01 bash scripts/distribuir_dados.sh pc05 pc06 pc07 pc08 &
ORIGEM=pc02 bash scripts/distribuir_dados.sh pc09 pc10 pc11 pc12 &
ORIGEM=pc03 bash scripts/distribuir_dados.sh pc13 pc14 pc15 pc16 &
ORIGEM=pc04 bash scripts/distribuir_dados.sh pc17 pc18 pc19 pc20 &
wait
```

`rsync` é retomável: uma cópia interrompida continua de onde parou.

---

## 6. Em cada máquina

### 6.1 O que precisa estar instalado

O `lab_setup.sh` instala tudo dentro de um venv, sem root. Só dois pré-requisitos
ficam fora do alcance dele:

- **driver NVIDIA** funcionando (`nvidia-smi` responde)
- **`python3` ≥ 3.10 com o módulo `venv`**

Em Ubuntu/Debian, `python3 -m venv` costuma exigir o pacote `python3-venv`, que
precisa de root. Teste antes:

```bash
python3 -m venv /tmp/teste-venv && echo OK || echo "falta python3-venv"
```

Se faltar e você não tiver root, peça ao administrador
(`sudo apt install python3-venv`), ou use `virtualenv` no espaço do usuário:

```bash
pip install --user virtualenv && python3 -m virtualenv .venv
```

### 6.2 Instalar

```bash
cd ~/Disaster-Damage
bash scripts/lab_setup.sh
```

O script detecta a versão de CUDA pelo driver e instala a roda correspondente do
torch (`cu124`, `cu121` ou `cu118`), depois `requirements-treino.txt` (numpy,
pandas, pyarrow, Pillow, opencv-headless, PyYAML, psutil, timm).

Essa detecção não é excesso de zelo: instalar `cu124` numa máquina cujo driver só
suporta 12.1 resulta num `torch.cuda.is_available() == False` silencioso — o
treino cai para CPU, roda dezenas de vezes mais devagar, e ninguém percebe até o
fim do dia.

Ao final o script confere GPU, imports e a presença dos dados. Leia a saída: se
disser `! sem GPU visivel para o torch` ou `! faltam imagens`, resolva antes de
seguir.

### 6.3 Disparar

```bash
export LAB_URL=https://lab.seudominio.com
export LAB_TOKEN=xxxx
bash scripts/lab_worker.sh
```

Para deixar rodando depois de fechar a sessão SSH (o caso normal):

```bash
cd ~/Disaster-Damage
export LAB_URL=https://lab.seudominio.com LAB_TOKEN=xxxx
setsid nohup bash scripts/lab_worker.sh > agente.out 2>&1 &
tail -f agente.out
```

### 6.4 Disparar nas 20 de uma vez

Depois de conferir que funciona em uma:

```bash
for m in $(grep -vE '^\s*(#|$)' experiments/maquinas.txt); do
  ssh "$m" "cd ~/Disaster-Damage && \
    LAB_URL=https://lab.seudominio.com LAB_TOKEN=xxxx \
    setsid nohup bash scripts/lab_worker.sh > agente.out 2>&1 &" &
done; wait
```

Em seguida abra o painel: em um ou dois minutos as 20 máquinas aparecem na tabela
**Máquinas** e os slots começam a ficar azuis.

---

## 7. Acompanhar

| O quê | Onde |
|---|---|
| Estado das 20 máquinas, progresso por época, placar | `https://lab.seudominio.com/` |
| Curvas de treino e validação, aba HPARAMS | `https://lab.seudominio.com/tb/` |
| Tabela final para análise | `https://lab.seudominio.com/api/resultados.csv` |

No TensorBoard, o seletor de runs aceita regex: `^baseline` mostra só as
referências, `^ablacao` só o estudo.

**Antes de concluir qualquer coisa das ablações**, olhe os três slots `ref`,
`base-semente-1337` e `base-semente-2024`. São a mesma receita com sementes
diferentes: o espalhamento entre eles é o ruído da medida. Uma ablação que se
afasta da referência por menos que esse espalhamento não mediu efeito, mediu a
semente.

---

## 8. Problemas comuns

**Um slot ficou amarelo (`perdido`).** A máquina parou de dar sinal por mais que
`LAB_LEASE_SEG`. O slot já voltou para a fila sozinho. Se a máquina voltar, ela
recebe o mesmo experimento de volta e retoma do checkpoint local.

**Quero devolver um slot travado para a fila manualmente:**

```bash
curl -X POST https://lab.seudominio.com/api/experimentos/NOME/liberar \
     -H "Authorization: Bearer $LAB_TOKEN"
```

**Um slot ficou vermelho (`erro`).** Fica parado de propósito — reenfileirar só
repetiria a mesma falha. O painel mostra a última linha do erro; o log completo
está na máquina, em `results/logs/NOME.log`.

**`/tb/` responde 502.** O TensorBoard morreu. O corpo do 502 traz o fim de
`/data/tensorboard.log`. Um redeploy no Coolify o levanta de novo.

**Métricas pararam de chegar mas o treino continua.** É o comportamento esperado
com a rede fora do ar. Os eventos estão no spool da máquina
(`results/spool/*.jsonl`) e sobem sozinhos quando a conexão voltar.

**Fim do dia, quero parar sem perder o progresso.** `Ctrl-C` no agente (ou
`pkill -INT -f xbd.agent`). O sinal chega no treino, que salva o checkpoint,
marca `interrompido` e devolve o slot. No dia seguinte, `bash scripts/lab_worker.sh`
retoma exatamente da época em que parou.

**Mudei a matriz e quero recarregar sem redeploy:** o `POST /api/matriz` da seção
4. Experimentos já rodando não são afetados; só a config dos que ainda estão
`pendente` é atualizada.
