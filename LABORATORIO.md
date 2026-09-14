# Runbook do dia de laboratório

Guia operacional: preparar os dados, subir o painel no Coolify, preparar o HD e
disparar os experimentos nas 10 máquinas Windows do laboratório.

---

## Ordem do dia

| # | Etapa | Onde | Tempo | Pode ser paralelo? |
|---|---|---|---|---|
| 1 | Preparar os dados (GeoTIFF → PNG) | sua máquina | ~25 min | — |
| 2 | Subir o painel no Coolify | navegador | ~10 min | sim, durante a 1 |
| 3 | Ensaio geral (matriz smoke) | 1 máquina | ~5 min | depois da 2 |
| 4 | Preparar o HD externo (exFAT + 33 GB) | sua máquina | ~15 min | sim, durante a 1 |
| 5 | Rodar `INSTALAR.bat` em cada máquina | 10 máquinas | ~15 min cada | não — um HD só |

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

Em **Domains**, ponha o FQDN (ex.: `https://experimento.tremdigital.com.br`). O Coolify
provisiona o HTTPS via Let's Encrypt. Depois **Deploy**.

### 2.6 Conferir

```bash
curl https://experimento.tremdigital.com.br/api/saude
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
https://experimento.tremdigital.com.br/entrar?t=SEU_TOKEN
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

- **Posse exclusiva da fila.** As 10 máquinas puxam trabalho em vez de receber
  atribuição fixa. Quem termina primeiro pega o próximo slot. Um `BEGIN IMMEDIATE`
  no SQLite garante que duas máquinas nunca recebem o mesmo experimento.
- **A rede não derruba o treino.** Cada métrica é gravada primeiro num spool em
  disco (`results/spool/`) e enviada por uma thread separada. Se a VPS cair, o
  treino continua e os eventos sobem quando a rede voltar.
- **Retomada na mesma máquina.** Os checkpoints são locais. Se o agente reiniciar,
  o painel devolve o mesmo experimento para a mesma máquina, que retoma da época
  em que parou em vez de recomeçar do zero.

### 3.1 Confirmar de dentro do laboratório

O `INSTALAR.bat` já testa isso sozinho, na etapa 5 de 6, e avisa se falhar. Mas
se quiser confirmar antes de comprometer as 10 máquinas, leve o HD a **uma** e
abra o PowerShell:

```
python E:\Disaster-Damage\scripts\testar_conexao.py
```

Esperado:

```
  ok    DNS                        92.113.39.129
  ok    TLS + certificado          TLSv1.3  emissor=Let's Encrypt
  ok    GET /api/saude             HTTP 200
  ok    GET /api/estado (token)    HTTP 200  24 experimentos (24 pendentes)
```

O teste usa `urllib.request`, a mesma pilha do `xbd/reporter.py`. Numa rede com
proxy de inspeção TLS — comum em instituição — o navegador confia porque a CA
corporativa está instalada para ele, e o Python rejeita com
`CERTIFICATE_VERIFY_FAILED`. Testar pelo navegador daria um falso OK, e o erro
só apareceria quando a primeira época tentasse subir.

Se for o caso, peça a CA ao administrador da rede e defina antes de instalar:

```
set SSL_CERT_FILE=C:\caminho\ca-corporativa.pem
```

## 4. Ensaio geral (antes de comprometer 20 GPUs)

Rode a matriz `smoke` na SUA máquina — 3 experimentos de 2 épocas, minutos, roda
até em CPU. Prova o caminho inteiro antes do investimento real.

No Coolify, mude temporariamente a variável `LAB_MATRIZ` para
`/app/experiments/matrix-smoke.yaml`... **ou**, mais simples, sem tocar no deploy:

```bash
curl -X POST https://experimento.tremdigital.com.br/api/matriz \
  -H "Authorization: Bearer $LAB_TOKEN" -H 'Content-Type: application/json' \
  --data "$(python3 -c '
import json,sys; print(json.dumps({"yaml": open("experiments/matrix-smoke.yaml").read()}))')"
```

Depois, na sua máquina mesmo:

```bash
export LAB_URL=https://experimento.tremdigital.com.br LAB_TOKEN=xxxx
.venv/bin/python -m xbd.agent --sair-vazio
```

Confira no painel que os 3 slots ficaram verdes e que aparecem curvas em `/tb/`.
Depois recarregue a matriz real:

```bash
curl -X POST https://experimento.tremdigital.com.br/api/matriz \
  -H "Authorization: Bearer $LAB_TOKEN" -H 'Content-Type: application/json' \
  --data "$(python3 -c '
import json; print(json.dumps({"yaml": open("experiments/matrix.yaml").read()}))')"
```

---

## 5. Preparar o HD externo

As 10 maquinas rodam **Windows**, voce **nao tem privilegio de administrador**
nelas, e a distribuicao e por HD. Isso define tres coisas:

| Restricao | Consequencia |
|---|---|
| HD lido no Windows | exFAT ou NTFS. `ext4` nao serve; FAT32 tambem nao, por causa do limite de 4 GB por arquivo |
| Sem administrador | Nada de instalar Python, driver ou servico. Tudo dentro da pasta do usuario |
| Windows | Os `.sh` nao rodam. O instalador e um `.py`, que serve aos dois sistemas |

### 5.1 Formatar em NTFS (ou exFAT)

```bash
lsblk -o NAME,SIZE,MODEL,TRAN,MOUNTPOINT      # confirme TRAN=usb antes!
sudo umount /dev/sdb1
sudo mkfs.ntfs -f -L XBD-LAB /dev/sdb1        # -f = formato rapido
```

**`-f` nao e opcional.** Sem ele, o `mkfs.ntfs` faz formato completo e zera os
931 GB do disco inteiro — horas de espera sem ganho nenhum.

NTFS em vez de exFAT porque `mkfs.ntfs` ja vem instalado (pacote `ntfs-3g`),
enquanto o exFAT exigiria `apt install exfatprogs`. Os dois sao lidos e escritos
pelo Windows sem driver extra e nao tem o limite de 4 GB por arquivo do FAT32.
Se preferir exFAT: `sudo mkfs.exfat -n XBD-LAB /dev/sdb1`.

> **Confira o dispositivo.** `/dev/sda` e o disco interno de 447 GB com o
> projeto, os 33 GB preparados e os 131 GB de GeoTIFF. Um `b` trocado por `a`
> destroi tudo isso.

Desplugue, replugue, e ajuste o dono (uma vez, so nesta maquina):

```bash
sudo chown $USER:$USER /media/$USER/XBD-LAB
```

### 5.2 Montar o HD

```bash
bash scripts/preparar_hd.sh /media/$USER/XBD-LAB
```

Leva ~10 min. Ao final o HD fica assim:

```
XBD-LAB/
├── INSTALAR.bat         <- clique duplo no Windows
├── INSTALAR.txt         instrucoes impressas
├── lab.env              LAB_URL + LAB_TOKEN
├── xbd-data.tar         33 GB
└── Disaster-Damage/     codigo
```

**Edite `lab.env` e cole o `LAB_TOKEN`** antes de sair. E o que evita digitar o
token dez vezes.

### 5.3 Conferir antes de sair

```bash
bash scripts/conferir_hd.sh /media/$USER/XBD-LAB
```

Confere estrutura, se o token foi preenchido, e lê o índice do tar para contar
as 11.034 cenas. ~2 min, e é a última chance de descobrir um problema ainda em
casa.

Depois, desmonte com segurança:

```bash
sync && udisksctl unmount -b /dev/sdb1
```

Não arranque o cabo direto. Depois de escrever 33 GB há cache pendente, e em
NTFS um desligamento sujo marca o volume como *dirty* — o Windows então exige
`chkdsk` antes de montar, e você descobre isso parado na frente da primeira
máquina.

Os dados viram um `.tar` unico em vez de 22.068 arquivos soltos porque copiar
milhares de arquivos pequenos de um HD USB e dominado por latencia de acesso:
cai para 20-40 MB/s, contra ~110 MB/s de leitura sequencial. Sao ~25 min contra
~6 min **por maquina**.

---

## 6. Em cada maquina (10x)

1. Plugue o HD.
2. Abra no Explorador de Arquivos.
3. Clique duas vezes em **`INSTALAR.bat`**.

Pelo terminal, se preferir (com `E:` sendo a letra do HD):

```
python E:\Disaster-Damage\scripts\instalar.py
```

O script faz tudo sem administrador: copia o codigo, extrai os 33 GB, monta o
venv, instala a versao do torch compativel com o driver **daquela** maquina,
testa a conexao com o painel e deixa o agente rodando. ~15 min, quase tudo copia
de disco. E retomavel — se travar, rode de novo.

Ao final imprime o que voce precisa anotar:

```
hostname : LAB-PC-03
IP       : 10.20.30.47        <-- ANOTE
usuario  : aluno
GPU      : NVIDIA GeForce RTX 3060
```

**E assim que voce descobre os IPs.** Voce ja esta na maquina; nao depende de
varredura de rede, que numa rede institucional costuma falhar por isolamento
entre clientes ou VLAN por sala.

### Se der errado

| Sintoma | Causa e saida |
|---|---|
| `python nao e reconhecido` | Python fora do PATH. Tente `py` no lugar de `python`; se nao houver, instale pela Microsoft Store (nao exige administrador) |
| `torch NAO enxerga a GPU` | Driver incompativel com a roda instalada. O script imprime a versao do driver que detectou; confira com `nvidia-smi` |
| `sem contato com o painel` | Rede bloqueando. Aceite seguir: o treino roda e as metricas ficam no spool local, subindo quando a conexao voltar |
| `extraiu N de 11034 cenas` | Tar incompleto ou HD com defeito. Refaca o HD |

### Religar uma maquina depois

```
cd %USERPROFILE%\Disaster-Damage
.venv\Scripts\python.exe scripts\rodar_agente.py
```

Retoma o experimento do checkpoint local, da epoca exata em que parou.

---

## 7. Acompanhar

| O quê | Onde |
|---|---|
| Estado das 10 máquinas, progresso por época, placar | `https://experimento.tremdigital.com.br/` |
| Curvas de treino e validação, aba HPARAMS | `https://experimento.tremdigital.com.br/tb/` |
| Tabela final para análise | `https://experimento.tremdigital.com.br/api/resultados.csv` |

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
curl -X POST https://experimento.tremdigital.com.br/api/experimentos/NOME/liberar \
     -H "Authorization: Bearer $LAB_TOKEN"
```

**Um slot ficou vermelho (`erro`).** Fica parado de propósito — reenfileirar só
repetiria a mesma falha. O painel mostra a última linha do erro; o log completo
está na máquina, em `%USERPROFILE%\Disaster-Damage\results\logs\NOME.log`.

**`/tb/` responde 502.** O TensorBoard morreu. O corpo do 502 traz o fim de
`/data/tensorboard.log`. Um redeploy no Coolify o levanta de novo.

**Métricas pararam de chegar mas o treino continua.** É o comportamento esperado
com a rede fora do ar. Os eventos estão no spool da máquina
(`results/spool/*.jsonl`) e sobem sozinhos quando a conexão voltar.

**Fim do dia, quero parar sem perder o progresso.** `Ctrl-C` na janela do agente.
O sinal chega no treino (via `CTRL_BREAK_EVENT` no Windows), que salva o
checkpoint, marca `interrompido` e devolve o slot. No dia seguinte,
`.venv\Scripts\python.exe scripts\rodar_agente.py` retoma da época exata.

**Mudei a matriz e quero recarregar sem redeploy:** o `POST /api/matriz` da seção
4. Experimentos já rodando não são afetados; só a config dos que ainda estão
`pendente` é atualizada.
