# Metodologia

Avaliação automática de danos em edificações a partir de pares de imagens de
satélite pré e pós-desastre, no dataset **xBD** / desafio **xView2** [1, 2].

Cada decisão abaixo tem origem declarada. Quando a origem é a análise
exploratória deste repositório (`notebooks/01_eda_xbd.ipynb`), o achado está
citado com o número medido; quando é literatura, está referenciado em
[EXPERIMENTOS.md § Fontes](EXPERIMENTOS.md#fontes).

---

## 1. Tarefa

Dadas duas imagens co-registradas da mesma cena — uma antes e uma depois de um
desastre —, produzir uma segmentação semântica em 5 classes por pixel:

| id | classe | significado |
|---|---|---|
| 0 | `background` | não é edificação |
| 1 | `no-damage` | edificação intacta |
| 2 | `minor-damage` | dano leve |
| 3 | `major-damage` | dano severo |
| 4 | `destroyed` | destruída |

O rótulo `un-classified` do dataset original **não** é um quinto nível de
severidade — é ausência de julgamento do anotador. Ele é mapeado para
`IGNORE_INDEX` (255) e removido de toda perda e de toda métrica
([config.py:15](xbd/config.py:15)). Tratá-lo como classe introduziria um alvo
que não corresponde a nenhum estado físico.

A tarefa é conjunta: **localização** (onde há edificação) e **classificação de
severidade** (quão danificada). A métrica oficial pesa as duas.

---

## 2. Dados

### 2.1 Fonte e volume

| | |
|---|---|
| Distribuição | GeoTIFF (`xview2_geotiff.tgz`, 52 GB) |
| Extraído | 139 GB, 44.136 arquivos |
| Cenas | 11.034 pares pré/pós, 1024×1024, 3 bandas, EPSG:4326 |
| Polígonos | 850.736, em WKT de pixel e de lon/lat |
| Splits originais | `tier1` 2.799 · `tier3` 6.369 · `hold` 933 · `test` 933 cenas |

### 2.2 Pré-processamento

Os GeoTIFF são `int16` sem compressão, mas com conteúdo de 8 bits. A conversão
para PNG `uint8` é **sem perda de informação** e reduz 139 GB para 33 GB. Isso
não é economia de disco por si: ler `int16` cru a cada época desperdiça banda de
I/O, que é o gargalo real em máquina sem SSD rápido.

Uma máscara por cena basta. A EDA mediu que os polígonos pré e pós são
geometricamente idênticos (mesmos uids, deslocamento mediano de 1e-7 px), então
a máscara de localização é simplesmente `máscara_dano > 0`.

Polígonos auto-intersectantes (80 casos) são corrigidos com `buffer(0)`, e a
rasterização ordena por área decrescente para que edificações pequenas
sobrepostas não desapareçam sob as grandes ([prep.py:83](xbd/prep.py:83)).

### 2.3 Propriedades que governaram o desenho

Cinco achados da EDA, cada um com consequência direta no pipeline:

**Desbalanceamento de ~10:1** a favor de `no-damage`, e a composição muda
drasticamente entre eventos: 99,5% `no-damage` em `mexico-earthquake` contra
17,9% em `hurricane-matthew`. → pesos por classe e reamostragem (§4, §5).

**97% dos pixels são fundo**, e 35% das cenas não têm nenhuma edificação. →
entropia cruzada pura converge para "preveja fundo"; daí o termo Dice (§4).

**Objetos pequenos**: mediana de 575 px² (~24×24 px), 71% classificados como
"small" pelo critério COCO [3]. Na escala mais grossa do encoder (stride 32) a
edificação mediana ocupa menos de um pixel. → skips em stride 2 e 4 são
essenciais (§3).

**Gap temporal mediano de 304 dias** entre as imagens pré e pós (5% acima de 3,3
anos), com off-nadir de 0° a 44°. → boa parte da diferença visual entre as datas
**não é dano**: é estação, iluminação, ângulo e construção nova. A diferença crua
entre pixels é um sinal ruim; a diferença em espaço de *features* é o que
interessa (§3).

**Vazamento espacial no split oficial**: em 8 dos 10 eventos avaliados, mais de
dois terços dos tiles de `hold`/`test` são adjacentes (<500 m) a um tile de
treino. → o número oficial mede generalização para o quarteirão vizinho, não
para um desastre novo (§6).

### 2.4 Augmentação

Calibrada por medição, não por convenção (`scripts/analyze_imagery.py`):

| Transformação | Faixa | Medida que a justifica |
|---|---|---|
| brilho | ±30 DN | 32% dos pares diferem >10 DN de brilho médio; 11% >25 DN |
| contraste | 0,75–1,35× | idem |
| nitidez | blur/sharpen | 35% dos pares diferem >2× na variância do laplaciano |
| ruído | σ 1–8 DN | σ medido entre 1,0 e 12,5 DN conforme o evento |
| matiz | **não usado** | bandas R,G,B com correlação 0,93–0,98: quase não há informação cromática independente |

O ponto crítico: as fotométricas são aplicadas **independentemente** em pré e
pós, porque é exatamente assim que o descasamento aparece no dataset real. As
geométricas (crop, flip, rot90) são aplicadas **em conjunto**, porque as duas
datas são co-registradas e compartilham a máscara.

---

## 3. Modelo

U-Net [4] siamesa com encoder do `timm` [5].

```
        pré ──┐
              ├── encoder compartilhado ──┐
        pós ──┘                           ├── fusão por escala ── decoder ── 5 classes
                                          │
        [f_pré | f_pós | |f_pós − f_pré|] ┘
```

**Encoder compartilhado entre as duas datas.** Os polígonos são idênticos nas
duas, então a rede não precisa reaprender a localizar — precisa comparar. Pesos
compartilhados forçam as duas imagens para o mesmo espaço de features e cortam
quase metade dos parâmetros. É a arquitetura siamesa clássica [6], na forma que
a literatura de detecção de mudança adotou para pares de imagens de satélite [7].

**Fusão por concatenação + diferença absoluta**, em cada escala. A diferença
crua entre imagens é ruim (§2.3), mas `|f_pós − f_pré|` em espaço de features é
o sinal que interessa. Esta é a hipótese central do modelo, e é testada
diretamente pelas ablações de `fuse_mode` ([EXPERIMENTOS.md §3.1](EXPERIMENTOS.md)).

**Skips em stride 2 e 4**, porque 71% das edificações são menores que 32×32 px.

**Uma cabeça de 5 classes, não duas.** A localização sai de `1 − p(fundo)`,
então as duas tarefas dividem todo o decoder e a perda de localização ainda
supervisiona a mesma saída. Duas cabeças independentes permitiriam que a rede
localizasse bem e classificasse mal sem que os gradientes se falassem.

Tamanhos medidos (decoder 256-128-64-32-16, 5 classes):

| encoder | total | só o encoder |
|---|---|---|
| `mobilenetv3_small_100` [8] | 4,59 M | 0,93 M |
| `efficientnet_b0` [9] | 6,19 M | 3,60 M |
| `resnet34` [10] — **referência** | 25,50 M | 21,28 M |
| `seresnext50_32x4d` [11, 12] | 51,26 M | 25,51 M |

---

## 4. Função de perda

Duas pressões governam a escolha, ambas vindas de §2.3.

**Termo de dano** (5 classes): Focal [13] + Dice [14].
Entropia cruzada pura converge para "preveja fundo" quando 97% dos pixels são
fundo. Dice mede sobreposição em vez de acerto por pixel, então não tem esse
ótimo local; Focal reduz o peso dos pixels já fáceis. Usar os dois juntos é o
que as soluções vencedoras do xView2 fazem [15].

**Termo de localização**: BCE + Dice binário, derivados da mesma saída de 5
classes. O logit binário de "existe edificação" sai exato das 5 saídas:

```
p(edif) = 1 − softmax(logits)[0] = Σ_{c≥1} softmax(logits)[c]
logit   = log(p/(1−p)) = logsumexp(logits[1:]) − logits[0]
```

porque o log-denominador do softmax cancela entre os dois termos. Usar o logit
(e não a probabilidade) é obrigatório sob precisão mista: `binary_cross_entropy`
é proibido em autocast, porque aplicar `log` a uma probabilidade já arredondada
em 16 bits perde precisão perto de 0 e 1 ([losses.py:147](xbd/losses.py:147)).

**Pesos por classe** derivados do inverso da frequência, amortecidos por raiz
quadrada e limitados a 10×. O inverso puro daria ~40× para `destroyed` e
desestabilizaria o treino.

---

## 5. Amostragem

Um `WeightedRandomSampler` com dois ajustes ([data.py:227](xbd/data.py:227)):

- cenas **com dano** recebem peso `oversample_dano` (3,0 na referência)
- cenas **sem nenhuma edificação** recebem peso 0,25

Sem isso, com 35% de cenas vazias e só 1 em 3 com dano, a maior parte dos passos
de otimização não veria nenhuma classe positiva. O valor 3,0 não é calibrado —
é o ponto de partida, e as ablações `abl-sem-oversample` e (na reserva) valores
alternativos medem quanto ele importa.

O crop de treino faz até duas tentativas de cair sobre uma edificação antes de
aceitar qualquer posição, o que é um segundo nível de reamostragem, dentro da
cena.

---

## 6. Divisões de dados

Dois modos, e a diferença entre eles é um resultado do estudo, não um detalhe:

**`oficial`** — reproduz a divisão do xView2: `tier1` (+`tier3`) treina, `hold`
valida. É o número comparável com a literatura.

**`grouped`** — agrupa por **desastre**: nenhum evento aparece nos dois lados,
com atribuição estável evento → fold. Mede generalização para um desastre novo,
que é o caso de uso real: um modelo de avaliação de danos serve para o próximo
furacão, não para o quarteirão vizinho do furacão que já foi anotado.

A EDA mostrou que o split oficial tem vazamento espacial forte (§2.3). Split
agrupado é a resposta padrão para dados com autocorrelação espacial [16], e a
diferença entre os dois números é o que quantifica o otimismo do protocolo
oficial. Cinco folds dão a queda média e o desvio entre eventos.

---

## 7. Métrica

A métrica oficial do xView2 [2]:

```
score = 0,3 · F1_localização + 0,7 · F1_dano
```

onde `F1_dano` é a **média harmônica** dos F1 das 4 classes de severidade.

A média harmônica é o ponto central do desenho: ela é dominada pelo pior termo,
então zerar `minor-damage` (a classe mais rara) derruba o score inteiro. Com
média aritmética daria para ignorar as classes raras e ainda pontuar bem — e é
exatamente isso que o desbalanceamento de 10:1 do xBD incentiva.

Também são registradas, por classe e por época: F1, IoU, **precisão** e
**recall**, além da **matriz de confusão completa**. Precisão e recall separados
importam porque o score é dominado pela pior classe: saber se `minor-damage` vai
mal por falta de recall (o modelo não acha o dano leve) ou de precisão (chama de
leve o que está intacto) aponta correções opostas. F1 e IoU sozinhos não
distinguem — são funções um do outro (`F1 = 2·IoU/(1+IoU)`).

A matriz de confusão é persistida junto das métricas, o que permite recalcular
qualquer métrica derivada depois, retroativamente, sem reexecutar nada.

---

## 8. Protocolo de treino

| | referência |
|---|---|
| Otimizador | AdamW [17], lr 3e-4, weight decay 1e-4 |
| lr do encoder | 0,1× o do decoder (pré-treinado aprende mais devagar) |
| Sem weight decay | em bias e parâmetros de normalização |
| Scheduler | warmup linear 3% + cosseno [18] |
| Épocas | 20 |
| Batch efetivo | 16, por acumulação de gradiente |
| Precisão | bf16 quando a GPU suporta, senão fp16, senão fp32 [19] |
| Crop | 512×512 |
| Clip de gradiente | 1,0 |

Batch por dispositivo, número de workers, precisão e uso de `channels_last` são
**derivados do hardware em tempo de execução** ([hardware.py](xbd/hardware.py)) e
não entram no hash de configuração — duas máquinas diferentes rodando a mesma
receita produzem o mesmo `run_id`, o que permite agregar os resultados.

Quando a VRAM aperta, o plano ativa gradient checkpointing [20] antes de reduzir
o crop, porque reduzir o crop mudaria o experimento.

---

## 9. Reprodutibilidade

**Identidade determinista do run.** O `run_id` é
`<nome>-<task>-f<fold>-s<seed>-<hash>`, onde o hash cobre exatamente os campos
que definem o experimento — e exclui os de execução. Mudar o batch por
dispositivo não muda o hash; mudar a perda muda.

**Validação de configuração.** Campos categóricos são verificados contra listas
de valores permitidos ([config.py:134](xbd/config.py:134)). Um `loss_dmg:
focal-dice` (com hífen) falha na carga em vez de cair silenciosamente num `else`
e virar um experimento diferente do que o nome promete — o modo de falha mais
caro num estudo de ablação, porque só aparece na análise.

**Conferência prévia da matriz.** `scripts/validar_matriz.py` constrói de
verdade a config de cada slot, detecta dois slots com `run_id` idêntico
(seriam o mesmo experimento com dois nomes, escrevendo no mesmo checkpoint), e
acusa ablações que não mudam nada ou que mudam mais de um fator.

**Retomada exata.** Todo checkpoint carrega modelo, otimizador, scheduler,
scaler, época e estado do RNG, então retomar continua a mesma trajetória em vez
de começar uma nova. Escrita atômica (grava em `.tmp` e renomeia): um processo
morto no meio nunca corrompe o checkpoint bom.

**Semente de augmentação derivada do índice**, para que a validação seja idêntica
entre máquinas.

**Proveniência.** Cada run registra SHA do git, se a árvore estava suja, versão
do Python e do torch, modelo de CPU e GPU, e o plano de execução completo.

---

## 10. Infraestrutura do experimento

10 máquinas Windows com GPU, sem privilégio de administrador, na rede do IBMEC;
painel numa VPS externa (Coolify). Ver [LABORATORIO.md](LABORATORIO.md) para o
procedimento operacional.

O ponto metodológico relevante: as máquinas **puxam** experimentos de uma fila
com posse exclusiva, em vez de receberem atribuição fixa. Com atribuição fixa, o
experimento de uma máquina que ficou ocupada, desligada ou com disco cheio
simplesmente não acontece, e isso só se descobre no fim. Puxando da fila, quem
termina primeiro pega o próximo, e um slot cuja máquina morreu volta para a fila
sozinho depois que o *lease* expira.

Métricas sobem por HTTP e são convertidas em tfevents no servidor, então um
único TensorBoard enxerga o laboratório inteiro sem diretório compartilhado.
Cada evento é gravado primeiro num spool local e enviado por thread separada:
rede caindo não derruba treino.

**Congelamento de código.** A partir do início da execução, `xbd/` não muda. Se
a máquina 3 rodar uma versão da perda e a máquina 8 outra, o delta entre elas
deixa de ser atribuível à ablação. Correções operacionais entram por ferramentas
que rodam do HD, e mudanças de análise entram pelo servidor — nenhum dos dois
toca o código de treino.

---

## 11. Limitações conhecidas

- **Co-registro assumido.** Os polígonos do pós são cópias exatas dos do pré, ou
  seja, o dataset assume alinhamento perfeito entre as datas. Qualquer
  desalinhamento real vira ruído de rótulo silencioso, e nenhum experimento aqui
  o mede.
- **`tier3` tem qualidade de anotação diferente de `tier1`**, e o estudo trata os
  dois como equivalentes. A ablação `abl-sem-tier3` mede o efeito agregado, não
  a diferença de qualidade.
- **20 épocas não é convergência.** O orçamento mira um dia de laboratório. Os
  números absolutos ficam abaixo do que a arquitetura alcança; as **comparações**
  entre slots continuam válidas, porque todos recebem o mesmo orçamento.
- **Uma semente por ablação.** Só a referência tem réplicas (3 sementes). O piso
  de ruído medido nelas é usado como aproximação para todos os slots, o que
  supõe que a variância não depende muito da configuração.
- **`hold` é usado como validação e como número reportado.** Não há conjunto de
  teste intocado neste estudo; `test` do xBD permanece sem uso.

---

Detalhamento dos 24 experimentos, hipóteses e bibliografia:
**[EXPERIMENTOS.md](EXPERIMENTOS.md)**.
