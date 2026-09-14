# Experimentos e ablações

24 slots: 1 referência, 8 baselines, 15 ablações. A fonte da verdade executável é
[`experiments/matrix.yaml`](experiments/matrix.yaml); este documento explica o
porquê de cada um. Método e justificativas de projeto em
[METODOLOGIA.md](METODOLOGIA.md).

---

## 1. Desenho

### 1.1 Um fator por vez

Toda ablação muda **exatamente um** campo em relação à referência. É o que torna
o delta interpretável: se dois campos mudam, o resultado não atribui a causa a
nenhum dos dois. A regra é verificada mecanicamente por
`scripts/validar_matriz.py`, que compara cada slot com a referência campo a campo
e acusa tanto ablações que mudam mais de um fator quanto ablações que não mudam
nada (override sem efeito, tipicamente por erro de digitação).

### 1.2 Configuração de referência

Todos os deltas são medidos contra este slot:

| campo | valor |
|---|---|
| `task` | `damage` (5 classes) |
| `split_mode` | `oficial` (tier1+tier3 → hold) |
| `encoder` | `resnet34`, pré-treinado no ImageNet |
| `siamese` / `fuse_mode` | `true` / `cat_diff` |
| `crop` | 512 |
| `norm` | `dataset` (médias medidas no xBD) |
| `usar_tier3` | `true` |
| `oversample_dano` | 3,0 |
| `loss_dmg` / `loss_loc` | `focal_dice` / `bce_dice` |
| `epochs` / `effective_batch` / `lr` | 20 / 16 / 3e-4 |
| `seed` | 42 |

### 1.3 O piso de ruído vem antes de tudo

Os slots 8 e 9 são a referência repetida com sementes diferentes (1337 e 2024).
Junto com o slot 1, dão três amostras da mesma configuração.

**O espalhamento entre elas é o ruído do próprio experimento.** Uma ablação que
se afasta da referência por menos que esse espalhamento não mediu efeito — mediu
a semente. Sem esses dois slots, todo delta pequeno da tabela final é
indistinguível de acaso, e é assim que se publica uma conclusão errada.

Isto é a primeira coisa a olhar na análise, antes de qualquer ablação.

### 1.4 Ondas

24 slots para 10 máquinas. A ordem da fila decide o que existe ao fim do primeiro
dia, então as prioridades formam ondas de 10:

| onda | prioridade | conteúdo |
|---|---|---|
| 1 | 0–5 | referência + 2 réplicas + 2 baselines + 5 ablações centrais |
| 2 | 20 | os 4 folds restantes + 6 ablações |
| reserva | 90 | 4 ablações extras, só se sobrar máquina |

A onda 1 não é "os 10 primeiros da lista" — é uma seleção. Com prioridade só por
grupo (referência < baseline < ablação), o primeiro dia inteiro seria 9 baselines
e 1 ablação: os 5 folds de generalização comeriam metade das máquinas antes de
qualquer ablação começar.

---

## 2. Baselines

### 2.1 `base-loc-only` — quanto do score é localização

`task: loc` (2 classes). Separa quanto do score xView2 vem de achar o prédio e
quanto vem de classificar a severidade.

É o piso informativo: como o score é `0,3·F1_loc + 0,7·F1_dano`, um modelo que só
localiza tem teto de 0,3. A distância entre este slot e a referência mede o que
o termo difícil está realmente entregando.

### 2.2 `base-grouped-f0` … `f4` — generalização honesta

`split_mode: grouped`, folds 0 a 4. Nenhum evento aparece no treino e na
validação.

A EDA mediu que, em 8 dos 10 eventos, mais de dois terços dos tiles de `hold` são
adjacentes (<500 m) a um tile de treino. O split oficial portanto mede
memorização do quarteirão vizinho, não generalização para um desastre novo — que
é o caso de uso real. Split agrupado é a resposta padrão para dados com
autocorrelação espacial [16].

**A diferença entre a referência e a média dos 5 folds é um resultado do estudo**,
não um detalhe: quantifica o otimismo do protocolo oficial. O desvio entre folds
mede quanto a dificuldade varia entre tipos de desastre.

### 2.3 `base-semente-1337`, `base-semente-2024` — o ruído

Ver §1.3.

---

## 3. Ablações

### 3.1 Arquitetura — a hipótese central do modelo

Três slots que atacam a mesma pergunta: **o sinal de dano mora na diferença em
espaço de features, e ela precisa ser dada de graça?**

| slot | mudança | parâmetros | o que testa |
|---|---|---|---|
| `abl-sem-siamese` | `siamese: false` | 24,44 M | a rede só vê a imagem **pré** |
| `abl-fusao-cat` | `fuse_mode: cat` | 25,14 M | `[f_pré \| f_pós]`, sem o termo `\|Δ\|` |
| `abl-fusao-diff` | `fuse_mode: diff` | 24,79 M | só `\|f_pós − f_pré\|`, sem aparência crua |
| *(referência)* | `cat_diff` | 25,50 M | `[f_pré \| f_pós \| \|Δ\|]` |

`abl-sem-siamese` deve desabar — é o teto de quem tenta inferir dano sem a imagem
pós. **Se não desabar, o achado é importante e ruim**: significa que o sinal de
dano está vazando pela aparência da cena (tipo de construção, densidade urbana,
região), e não pela comparação entre as datas. Um modelo assim não generalizaria
para um desastre novo.

`abl-fusao-cat` é a comparação honesta: a rede tem as duas imagens e teria que
aprender a subtrair sozinha. Se empatar com a referência, o termo explícito é
redundante. Esta é a mesma dicotomia que a literatura de detecção de mudança
formalizou como *FC-Siam-conc* contra *FC-Siam-diff* [7].

`abl-fusao-diff` testa o oposto: se empatar, a aparência crua é redundante e o
modelo pode ficar ~3% menor.

### 3.2 `abl-sem-pretrained` — quanto vale o ImageNet aqui

`pretrained: false`. Imagem de satélite nadir não se parece com ImageNet
(perspectiva, escala, distribuição de cor). Quantifica quanto o pré-treino ainda
paga neste domínio, com 20 épocas de orçamento.

### 3.3 Amostragem

| slot | mudança | hipótese |
|---|---|---|
| `abl-sem-oversample` | `oversample_dano: 1.0` | sampler uniforme |
| `abl-excluir-vazias` *(reserva)* | `excluir_cenas_vazias: true` | remove as 35% de cenas sem edificação em vez de dar peso 0,25 |

Com 10:1 a favor de `no-damage` e só 1 em 3 cenas com dano, sem reamostragem a
maioria dos passos não veria classe positiva. Deve derrubar sobretudo o F1 de
`minor-damage` — e como o score usa média **harmônica**, degradar a pior classe
derruba o total desproporcionalmente. É a ablação em que precisão e recall
separados (§7 da metodologia) mais vão importar na leitura.

### 3.4 `abl-sem-tier3` — o tier3 paga o que custa?

`usar_tier3: false`: treina com 2.799 cenas em vez de 9.168. Se o score cair
pouco, as 6.369 cenas de `tier3` estão pagando pouco pelo tempo de treino que
custam — e todos os experimentos futuros ficam 3× mais baratos.

### 3.5 Resolução

`abl-crop256` e `abl-crop768`, que com a referência (512) fecham uma curva de três
pontos.

Mais contexto por amostra contra mais amostras por passo. Com 71% das edificações
abaixo de 32×32 px [3], o contexto pode importar menos que o número de amostras —
mas a classificação de severidade talvez precise ver o entorno (escombros na rua,
telhado parcialmente ausente) para distinguir `minor` de `major`.

### 3.6 Capacidade

| slot | encoder | parâmetros |
|---|---|---|
| `abl-encoder-seresnext50` | `seresnext50_32x4d` [11, 12] | 51,26 M |
| `abl-encoder-effnet` *(reserva)* | `efficientnet_b0` [9] | 6,19 M |
| *(referência)* | `resnet34` [10] | 25,50 M |

Separa "falta capacidade" de "falta sinal". Se dobrar o tamanho der ganho
marginal, o gargalo é a qualidade do rótulo, não o modelo — e a conclusão prática
é investir em anotação, não em GPU.

### 3.7 `abl-norm-imagenet` — a constante copiada sem verificar

`norm: imagenet`. As médias do ImageNet (0,485/0,456/0,406) contra as medidas no
próprio xBD (0,329/0,357/0,269), com desvios também bem diferentes.

Barato de testar e quase sempre copiado sem verificação em trabalhos de sensoriamento
remoto. Vale saber se custa algo aqui.

### 3.8 `abl-perda-ce` — o Focal paga o próprio custo?

`loss_dmg: ce_dice`: entropia cruzada ponderada + Dice, sem o termo Focal.

Com pesos por classe já aplicados, parte do trabalho do Focal [13] é redundante —
os dois atacam desbalanceamento, por mecanismos diferentes (reponderação por
frequência contra modulação por dificuldade). Esta é a comparação controlada:
mesma estrutura, mesmo Dice, sem o `(1−p)^γ`.

### 3.9 Otimização *(reserva)*

| slot | mudança | hipótese |
|---|---|---|
| `abl-ema` | `ema_decay: 0.999` | média móvel dos pesos [21] |
| `abl-lr-encoder-igual` | `lr_encoder_mult: 1.0` | encoder com o mesmo lr do decoder |

---

## 4. Como ler os resultados

**Ordem obrigatória:**

1. **Piso de ruído.** Espalhamento entre `ref`, `base-semente-1337` e
   `base-semente-2024`. Chame-o de σ.
2. **Qualquer ablação com |Δ| < σ não mediu efeito.** Reporte como
   "indistinguível de ruído", não como "pequeno efeito positivo/negativo".
3. **Generalização.** Média e desvio dos 5 folds agrupados contra o número
   oficial. A diferença é o otimismo do protocolo.
4. **Só então** as ablações individuais, cada uma como delta contra a referência.

**Decomponha o score antes de concluir.** O `score_xview2` é
`0,3·F1_loc + 0,7·média_harmônica(F1 das 4 severidades)`. Uma queda pode vir de
localização ou de severidade, e a média harmônica é dominada pela pior classe.
Olhe `score/f1_localizacao`, `score/f1_dano_harmonico` e os `f1/<classe>`
individuais antes de atribuir causa.

**Precisão e recall apontam correções opostas.** Quando `minor-damage` puxar o
score para baixo — e vai, é a classe mais rara —, `recall/minor-damage` baixo
pede mais oversampling e mais peso; `precisao/minor-damage` baixo pede o
contrário. F1 sozinho não distingue.

**Fontes de dados para a análise:**

| o quê | onde |
|---|---|
| curvas por época, aba HPARAMS | `/tb/` no painel |
| tabela final, uma linha por run | `/api/resultados.csv` |
| matriz de confusão por época | campo `matriz` nas métricas, no banco |
| pesos do melhor checkpoint | `melhor.pt` de cada máquina, recolhido pelo `COLETAR.bat` |

---

## 5. Fontes

> **Verifique os identificadores antes de citar em texto formal.** A lista abaixo
> foi montada de memória e serve para localizar os trabalhos, não como
> bibliografia auditada. Confira autores, ano, veículo e DOI/arXiv na fonte
> primária antes de submeter qualquer coisa.

### Dataset e desafio

[1] Gupta, R. et al. *Creating xBD: A Dataset for Assessing Building Damage from
Satellite Imagery.* CVPR Workshops, 2019. arXiv:1911.09296.

[2] **xView2 Challenge** — xview2.org. Define a métrica oficial
(`0,3·F1_loc + 0,7·F1_dano` com média harmônica) e os splits `tier1`/`tier3`/
`hold`/`test`.

[15] **Soluções vencedoras do xView2** — repositórios em
`github.com/DIUx-xView`. A primeira colocada (V. Durnov) usa U-Net siamesa com
combinação de perdas focal e dice, e foi a referência prática para o desenho de
§3 e §4 da metodologia.

### Arquitetura

[4] Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for
Biomedical Image Segmentation.* MICCAI, 2015. arXiv:1505.04597.

[6] Bromley, J. et al. *Signature Verification using a "Siamese" Time Delay
Neural Network.* NIPS, 1993. Origem da arquitetura de pesos compartilhados.

[7] Daudt, R. C., Le Saux, B., Boulch, A. *Fully Convolutional Siamese Networks
for Change Detection.* ICIP, 2018. arXiv:1810.08462. Formaliza a distinção
*FC-EF* / *FC-Siam-conc* / *FC-Siam-diff*, que é exatamente a ablação de
`fuse_mode` (§3.1).

[10] He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
arXiv:1512.03385. (`resnet34`, encoder da referência.)

[11] Xie, S. et al. *Aggregated Residual Transformations for Deep Neural
Networks.* CVPR, 2017. arXiv:1611.05431. (ResNeXt.)

[12] Hu, J., Shen, L., Sun, G. *Squeeze-and-Excitation Networks.* CVPR, 2018.
arXiv:1709.01507. (O `se` de `seresnext50_32x4d`.)

[9] Tan, M., Le, Q. *EfficientNet: Rethinking Model Scaling for Convolutional
Neural Networks.* ICML, 2019. arXiv:1905.11946.

[8] Howard, A. et al. *Searching for MobileNetV3.* ICCV, 2019. arXiv:1905.02244.
(Usado só no preset `smoke`.)

[5] Wightman, R. *PyTorch Image Models (`timm`).*
`github.com/huggingface/pytorch-image-models`. Fonte dos encoders e dos pesos
pré-treinados.

### Perdas

[13] Lin, T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV, 2017.
arXiv:1708.02002.

[14] Milletari, F., Navab, N., Ahmadi, S.-A. *V-Net: Fully Convolutional Neural
Networks for Volumetric Medical Image Segmentation.* 3DV, 2016. arXiv:1606.04797.
(Dice como função de perda.)

### Otimização e treino

[17] Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization.* ICLR,
2019. arXiv:1711.05101. (AdamW.)

[18] Loshchilov, I., Hutter, F. *SGDR: Stochastic Gradient Descent with Warm
Restarts.* ICLR, 2017. arXiv:1608.03983. (Agendamento por cosseno.)

[19] Micikevicius, P. et al. *Mixed Precision Training.* ICLR, 2018.
arXiv:1710.03740.

[20] Chen, T. et al. *Training Deep Nets with Sublinear Memory Cost.* 2016.
arXiv:1604.06174. (Gradient checkpointing.)

[21] Tarvainen, A., Valpola, H. *Mean teachers are better role models.* NeurIPS,
2017. arXiv:1703.01780. Média móvel exponencial de pesos; a ideia remonta à
média de Polyak (Polyak & Juditsky, *SIAM J. Control Optim.*, 1992).

### Avaliação

[3] Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context.* ECCV, 2014.
arXiv:1405.0312. Define o critério de objeto "small" (área < 32² px) usado em §2.3
da metodologia.

[16] Roberts, D. R. et al. *Cross-validation strategies for data with temporal,
spatial, hierarchical, or phylogenetic structure.* Ecography, 2017. Fundamenta o
split agrupado por evento (§2.2, §6 da metodologia).

---

## 6. Tabela completa dos slots

| # | slot | grupo | eixo | delta vs referência | onda |
|---|---|---|---|---|---|
| 1 | `ref` | referência | — | — | 1 |
| 2 | `base-loc-only` | baseline | tarefa | `task: damage → loc` | 1 |
| 3 | `base-grouped-f0` | baseline | generalização | `split_mode: oficial → grouped` | 1 |
| 4 | `base-grouped-f1` | baseline | generalização | `+ fold: 1` | 2 |
| 5 | `base-grouped-f2` | baseline | generalização | `+ fold: 2` | 2 |
| 6 | `base-grouped-f3` | baseline | generalização | `+ fold: 3` | 2 |
| 7 | `base-grouped-f4` | baseline | generalização | `+ fold: 4` | 2 |
| 8 | `base-semente-1337` | baseline | ruído | `seed: 42 → 1337` | 1 |
| 9 | `base-semente-2024` | baseline | ruído | `seed: 42 → 2024` | 1 |
| 10 | `abl-sem-siamese` | ablação | arquitetura | `siamese: true → false` | 1 |
| 11 | `abl-fusao-cat` | ablação | arquitetura | `fuse_mode: cat_diff → cat` | 1 |
| 12 | `abl-fusao-diff` | ablação | arquitetura | `fuse_mode: cat_diff → diff` | 2 |
| 13 | `abl-sem-pretrained` | ablação | inicialização | `pretrained: true → false` | 1 |
| 14 | `abl-sem-oversample` | ablação | amostragem | `oversample_dano: 3,0 → 1,0` | 1 |
| 15 | `abl-sem-tier3` | ablação | dados | `usar_tier3: true → false` | 2 |
| 16 | `abl-crop256` | ablação | resolução | `crop: 512 → 256` | 1 |
| 17 | `abl-crop768` | ablação | resolução | `crop: 512 → 768` | 2 |
| 18 | `abl-encoder-seresnext50` | ablação | capacidade | `encoder: resnet34 → seresnext50_32x4d` | 2 |
| 19 | `abl-norm-imagenet` | ablação | normalização | `norm: dataset → imagenet` | 2 |
| 20 | `abl-perda-ce` | ablação | perda | `loss_dmg: focal_dice → ce_dice` | 2 |
| 21 | `abl-ema` | ablação | otimização | `ema_decay: 0,0 → 0,999` | reserva |
| 22 | `abl-excluir-vazias` | ablação | amostragem | `excluir_cenas_vazias: false → true` | reserva |
| 23 | `abl-encoder-effnet` | ablação | capacidade | `encoder: resnet34 → efficientnet_b0` | reserva |
| 24 | `abl-lr-encoder-igual` | ablação | otimização | `lr_encoder_mult: 0,1 → 1,0` | reserva |

Confira a qualquer momento com:

```bash
.venv/bin/python scripts/validar_matriz.py
```
