# Resultados

24 experimentos, 10 máquinas, um dia. Todos os números abaixo vêm da
**reavaliação em cena inteira** (1024×1024) sobre os checkpoints coletados —
`results/avaliacao_validacao.json`, gerado por `scripts/avaliar.py`. Ver
[METODOLOGIA.md §11](METODOLOGIA.md) para por que este é o número a reportar, e
não o do treino.

Desenho dos experimentos em [EXPERIMENTOS.md](EXPERIMENTOS.md).

---

## 0. Entrega final: baseline, modelo final e teste

**Baseline** = `ref`: U-Net siamesa com `resnet34` pré-treinado, fusão
`[f_pré | f_pós | |Δ|]`, Focal + Dice, AdamW com lr 3e-4 (3e-5 no encoder), batch
efetivo 24 e 20 épocas. A ficha completa está no notebook 03 §2.

**Modelo final** = `abl-lr-encoder-igual`: o baseline com o encoder aprendendo à
mesma taxa do decoder. Foi escolhido por ter o maior score de validação
(+0,0411, 6× o piso de ruído; §3.6). Como essa escolha usou `hold`, o número dele
em `hold` é otimista por construção. Por isso o modelo final foi avaliado também no
split `test`, que nenhuma decisão usou (`scripts/avaliar.py --split teste`).

| | validação (`hold`) | teste (933 cenas) | teste sem as 156 cópias | F1 loc (teste) | F1 dano (teste) |
|---|---|---|---|---|---|
| baseline (`ref`) | 0,6583 | — | — | — | — |
| **modelo final** | 0,6994 | **0,6900** | 0,6907 | 0,8559 | 0,6189 |

- **Queda pequena da validação para o teste:** −0,009, perto do piso de ruído.
  A escolha em `hold` não sobreajustou.
- **O ganho sobre o baseline não vem do otimismo da escolha:** mesmo no teste, o
  modelo final fica 0,032 acima do score do baseline na validação.
- **Por que o baseline não tem número de teste:** baseline e ablações são
  comparados em `hold`, onde a escolha foi feita, e só o modelo escolhido vai ao
  teste. Para medir o baseline no teste também, rode
  `scripts/avaliar.py --split teste --so ref` (~80 min em CPU).
- **As cópias `socal-fire`/`woolsey-fire` não inflaram o teste:** sem elas, o
  número fica igual.

Modelo final no teste, por classe (por pixel; F1 = Dice):

| classe | precisão | revocação | F1 | IoU |
|---|---|---|---|---|
| fundo | 0,992 | 0,991 | 0,991 | 0,983 |
| intacta | 0,822 | 0,806 | 0,814 | 0,686 |
| dano leve | 0,399 | 0,455 | 0,425 | 0,270 |
| dano severo | 0,599 | 0,710 | 0,650 | 0,481 |
| destruída | 0,764 | 0,725 | 0,744 | 0,592 |

O mIoU das 5 classes é 0,603. O dano leve continua sendo o gargalo (§4): só
45,5% dos seus pixels acertam, e o resto se divide quase por igual entre
intacta, dano severo e fundo. A análise por evento, os exemplos de acertos e
erros e a demonstração estão no
[notebook 03](notebooks/03_resultados_e_demo.ipynb).

---

## 1. Piso de ruído

A mesma receita, três sementes:

| slot | score |
|---|---|
| `ref` (seed 42) | 0,6583 |
| `base-semente-1337` | 0,6518 |
| `base-semente-2024` | 0,6516 |

**Amplitude 0,0068 · desvio 0,0039.**

Esta é a régua. Qualquer ablação a menos de 0,0068 da referência não mediu
efeito — mediu a semente. Aplicada rigorosamente na tabela da §3.

> Ressalva: o piso foi medido só na configuração de referência, e é usado como
> aproximação para todas as outras. Isso supõe que a variância não depende muito
> da configuração — plausível, não verificado.

---

## 2. O achado principal: localização generaliza, severidade não

### 2.1 Split agrupado por desastre

| fold | score | F1 loc | F1 dano | cenas |
|---|---|---|---|---|
| f0 | 0,5940 | 0,8269 | 0,4941 | 2.199 |
| f1 | 0,3195 | 0,8113 | 0,1087 | 1.181 |
| f2 | 0,3780 | 0,8144 | 0,1910 | 2.903 |
| f3 | 0,3854 | 0,7899 | 0,2120 | 3.597 |
| f4 | 0,5527 | 0,8543 | 0,4235 | 1.154 |
| **média** | **0,4459** | **0,8194** | **0,2859** | |
| *(oficial)* | *0,6583* | *0,8455* | *0,5781* | *933* |

**A queda é de 32,3%** (0,6583 → 0,4459) quando nenhum desastre aparece nos dois
lados. Isso confirma quantitativamente o vazamento espacial que a EDA previu: no
split oficial, mais de dois terços dos tiles de `hold` são adjacentes a um tile
de treino, e o número oficial mede memorização do quarteirão, não generalização.

**Mas a queda não é uniforme, e é aí que está o resultado:**

| termo | oficial | agrupado | queda | desvio entre folds |
|---|---|---|---|---|
| F1 localização | 0,8455 | 0,8194 | **−3,1%** | 0,024 |
| F1 dano (harmônica) | 0,5781 | 0,2859 | **−50,6%** | 0,164 |

Localizar edificação transfere quase intacto para um desastre novo. **Classificar
severidade não transfere.** O desvio entre folds é 7× maior no termo de dano
(0,164 contra 0,024), e o pior fold chega a 0,1087 — o modelo praticamente perde
a capacidade de graduar dano em eventos que não viu.

A leitura prática: a aparência do dano é específica do tipo de desastre.
Enchente, terremoto e furacão destroem de maneiras visualmente diferentes, e um
modelo treinado em quatro tipos não extrapola para o quinto. Um sistema de
resposta a desastres que dependesse desse número oficial estaria contando com
uma capacidade que ele não tem.

### 2.2 A mesma dissociação aparece na ablação siamesa

| | F1 loc | F1 dano | F1 `destroyed` |
|---|---|---|---|
| `ref` | 0,8455 | 0,5781 | 0,731 |
| `abl-sem-siamese` (só imagem pré) | **0,8416** | **0,3799** | **0,378** |

Removendo a imagem pós inteira, a localização **não muda** (−0,004, dentro do
ruído) e o termo de dano cai 34%. Duas conclusões:

- **A hipótese central do modelo se sustenta.** O sinal de dano vem da comparação
  entre datas, não da aparência da cena. Se estivesse vazando por tipo de
  construção ou densidade urbana, o slot não teria desabado.
- **Ainda sobra 0,38 de F1 de dano sem ver o pós.** Não é zero. Parte da
  severidade é inferível do contexto — provavelmente porque o tipo de construção
  correlaciona com o dano que ela sofre. É um piso de vazamento que vale
  investigar.

---

## 3. Ablações

Delta contra `ref` = 0,6583. Piso de ruído = 0,0068.

| experimento | score | Δ | veredito |
|---|---|---|---|
| `abl-lr-encoder-igual` | 0,6994 | **+0,0411** | melhor |
| `abl-encoder-seresnext50` | 0,6878 | **+0,0295** | melhor |
| `abl-encoder-effnet` | 0,6862 | **+0,0279** | melhor |
| `abl-perda-ce` | 0,6613 | +0,0030 | dentro do ruído |
| `ref` | 0,6583 | — | — |
| `abl-fusao-cat` | 0,6563 | −0,0020 | dentro do ruído |
| `abl-ema` | 0,6516 | −0,0067 | dentro do ruído |
| `abl-crop768` | 0,6495 | −0,0088 | pior |
| `abl-sem-oversample` | 0,6412 | −0,0172 | pior |
| `abl-norm-imagenet` | 0,6382 | −0,0202 | pior |
| `abl-excluir-vazias` | 0,6331 | −0,0253 | pior |
| `abl-crop256` | 0,6248 | −0,0335 | pior |
| `abl-fusao-diff` | 0,6080 | −0,0503 | pior |
| `abl-sem-tier3` | 0,5922 | −0,0662 | pior |
| `abl-sem-pretrained` | 0,5311 | −0,1272 | pior |
| `abl-sem-siamese` | 0,5184 | −0,1399 | pior |

### 3.1 Três resultados negativos limpos

**O termo `|f_pós − f_pré|` explícito não se paga** (`abl-fusao-cat`, −0,0020,
dentro do ruído). A rede aprende a comparar sozinha quando recebe as duas
features concatenadas. Isso **contraria a premissa do desenho** — a diferença em
espaço de features foi dada de graça na referência justamente por se supor que
ela ajudaria. Não ajudou.

Mas o oposto não vale: `abl-fusao-diff` (só a diferença, sem aparência crua)
perde 0,0503. A aparência não é redundante. Conclusão combinada: **concatenar
basta, subtrair não basta.**

**EMA não fez diferença** (−0,0067, na fronteira do ruído), contra a expectativa
de "~1 ponto de F1 de graça" que estava no código.

**Focal não se pagou.** `abl-perda-ce` (entropia cruzada ponderada + dice, sem o
termo focal) ficou +0,0030 — dentro do ruído. Com pesos por classe já aplicados,
a modulação por dificuldade é redundante. O `gamma` pode sair sem custo.

### 3.2 Capacidade não é o gargalo

| encoder | parâmetros | score |
|---|---|---|
| `efficientnet_b0` | 6,19 M | 0,6862 |
| `resnet34` *(ref)* | 25,50 M | 0,6583 |
| `seresnext50_32x4d` | 51,26 M | 0,6878 |

Um modelo **8× menor** que o `seresnext50` empata com ele (0,6862 contra 0,6878,
diferença dentro do ruído). E os dois batem o `resnet34` por ~0,03.

Isso não é uma curva de capacidade — é evidência de que **o `resnet34` é a
escolha ruim do trio**, não de que mais parâmetros ajudam. Se capacidade fosse o
gargalo, o `seresnext50` teria se distanciado do `efficientnet_b0`. Não se
distanciou.

A conclusão prática inverte a intuição: dá para rodar este problema com 6 M de
parâmetros sem perder nada, e o orçamento deveria ir para dados ou anotação, não
para GPU.

### 3.3 O que mais custa quando falta

| removido | custo |
|---|---|
| imagem pós (`sem-siamese`) | −0,1399 |
| pré-treino ImageNet | −0,1272 |
| tier3 (6.369 cenas) | −0,0662 |

**O pré-treino ImageNet ainda paga muito**, apesar de imagem de satélite nadir
não se parecer com ImageNet. −0,13 é quase tanto quanto remover a imagem pós.

**O tier3 se paga**, mas não espetacularmente: 6.369 cenas extras (3,3× mais
dados) valem 0,066. Quem precisar de experimentos mais baratos pode treinar só
com tier1 aceitando essa perda.

### 3.4 Curva de resolução — agora limpa

Reavaliados todos em cena inteira, o que corrige o defeito de desenho descrito em
[METODOLOGIA.md §11](METODOLOGIA.md):

| crop de treino | score | F1 loc | F1 dano |
|---|---|---|---|
| 256 | 0,6248 | 0,8389 | 0,5331 |
| **512** | **0,6583** | 0,8455 | 0,5781 |
| 768 | 0,6495 | 0,8465 | 0,5651 |

**Não é monotônica: 512 é ótimo.** 256 perde contexto (−0,034); 768 perde
amostras por passo no mesmo orçamento (−0,009). A localização é insensível
(0,839–0,847); toda a variação está no termo de dano, que é o que precisa de
contexto.

### 3.5 Amostragem

`abl-sem-oversample` (−0,0172) e `abl-excluir-vazias` (−0,0253) confirmam que a
reamostragem ajuda, mas o efeito é modesto perto do que se esperava de um
desbalanceamento de 10:1. E `abl-excluir-vazias` ser **pior** que a referência
mostra que as cenas sem edificação carregam sinal útil — reduzir o peso para 0,25
é melhor que removê-las.

### 3.6 O vencedor: não frear o encoder

`abl-lr-encoder-igual` (+0,0411) é o maior ganho do estudo, e era um slot de
**reserva**. Dar ao encoder pré-treinado o mesmo `lr` do decoder, em vez de 0,1×,
melhora tudo — e melhora mais o `minor-damage` (F1 0,454 contra 0,399 da
referência), que é a classe que domina a métrica.

A prática de frear o encoder vem de regimes de *fine-tuning* longo. Em 20 épocas,
ela está impedindo o encoder de se adaptar ao domínio.

---

## 4. Onde o modelo erra

### 4.1 `minor-damage` é o gargalo

O score é média **harmônica** dos F1 das 4 severidades, logo é dominado pela
pior. Na referência:

| classe | F1 | precisão | recall |
|---|---|---|---|
| `no-damage` | 0,779 | 0,791 | 0,768 |
| **`minor-damage`** | **0,399** | **0,372** | **0,430** |
| `major-damage` | 0,567 | 0,514 | 0,633 |
| `destroyed` | 0,731 | 0,739 | 0,724 |

`minor-damage` tem metade do F1 de `no-damage`. Como a harmônica é dominada pelo
pior termo, **melhorar essa classe sozinha vale mais que melhorar as outras três
juntas.**

**O gargalo é precisão, não recall** (0,372 contra 0,430) — e isso vale para 15
dos 18 experimentos com split oficial. O modelo marca dano leve onde não há, mais
do que deixa de achar dano leve que existe.

Isso **inverte a correção intuitiva**: mais oversampling e mais peso de classe
empurrariam o recall para cima às custas da precisão, que já é o lado fraco. O
resultado de `abl-sem-oversample` é consistente com isso — sem reamostragem, a
classe fica equilibrada (0,356 / 0,370) mas com F1 pior.

### 4.2 Para onde vaza

Matriz de confusão da referência, em % de cada linha verdadeira:

| verdade \ predito | background | no-damage | minor | major | destroyed |
|---|---|---|---|---|---|
| background | **99,0%** | 0,7% | 0,1% | 0,1% | 0,1% |
| no-damage | 14,5% | **76,8%** | 5,7% | 2,9% | 0,2% |
| minor-damage | 16,9% | 23,0% | **43,0%** | 16,4% | 0,6% |
| major-damage | 13,4% | 11,2% | 10,1% | **63,3%** | 2,1% |
| destroyed | 15,8% | 2,3% | 0,9% | 8,7% | **72,4%** |

Dos pixels que são `minor-damage`:

- **23,0%** viram `no-damage` — dano leve não detectado como dano
- **16,9%** viram `background` — o prédio nem foi encontrado
- **16,4%** viram `major-damage` — severidade superestimada
- só **43,0%** acertam

O erro é quase simétrico entre "subestimar" (23% para `no-damage`) e
"superestimar" (16,4% para `major`), o que sugere que a fronteira entre níveis
é genuinamente ambígua, não um viés do modelo. Isso é consistente com o que a
EDA já apontava sobre a anotação ordinal.

Note também que **13–17% de toda edificação danificada é perdida como
`background`** em todas as classes de dano — um erro de localização, não de
classificação, que a métrica de dano paga junto.

---

## 5. Notas de validade

**Nem todos os modelos convergiram.** O melhor checkpoint ficou em média na época
16,8 de 19, e só 9 dos 18 experimentos com split oficial atingiram o pico na
última época. A correlação entre época do pico e score é −0,285, ou seja, os
vencedores **não** venceram por terem treinado mais. Ainda assim, com 20 épocas
os números absolutos ficam abaixo do que estas arquiteturas alcançam.

**O recorte central era levemente otimista.** A reavaliação em cena inteira
baixou o score em média 0,0059 (de −0,0130 em `ref` a +0,0068 em `abl-crop256`).
Pequeno, mas sistemático, e mais uma razão para reportar o número reavaliado.

**O batch efetivo não foi o mesmo em todos os slots.** A matriz pede 16, mas o
planejador de hardware arredondou para 12 × 2 = 24 nas GPUs de 16 GB, e é esse
24 que a referência e quase todos os slots usaram (`plano` em
`results/runs/*.json`). Quatro slots rodaram com outro valor: `abl-encoder-effnet`
e `abl-crop256` com 16, `abl-encoder-seresnext50` com 18 e `abl-crop768` com 20.
Nesses quatro, o delta mistura o fator ablacionado com o batch. A comparação
entre a referência e `abl-lr-encoder-igual` não é afetada, porque os dois usaram 24.

**Uma semente por ablação.** Os deltas de +0,03 a +0,04 são 4–6× o piso de ruído,
então provavelmente reais. Os de ±0,01 a ±0,02 mereceriam réplica antes de virar
afirmação.

**`base-loc-only` não entra nas comparações.** Com `task: loc` o
`score_xview2` degenera no F1 de localização puro (sem o termo de dano), então
seu 0,8446 mede outra coisa. O que ele informa: localização sozinha chega a 0,84,
logo o que puxa a referência para 0,66 é inteiramente a classificação de
severidade.

**Os folds agrupados validam em conjuntos diferentes entre si**, com 1.154 a
3.597 cenas. O desvio de 0,164 no termo de dano mistura variação de dificuldade
entre eventos com variação de treino — não é ruído no mesmo sentido do §1.

**Estes runs são anteriores à correção das duplicatas.** A auditoria de
13/10/2026 achou que `socal-fire` e `woolsey-fire` são os mesmos tiles (725 pares
com a imagem pré idêntica). Nos runs acima, 143 das 933 cenas de `hold` tinham
cópia no treino, e os folds 0 (que contém `socal-fire`) e 3 (`woolsey-fire`)
compartilhavam os 725 pares. Logo, o score oficial e os folds 0 e 3 são
otimistas por um valor ainda não medido. O fold 0 é justamente o melhor dos cinco
(0,594). Ver [METODOLOGIA.md §2.3 e §6](METODOLOGIA.md).

---

## 6. O que fazer a seguir

Em ordem de retorno esperado:

1. **`lr_encoder_mult: 1.0` na referência.** +0,041 por mudar um número. Com
   réplica de semente para confirmar.
2. **Trocar `resnet34` por `efficientnet_b0`.** Mesmo ganho que o `seresnext50`
   com 8× menos parâmetros — experimentos futuros ficam mais baratos.
3. **Atacar a precisão de `minor-damage`**, não o recall. O caminho óbvio
   (mais peso, mais oversampling) empurra o lado errado.
4. **Tratar o número agrupado como o número real.** Se o objetivo é resposta a
   desastres, 0,446 é a estimativa honesta, e a variância entre eventos (0,32 a
   0,59) é parte do resultado.
5. **Simplificar o modelo e a perda.** `fuse_mode: cat` e `loss_dmg: ce_dice`
   empatam com a referência — menos parâmetros e menos hiperparâmetros, de graça.

---

## 7. Reproduzir

```bash
# tabela reavaliada, a partir dos checkpoints coletados
.venv/bin/python scripts/avaliar.py --checkpoints colheita --dados data

# teste: so o modelo final (~80 min em CPU); --so ref acrescenta o baseline
.venv/bin/python scripts/avaliar.py --checkpoints colheita --dados data --split teste \
    --so abl-lr-encoder-igual --saida results/avaliacao_teste.json
```

Dados brutos:

| o quê | onde |
|---|---|
| tabela reavaliada na validação (24 modelos, todas as métricas, matrizes) | `results/avaliacao_validacao.json` (cópia de `colheita/avaliacao.json`) |
| teste do modelo final, com a matriz de cada cena | `results/avaliacao_teste.json`, `results/avaliacao_teste.por_cena.json` |
| manifesto de cada run (config, hardware, batch efetivo) | `results/runs/*.json` |
| métricas por época | `results/history/*.jsonl` |
| pesos do baseline e do modelo final | release v1.0 (`baseline_ref.pt`, `modelo_final.pt`) |
| pesos dos outros runs | `colheita/<máquina>/checkpoints/<run_id>/melhor.pt` |
| curvas ao vivo | painel, aba TensorBoard |

`colheita/` não é versionada (10 GB). Tudo o que está em `results/` é versionado.
