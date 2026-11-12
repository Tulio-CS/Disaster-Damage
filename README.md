# xBD / xView2 — avaliação de danos em edificações

Estudo de ablação sobre segmentação de danos em edificações a partir de pares de
imagens de satélite pré e pós-desastre (dataset **xBD** / desafio **xView2**).

## Entrega final

| | |
|---|---|
| Apresentação (documentação do trabalho) | [apresentacao/apresentacao.pdf](apresentacao/apresentacao.pdf) · [.pptx](apresentacao/apresentacao.pptx) |
| Resultados, análise de erros e demo, executados | [notebooks/03_resultados_e_demo.ipynb](notebooks/03_resultados_e_demo.ipynb) |
| Demonstração pela linha de comando | [scripts/demo.py](scripts/demo.py), sobre as cenas de teste em [amostras/](amostras/) |
| Pesos do modelo final e do baseline | [release v1.0](https://github.com/Tulio-CS/Disaster-Damage/releases/tag/v1.0); a demo baixa sozinha |

**Pergunta.** A partir de um par de imagens de satélite da mesma área, antes e
depois do desastre, uma rede neural consegue localizar as edificações e graduar o
dano de cada uma bem o bastante para apoiar a triagem? E esse desempenho se mantém
em um desastre que o modelo não viu no treino?

**Resposta curta.** Em parte.
- **Localizar edificações: sim.** O F1 de localização é 0,856 no teste e cai só 3%
  num desastre que o modelo não viu no treino.
- **Graduar o dano: só em parte.** O F1 de dano no teste é 0,619: bom para
  `destruída` (0,744) e fraco para `dano leve` (0,425). Num desastre fora do treino,
  cai para 0,29.

O modelo final faz score xView2 **0,690** no teste. O baseline faz 0,658 e o
modelo final 0,699 na validação, onde a escolha foi feita. Serve para triagem
(separar áreas intactas de destruição evidente), não para substituir a avaliação
humana dos níveis intermediários.

### Rodar a demonstração

CPU basta: ~5 s e ~1,3 GB de RAM por cena 1024×1024.

```bash
git clone https://github.com/Tulio-CS/Disaster-Damage.git && cd Disaster-Damage
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-demo.txt
.venv/bin/python scripts/demo.py
```

A demo roda o fluxo imagem → pré-processamento → modelo → resultado nas 4 cenas
de `amostras/`, que são do split `test` e nunca foram vistas no treino nem na
escolha do modelo. O resultado vai para `saidas/demo/`: uma figura com pré, pós,
anotação e predição, a máscara predita e a contagem de edificações por nível de
dano. Para outro par de imagens, inclusive o GeoTIFF original:
`scripts/demo.py --pre a_pre.png --post a_post.png`.

### Onde está cada item pedido

| item | onde |
|---|---|
| tema, pergunta, tarefa, dataset, distribuição das imagens | notebook 03 §1 · [notebook 01](notebooks/01_eda_xbd.ipynb) (EDA) · [METODOLOGIA.md](METODOLOGIA.md) §1–2 |
| preparação, normalização, aumento de dados | [xbd/prep.py](xbd/prep.py) · [xbd/data.py](xbd/data.py) · [notebook 02](notebooks/02_pipeline_dados.ipynb) · METODOLOGIA §2.2–2.4 |
| divisão treino/validação/teste e cuidados com vazamento | [xbd/splits.py](xbd/splits.py) · [manifestos/](manifestos/) · notebook 02 §2–3 · METODOLOGIA §6 |
| baseline neural: arquitetura, entrada/saída, perda, otimizador, taxa, batch, épocas, regularização | notebook 03 §2 · [xbd/models.py](xbd/models.py) · [xbd/losses.py](xbd/losses.py) · METODOLOGIA §3–5 |
| experimentos: o que mudou, hipótese, resultado, decisão | notebook 03 §3 · [EXPERIMENTOS.md](EXPERIMENTOS.md) · [RESULTADOS.md](RESULTADOS.md) §1–3 · [experiments/matrix.yaml](experiments/matrix.yaml) |
| modelo final e justificativa da escolha | notebook 03 §3 e §5 · RESULTADOS §0 |
| métricas (IoU, Dice, precisão e revocação por pixel), curvas de treino e validação | notebook 03 §4–5 · [results/](results/) |
| exemplos de acertos e erros, causas dos erros, limitações | notebook 03 §6–7 e §9 · RESULTADOS §4–5 |
| demonstração em imagens fora do treino | [scripts/demo.py](scripts/demo.py) · [xbd/inferencia.py](xbd/inferencia.py) · notebook 03 §8 |
| conclusão | notebook 03 §9 · slide 12 |

### Treino, validação e teste

- **Treino:** `tier1` + `tier3`, 9.168 cenas.
- **Validação:** `hold`, 933 cenas. Escolhe a melhor época de cada run e o melhor
  dos 24 runs, então o número de validação do modelo final é otimista por
  construção.
- **Teste:** `test`, 933 cenas. Não entrou em nenhuma decisão e só foi usado na
  avaliação final (`scripts/avaliar.py --split teste`).

Os 24 runs são anteriores à auditoria de duplicatas, então 156 cenas de `test` têm
cópia idêntica no treino. Por isso o teste é reportado com e sem essas cenas.

---

| documento | conteúdo |
|---|---|
| [METODOLOGIA.md](METODOLOGIA.md) | tarefa, dados, modelo, perda, métrica, protocolo, limitações |
| [EXPERIMENTOS.md](EXPERIMENTOS.md) | os 24 experimentos, hipóteses, como ler os resultados, bibliografia |
| [RESULTADOS.md](RESULTADOS.md) | o que os 24 experimentos mediram, e o que concluir |
| [LABORATORIO.md](LABORATORIO.md) | procedimento operacional das 10 máquinas |
| [manifestos/](manifestos/) | manifesto por cena, splits `train`/`validation`/`test`, duplicatas e auditoria |
| este arquivo | dados brutos e achados da análise exploratória |

## Estado dos dados

| | |
|---|---|
| Fonte | xBD / xView2 Challenge — https://xview2.org/dataset (Gupta et al., 2019, arXiv:1911.09296) |
| Licenca | CC BY-NC-SA 4.0: atribuicao, uso nao comercial, derivados sob a mesma licenca. Download exige cadastro. O repositorio so redistribui as 4 cenas de `amostras/`, sob a mesma licenca, para a demo |
| Tarball | `xview2_geotiff.tgz`, 52 GB, SHA-256 `5643e8c7b3ca790864d93d2507909c2b5ed12a22146e72cd9e146b2950370cc8` |
| Extraido | `geotiffs/`, 139 GB, 44.136 arquivos |
| Verificacao | indice do tarball conferido contra o disco — 0 diferencas, 0 erros de leitura |
| Imagens | 22.068 GeoTIFF (11.034 pares pre/pos), 1024x1024, 3 bandas, `int16`, EPSG:4326 |
| Labels | 22.068 JSON com 850.736 poligonos de edificacao (WKT em pixel e em lon/lat) |
| Splits | `tier1` 5.598 · `tier3` 12.738 · `hold` 1.866 · `test` 1.866 |

```
geotiffs/{tier1,tier3,hold,test}/
  images/<desastre>_<id>_{pre,post}_disaster.tif
  labels/<desastre>_<id>_{pre,post}_disaster.json
```

## Estrutura

```
notebooks/01_eda_xbd.ipynb   analise exploratoria (ja executada, com saidas)
notebooks/02_pipeline_dados.ipynb  split, testes de vazamento, transformacoes, smoke test
notebooks/03_resultados_e_demo.ipynb  baseline, experimentos, teste, erros e demo (entrega final)
apresentacao/                apresentacao (PPTX e PDF) e as figuras que ela usa
amostras/                    4 cenas do split test (pre, pos, mascara) para a demo
scripts/build_index.py       varre labels + headers -> cache/*.parquet
scripts/xbd_viz.py           tema de graficos e renderizacao de cenas
scripts/gerar_manifesto.py   hashes de pixel, duplicatas e manifestos -> manifestos/
scripts/verificar_reproducao.py  refaz os pixels dos GeoTIFF e confere com o manifesto
cache/*.parquet              indice tabular derivado (nao versionado)
manifestos/                  manifesto.csv, train/validation/test.csv, duplicatas.csv

xbd/                         pacote de treino (config, dados, modelo, perdas, metricas)
xbd/prep.py                  GeoTIFF -> PNG uint8 + mascara (data/, nao versionado)
xbd/splits.py                split oficial e agrupado, com a remocao de duplicatas
xbd/agent.py                 agente que puxa experimentos da fila do painel
xbd/reporter.py              cliente HTTP com spool em disco
server/                      painel: fila, ingestao de metricas e TensorBoard
experiments/matrix.yaml      os 24 experimentos (referencia, baselines, ablacoes)
scripts/validar_matriz.py    confere a matriz antes de rodar
scripts/instalar.py          instala e dispara uma maquina, a partir do HD
scripts/coletar.py           recolhe modelos e resultados de volta para o HD
scripts/painel.py            administra a fila pela linha de comando
scripts/avaliar.py           reavalia os checkpoints em cena inteira (validacao ou --split teste)
xbd/inferencia.py            imagem -> pre-processamento -> modelo -> resultado
scripts/demo.py              demonstracao de ponta a ponta pela linha de comando
results/                     manifesto e historico por epoca dos 24 runs, avaliacoes
```

## Como rodar

```bash
sha256sum xview2_geotiff.tgz                              # confira com a tabela acima
tar -xzf xview2_geotiff.tgz                               # -> geotiffs/
python3 -m venv .venv
.venv/bin/pip install -r requirements-dados.txt           # versoes fixas
.venv/bin/python scripts/build_index.py --workers 10      # ~4 min, gera cache/
.venv/bin/python -m xbd.prep --workers 10                 # GeoTIFF -> data/ (PNG + mascaras)
.venv/bin/python -m xbd.prep_merge --out data             # junta os indices parciais
.venv/bin/python scripts/gerar_manifesto.py               # ~5 min, gera manifestos/
.venv/bin/jupyter lab notebooks/
```

Para so conferir que os seus dados sao os mesmos destes manifestos, basta o
`geotiffs/` extraido e o repositorio: `verificar_reproducao.py` refaz os pixels de
cada imagem e mascara com as funcoes de `xbd/prep.py` e compara o SHA-256 com
`manifestos/manifesto.csv` (~15 min, sem gravar nada).

```bash
.venv/bin/python scripts/verificar_reproducao.py --geotiffs geotiffs --workers 4
```

O split e deterministico (sem sorteio): `xbd/splits.py` o deriva da pasta de origem
do xView2 e de `manifestos/duplicatas.csv`, e o notebook 02 confere que ele coincide
com o manifesto.

`build_index.py` produz quatro tabelas: `images` (1 linha por imagem, com metadados de
aquisicao, header do raster e contagens por classe), `buildings` (1 linha por poligono,
com area em px e m²), `registration` (deslocamento pre→pos do mesmo predio) e
`band_stats` (estatisticas de pixel numa amostra estratificada).

## Principais achados

- **Formato.** Os GeoTIFF sao `int16` sem compressao mas com conteudo 8-bit — converter
  para `uint8` corta ~70 GB sem perda. O pixel nao e quadrado no chao (0,42 m em X contra
  0,50 m em Y), e o fator varia com a latitude.
- **Classes.** 4 niveis ordinais de dano + `un-classified`. Desbalanceamento de ~10:1 a
  favor de `no-damage`, e a composicao muda drasticamente entre eventos (99,5%
  `no-damage` em `mexico-earthquake` contra 17,9% em `hurricane-matthew`).
- **Objetos.** Mediana de 575 px² (~24x24 px); 71% sao "small" pelo criterio COCO. A
  imagem media e 97% fundo e 35% das cenas nao tem nenhuma edificacao.
- **Aquisicao.** Off-nadir de 0 a 44°, e mediana de 304 dias entre a imagem pre e a pos
  (5% acima de 3,3 anos) — boa parte da diferenca visual entre as datas nao e dano.
- **Avaliacao.** Os poligonos do pos sao copias exatas dos do pre, entao o dataset assume
  co-registro perfeito e qualquer desalinhamento vira ruido de rotulo silencioso. Alem
  disso, em 7 dos 10 eventos avaliados mais de dois tercos dos tiles de `hold`/`test` sao
  adjacentes (<500 m) a um tile de treino — a avaliacao oficial mede generalizacao para o
  quarteirao vizinho, nao para um desastre novo.
- **Duplicatas.** `socal-fire` e `woolsey-fire` sao os mesmos tiles com a mesma imagem
  pre (725 pares; 143 cenas de `hold` e 156 de `test` tinham copia no treino). As copias
  saem do treino e os dois eventos viram um grupo so no split agrupado
  (`remover_duplicatas`, padrao). Os 24 runs de RESULTADOS.md sao anteriores a correcao.

Como cada achado virou decisao de projeto: [METODOLOGIA.md §2.3](METODOLOGIA.md).

## Estado

Concluido. Os 24 experimentos foram treinados em 14/09/2026, em 10 maquinas com
GPU que puxavam slots de uma fila central (detalhes em [LABORATORIO.md](LABORATORIO.md)).
Os checkpoints foram reavaliados em cena inteira na validacao, e em 11/11/2026 o
modelo final e o baseline foram avaliados no split de teste. Os pesos dos dois
estao no release v1.0. Os dos outros 22 runs (10 GB) ficam fora do repositorio.
