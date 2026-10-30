# xBD / xView2 — avaliação de danos em edificações

Estudo de ablação sobre segmentação de danos em edificações a partir de pares de
imagens de satélite pré e pós-desastre (dataset **xBD** / desafio **xView2**).

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
| Licenca | CC BY-NC-SA 4.0: atribuicao, uso nao comercial, derivados sob a mesma licenca. Download exige cadastro; as imagens nao sao redistribuidas aqui |
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

Pipeline de treino implementado e em execucao em 10 maquinas. Os 24 experimentos
sao puxados de uma fila central; metricas e curvas em tempo real no painel, e os
pesos ficam no disco de cada maquina ate a coleta. Detalhes operacionais em
[LABORATORIO.md](LABORATORIO.md).
