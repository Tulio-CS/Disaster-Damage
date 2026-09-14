# xBD / xView2 — avaliação de danos em edificações

Estudo de ablação sobre segmentação de danos em edificações a partir de pares de
imagens de satélite pré e pós-desastre (dataset **xBD** / desafio **xView2**).

| documento | conteúdo |
|---|---|
| [METODOLOGIA.md](METODOLOGIA.md) | tarefa, dados, modelo, perda, métrica, protocolo, limitações |
| [EXPERIMENTOS.md](EXPERIMENTOS.md) | os 24 experimentos, hipóteses, como ler os resultados, bibliografia |
| [RESULTADOS.md](RESULTADOS.md) | o que os 24 experimentos mediram, e o que concluir |
| [LABORATORIO.md](LABORATORIO.md) | procedimento operacional das 10 máquinas |
| este arquivo | dados brutos e achados da análise exploratória |

## Estado dos dados

| | |
|---|---|
| Tarball | `xview2_geotiff.tgz`, 52 GB |
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
scripts/build_index.py       varre labels + headers -> cache/*.parquet
scripts/xbd_viz.py           tema de graficos e renderizacao de cenas
cache/*.parquet              indice tabular derivado (nao versionado)

xbd/                         pacote de treino (config, dados, modelo, perdas, metricas)
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
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib rasterio shapely pyarrow jupyter
.venv/bin/python scripts/build_index.py --workers 10      # ~4 min, gera cache/
.venv/bin/jupyter lab notebooks/01_eda_xbd.ipynb
```

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
  disso, em 8 dos 10 eventos avaliados mais de dois tercos dos tiles de `hold`/`test` sao
  adjacentes (<500 m) a um tile de treino — a avaliacao oficial mede generalizacao para o
  quarteirao vizinho, nao para um desastre novo.

A secao 14 do notebook traz a tabela completa de consequencias para o pipeline.
Como cada achado virou decisao de projeto: [METODOLOGIA.md §2.3](METODOLOGIA.md).

## Estado

Pipeline de treino implementado e em execucao em 10 maquinas. Os 24 experimentos
sao puxados de uma fila central; metricas e curvas em tempo real no painel, e os
pesos ficam no disco de cada maquina ate a coleta. Detalhes operacionais em
[LABORATORIO.md](LABORATORIO.md).
