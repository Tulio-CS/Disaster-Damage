# Relatório em LaTeX

```bash
cd paper
pdflatex relatorio.tex && pdflatex relatorio.tex   # duas vezes: referências cruzadas
```

Sem dependências externas: a bibliografia é `thebibliography` embutida, não
precisa de BibTeX. Pacotes usados: `tikz`, `pgfplots`, `booktabs`, `graphicx`,
`babel/brazil`, `hyperref`.

## A foto do laboratório

Salve como **`figuras/laboratorio.jpg`** e recompile. O documento usa
`\IfFileExists`, então compila normalmente sem ela — aparece um quadro de
marcação no lugar, e a fotografia o substitui automaticamente quando existir.

## Se faltar TeX na máquina

```bash
sudo apt install texlive-latex-recommended texlive-pictures texlive-lang-portuguese
```

Ou, sem instalar nada, cole o `.tex` no Overleaf.

## Números

Todos vêm de `../colheita/avaliacao.json` e foram conferidos programaticamente
contra ele. Ao reexecutar experimentos, atualize as tabelas das seções 6 e 7 —
elas estão em texto literal, não geradas.

## Imagens que faltam

Três arquivos não estão no repositório e o documento os espera. Ele **compila
sem eles** (aparece um quadro de marcação no lugar), mas ficam melhores com:

| arquivo | o quê |
|---|---|
| `figuras/mapa_xbd.png` | mapa de cobertura geográfica do xBD |
| `figuras/leaderboard.png` | captura da tabela de classificação oficial |
| `figuras/laboratorio.jpg` | fotografia das 10 estações em operação |

## Figuras geradas

```bash
.venv/bin/python scripts/figuras.py        # curvas, folds, prec_recall, confusao, leaderboard
.venv/bin/python scripts/figuras_pred.py   # exemplos.png (roda o modelo)
```

O primeiro lê apenas `colheita/avaliacao.json` e o histórico por época. O
segundo carrega o melhor checkpoint e roda inferência em 4 cenas de `hold`
escolhidas pela composição de severidade do recorte avaliado.
