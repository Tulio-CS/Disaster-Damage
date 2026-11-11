# Amostras para a demonstração

Quatro cenas do split `test` do xView2. Nenhuma delas foi usada no treino, na
escolha de época ou na escolha do modelo. Nenhuma pertence a `socal-fire` ou a
`woolsey-fire`, os dois eventos com cópias no treino.

| cena | evento | o que ilustra |
|---|---|---|
| `hurricane-matthew_00000105` | furacão, Haiti, 2016 | dano leve predominante, a classe mais difícil |
| `hurricane-michael_00000197` | furacão, EUA, 2018 | mistura de intacta, dano leve e dano severo |
| `palu-tsunami_00000143` | tsunami, Indonésia, 2018 | intacta, dano severo e destruída na mesma cena |
| `santa-rosa-wildfire_00000038` | incêndio, EUA, 2017 | quase tudo destruído |

Para cada cena há três arquivos: `<cena>_pre.png` e `<cena>_post.png` (RGB `uint8`,
1024×1024, convertidos dos GeoTIFF por `xbd/prep.py`) e `<cena>_mask.png`, com a
anotação rasterizada (0 fundo, 1 intacta, 2 dano leve, 3 dano severo, 4 destruída,
5 não classificada).

**Fonte e licença.** xBD, de Gupta et al. (2019), *Creating xBD: A Dataset for
Assessing Building Damage from Satellite Imagery*, arXiv:1911.09296,
<https://xview2.org/dataset>. As imagens são redistribuídas sob a licença
original, CC BY-NC-SA 4.0: com atribuição, sem uso comercial e com derivados sob a
mesma licença.
