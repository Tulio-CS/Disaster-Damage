#!/usr/bin/env python3
"""
Manifesto do dataset preparado: uma linha por cena (par pre/pos + mascara).

Junta o que ja existe (`cache/images.parquet`, `cache/buildings.parquet`,
`data/index.parquet`) com uma passada de leitura sobre os PNG de `data/`, que
calcula por imagem:

  sha256    dos PIXELS decodificados, nao dos bytes do arquivo: dois PNG com o
            mesmo conteudo mas zlib diferente (outra maquina) tem o mesmo hash
  dHash     256 bits (17x16, tons de cinza), para duplicatas proximas
  zeros     fracao de pixels zerados nas 3 bandas (borda sem dado)

Saida:
    manifestos/manifesto.csv     todas as cenas, todas as colunas (inclui as excluidas)
    manifestos/train.csv         split oficial: tier1 + tier3, sem as copias de hold/test
    manifestos/validation.csv    split oficial: hold
    manifestos/test.csv          split oficial: test (nao usado em nenhum run oficial)
    manifestos/duplicatas.csv    pares de cenas com imagem identica (lido por xbd/splits.py)
    manifestos/auditoria.json    resumo das checagens de duplicata e integridade
    manifestos/duplicatas.png    pares de cenas duplicadas, lado a lado (evidencia visual)
    cache/hashes.parquet         hashes por cena (reaproveitado na proxima execucao)

O split e deterministico, sem sorteio, e sai das mesmas funcoes que o treino usa
(`xbd/splits.py`): o oficial vem da pasta de origem do xView2, menos as copias de
treino de cenas de hold/test (split = "excluir"); o agrupado une os eventos que
cobrem os mesmos tiles e distribui os grupos em 5 folds por ordem alfabetica. Os
dois estao no manifesto: `split` e `fold_agrupado`.

Uso:
    python scripts/gerar_manifesto.py --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from xbd.splits import copias_de_avaliacao, folds, grupo_de  # noqa: E402

SPLIT_OFICIAL = {"tier1": "train", "tier3": "train", "hold": "validation", "test": "test"}
N_FOLDS = 5
LICENCA = "CC BY-NC-SA 4.0"
SEVERIDADE = [("px_destroyed", "destroyed"), ("px_major", "major-damage"),
              ("px_minor", "minor-damage"), ("px_no_damage", "no-damage")]

# Duplicata proxima: distancia de Hamming entre dHash de 256 bits. Imagens quase
# sem textura (agua, borda preta) colidem entre si sem serem a mesma cena, entao
# ficam fora da comparacao.
LIMIAR_HAMMING = 16
DESVIO_MINIMO = 8.0       # desvio dos tons de cinza na miniatura 64x64
ZEROS_MAXIMO = 0.5


# ------------------------------------------------------------------- hashing


def _dhash(cinza: Image.Image) -> np.ndarray:
    a = np.asarray(cinza.resize((17, 16), Image.BOX), dtype=np.int16)
    # 4 x int64 com sinal: parquet e pandas guardam sem surpresa; a comparacao reinterpreta
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel()).view(">i8").astype(np.int64)


def _imagem(caminho: Path) -> dict:
    arr = np.asarray(Image.open(caminho))
    cinza = Image.fromarray(arr).convert("L")
    mini = np.asarray(cinza.resize((64, 64), Image.BOX), dtype=np.float32)
    return {
        "sha": hashlib.sha256(arr.tobytes()).hexdigest(),
        "dhash": _dhash(cinza),
        "zeros": float((arr.reshape(-1, arr.shape[-1]) == 0).all(axis=1).mean()),
        "desvio": float(mini.std()),
        "shape": arr.shape,
    }


def _cena(tarefa: tuple[str, str, str]) -> dict:
    dados, split, cena = Path(tarefa[0]), tarefa[1], tarefa[2]
    rec = {"scene_id": cena}
    try:
        for fase in ("pre", "post"):
            r = _imagem(dados / "images" / split / f"{cena}_{fase}.png")
            rec[f"sha256_{fase}"] = r["sha"]
            rec[f"zeros_{fase}"] = r["zeros"]
            rec[f"desvio_{fase}"] = r["desvio"]
            rec[f"shape_{fase}"] = "x".join(map(str, r["shape"]))
            for k, v in enumerate(r["dhash"]):
                rec[f"dhash_{fase}_{k}"] = int(v)
        msk = np.asarray(Image.open(dados / "masks" / split / f"{cena}.png"))
        rec["sha256_mask"] = hashlib.sha256(msk.tobytes()).hexdigest()
        rec["shape_mask"] = "x".join(map(str, msk.shape))
        rec["erro"] = ""
    except Exception as exc:  # arquivo ausente ou truncado: vira linha de auditoria
        rec["erro"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def calcular_hashes(index: pd.DataFrame, dados: Path, workers: int) -> pd.DataFrame:
    tarefas = [(str(dados), s, c) for s, c in index[["split", "scene_id"]].itertuples(index=False)]
    linhas = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(_cena, tarefas, chunksize=16), 1):
            linhas.append(rec)
            if i % 1000 == 0:
                print(f"  {i:,}/{len(tarefas):,}", flush=True)
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- duplicatas


class Componentes:
    """Union-find simples: cenas ligadas por qualquer imagem repetida."""

    def __init__(self, itens):
        self.pai = {i: i for i in itens}

    def achar(self, i):
        while self.pai[i] != i:
            self.pai[i] = self.pai[self.pai[i]]
            i = self.pai[i]
        return i

    def unir(self, a, b):
        ra, rb = self.achar(a), self.achar(b)
        if ra != rb:
            self.pai[max(ra, rb)] = min(ra, rb)   # representante = menor scene_id


def duplicatas(h: pd.DataFrame) -> tuple[list[tuple], list[tuple]]:
    """(pares exatos, pares proximos) entre imagens de cenas DIFERENTES."""
    imgs = pd.concat([
        pd.DataFrame({"scene_id": h.scene_id, "fase": f, "sha": h[f"sha256_{f}"],
                      "zeros": h[f"zeros_{f}"], "desvio": h[f"desvio_{f}"],
                      **{f"d{k}": h[f"dhash_{f}_{k}"] for k in range(4)}})
        for f in ("pre", "post")
    ], ignore_index=True)

    exatos = []
    for _, g in imgs.groupby("sha"):
        if g.scene_id.nunique() > 1:
            itens = list(g[["scene_id", "fase"]].itertuples(index=False, name=None))
            exatos += [(itens[0], it, 0) for it in itens[1:]]

    ok = imgs[(imgs.desvio >= DESVIO_MINIMO) & (imgs.zeros <= ZEROS_MAXIMO)].reset_index(drop=True)
    bits = ok[[f"d{k}" for k in range(4)]].to_numpy(np.int64).view(np.uint64)
    cenas = ok.scene_id.to_numpy()
    proximos = []
    for i0 in range(0, len(ok), 128):
        bloco = bits[i0:i0 + 128]
        dist = np.bitwise_count(bloco[:, None, :] ^ bits[None, :, :]).sum(-1)
        for a, b in zip(*np.nonzero(dist <= LIMIAR_HAMMING)):
            i = i0 + a
            if b > i and cenas[i] != cenas[b]:
                proximos.append(((cenas[i], ok.fase[i]), (cenas[b], ok.fase[b]), int(dist[a, b])))
    return exatos, proximos


# ------------------------------------------------------------------ manifesto


def rotulo(r) -> str:
    """Rotulo da cena = severidade maxima presente (a tarefa e por pixel)."""
    for col, nome in SEVERIDADE:
        if r[col] > 0:
            return nome
    return "un-classified" if r.px_unclassified > 0 else "sem-edificacao"


def pares_de_cenas(exatos, proximos) -> dict[tuple, dict]:
    """Junta os pares de imagens em pares de cenas: {(a, b): {fases, exato}}."""
    pares = {}
    for (a, fa), (b, fb), d in exatos + proximos:
        chave = tuple(sorted((a, b)))
        p = pares.setdefault(chave, {"fases": set(), "exato": False})
        p["fases"].add(fa if fa == fb else f"{fa}~{fb}")
        p["exato"] |= d == 0
    return pares


def montar(index, images, buildings, h, grupo_dup, pares, fora) -> pd.DataFrame:
    gemeas = {}
    for (a, b), p in pares.items():
        fases = "+".join(sorted(p["fases"]))
        tipo = "identica" if p["exato"] else "quase identica"
        gemeas.setdefault(a, []).append(f"imagem {fases} {tipo} a de {b}")
        gemeas.setdefault(b, []).append(f"imagem {fases} {tipo} a de {a}")
    pre = images[images.phase == "pre"].set_index("scene_id")
    post = images[images.phase == "post"].set_index("scene_id")
    invalidas = buildings[~buildings.is_valid].groupby(["scene_id", "phase"]).size().unstack(fill_value=0)

    grupo = {d: grupo_de(d) for d in index.disaster.unique()}
    fold_de = folds(grupo.values(), N_FOLDS)
    gemea_aval = {}
    for (a, b) in pares:
        if a in fora:
            gemea_aval[a] = b
        if b in fora:
            gemea_aval[b] = a

    m = index.merge(h, on="scene_id", how="left").set_index("scene_id", drop=False)
    info_tier = dict(zip(index.scene_id, index.split))
    gap = (post.capture_date - pre.capture_date).dt.days

    linhas = []
    for c, r in m.iterrows():
        tier, p, q = r.split, post.loc[c], pre.loc[c]
        n_pos = int(invalidas.at[c, "post"]) if c in invalidas.index and "post" in invalidas else 0
        n_pre = int(invalidas.at[c, "pre"]) if c in invalidas.index and "pre" in invalidas else 0

        problemas = []
        if r.erro:
            problemas.append(f"leitura falhou: {r.erro}")
        if n_pos:
            problemas.append(f"{n_pos} poligono(s) invalido(s) no label pos, corrigido(s) com buffer(0)")
        if n_pre:
            problemas.append(f"{n_pre} poligono(s) invalido(s) so no label pre (nao entra na mascara)")
        n_unc = int(p["n_un-classified"])
        if n_unc:
            problemas.append(f"{n_unc} un-classified -> ignore_index 255")
        if p.n_buildings == 0:
            problemas.append("sem edificacao")
        z = max(r.zeros_pre, r.zeros_post)
        if z > 0.2:
            problemas.append(f"borda sem dado: {z:.0%} dos pixels zerados")
        if gap[c] > 3 * 365:
            problemas.append(f"gap pre/pos de {gap[c]} dias")
        problemas += gemeas.get(c, [])

        integro = (not r.erro) and bool(p.tif_exists) and bool(q.tif_exists) \
            and r.shape_pre == r.shape_post == "1024x1024x3"
        linhas.append({
            "sample_id": c,
            "file_path": f"data/images/{tier}/{c}_pre.png;data/images/{tier}/{c}_post.png",
            "label": rotulo(r),
            "group_id": grupo[r.disaster],
            "source": f"xBD {tier}",
            "license": LICENCA,
            "width": int(p.width), "height": int(p.height), "channels": int(p.bands),
            "annotation_path": f"geotiffs/{tier}/labels/{c}_post_disaster.json",
            "object_count": int(p.n_buildings),
            "integrity_status": "OK" if integro else "Corrompido",
            "issue_type": "; ".join(problemas),
            "split": "excluir" if c in fora else SPLIT_OFICIAL[tier],
            "file_hash": f"{r.sha256_pre[:16]}/{r.sha256_post[:16]}" if not r.erro else "",
            "duplicate_group": grupo_dup.get(c, c),
            "capture_group": str(p.catalog_id),
            "notes": f"fold_agrupado={fold_de[grupo[r.disaster]]}; tipo={p.disaster_type}",
            "include": "Não" if c in fora else "Sim",
            "exclusion_reason": (f"copia identica de {gemea_aval[c]} ({SPLIT_OFICIAL[info_tier[gemea_aval[c]]]}): "
                                 "sai do treino oficial" if c in fora else ""),
            # ---- colunas extras (so no CSV) ----
            "disaster": r.disaster,
            "tier": tier,
            "fold_agrupado": fold_de[grupo[r.disaster]],
            "disaster_type": str(p.disaster_type),
            "mask_path": f"data/masks/{tier}/{c}.png",
            "geotiff_pre": f"geotiffs/{tier}/images/{c}_pre_disaster.tif",
            "geotiff_post": f"geotiffs/{tier}/images/{c}_post_disaster.tif",
            "sha256_pre": r.sha256_pre, "sha256_post": r.sha256_post, "sha256_mask": r.sha256_mask,
            "catalog_id_pre": str(q.catalog_id), "catalog_id_post": str(p.catalog_id),
            "sensor_post": str(p.sensor),
            "data_pre": q.capture_date.date().isoformat(), "data_post": p.capture_date.date().isoformat(),
            "gap_dias": int(gap[c]),
            "off_nadir_post": round(float(p.off_nadir_angle), 2),
            "lon": round(float(p.lon_center), 6), "lat": round(float(p.lat_center), 6),
            "n_no_damage": int(p["n_no-damage"]), "n_minor": int(p["n_minor-damage"]),
            "n_major": int(p["n_major-damage"]), "n_destroyed": int(p["n_destroyed"]),
            "n_unclassified": n_unc,
            "frac_edificacao": round(r.px_edificacao / (1024 * 1024), 5),
            "zeros_pre": round(r.zeros_pre, 4), "zeros_post": round(r.zeros_post, 4),
        })
    ordem = {"train": 0, "validation": 1, "test": 2, "excluir": 3}
    return pd.DataFrame(linhas).sort_values(["split", "group_id", "sample_id"],
                                            key=lambda s: s.map(ordem) if s.name == "split" else s
                                            ).reset_index(drop=True)


def figura_duplicatas(info: pd.DataFrame, tab: pd.DataFrame, dados: Path, destino: Path) -> None:
    """Um par por combinacao de splits (o primeiro em ordem alfabetica que tenha edificacao)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = matplotlib.colors.ListedColormap(["black", "#2ca02c", "#ffdd00", "#ff7f0e", "#d62728", "#888888"])
    escolhidos = []
    for _, g in tab.groupby("splits", sort=True):
        com_predio = g[(info.loc[g.a, "object_count"].to_numpy() > 0) & (info.loc[g.b, "object_count"].to_numpy() > 0)]
        escolhidos.append((com_predio if len(com_predio) else g).iloc[0])
    fig, ax = plt.subplots(2 * len(escolhidos), 3, figsize=(10.5, 7 * len(escolhidos)))
    ax = ax.reshape(-1, 3)
    for k, par in enumerate(escolhidos):
        for i, c in enumerate((par.a, par.b)):
            r, lin = info.loc[c], 2 * k + i
            ax[lin, 0].imshow(Image.open(dados / "images" / r.tier / f"{c}_pre.png"))
            ax[lin, 0].set_title(f"{c}  [{r.split}]\npre {r.data_pre}  captura {r.catalog_id_pre}", fontsize=8)
            ax[lin, 1].imshow(Image.open(dados / "images" / r.tier / f"{c}_post.png"))
            ax[lin, 1].set_title(f"pos {r.data_post}  captura {r.catalog_id_post}", fontsize=8)
            ax[lin, 2].imshow(np.asarray(Image.open(dados / "masks" / r.tier / f"{c}.png")), cmap=cores,
                              vmin=0, vmax=5, interpolation="nearest")
            ax[lin, 2].set_title(f"mascara ({r.object_count} poligonos)", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(destino, dpi=80)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", type=Path, default=RAIZ / "data")
    ap.add_argument("--cache", type=Path, default=RAIZ / "cache")
    ap.add_argument("--saida", type=Path, default=RAIZ / "manifestos")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refazer-hashes", action="store_true")
    args = ap.parse_args()

    index = pd.read_parquet(args.dados / "index.parquet")
    images = pd.read_parquet(args.cache / "images.parquet")
    buildings = pd.read_parquet(args.cache / "buildings.parquet", columns=["scene_id", "phase", "is_valid"])
    for df in (index, images, buildings):
        for col in df.select_dtypes(["category", "string"]).columns:
            df[col] = df[col].astype(str)

    arq_h = args.cache / "hashes.parquet"
    if arq_h.exists() and not args.refazer_hashes:
        h = pd.read_parquet(arq_h)
        print(f"hashes reaproveitados de {arq_h}")
    else:
        print(f"lendo {len(index):,} cenas de {args.dados} ...")
        h = calcular_hashes(index, args.dados, args.workers)
        h.to_parquet(arq_h, index=False)

    exatos, proximos = duplicatas(h)
    pares = pares_de_cenas(exatos, proximos)
    uf = Componentes(index.scene_id)
    for a, b in pares:
        uf.unir(a, b)
    grupo_dup = {c: uf.achar(c) for c in index.scene_id}

    # pares primeiro: xbd/splits.py le este arquivo para decidir o que sai do treino
    args.saida.mkdir(parents=True, exist_ok=True)
    ev = dict(zip(index.scene_id, index.disaster))
    tr = dict(zip(index.scene_id, index.split))
    pd.DataFrame([{"a": a, "b": b, "exato": p["exato"], "imagens": "+".join(sorted(p["fases"])),
                   "evento_a": ev[a], "evento_b": ev[b], "tier_a": tr[a], "tier_b": tr[b]}
                  for (a, b), p in sorted(pares.items())]
                 ).to_csv(args.saida / "duplicatas.csv", index=False)
    fora = copias_de_avaliacao(index, args.saida / "duplicatas.csv")

    m = montar(index, images, buildings, h, grupo_dup, pares, fora)
    extras = list(m.columns[20:])
    m.to_csv(args.saida / "manifesto.csv", index=False)
    for sp in ("train", "validation", "test"):
        m[m.split == sp].to_csv(args.saida / f"{sp}.csv", index=False)

    # antes da correcao: split oficial como o xView2 entrega; depois: manifesto
    info = m.set_index("sample_id")
    oficial = {c: SPLIT_OFICIAL[t] for c, t in tr.items()}
    tab = pd.DataFrame([{"a": a, "b": b, "exato": p["exato"], "fases": "+".join(sorted(p["fases"])),
                         "eventos": " | ".join(sorted({ev[a], ev[b]})),
                         "splits": " | ".join(sorted({oficial[a], oficial[b]})),
                         "splits_corrigidos": " | ".join(sorted({info.at[a, "split"], info.at[b, "split"]} - {"excluir"})),
                         "folds_antes": " | ".join(sorted({str(folds(ev.values(), N_FOLDS)[ev[a]]),
                                                           str(folds(ev.values(), N_FOLDS)[ev[b]])})),
                         "folds": " | ".join(sorted({str(info.at[a, "fold_agrupado"]), str(info.at[b, "fold_agrupado"])}))}
                        for (a, b), p in sorted(pares.items())])
    cruza_split = tab[tab.splits.str.contains("|", regex=False)]
    gemea_no_treino = {sp: int((cruza_split.splits == f"train | {sp}").sum() + (cruza_split.splits == f"{sp} | train").sum())
                       for sp in ("validation", "test")}
    incl = m[m.split != "excluir"]
    eventos_cruzados = incl.groupby("group_id").split.nunique()
    grupos_em_folds = m.groupby("group_id").fold_agrupado.nunique()
    info_fig = info.assign(split=info.index.map(oficial))
    figura_duplicatas(info_fig, tab, args.dados, args.saida / "duplicatas.png")
    resumo = {
        "cenas": len(m),
        "imagens_lidas": int(2 * (h.erro == "").sum()),
        "falhas_de_leitura": int((h.erro != "").sum()),
        "fora_de_1024x1024x3": int((m.integrity_status != "OK").sum()),
        "pre_igual_pos_na_mesma_cena": int((h.sha256_pre == h.sha256_post).sum()),
        "pares_de_imagens_identicas": len(exatos),
        "pares_de_imagens_quase_identicas": {"limiar_hamming_256": LIMIAR_HAMMING, "pares": len(proximos),
                                             "imagens_fora_por_baixa_textura":
                                                 int(((h.desvio_pre < DESVIO_MINIMO) | (h.zeros_pre > ZEROS_MAXIMO)).sum()
                                                     + ((h.desvio_post < DESVIO_MINIMO) | (h.zeros_post > ZEROS_MAXIMO)).sum())},
        "pares_de_cenas_duplicadas": {
            "total": len(tab),
            "com_imagem_identica": int(tab.exato.sum()),
            "por_eventos_e_imagem": {f"{k[0]} [{k[1]}]": int(v) for k, v in tab.groupby(["eventos", "fases"]).size().items()},
        },
        "antes_da_correcao": {
            "pares_que_atravessam_o_split_oficial": len(cruza_split),
            "por_splits": {k: int(v) for k, v in cruza_split.splits.value_counts().items()},
            "cenas_de_val_test_com_copia_no_treino": gemea_no_treino,
            "pares_que_atravessam_folds_agrupados": int(tab.folds_antes.str.contains("|", regex=False).sum()),
        },
        "correcao": {
            "cenas_excluidas_do_treino_oficial": len(fora),
            "eventos_unidos_no_split_agrupado": sorted({g for g in m.group_id if "+" in g}),
        },
        "depois_da_correcao": {
            "pares_que_atravessam_o_split_oficial": int(tab.splits_corrigidos.str.contains("|", regex=False).sum()),
            "pares_que_atravessam_folds_agrupados": int(tab.folds.str.contains("|", regex=False).sum()),
            "grupos_em_mais_de_um_fold": int((grupos_em_folds > 1).sum()),
            "grupos_em_mais_de_um_split_oficial": int((eventos_cruzados > 1).sum()),
            "cenas_desses_grupos": int(incl.group_id.isin(eventos_cruzados[eventos_cruzados > 1].index).sum()),
        },
        "exemplos_de_pares": tab.head(20).to_dict("records"),
        "cenas_com_mais_de_20pct_zerados": int((m[["zeros_pre", "zeros_post"]].max(axis=1) > 0.2).sum()),
        "cenas_por_split": {k: int(v) for k, v in m.split.value_counts().items()},
        "cenas_por_fold_agrupado": {int(k): int(v) for k, v in m.fold_agrupado.value_counts().sort_index().items()},
        "colunas_extras_do_csv": extras,
    }
    (args.saida / "auditoria.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in resumo.items() if k != "exemplos_de_pares"}, indent=2, ensure_ascii=False))
    print(f"-> {args.saida}/manifesto.csv ({len(m):,} linhas) + train/validation/test.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
