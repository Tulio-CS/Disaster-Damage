#!/usr/bin/env python3
"""
Reavalia os checkpoints coletados sob um protocolo comum, em cena inteira.

    python scripts/avaliar.py --checkpoints E:\\colheita --dados %USERPROFILE%\\Disaster-Damage\\data

Por que existe
--------------
O treino valida num recorte CENTRAL de `cfg.crop` pixels, e `cfg.crop` e um dos
fatores ablacionados. Consequencia: `abl-crop256`, `ref` e `abl-crop768` foram
avaliados sobre 6%, 25% e 56% da cena, respectivamente — conjuntos de validacao
diferentes. O delta entre eles confunde "efeito do crop no treino" com "efeito
de avaliar em mais area", e a segunda parte e artefato.

Aqui todo modelo e avaliado na cena INTEIRA (1024x1024), o que:
  - torna a ablacao de resolucao uma comparacao limpa;
  - aproxima os numeros do protocolo do xView2, que pontua a cena inteira.

Cada modelo e avaliado no split que a PROPRIA config define. Os folds agrupados
validam em desastres que nao viram no treino — avalia-los em `hold` daria um
numero otimista e sem significado, porque `hold` contem eventos que eles usaram
para treinar. A referencia e todas as ablacoes usam `hold`, entao continuam
mutuamente comparaveis.

Nada aqui toca o codigo de treino: e analise pos-hoc sobre os pesos salvos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import pandas as pd                                          # noqa: E402
import torch                                                 # noqa: E402
from torch.utils.data import DataLoader                      # noqa: E402

from xbd.config import Config                                # noqa: E402
from xbd.data import XBDDataset, montar_splits               # noqa: E402
from xbd.metrics import Confusao                             # noqa: E402
from xbd.models import build_model                           # noqa: E402

WINDOWS = os.name == "nt"
LADO = 1024          # cena inteira do xBD


def achar_checkpoints(raiz: Path) -> list[Path]:
    """Todos os melhor.pt sob a raiz, em qualquer profundidade."""
    return sorted(raiz.rglob("melhor.pt"))


def limpar_chaves(estado: dict) -> dict:
    """Remove o prefixo que torch.compile acrescenta, se houver."""
    if any(k.startswith("_orig_mod.") for k in estado):
        return {k.replace("_orig_mod.", "", 1): v for k, v in estado.items()}
    return estado


@torch.no_grad()
def avaliar(ck: Path, index: pd.DataFrame, dados: Path, dev: torch.device,
            batch: int, workers: int, usar_ema: bool, limite: int | None) -> dict:
    bruto = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(bruto["config"])
    cfg.data_dir = str(dados)

    # Split que ESTE experimento define. `crop` vai para 1024 apenas na avaliacao:
    # `Aug` faz `c = min(crop, h, w)`, entao 1024 numa cena 1024x1024 e o mesmo
    # que nao recortar. A config original fica intacta no checkpoint.
    _, val_df = montar_splits(index, cfg)
    cfg_eval = Config.from_dict({**cfg.to_dict(), "crop": LADO})

    ds = XBDDataset(val_df, dados, cfg_eval, treino=False, norm=cfg.norm)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=(dev.type == "cuda"))

    modelo = build_model(cfg)
    chave = "ema" if (usar_ema and bruto.get("ema")) else "modelo"
    modelo.load_state_dict(limpar_chaves(bruto[chave]))
    modelo = modelo.to(dev).eval()

    amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else None
    conf = Confusao(cfg.n_classes, device=dev)
    n_vistas, t0 = 0, time.time()

    for i, lote in enumerate(dl):
        if limite and n_vistas >= limite:
            break
        pre = lote["pre"].to(dev, non_blocking=True)
        post = lote["post"].to(dev, non_blocking=True)
        alvo = lote["mask"].to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            logits = modelo(pre, post)
        conf.update(logits.argmax(1), alvo)
        n_vistas += pre.shape[0]

    dt = time.time() - t0
    m = conf.resumo()
    del modelo
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "nome": cfg.nome, "run_id": ck.parent.name, "maquina": ck.parents[2].name,
        "split_mode": cfg.split_mode, "fold": cfg.fold, "crop_treino": cfg.crop,
        "epoca_do_ckpt": bruto.get("epoca"), "usou_ema": chave == "ema",
        "cenas_avaliadas": n_vistas, "seg": round(dt, 1),
        "img_por_s": round(n_vistas / max(dt, 1e-9), 2),
        "score_treino": (bruto.get("metricas") or {}).get("score_xview2"),
        **{k: v for k, v in m.items() if k != "matriz"},
        "matriz": m["matriz"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", type=Path, required=True,
                    help="raiz da colheita (varre recursivamente por melhor.pt)")
    ap.add_argument("--dados", type=Path, default=RAIZ / "data")
    ap.add_argument("--saida", type=Path, default=None,
                    help="padrao: <checkpoints>/avaliacao.json")
    ap.add_argument("--batch", type=int, default=0, help="0 = escolhe pela VRAM")
    ap.add_argument("--workers", type=int, default=2 if WINDOWS else 4)
    ap.add_argument("--sem-ema", action="store_true", help="ignora os pesos EMA")
    ap.add_argument("--rapido", type=int, metavar="N", default=0,
                    help="so N cenas por modelo — para estimar o tempo total")
    ap.add_argument("--so", nargs="*", default=None, help="so estes experimentos")
    args = ap.parse_args()

    idx_path = args.dados / "index.parquet"
    if not idx_path.exists():
        print(f"!! {idx_path} nao existe. Aponte --dados para a pasta data/.")
        return 1
    index = pd.read_parquet(idx_path)

    cks = achar_checkpoints(args.checkpoints)
    if not cks:
        print(f"!! nenhum melhor.pt sob {args.checkpoints}")
        return 1

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch:
        batch = args.batch
    elif dev.type != "cuda":
        batch = 1
    else:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch = max(1, min(8, int(vram // 3)))   # ~3 GB por cena 1024 em inferencia

    print("=" * 78)
    print(f"  {len(cks)} checkpoint(s)  |  dispositivo: {dev.type}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if dev.type == 'cuda' else ''}")
    print(f"  batch {batch}  |  avaliacao em cena inteira {LADO}x{LADO}")
    if dev.type != "cuda":
        print("  ! sem GPU: isto vai levar horas. Rode numa maquina do laboratorio.")
    if args.rapido:
        print(f"  ! modo rapido: {args.rapido} cenas por modelo (numeros NAO finais)")
    print("=" * 78)

    saida = args.saida or (args.checkpoints / "avaliacao.json")
    resultados: list[dict] = []
    if saida.exists():                            # retomavel
        try:
            resultados = json.loads(saida.read_text(encoding="utf-8"))
            print(f"\nretomando: {len(resultados)} ja avaliado(s) em {saida.name}")
        except json.JSONDecodeError:
            pass
    feitos = {r["run_id"] for r in resultados}

    t_inicio = time.time()
    for i, ck in enumerate(cks, 1):
        if ck.parent.name in feitos:
            continue
        try:
            r = avaliar(ck, index, args.dados, dev, batch, args.workers,
                        not args.sem_ema, args.rapido or None)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{i}/{len(cks)}] {ck.parent.name}: sem VRAM — use --batch menor")
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            print(f"  [{i}/{len(cks)}] {ck.parent.name}: FALHOU ({type(exc).__name__}: {exc})")
            continue

        resultados.append(r)
        saida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                         encoding="utf-8")     # grava a cada modelo: queda nao perde
        antes = r["score_treino"]
        delta = f"{r['score_xview2'] - antes:+.4f}" if isinstance(antes, (int, float)) else "—"
        print(f"  [{i}/{len(cks)}] {r['nome']:<26} score {r['score_xview2']:.4f}  "
              f"(treino {antes if antes is None else f'{antes:.4f}'}, {delta})  "
              f"{r['cenas_avaliadas']} cenas, {r['seg']:.0f}s, {r['img_por_s']:.1f} img/s")

        if i == 1 and not args.rapido:
            resta = (len(cks) - 1) * r["seg"] / 60
            print(f"       estimativa para os {len(cks)-1} restantes: ~{resta:.0f} min")

    # ---- tabela final ----
    print()
    print("=" * 78)
    print(f"  {len(resultados)} modelo(s) em {(time.time()-t_inicio)/60:.1f} min  ->  {saida}")
    if resultados:
        ref = next((r for r in resultados if r["nome"] == "ref"), None)
        print()
        print(f"  {'experimento':<26}{'score':<9}{'Δ ref':<9}{'loc':<8}{'dano':<8}crop")
        print("  " + "-" * 68)
        for r in sorted(resultados, key=lambda x: -x["score_xview2"]):
            d = (f"{r['score_xview2'] - ref['score_xview2']:+.4f}"
                 if ref and r is not ref else ("—" if ref else "?"))
            print(f"  {r['nome']:<26}{r['score_xview2']:<9.4f}{d:<9}"
                  f"{r['f1_loc']:<8.4f}{r.get('f1_dano_harmonico', float('nan')):<8.4f}"
                  f"{r['crop_treino']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    codigo = main()
    if WINDOWS:
        input("\nEnter para fechar...")
    raise SystemExit(codigo)
