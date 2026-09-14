"""Junta os indices parciais dos shards em `data/index.parquet`.

Roda depois que todas as maquinas terminaram o `xbd.prep`. Os indices parciais
sao pequenos (algumas centenas de KB), entao dao para trafegar por git; os PNG
seguem por rsync/rclone. Ver RUNBOOK.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    aqui = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", type=Path, default=aqui / "data")
    ap.add_argument("--strict", action="store_true",
                    help="falha se algum shard estiver faltando")
    args = ap.parse_args()

    parciais = sorted(args.out.glob("index_shard*of*.parquet"))
    if not parciais:
        raise SystemExit(f"nenhum index_shard*.parquet em {args.out}")

    esperado = int(parciais[0].stem.split("of")[1])
    vistos = {int(p.stem.split("of")[0].split("shard")[1]) for p in parciais}
    faltando = sorted(set(range(esperado)) - vistos)
    if faltando:
        msg = f"shards ausentes: {faltando} (de {esperado})"
        if args.strict:
            raise SystemExit("ERRO: " + msg)
        print("AVISO:", msg)

    df = pd.concat([pd.read_parquet(p) for p in parciais], ignore_index=True)
    antes = len(df)
    df = df.drop_duplicates(subset=["split", "scene_id"], keep="last")
    if len(df) != antes:
        print(f"  {antes - len(df)} duplicatas removidas (shards sobrepostos)")

    destino = args.out / "index.parquet"
    df.to_parquet(destino, index=False)
    print(f"-> {len(df):,} cenas em {destino}")
    print(f"   {df.tem_edificacao.sum():,} com edificacao, {df.tem_dano.sum():,} com dano")
    print(f"   {df.bytes.sum()/1e9:.1f} GB em PNG")
    print("\npor split:")
    print(df.groupby("split").agg(cenas=("scene_id", "size"),
                                  com_dano=("tem_dano", "sum"),
                                  GB=("bytes", lambda s: round(s.sum() / 1e9, 1))).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
