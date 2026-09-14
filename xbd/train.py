"""
CLI de treino. Detecta a maquina, deriva o plano e roda.

    python -m xbd.train --preset smoke                 # valida o pipeline (roda ate em CPU)
    python -m xbd.train --preset base                  # GPU de 6-12 GB
    python -m xbd.train --preset grouped --fold 2       # split honesto, fold 2
    python -m xbd.train --preset base --autotune        # acha o maior batch que cabe
    python -m xbd.train --config exp/meu.yaml           # config explicita

O `run_id` deriva de um hash dos campos que definem o experimento, sem incluir os
de execucao. Duas maquinas rodando a mesma receita com batches diferentes geram o
MESMO hash de config, o que permite comparar e agregar os resultados depois.
"""

from __future__ import annotations

import argparse
import os
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config, PRESETS, from_preset
from .data import build_loaders, contagens_de_classe
from .engine import (EMA, ajustar_batch_automaticamente, build_scheduler,
                     carregar_para_retomar, salvar_checkpoint, treinar_epoca, validar)
from .hardware import plan, probe
from .losses import build_loss
from .models import build_model
from .reporter import do_ambiente
from .runs import Run


def semear(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(__file__).resolve().parent.parent

    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=raiz / "data")
    ap.add_argument("--results-dir", type=Path, default=raiz / "results")
    ap.add_argument("--ckpt-dir", type=Path, default=raiz / "checkpoints")

    # sobrescritas comuns
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--crop", type=int)
    ap.add_argument("--encoder")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--effective-batch", type=int)
    ap.add_argument("--task", choices=["loc", "damage"])
    ap.add_argument("--split-mode", choices=["oficial", "grouped"])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--norm", choices=["dataset", "imagenet", "per_image"], default=None,
                    help="sobrescreve cfg.norm")
    ap.add_argument("--fuse-mode", choices=["cat_diff", "cat", "diff"], default=None)

    # execucao
    ap.add_argument("--batch-size", type=int, help="sobrescreve a heuristica de hardware")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--autotune", action="store_true", help="acha empiricamente o maior batch")
    ap.add_argument("--no-compile", action="store_true",
                    help="desliga torch.compile mesmo onde o plano o habilitaria")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="mostra o plano e sai")
    args = ap.parse_args(argv)

    # ---------------------------------------------------------------- config
    if args.config:
        cfg = Config.load(args.config)
    elif args.preset:
        cfg = from_preset(args.preset)
    else:
        ap.error("informe --preset ou --config")

    for campo in ("epochs", "fold", "seed", "crop", "encoder", "lr", "task",
                  "split_mode", "max_steps", "effective_batch", "norm", "fuse_mode"):
        v = getattr(args, campo, None)
        if v is not None:
            setattr(cfg, campo, v)
    cfg.data_dir = str(args.data_dir)
    cfg.results_dir = str(args.results_dir)
    cfg.ckpt_dir = str(args.ckpt_dir)
    if args.no_resume:
        cfg.resume = False
    cfg.validar()   # sobrescritas da CLI tambem passam pela checagem

    # ---------------------------------------------------------------- hardware
    hw = probe(disk_path=str(args.ckpt_dir.parent))
    print("=" * 70)
    print(hw.summary())
    print("=" * 70)

    plano = plan(hw, crop=cfg.crop, encoder=cfg.encoder,
                 effective_batch=cfg.effective_batch, batch_size=args.batch_size,
                 num_workers=args.num_workers, precision=args.precision)
    cfg.host = hw.hostname

    # ---------------------------------------------------------------- dados
    idx_path = Path(cfg.data_dir) / "index.parquet"
    if not idx_path.exists():
        parciais = sorted(Path(cfg.data_dir).glob("index_shard*.parquet"))
        if not parciais:
            raise SystemExit(f"{idx_path} nao existe. Rode antes:\n"
                             f"  python -m xbd.prep --shard 0 --of 1\n"
                             f"  python -m xbd.prep_merge")
        print(f"! index.parquet ausente; usando {len(parciais)} indices parciais")
        index = pd.concat([pd.read_parquet(p) for p in parciais], ignore_index=True)
    else:
        index = pd.read_parquet(idx_path)

    semear(cfg.seed)
    modelo = build_model(cfg, grad_checkpointing=plano.grad_checkpointing)

    if args.autotune:
        criterio_tmp = build_loss(cfg)
        antes = plano.batch_size
        plano.batch_size = ajustar_batch_automaticamente(modelo, criterio_tmp, plano, cfg)
        plano.accum_steps = max(1, -(-cfg.effective_batch // (plano.batch_size * max(plano.n_gpus, 1))))
        plano.effective_batch = plano.batch_size * plano.accum_steps * max(plano.n_gpus, 1)
        plano.notas.append(f"autotune: batch {antes} -> {plano.batch_size}")
        modelo = build_model(cfg, grad_checkpointing=plano.grad_checkpointing)

    print(plano.summary())
    print("=" * 70)

    dl_tr, dl_va, df_tr, df_va = build_loaders(index, cfg, plano)
    print(f"treino  {len(df_tr):,} cenas ({df_tr.tem_dano.sum():,} com dano)")
    print(f"val     {len(df_va):,} cenas ({df_va.tem_dano.sum():,} com dano)")
    if cfg.split_mode == "grouped":
        print(f"        fold {cfg.fold}: {sorted(df_va.disaster.unique())}")
    print(f"run_id  {cfg.run_id}")
    print("=" * 70)

    if args.dry_run:
        print("\n--dry-run: nada foi treinado.")
        return 0

    # ---------------------------------------------------------------- objetos
    dev = torch.device(plano.device)
    modelo = modelo.to(dev)
    if plano.channels_last:
        modelo = modelo.to(memory_format=torch.channels_last)

    if cfg.class_weights:
        pesos = torch.tensor(cfg.class_weights, dtype=torch.float32)
    else:
        from .losses import pesos_por_frequencia
        pesos = pesos_por_frequencia(contagens_de_classe(df_tr, cfg.n_classes)).float()
    print("pesos de classe:", np.round(pesos.numpy(), 3).tolist())

    criterio = build_loss(cfg, class_weights=pesos.to(dev))
    otim = torch.optim.AdamW(
        modelo.grupos_de_parametros(cfg.lr, cfg.lr_encoder_mult, cfg.weight_decay))
    passos_por_epoca = max(1, len(dl_tr) // plano.accum_steps)
    sched = build_scheduler(otim, cfg, passos_por_epoca)
    scaler = torch.amp.GradScaler("cuda") if plano.precision == "fp16" and plano.device == "cuda" else None
    ema = EMA(modelo, cfg.ema_decay) if cfg.ema_decay > 0 else None

    if plano.compile_model and not args.no_compile:
        # `torch.compile` e PREGUICOSO: esta chamada so embrulha o modulo e
        # retorna. A compilacao de verdade acontece no primeiro forward, ja
        # dentro de `treinar_epoca` — fora deste try, que portanto nunca captura
        # a falha real. O fallback tem que ser configurado no proprio dynamo.
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True   # grafo que falhar cai
            modelo = torch.compile(modelo)                # para eager, sem matar
        except Exception as exc:                          # o treino
            print(f"! torch.compile indisponivel ({exc}); seguindo sem compilar")
    elif args.no_compile and plano.compile_model:
        print("torch.compile desligado por --no-compile")

    ckpt_dir = Path(cfg.ckpt_dir) / cfg.run_id
    ck_ultimo, ck_melhor = ckpt_dir / "ultimo.pt", ckpt_dir / "melhor.pt"
    epoca_inicial = carregar_para_retomar(ck_ultimo, modelo, otim, sched, scaler, ema) if cfg.resume else 0
    if epoca_inicial:
        print(f"retomando da epoca {epoca_inicial}")

    reporter = do_ambiente(Path(cfg.results_dir) / "spool", nome_spool=cfg.run_id)
    if reporter:
        print(f"painel: {os.environ.get('LAB_URL')}  (spool {reporter.spool.name})")
    run = Run(cfg, hw, plano, results_dir=cfg.results_dir, reporter=reporter)

    # Recupera o melhor score ja alcancado: sem isso, a primeira validacao apos
    # uma retomada sobrescreveria `melhor.pt` mesmo sendo pior que o salvo.
    melhor_score = -1.0
    if epoca_inicial and ck_melhor.exists():
        try:
            anterior = torch.load(ck_melhor, map_location="cpu", weights_only=False)
            melhor_score = float(anterior.get("metricas", {}).get("score_xview2", -1.0))
            print(f"melhor score anterior: {melhor_score:.4f}")
        except Exception as exc:
            print(f"! nao consegui ler {ck_melhor} ({exc}); recomecando o ranking")

    # ---------------------------------------------------------------- laco
    try:
        for epoca in range(epoca_inicial, cfg.epochs):
            m_tr = treinar_epoca(modelo, dl_tr, criterio, otim, sched, scaler,
                                 plano, cfg, epoca, ema)

            metricas = dict(m_tr)
            if (epoca + 1) % cfg.val_every == 0 or epoca == cfg.epochs - 1:
                alvo_val = ema.shadow if ema else modelo
                m_va = validar(alvo_val, dl_va, criterio, plano, cfg,
                               max_lotes=cfg.max_steps)
                conf = m_va.pop("_confusao")
                metricas.update(m_va)
                print(f"\n  epoca {epoca}: score {m_va['score_xview2']:.4f} "
                      f"(loc {m_va['f1_loc']:.4f}"
                      + (f", dano {m_va.get('f1_dano_harmonico', float('nan')):.4f}"
                         if cfg.n_classes == 5 else "") + ")")
                print(conf.tabela())
                print()

                if m_va["score_xview2"] > melhor_score:
                    melhor_score = m_va["score_xview2"]
                    salvar_checkpoint(ck_melhor, modelo=modelo.state_dict(), epoca=epoca,
                                      metricas=m_va, config=cfg.to_dict(),
                                      ema=ema.shadow.state_dict() if ema else None)
                    run.registrar_melhor(m_va, ck_melhor)
                    print(f"  * melhor ate agora: {melhor_score:.4f} -> {ck_melhor}")

            run.log_epoca(epoca, metricas)
            salvar_checkpoint(ck_ultimo, modelo=modelo.state_dict(), otim=otim.state_dict(),
                              sched=sched.state_dict(),
                              scaler=scaler.state_dict() if scaler else None,
                              ema=ema.shadow.state_dict() if ema else None,
                              epoca=epoca, config=cfg.to_dict(),
                              rng_torch=torch.get_rng_state().numpy().tolist(),
                              rng_numpy=np.random.get_state())

        run.finalizar("concluido")
        print(f"\nconcluido. melhor score {melhor_score:.4f}")
        print(f"manifesto: {run.path_manifesto}")

    except KeyboardInterrupt:
        run.finalizar("interrompido")
        print("\ninterrompido; retome com o mesmo comando (--resume e o padrao)")
        return 130
    except Exception:
        run.finalizar("erro", traceback.format_exc())
        raise
    finally:
        # O spool ainda pode ter as ultimas epocas; da um tempo para subirem antes
        # do processo sair. O que nao subir fica no disco e vai no proximo run.
        if reporter:
            reporter.parar(timeout=30)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
