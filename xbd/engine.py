"""
Loop de treino: AMP, acumulacao de gradiente, EMA, retomada e checkpoint atomico.

Retomada e requisito de projeto, nao conforto: em maquina de laboratorio o
processo cai (sessao encerrada, outro usuario, OOM). Todo checkpoint carrega
modelo, otimizador, scheduler, scaler, epoca e o estado do RNG, entao retomar
continua a mesma trajetoria em vez de comecar uma nova.
"""

from __future__ import annotations

import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import Confusao


# ------------------------------------------------------------------------- apoio


class EMA:
    """Media movel exponencial dos pesos. Costuma valer ~1 ponto de F1 de graca."""

    def __init__(self, modelo: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(modelo).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, modelo: nn.Module) -> None:
        for s, m in zip(self.shadow.state_dict().values(), modelo.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)


def build_scheduler(otim, cfg, passos_por_epoca: int):
    total = max(1, cfg.epochs * passos_por_epoca)
    warmup = max(1, int(total * cfg.warmup_frac))

    def lr_lambda(passo: int) -> float:
        if passo < warmup:
            return passo / warmup
        if cfg.scheduler == "cosine":
            p = (passo - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(otim, lr_lambda)


def _amp_dtype(precision: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)


def salvar_checkpoint(path: Path, **estado) -> None:
    """Escreve em tmp e renomeia: um kill no meio nao corrompe o checkpoint bom."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(estado, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------------ epocas


def treinar_epoca(modelo, loader: DataLoader, criterio, otim, sched, scaler, plano,
                  cfg, epoca: int, ema: EMA | None = None, log_cada: int = 50) -> dict:
    modelo.train()
    dev = torch.device(plano.device)
    dtype = _amp_dtype(plano.precision)
    usar_amp = dtype is not None and plano.device == "cuda"

    soma, n_passos, t0 = 0.0, 0, time.time()
    otim.zero_grad(set_to_none=True)
    n_lotes = len(loader)
    limite = min(n_lotes, cfg.max_steps * plano.accum_steps) if cfg.max_steps else n_lotes

    for i, lote in enumerate(loader):
        if i >= limite:
            break
        pre = lote["pre"].to(dev, non_blocking=True)
        post = lote["post"].to(dev, non_blocking=True)
        alvo = lote["mask"].to(dev, non_blocking=True)
        if plano.channels_last:
            pre = pre.to(memory_format=torch.channels_last)
            post = post.to(memory_format=torch.channels_last)

        with torch.autocast(device_type=plano.device, dtype=dtype, enabled=usar_amp):
            logits = modelo(pre, post)
            perda, _ = criterio(logits, alvo)
            perda = perda / plano.accum_steps

        if scaler is not None:
            scaler.scale(perda).backward()
        else:
            perda.backward()

        if (i + 1) % plano.accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(otim)
            if cfg.clip_grad:
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.clip_grad)
            if scaler is not None:
                scaler.step(otim)
                scaler.update()
            else:
                otim.step()
            otim.zero_grad(set_to_none=True)
            sched.step()
            if ema is not None:
                ema.update(modelo)
            n_passos += 1

        soma += float(perda.detach()) * plano.accum_steps

        if log_cada and i % log_cada == 0:
            visto = i + 1
            ips = visto * plano.batch_size / max(time.time() - t0, 1e-9)
            print(f"    ep{epoca} {visto}/{limite}  perda {soma/visto:.4f}  "
                  f"lr {sched.get_last_lr()[-1]:.2e}  {ips:.1f} img/s", flush=True)

    dt = time.time() - t0
    return {
        "perda_treino": soma / max(1, min(limite, n_lotes)),
        "passos_otim": n_passos,
        "seg_treino": round(dt, 1),
        "img_por_s": round(limite * plano.batch_size / max(dt, 1e-9), 1),
        "lr_final": sched.get_last_lr()[-1],
    }


@torch.no_grad()
def validar(modelo, loader: DataLoader, criterio, plano, cfg, max_lotes: int | None = None) -> dict:
    modelo.eval()
    dev = torch.device(plano.device)
    dtype = _amp_dtype(plano.precision)
    usar_amp = dtype is not None and plano.device == "cuda"

    conf = Confusao(cfg.n_classes, device=dev)
    soma, n, t0 = 0.0, 0, time.time()

    for i, lote in enumerate(loader):
        if max_lotes and i >= max_lotes:
            break
        pre = lote["pre"].to(dev, non_blocking=True)
        post = lote["post"].to(dev, non_blocking=True)
        alvo = lote["mask"].to(dev, non_blocking=True)
        if plano.channels_last:
            pre = pre.to(memory_format=torch.channels_last)
            post = post.to(memory_format=torch.channels_last)

        with torch.autocast(device_type=plano.device, dtype=dtype, enabled=usar_amp):
            logits = modelo(pre, post)
            perda, _ = criterio(logits, alvo)

        conf.update(logits.argmax(1), alvo)
        soma += float(perda.detach())
        n += 1

    out = {"perda_val": soma / max(n, 1), "seg_val": round(time.time() - t0, 1)}
    out.update(conf.resumo())
    out["_confusao"] = conf
    return out


# -------------------------------------------------------------------- orquestracao


def ajustar_batch_automaticamente(modelo, criterio, plano, cfg, max_tentativas: int = 6) -> int:
    """Sobe o batch ate a GPU recusar, e devolve o ultimo que coube.

    A tabela de VRAM em `hardware.py` e conservadora de proposito. Em GPU
    dedicada, este teste empirico costuma render 1.5-2x mais throughput.
    """
    if plano.device != "cuda":
        return plano.batch_size

    dev = torch.device("cuda")
    modelo = modelo.to(dev)
    b, melhor = plano.batch_size, plano.batch_size
    dtype = _amp_dtype(plano.precision)

    for _ in range(max_tentativas):
        try:
            torch.cuda.empty_cache()
            pre = torch.randn(b, 3, cfg.crop, cfg.crop, device=dev)
            post = torch.randn_like(pre)
            alvo = torch.randint(0, cfg.n_classes, (b, cfg.crop, cfg.crop), device=dev)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                perda, _ = criterio(modelo(pre, post), alvo)
            perda.backward()
            modelo.zero_grad(set_to_none=True)
            melhor, b = b, b * 2
        except torch.cuda.OutOfMemoryError:
            break
        finally:
            for v in ("pre", "post", "alvo"):
                if v in dir():
                    del v
            torch.cuda.empty_cache()

    # margem de seguranca: fragmentacao ao longo do treino costuma comer o ultimo passo
    return max(1, int(melhor * 0.8))


def carregar_para_retomar(path: Path, modelo, otim, sched, scaler, ema) -> int:
    """Devolve a epoca inicial. 0 se nao houver checkpoint."""
    if not path.exists():
        return 0
    ck = torch.load(path, map_location="cpu", weights_only=False)
    modelo.load_state_dict(ck["modelo"])
    if otim and ck.get("otim"):
        otim.load_state_dict(ck["otim"])
    if sched and ck.get("sched"):
        sched.load_state_dict(ck["sched"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if ema and ck.get("ema"):
        ema.shadow.load_state_dict(ck["ema"])
    if ck.get("rng_torch") is not None:
        torch.set_rng_state(torch.as_tensor(ck["rng_torch"], dtype=torch.uint8))
    if ck.get("rng_numpy") is not None:
        np.random.set_state(ck["rng_numpy"])
    return int(ck.get("epoca", -1)) + 1
