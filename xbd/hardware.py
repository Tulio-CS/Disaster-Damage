"""
Sonda a maquina e deriva um plano de treino compativel com ela.

O objetivo nao e so "caber na memoria": e fazer com que a MESMA config rode em
maquinas diferentes produzindo resultados COMPARAVEIS. Por isso o que se mantem
fixo entre maquinas e o **batch efetivo** (`batch_size x accum_steps x n_gpus`),
e o que varia e como cada maquina chega la. Uma GPU de 24 GB usa batch 16 e
accum 1; o notebook sem CUDA usa batch 1 e accum 16. O gradiente medio e o
mesmo, entao as curvas de treino sao comparaveis e os checkpoints, combinaveis.

Uso:
    from xbd.hardware import probe, plan
    hw = probe()
    p = plan(hw, crop=512, encoder="resnet34", effective_batch=16)
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

# VRAM livre (GB) -> batch base para crop 512, encoder ~resnet34, AMP ligado.
# Deliberadamente conservador: use `--autotune` para subir empiricamente.
_VRAM_BATCH = ((3.5, 1), (6.0, 2), (10.0, 4), (16.0, 8), (24.0, 12), (40.0, 16), (10**9, 24))

# Custo relativo de memoria por encoder, medido contra resnet34 = 1.0.
_ENCODER_COST = {
    "mobilenetv3_small_100": 0.35,
    "mobilenetv3_large_100": 0.5,
    "efficientnet_b0": 0.7,
    "resnet18": 0.75,
    "resnet34": 1.0,
    "resnet50": 1.6,
    "seresnext50_32x4d": 2.0,
    "convnext_tiny": 2.2,
    "tf_efficientnet_b4": 2.6,
}


@dataclass
class GPU:
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: tuple[int, int] | None = None

    @property
    def supports_bf16(self) -> bool:
        """bf16 exige Ampere (SM 8.0) ou mais novo; abaixo disso, fp16."""
        return self.compute_capability is not None and self.compute_capability[0] >= 8


@dataclass
class Hardware:
    hostname: str
    platform: str
    cpu_model: str
    cpu_logical: int
    cpu_physical: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpus: list[GPU] = field(default_factory=list)
    torch_version: str | None = None
    cuda_version: str | None = None

    @property
    def has_cuda(self) -> bool:
        return len(self.gpus) > 0

    @property
    def vram_free_gb(self) -> float:
        return min((g.vram_free_gb for g in self.gpus), default=0.0)

    def summary(self) -> str:
        linhas = [
            f"host        {self.hostname}  ({self.platform})",
            f"cpu         {self.cpu_model}  —  {self.cpu_physical} nucleos / {self.cpu_logical} threads",
            f"ram         {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB livres",
            f"disco       {self.disk_free_gb:.1f} GB livres",
            f"torch       {self.torch_version or 'ausente'}  cuda={self.cuda_version or 'nao'}",
        ]
        if self.gpus:
            for g in self.gpus:
                cc = f"sm_{g.compute_capability[0]}{g.compute_capability[1]}" if g.compute_capability else "?"
                linhas.append(f"gpu {g.index}       {g.name}  {g.vram_total_gb:.1f} GB "
                              f"({g.vram_free_gb:.1f} livres, {cc})")
        else:
            linhas.append("gpu         nenhuma GPU CUDA — o treino rodara em CPU")
        return "\n".join(linhas)


@dataclass
class Plan:
    """Parametros de execucao derivados do hardware."""
    device: str
    batch_size: int
    accum_steps: int
    effective_batch: int
    num_workers: int
    precision: str            # "bf16" | "fp16" | "fp32"
    crop: int
    encoder: str
    channels_last: bool
    compile_model: bool
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    grad_checkpointing: bool
    n_gpus: int
    notas: list[str] = field(default_factory=list)

    def summary(self) -> str:
        linhas = [
            f"device            {self.device}  ({self.n_gpus} GPU)" if self.n_gpus
            else f"device            {self.device}",
            f"encoder           {self.encoder}",
            f"crop              {self.crop}",
            f"batch             {self.batch_size} x accum {self.accum_steps}"
            f" x {max(self.n_gpus,1)} gpu = efetivo {self.effective_batch}",
            f"precisao          {self.precision}",
            f"dataloader        {self.num_workers} workers"
            f"{', persistentes' if self.persistent_workers else ''}"
            f"{f', prefetch {self.prefetch_factor}' if self.prefetch_factor else ''}",
            f"channels_last     {self.channels_last}",
            f"torch.compile     {self.compile_model}",
            f"grad checkpoint   {self.grad_checkpointing}",
        ]
        return "\n".join(linhas) + ("".join(f"\n  ! {n}" for n in self.notas) if self.notas else "")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- sonda


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for linha in fh:
                if linha.startswith("model name"):
                    return linha.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconhecido"


def _nvidia_smi() -> list[GPU]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    gpus = []
    for linha in out.strip().splitlines():
        partes = [p.strip() for p in linha.split(",")]
        if len(partes) < 4:
            continue
        cc = None
        if len(partes) >= 5 and "." in partes[4]:
            maior, menor = partes[4].split(".")[:2]
            cc = (int(maior), int(menor))
        gpus.append(GPU(index=int(partes[0]), name=partes[1],
                        vram_total_gb=float(partes[2]) / 1024,
                        vram_free_gb=float(partes[3]) / 1024,
                        compute_capability=cc))
    return gpus


def probe(disk_path: str | None = None) -> Hardware:
    """Mede a maquina atual. Nao importa torch se ele nao estiver instalado."""
    try:
        import psutil
        ram_total = psutil.virtual_memory().total / 1e9
        ram_avail = psutil.virtual_memory().available / 1e9
        cpu_phys = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        ram_total = ram_avail = 0.0
        cpu_phys = os.cpu_count() or 1

    torch_v = cuda_v = None
    gpus = _nvidia_smi()
    try:
        import torch
        torch_v = torch.__version__
        if torch.cuda.is_available():
            cuda_v = torch.version.cuda
            # torch e a fonte mais confiavel para capability; nvidia-smi para VRAM livre
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                livre, total = torch.cuda.mem_get_info(i)
                achado = next((g for g in gpus if g.index == i), None)
                if achado:
                    achado.compute_capability = (props.major, props.minor)
                    achado.vram_free_gb = livre / 1e9
                else:
                    gpus.append(GPU(i, props.name, total / 1e9, livre / 1e9,
                                    (props.major, props.minor)))
        else:
            gpus = []  # nvidia-smi pode ver a GPU sem o torch conseguir usa-la
    except ImportError:
        pass

    du = shutil.disk_usage(disk_path or os.getcwd())
    return Hardware(
        hostname=platform.node(),
        platform=f"{platform.system()} {platform.release()}",
        cpu_model=_cpu_model(),
        cpu_logical=os.cpu_count() or 1,
        cpu_physical=cpu_phys,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        disk_free_gb=du.free / 1e9,
        gpus=gpus,
        torch_version=torch_v,
        cuda_version=cuda_v,
    )


# --------------------------------------------------------------------------- plano


def _batch_base(vram_gb: float) -> int:
    for limite, b in _VRAM_BATCH:
        if vram_gb < limite:
            return b
    return _VRAM_BATCH[-1][1]


def plan(hw: Hardware, *, crop: int = 512, encoder: str = "resnet34",
         effective_batch: int = 16, batch_size: int | None = None,
         num_workers: int | None = None, precision: str | None = None) -> Plan:
    """Deriva os parametros de execucao. Argumentos explicitos vencem a heuristica."""
    notas: list[str] = []
    n_gpus = len(hw.gpus)

    # ---- batch por dispositivo ----
    if batch_size is None:
        if hw.has_cuda:
            b = _batch_base(hw.vram_free_gb)
            # a memoria de ativacao cresce com a area do crop e com o encoder
            b = b / max(_ENCODER_COST.get(encoder, 1.0), 0.1) / ((crop / 512) ** 2)
            batch_size = max(1, int(b))
        else:
            batch_size = 1
            notas.append("sem CUDA: batch 1 em CPU. Use apenas para smoke test.")
    batch_size = max(1, batch_size)

    # ---- accum para fechar o batch efetivo ----
    # Se a maquina couber MAIS que o batch efetivo pedido, encolhe o batch por
    # dispositivo em vez de estourar o alvo: manter o efetivo fixo e o que torna
    # os runs de maquinas diferentes comparaveis entre si.
    if batch_size * max(n_gpus, 1) > effective_batch:
        batch_size = max(1, effective_batch // max(n_gpus, 1))
    por_passo = batch_size * max(n_gpus, 1)
    accum = max(1, math.ceil(effective_batch / por_passo))
    efetivo_real = por_passo * accum
    if efetivo_real != effective_batch:
        notas.append(f"batch efetivo ajustado para {efetivo_real}: "
                     f"{effective_batch} nao e divisivel por {por_passo} por passo")

    # ---- workers: limitados pela RAM, nao pela CPU ----
    if num_workers is None:
        por_cpu = max(0, hw.cpu_logical - 2)
        # cada worker segura batch de imagens 1024x1024 descomprimidas: ~1.5 GB de folga
        por_ram = int(max(0, hw.ram_available_gb - 2.0) / 1.5) if hw.ram_available_gb else 4
        # No Windows o multiprocessing usa spawn, nao fork: cada worker sobe um
        # interpretador novo e reimporta torch (~3 s cada). Com 8 workers isso
        # custa meio minuto por epoca so de partida, e `persistent_workers` nao
        # cobre a validacao. 4 e o ponto em que o custo para de compensar.
        teto = 4 if platform.system() == "Windows" else 8
        num_workers = max(0, min(por_cpu, por_ram, teto))
        if por_ram < por_cpu:
            notas.append(f"workers limitados pela RAM livre ({hw.ram_available_gb:.1f} GB), nao pela CPU")
        elif teto < por_cpu:
            notas.append(f"workers limitados a {teto}: spawn do Windows encarece cada worker")

    # ---- precisao ----
    if precision is None:
        if not hw.has_cuda:
            precision = "fp32"
        elif all(g.supports_bf16 for g in hw.gpus):
            precision = "bf16"
        else:
            precision = "fp16"
            notas.append("GPU pre-Ampere: fp16 com GradScaler (bf16 indisponivel)")

    # ---- avisos de viabilidade ----
    grad_ckpt = False
    if hw.has_cuda and hw.vram_free_gb < 6:
        grad_ckpt = True
        notas.append(f"VRAM baixa ({hw.vram_free_gb:.1f} GB): gradient checkpointing ligado")
    if hw.ram_total_gb and hw.ram_total_gb < 8:
        notas.append(f"RAM total de {hw.ram_total_gb:.1f} GB: prefira --preset smoke nesta maquina")

    torch_ok = hw.torch_version is not None and int(hw.torch_version.split(".")[0]) >= 2

    return Plan(
        device="cuda" if hw.has_cuda else "cpu",
        batch_size=batch_size,
        accum_steps=accum,
        effective_batch=efetivo_real,
        num_workers=num_workers,
        precision=precision,
        crop=crop,
        encoder=encoder,
        channels_last=hw.has_cuda,
        compile_model=bool(torch_ok and hw.has_cuda),
        pin_memory=hw.has_cuda,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        grad_checkpointing=grad_ckpt,
        n_gpus=n_gpus,
        notas=notas,
    )
