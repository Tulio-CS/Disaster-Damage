"""Configuracao de treino: dataclass + YAML + identidade determinista do run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

# Ids de classe nas mascaras rasterizadas.
BACKGROUND, NO_DAMAGE, MINOR, MAJOR, DESTROYED, UNCLASSIFIED = range(6)
CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
IGNORE_INDEX = 255  # `un-classified` vira isto na perda: nao e um nivel de severidade


@dataclass
class Config:
    # ---- identidade ----
    nome: str = "baseline"
    task: str = "damage"           # "loc" (so localizacao) | "damage" (multitarefa)
    seed: int = 42
    fold: int = 0                  # indice do fold quando split_mode="grouped"

    # ---- dados ----
    data_dir: str = "data"
    split_mode: str = "oficial"    # "oficial" (tier1+tier3 / hold+test) | "grouped" (por desastre)
    n_folds: int = 5
    crop: int = 512
    usar_tier3: bool = True
    oversample_dano: float = 3.0   # peso relativo das cenas com dano no sampler
    excluir_cenas_vazias: bool = False
    remover_duplicatas: bool = True  # copias socal/woolsey fora do treino (ver xbd/splits.py)
    norm: str = "dataset_treino"   # "dataset_treino" | "dataset" (amostra com hold/test) | "imagenet" | "per_image"

    # ---- modelo ----
    encoder: str = "resnet34"
    pretrained: bool = True
    decoder_ch: tuple[int, ...] = (256, 128, 64, 32, 16)
    siamese: bool = True           # encoder compartilhado entre pre e pos
    fuse_mode: str = "cat_diff"    # "cat_diff" | "cat" | "diff" (ver models.FuseBlock)

    # ---- otimizacao ----
    epochs: int = 30
    effective_batch: int = 16
    lr: float = 3e-4
    lr_encoder_mult: float = 0.1   # encoder pre-treinado aprende mais devagar
    weight_decay: float = 1e-4
    warmup_frac: float = 0.03
    scheduler: str = "cosine"
    clip_grad: float = 1.0
    ema_decay: float = 0.0         # 0 desliga

    # ---- perda ----
    loss_loc: str = "bce_dice"
    loss_dmg: str = "focal_dice"
    focal_gamma: float = 2.0
    peso_loc: float = 1.0
    peso_dmg: float = 1.0
    class_weights: tuple[float, ...] | None = None  # None = derivado da frequencia

    # ---- execucao (preenchido pelo hardware, mas sobrescrevivel) ----
    batch_size: int | None = None
    accum_steps: int | None = None
    num_workers: int | None = None
    precision: str | None = None
    max_steps: int | None = None   # corta a epoca; util para smoke test
    val_every: int = 1
    ckpt_dir: str = "checkpoints"
    results_dir: str = "results"
    resume: bool = True

    # ---- proveniencia (nao entra no hash) ----
    host: str = ""
    notas: str = ""

    _NAO_HASH = ("host", "notas", "batch_size", "accum_steps", "num_workers",
                 "precision", "ckpt_dir", "results_dir", "resume", "data_dir")

    # ------------------------------------------------------------------ util

    @property
    def n_classes(self) -> int:
        return 2 if self.task == "loc" else 5

    def config_hash(self) -> str:
        """Hash dos campos que definem o EXPERIMENTO.

        Campos de execucao (batch por dispositivo, workers, precisao) ficam de
        fora de proposito: duas maquinas rodando a mesma receita produzem o mesmo
        hash, o que permite juntar e comparar os resultados depois.
        """
        d = {f.name: getattr(self, f.name) for f in fields(self)
             if not f.name.startswith("_") and f.name not in self._NAO_HASH}
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    @property
    def run_id(self) -> str:
        return f"{self.nome}-{self.task}-f{self.fold}-s{self.seed}-{self.config_hash()}"

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()
                if not k.startswith("_")}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        validos = {f.name for f in fields(cls) if not f.name.startswith("_")}
        desconhecidos = set(d) - validos
        if desconhecidos:
            raise ValueError(f"campos desconhecidos na config: {sorted(desconhecidos)}")
        cfg = cls(**{k: v for k, v in d.items() if k in validos})
        if isinstance(cfg.decoder_ch, list):
            cfg.decoder_ch = tuple(cfg.decoder_ch)
        if isinstance(cfg.class_weights, list):
            cfg.class_weights = tuple(cfg.class_weights)
        cfg.validar()
        return cfg

    # ------------------------------------------------------------------ validacao

    # Campos categoricos. Sem esta checagem, um typo na matriz de experimentos
    # (`loss_dmg: focal-dice`) cairia no `else` de algum builder e viraria um
    # experimento silenciosamente diferente do que o nome promete — o modo de
    # falha mais caro num estudo de ablacao, porque so aparece na analise.
    _OPCOES = {
        "task": {"loc", "damage"},
        "split_mode": {"oficial", "grouped"},
        "norm": {"dataset_treino", "dataset", "imagenet", "per_image"},
        "fuse_mode": {"cat_diff", "cat", "diff"},
        "scheduler": {"cosine", "constante"},
        "loss_dmg": {"focal_dice", "ce_dice", "focal", "dice"},
        "loss_loc": {"bce_dice", "bce", "nenhuma"},
        "precision": {"bf16", "fp16", "fp32", None},
    }

    def validar(self) -> "Config":
        for campo, permitidos in self._OPCOES.items():
            v = getattr(self, campo)
            if v not in permitidos:
                validos = sorted(x for x in permitidos if x is not None)
                raise ValueError(f"{campo}={v!r} invalido. Use um de: {validos}")
        if self.task == "loc" and self.loss_dmg != "focal_dice":
            raise ValueError("loss_dmg nao tem efeito com task='loc'; deixe o padrao")
        if not 0 <= self.fold < self.n_folds:
            raise ValueError(f"fold={self.fold} fora de [0, {self.n_folds})")
        if self.oversample_dano <= 0:
            raise ValueError("oversample_dano deve ser > 0 (1.0 desliga o reforco)")
        return self


# --------------------------------------------------------------------- presets

PRESETS: dict[str, dict] = {
    # Valida o pipeline inteiro em minutos, em qualquer maquina (inclusive CPU).
    "smoke": dict(nome="smoke", epochs=1, crop=256, encoder="mobilenetv3_small_100",
                  effective_batch=4, max_steps=8, usar_tier3=False, pretrained=False),
    # Roda em GPU modesta (6-12 GB). Ponto de partida recomendado.
    "base": dict(nome="base", epochs=30, crop=512, encoder="resnet34",
                 effective_batch=16, lr=3e-4),
    # Para 16-24 GB de VRAM.
    "forte": dict(nome="forte", epochs=45, crop=640, encoder="seresnext50_32x4d",
                  effective_batch=32, lr=2e-4, ema_decay=0.999),
    # Split honesto: agrupado por desastre, para estimar generalizacao real.
    "grouped": dict(nome="grouped", split_mode="grouped", n_folds=5, epochs=30,
                    crop=512, encoder="resnet34", effective_batch=16),
}


def from_preset(preset: str, **overrides) -> Config:
    """O parametro se chama `preset`, e nao `nome`, porque `nome` e um campo da
    propria Config: `from_preset("base", nome="abl-crop256")` precisa funcionar."""
    if preset not in PRESETS:
        raise KeyError(f"preset '{preset}' nao existe. Disponiveis: {sorted(PRESETS)}")
    return Config.from_dict({**PRESETS[preset], **overrides})
