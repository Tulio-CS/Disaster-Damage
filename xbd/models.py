"""
U-Net siamesa com encoder do timm.

Desenho vindo direto dos achados da EDA:

- **Encoder compartilhado entre pre e pos.** Os poligonos sao identicos nas duas
  datas, entao a rede nao precisa reaprender a localizar; o que ela precisa e
  comparar. Pesos compartilhados forcam as duas imagens para o mesmo espaco de
  features e cortam quase metade dos parametros.
- **Fusao por concatenacao + diferenca absoluta** em cada escala. A diferenca
  crua entre imagens e ruim (gap temporal mediano de 304 dias), mas a diferenca
  no espaco de FEATURES e o sinal que interessa.
- **Skips em stride 2 e 4.** 71% das edificacoes sao menores que 32x32 px; na
  escala mais grossa (stride 32) a edificacao mediana ocupa menos de um pixel.
- **Uma cabeca de 5 classes, nao duas.** Localizacao sai como `1 - p(fundo)`,
  entao as duas tarefas dividem todo o decoder e a perda de localizacao ainda
  supervisiona a mesma saida.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, k: int = 3):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )


class FuseBlock(nn.Module):
    """Funde features pre e pos numa escala, e reduz de volta para `ch` canais.

    `mode` existe para a ablacao da hipotese central do modelo — que o sinal de
    dano mora na DIFERENCA em espaco de features, nao nas duas imagens lado a
    lado:

        cat_diff  [pre | pos | |pos-pre|]   padrao
        cat       [pre | pos]               a rede teria que aprender a diferenca
        diff      [|pos-pre|]               so a diferenca, sem a aparencia crua
    """

    N_ENTRADAS = {"cat_diff": 3, "cat": 2, "diff": 1}

    def __init__(self, ch: int, mode: str = "cat_diff"):
        super().__init__()
        if mode not in self.N_ENTRADAS:
            raise ValueError(f"fuse_mode desconhecido: {mode}")
        self.mode = mode
        self.reduce = nn.Sequential(
            nn.Conv2d(ch * self.N_ENTRADAS[mode], ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, f_pre: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
        if self.mode == "cat":
            x = torch.cat([f_pre, f_post], dim=1)
        elif self.mode == "diff":
            x = (f_post - f_pre).abs()
        else:
            x = torch.cat([f_pre, f_post, (f_post - f_pre).abs()], dim=1)
        return self.reduce(x)


class DecoderBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv1 = ConvBNAct(c_in + c_skip, c_out)
        self.conv2 = ConvBNAct(c_out, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SiameseUNet(nn.Module):
    """Recebe (pre, post) e devolve logits (B, n_classes, H, W) no tamanho da entrada."""

    def __init__(self, encoder: str = "resnet34", n_classes: int = 5,
                 pretrained: bool = True, siamese: bool = True,
                 decoder_ch: tuple[int, ...] = (256, 128, 64, 32, 16),
                 grad_checkpointing: bool = False, fuse_mode: str = "cat_diff"):
        super().__init__()
        import timm

        self.siamese = siamese
        self.n_classes = n_classes
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True,
                                         in_chans=3)
        if grad_checkpointing and hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(True)

        enc_ch = list(self.encoder.feature_info.channels())   # do mais fino ao mais grosso
        self.reducoes = self.encoder.feature_info.reduction()

        self.fuse = (nn.ModuleList([FuseBlock(c, fuse_mode) for c in enc_ch])
                     if siamese else None)

        # caminho de subida: comeca no mapa mais grosso e sobe usando os skips
        dec = list(decoder_ch)[: len(enc_ch)]
        blocos = []
        c_in = enc_ch[-1]
        for i, c_out in enumerate(dec):
            c_skip = enc_ch[-(i + 2)] if i + 2 <= len(enc_ch) else 0
            blocos.append(DecoderBlock(c_in, c_skip, c_out))
            c_in = c_out
        self.decoder = nn.ModuleList(blocos)
        self.head = nn.Conv2d(c_in, n_classes, 1)

    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)

    def forward(self, pre: torch.Tensor, post: torch.Tensor | None = None) -> torch.Tensor:
        h, w = pre.shape[-2:]
        f_pre = self._encode(pre)

        if self.siamese:
            if post is None:
                raise ValueError("modelo siames exige a imagem pos")
            f_post = self._encode(post)
            feats = [fu(a, b) for fu, a, b in zip(self.fuse, f_pre, f_post)]
        else:
            feats = f_pre

        x = feats[-1]
        for i, bloco in enumerate(self.decoder):
            idx = len(feats) - (i + 2)
            x = bloco(x, feats[idx] if idx >= 0 else None)

        logits = self.head(x)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits

    # ------------------------------------------------------------------

    def grupos_de_parametros(self, lr: float, lr_encoder_mult: float, weight_decay: float):
        """Encoder pre-treinado com lr menor; sem weight decay em norm/bias."""
        enc_decay, enc_no, dec_decay, dec_no = [], [], [], []
        for nome, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_enc = nome.startswith("encoder.")
            sem_decay = p.ndim <= 1  # bias e parametros de normalizacao
            alvo = (enc_no if sem_decay else enc_decay) if is_enc else (dec_no if sem_decay else dec_decay)
            alvo.append(p)
        return [
            {"params": enc_decay, "lr": lr * lr_encoder_mult, "weight_decay": weight_decay},
            {"params": enc_no, "lr": lr * lr_encoder_mult, "weight_decay": 0.0},
            {"params": dec_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": dec_no, "lr": lr, "weight_decay": 0.0},
        ]


def build_model(cfg, grad_checkpointing: bool = False) -> SiameseUNet:
    return SiameseUNet(
        encoder=cfg.encoder,
        n_classes=cfg.n_classes,
        pretrained=cfg.pretrained,
        siamese=(cfg.task == "damage" and cfg.siamese),
        decoder_ch=cfg.decoder_ch,
        grad_checkpointing=grad_checkpointing,
        fuse_mode=cfg.fuse_mode,
    )
