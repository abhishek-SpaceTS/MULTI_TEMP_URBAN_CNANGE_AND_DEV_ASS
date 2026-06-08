"""
segmentation/model.py
=====================
TE-UNet — Scratch U-Net encoder + TFR at every scale.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F



# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _gn(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = num_groups
    while channels % g != 0:
        g //= 2
    if g < 2:
        raise ValueError(f"Cannot find valid num_groups for channels={channels}")
    return nn.GroupNorm(g, channels)


# ══════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )
    def forward(self, x): return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class GFAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.local    = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _gn(channels), nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ch_mlp   = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.ch_sig  = nn.Sigmoid()
        self.sp_conv = nn.Conv2d(2, 1, spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)
        self.sp_sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out  = self.local(x)
        w_ch = self.ch_sig(self.ch_mlp(self.avg_pool(out)) +
                           self.ch_mlp(self.max_pool(out)))
        out  = out * w_ch
        avg_map = torch.mean(out, dim=1, keepdim=True)
        max_map, _ = torch.max(out, dim=1, keepdim=True)
        w_sp = self.sp_sig(self.sp_conv(torch.cat([avg_map, max_map], dim=1)))
        return out * w_sp + identity


class TFRModule(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4,
                 n_layers: int = 2, max_t: int = 8):
        super().__init__()
        actual_heads = n_heads
        while channels % actual_heads != 0:
            actual_heads //= 2

        self.register_buffer(
            'pos_enc',
            self._sinusoidal(max_t, channels),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=actual_heads,
            dim_feedforward=channels * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,
        )

    @staticmethod
    def _sinusoidal(seq_len: int, d: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d//2])
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).contiguous().reshape(B * H * W, T, C)
        tokens = tokens + self.pos_enc[:, :T, :]
        tokens = self.encoder(tokens)
        out = tokens.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return out


class TemporalAttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1, bias=True)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = ctx.shape
        scores  = self.score(ctx.mean(dim=(-2,-1))).squeeze(-1)
        weights = torch.softmax(scores, dim=1).view(B, T, 1, 1, 1)
        return (ctx * weights).sum(dim=1)


class TemporalSkipContext(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, first: torch.Tensor, last: torch.Tensor):
        ctx = self.gate(torch.cat([self.pool(first), self.pool(last)], dim=1))
        return first * ctx, last * ctx


class TemporalFuse(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            _gn(channels), nn.ReLU(inplace=True),
        )
    def forward(self, first, last, pooled):
        return self.proj(torch.cat(
            [first, last, pooled, torch.abs(last - first)], dim=1
        ))


class AuxHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            _gn(in_ch // 2), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, 1),
        )
    def forward(self, x, target_size):
        return F.interpolate(self.head(x), size=target_size,
                             mode="bilinear", align_corners=False)


# ══════════════════════════════════════════════════════════════
# MAIN MODEL
# ══════════════════════════════════════════════════════════════

class TE_UNet(nn.Module):
    """
    TE-UNet v4 — Scratch encoder + TFR at every scale.

    Args:
        in_ch_per_frame : RGB channels per month (3)
        n_months        : timesteps (5)
        fbc_ch          : 3 = footprint+boundary+contact
        topology        : encoder channel widths [32,64,128,256,512]
        n_heads         : TFR attention heads
        n_layers        : TFR encoder layers per scale
        deep_supervision: aux heads at 1/8 resolution (training only)

    Outputs (dict):
        "fbc"        : [B, n_months*fbc_ch, H, W]
        "change"     : [B, 1, H, W]
        "aux_fbc"    : [B, n_months*fbc_ch, H, W]  (training only)
        "aux_change" : [B, 1, H, W]                 (training only)
    """

    def __init__(
        self,
        in_ch_per_frame:  int   = 3,
        n_months:         int   = 5,
        fbc_ch:           int   = 3,
        base:             int   = 32,
        topology:         list  = None,
        n_heads:          int   = 4,
        n_layers:         int   = 2,
        d_model:          int   = 256,
        encoder_weights:  str   = None,
        freeze_encoder:   bool  = False,
        deep_supervision: bool  = True,
    ):
        super().__init__()
        self.in_ch_per_frame  = in_ch_per_frame
        self.n_months         = n_months
        self.fbc_ch           = fbc_ch
        self.deep_supervision = deep_supervision

        if topology is None:
            topology = [32, 64, 128, 256, 512]
        t0, t1, t2, t3, t4 = topology
        self.topology = topology

        # ── Scratch encoder ───────────────────────────────────
        self.inc   = DoubleConv(in_ch_per_frame, t0)
        self.down1 = Down(t0, t1)
        self.down2 = Down(t1, t2)
        self.down3 = Down(t2, t3)
        self.down4 = Down(t3, t4)

        # ── TFR at 16×16, 32×32, 64×64 ───────────────────────
        # max_t tied to n_months so pos_enc covers exactly the sequence length
        self.tfr_b  = TFRModule(t4, n_heads=n_heads, n_layers=n_layers, max_t=n_months)
        self.tfr_s3 = TFRModule(t3, n_heads=n_heads, n_layers=n_layers, max_t=n_months)
        self.tfr_s2 = TFRModule(t2, n_heads=n_heads, n_layers=n_layers, max_t=n_months)

        # ── GFAM at 16×16, 32×32, and 64×64 ─────────────────
        self.gfam_b  = GFAM(t4)
        self.gfam_s3 = GFAM(t3)
        self.gfam_s2 = GFAM(t2)

        # ── Temporal attention pool — all 5 scales ────────────
        self.ctx_pool_b  = TemporalAttentionPool(t4)
        self.ctx_pool_s3 = TemporalAttentionPool(t3)
        self.ctx_pool_s2 = TemporalAttentionPool(t2)
        self.ctx_pool_s1 = TemporalAttentionPool(t1)
        self.ctx_pool_s0 = TemporalAttentionPool(t0)

        # ── TemporalSkipContext — all 5 scales ────────────────
        self.ctx_b  = TemporalSkipContext(t4)
        self.ctx_s3 = TemporalSkipContext(t3)
        self.ctx_s2 = TemporalSkipContext(t2)
        self.ctx_s1 = TemporalSkipContext(t1)
        self.ctx_s0 = TemporalSkipContext(t0)

        # ── TemporalFuse — all 5 scales ───────────────────────
        self.fuse_b  = TemporalFuse(t4)
        self.fuse_s3 = TemporalFuse(t3)
        self.fuse_s2 = TemporalFuse(t2)
        self.fuse_s1 = TemporalFuse(t1)
        self.fuse_s0 = TemporalFuse(t0)

        # ── FBC decoder ───────────────────────────────────────
        self.fbc_up4 = Up(t4, t3, t3)
        self.fbc_up3 = Up(t3, t2, t2)
        self.fbc_up2 = Up(t2, t1, t1)
        self.fbc_up1 = Up(t1, t0, t0)
        self.fbc_out = nn.Conv2d(t0, fbc_ch, 1)

        # ── Change decoder ────────────────────────────────────
        self.chg_up4 = Up(t4, t3, t3)
        self.chg_up3 = Up(t3, t2, t2)
        self.chg_up2 = Up(t2, t1, t1)
        self.chg_up1 = Up(t1, t0, t0)
        self.chg_out = nn.Conv2d(t0, 1, 1)

        # ── Deep supervision aux heads ────────────────────────
        if deep_supervision:
            self.aux_fbc_head = AuxHead(t3, fbc_ch)
            self.aux_chg_head = AuxHead(t3, 1)

    # ── Forward ───────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, T*in_ch_per_frame, H, W]  e.g. [2, 15, 256, 256]"""
        B, C, H, W = x.shape
        T, cpf = self.n_months, self.in_ch_per_frame
        assert C == T * cpf, f"expected {T*cpf} channels, got {C}"

        # Fold T into batch
        x_bt = x.reshape(B, T, cpf, H, W).reshape(B * T, cpf, H, W)

        # ── Scratch encoder ───────────────────────────────────
        s0 = self.inc(x_bt)    # [B*T, t0=32,  256, 256]
        s1 = self.down1(s0)    # [B*T, t1=64,  128, 128]
        s2 = self.down2(s1)    # [B*T, t2=128,  64,  64]
        s3 = self.down3(s2)    # [B*T, t3=256,  32,  32]
        b  = self.down4(s3)    # [B*T, t4=512,  16,  16]

        def seq(t): return t.reshape(B, T, *t.shape[1:])
        b_seq  = seq(b)
        s3_seq = seq(s3)
        s2_seq = seq(s2)
        s1_seq = seq(s1)
        s0_seq = seq(s0)

        # ── TFR at 16×16, 32×32, 64×64 ───────────────────────
        b_seq  = self.tfr_b(b_seq)
        s3_seq = self.tfr_s3(s3_seq)
        s2_seq = self.tfr_s2(s2_seq)

        # ── GFAM on bottleneck, s3, and s2 ───────────────────
        b_refined  = self.gfam_b( b_seq.reshape(B*T, *b_seq.shape[2:]))
        s3_refined = self.gfam_s3(s3_seq.reshape(B*T, *s3_seq.shape[2:]))
        s2_refined = self.gfam_s2(s2_seq.reshape(B*T, *s2_seq.shape[2:]))
        b_seq  = b_refined.reshape(B, T, *b_refined.shape[1:])
        s3_seq = s3_refined.reshape(B, T, *s3_refined.shape[1:])
        s2_seq = s2_refined.reshape(B, T, *s2_refined.shape[1:])

        # ── FBC branch ────────────────────────────────────────
        b_bt  = b_seq.reshape(B*T,  *b_seq.shape[2:])
        s3_bt = s3_seq.reshape(B*T, *s3_seq.shape[2:])
        s2_bt = s2_seq.reshape(B*T, *s2_seq.shape[2:])
        s1_bt = s1_seq.reshape(B*T, *s1_seq.shape[2:])
        s0_bt = s0_seq.reshape(B*T, *s0_seq.shape[2:])

        d      = self.fbc_up4(b_bt,  s3_bt)
        d_aux  = d
        d      = self.fbc_up3(d,     s2_bt)
        d      = self.fbc_up2(d,     s1_bt)
        d      = self.fbc_up1(d,     s0_bt)
        fbc_bt = self.fbc_out(d)
        fbc    = fbc_bt.reshape(B, T, self.fbc_ch, H, W).reshape(
            B, T * self.fbc_ch, H, W
        )

        # ── Change branch ─────────────────────────────────────
        ctx_pooled_b  = self.ctx_pool_b(b_seq)
        ctx_pooled_s3 = self.ctx_pool_s3(s3_seq)
        ctx_pooled_s2 = self.ctx_pool_s2(s2_seq)
        ctx_pooled_s1 = self.ctx_pool_s1(s1_seq)
        ctx_pooled_s0 = self.ctx_pool_s0(s0_seq)

        b_f,  b_l  = self.ctx_b( b_seq[:,0],  b_seq[:,-1])
        s3_f, s3_l = self.ctx_s3(s3_seq[:,0], s3_seq[:,-1])
        s2_f, s2_l = self.ctx_s2(s2_seq[:,0], s2_seq[:,-1])
        s1_f, s1_l = self.ctx_s1(s1_seq[:,0], s1_seq[:,-1])
        s0_f, s0_l = self.ctx_s0(s0_seq[:,0], s0_seq[:,-1])

        fused_b = self.fuse_b( b_f,  b_l,  ctx_pooled_b)
        s3d     = self.fuse_s3(s3_f, s3_l, ctx_pooled_s3)
        s2d     = self.fuse_s2(s2_f, s2_l, ctx_pooled_s2)
        s1d     = self.fuse_s1(s1_f, s1_l, ctx_pooled_s1)
        s0d     = self.fuse_s0(s0_f, s0_l, ctx_pooled_s0)

        c      = self.chg_up4(fused_b, s3d)
        c_aux  = c
        c      = self.chg_up3(c,       s2d)
        c      = self.chg_up2(c,       s1d)
        c      = self.chg_up1(c,       s0d)
        change = self.chg_out(c)

        out = {"fbc": fbc, "change": change}

        # ── Deep supervision (training only) ──────────────────
        if self.deep_supervision and self.training:
            aux_fbc_bt = self.aux_fbc_head(d_aux, (H, W))
            out["aux_fbc"] = aux_fbc_bt.reshape(
                B, T, self.fbc_ch, H, W
            ).reshape(B, T * self.fbc_ch, H, W)
            out["aux_change"] = self.aux_chg_head(c_aux, (H, W))

        return out