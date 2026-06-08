"""
segmentation/config.py
======================

CHANGES FOR V4:
  V4: scratch encoder, topology=[32,64,128,256,512], bottleneck at 16x16
  TFR at every scale, GFAM at 16x16+32x32, input_conv 256x256 skip
  n_heads=4 (was 8), n_layers=2 (was 3) — simpler transformer for scratch
  encoder_weights=None (no pretrained), topology added to model_kwargs
"""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class Config:
    # ── Data / loader ────────────────────────────────────────────
    n_months: int          = 5
    batch_size: int        = 8
    num_workers: int       = 2
    pin_memory: bool       = True
    samples_per_epoch: int = 25_000

    # ── Model ────────────────────────────────────────────────────
    in_ch_per_frame: int  = 3
    fbc_ch: int           = 1           # CHANGED: footprint+boundary+contact
    topology: list        = None          # set in __post_init__
    n_heads: int          = 4             # v4: 4 heads (was 8 for EfficientNet)
    n_layers: int         = 2             # v4: 2 layers (was 3)
    d_model: int          = 256           # kept for compat, unused in v4
    encoder_weights: str  = None          # v4: scratch encoder, no pretrained
    deep_supervision: bool = True         # aux heads during training

    # ── Optimisation ─────────────────────────────────────────────
    epochs: int         = 100
    lr: float           = 1e-4
    weight_decay: float = 1e-3
    warmup_epochs: int  = 5
    grad_clip: float    = 0.5
    amp: bool           = False          # keep False on WSL2
    seed: int           = 42

    # ── Loss weights ─────────────────────────────────────────────
    # w_fbc stays CONSTANT — do not decay it
    w_fbc: float        = 1.0
    w_change: float     = 1.0           # conservative — stable with NaN guard
    change_pos_w: float = 7.0          # data-driven from registry stats

    # ── Evaluation / IO ──────────────────────────────────────────
    threshold: float = 0.5
    out_dir: str     = "checkpoints"

    def __post_init__(self):
        if self.topology is None:
            self.topology = [32, 64, 128, 256, 512]

    def model_kwargs(self) -> Dict:
        return {
            "in_ch_per_frame":  self.in_ch_per_frame,
            "n_months":         self.n_months,
            "fbc_ch":           self.fbc_ch,
            "base":             self.base,
            "topology":         self.topology,
            "n_heads":          self.n_heads,
            "n_layers":         self.n_layers,
            "d_model":          self.d_model,           # kept for compat
            "encoder_weights":  self.encoder_weights,   # None for v4
            "deep_supervision": self.deep_supervision,
        }

    def as_dict(self) -> Dict:
        return asdict(self)


cfg = Config()

if __name__ == "__main__":
    import json
    print(json.dumps(cfg.as_dict(), indent=2))