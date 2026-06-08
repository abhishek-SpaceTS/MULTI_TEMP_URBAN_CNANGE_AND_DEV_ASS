"""
segmentation/loss.py
====================

Loss for TE-UNet v2 with fbc_ch=3 boundary supervision + deep supervision.

CHANGES FROM v1:
  1. WeightedFBCLoss replaces plain DiceBCELoss for FBC head.
     Boundary (ch1) and contact (ch2) are weighted 3x higher than footprint (ch0).
     Boundary/contact pixels are ~5-10% of footprint pixels — without extra
     weight the model ignores them (class imbalance within the FBC task).

  2. MultiTaskLoss handles optional aux_fbc / aux_change from deep supervision.
     Aux losses use weight 0.4 (less than main head weight 1.0) — they guide
     gradient flow but the main head is still the primary learning signal.
     Aux outputs are only present during training (model.training=True).

  3. FBC pos_weight now applied per-channel:
       footprint: 4.1  (from building coverage 19.7%)
       boundary:  15.0 (boundary pixels ~5% of all pixels → 95/5 ≈ 19 → use 15)
       contact:   20.0 (contact pixels even rarer)

VISUALIZATION:
  At inference use only fbc[:, ::fbc_ch] (footprint channels).
  Boundary/contact logits are never shown to the user.
  white = building, black = background.

ALL INPUTS ARE LOGITS — do NOT sigmoid before passing in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Core losses ──────────────────────────────────────────────────────────────

def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss. [B,C,H,W] targets in {0,1}. Per (B,C) then mean."""
    probs   = torch.sigmoid(logits)
    probs   = probs.reshape(probs.shape[0],   probs.shape[1],   -1)
    targets = targets.reshape(targets.shape[0], targets.shape[1], -1)
    inter   = (probs * targets).sum(-1)
    union   = probs.sum(-1) + targets.sum(-1)
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


class DiceBCELoss(nn.Module):
    """Dice + BCE for one task with optional pos_weight."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight_val = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight_val is not None:
            pw  = torch.as_tensor(self.pos_weight_val,
                                  device=logits.device, dtype=logits.dtype)
            bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice_loss(logits, targets)


class WeightedFBCLoss(nn.Module):
    """
    Channel-weighted Dice+BCE for the 3-channel FBC head.

    Applies different pos_weight AND channel weight per FBC channel type:
      footprint (every fbc_ch-th channel starting at 0): weight=1.0, pos_weight=4.1
      boundary  (every fbc_ch-th channel starting at 1): weight=3.0, pos_weight=15.0
      contact   (every fbc_ch-th channel starting at 2): weight=3.0, pos_weight=20.0

    WHY HIGHER WEIGHT FOR BOUNDARY/CONTACT:
      Boundary pixels are ~5% of building pixels.
      Contact pixels are even rarer.
      Without 3x weight, the model ignores them (tiny contribution to loss).
      With 3x weight, getting boundary/contact right matters as much as footprint.

    With fbc_ch=1: falls back to plain DiceBCELoss (footprint only).
    With fbc_ch=3: applies per-channel weighting across all months.
    """

    def __init__(self, fbc_ch: int = 3, n_months: int = 5):
        super().__init__()
        self.fbc_ch   = fbc_ch
        self.n_months = n_months

        if fbc_ch == 3:
            # Per-channel criteria: footprint, boundary, contact
            #
            # FBC masks use BOUNDARY_EROSION_PX=1, CONTACT_DILATE_PX=1
            # producing very thin 1px rings (~1.2m wide).
            # Compensate with higher weights so model is FORCED to learn them:
            #   channel_weight: 3→5  (boundary/contact 5x more important)
            #   pos_weight:    15→30 for boundary (1px ring = very few pixels)
            #   pos_weight:    20→40 for contact  (rarest channel)
            #
            # If FBC masks are regenerated with BOUNDARY_EROSION_PX=3,
            # reduce back to: weight=3.0, pos_weight=15.0/20.0
            self.criteria = nn.ModuleList([
                DiceBCELoss(bce_weight=0.5, dice_weight=0.5, pos_weight=4.1),   # footprint
                DiceBCELoss(bce_weight=0.5, dice_weight=0.5, pos_weight=10.0),  # boundary ↑
                DiceBCELoss(bce_weight=0.5, dice_weight=0.5, pos_weight=10.0),  # contact  ↑
            ])
            self.channel_weights = [1.0, 3.0, 3.0]
        else:
            # fbc_ch=1 fallback
            self.criteria = nn.ModuleList([
                DiceBCELoss(bce_weight=0.5, dice_weight=0.5, pos_weight=4.1),
            ])
            self.channel_weights = [1.0]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits/targets: [B, N_MONTHS * fbc_ch, H, W]
        Layout: [fp_m0, bd_m0, ct_m0, fp_m1, bd_m1, ct_m1, ...]
        """
        total = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        weight_sum = sum(self.channel_weights)

        for ch_offset, (criterion, w) in enumerate(
            zip(self.criteria, self.channel_weights)
        ):
            # Select this channel type across all months
            # e.g. footprint (ch_offset=0): indices [0, 3, 6, 9, 12]
            idx    = list(range(ch_offset, logits.shape[1], self.fbc_ch))
            total  = total + w * criterion(logits[:, idx], targets[:, idx])

        return total / weight_sum


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for TE-UNet v2.

    total = w_fbc * fbc_loss(fbc, fbc_target)
          + w_change * change_loss(change, change_target)
          + 0.4 * w_fbc * aux_fbc_loss    (if deep supervision)
          + 0.4 * w_change * aux_chg_loss  (if deep supervision)

    Aux loss weight 0.4: guides gradient flow without dominating main head.
    Aux outputs only present during training (model.training=True).

    Args:
        fbc_ch        : 3 (footprint+boundary+contact) or 1 (footprint only)
        n_months      : number of months per window (5)
        w_fbc         : weight on FBC task (constant — do not decay)
        w_change      : weight on change task
        change_pos_w  : pos_weight for change BCE
        aux_weight    : weight multiplier for auxiliary deep supervision losses
    """

    def __init__(
        self,
        fbc_ch:       int   = 3,
        n_months:     int   = 5,
        w_fbc:        float = 1.0,
        w_change:     float = 1.0,
        change_pos_w: float = 11.0,
        aux_weight:   float = 0.4,
    ):
        super().__init__()
        self.w_fbc      = w_fbc
        self.w_change   = w_change
        self.aux_weight = aux_weight

        self.fbc_loss    = WeightedFBCLoss(fbc_ch=fbc_ch, n_months=n_months)
        self.change_loss = DiceBCELoss(bce_weight=0.5, dice_weight=0.5,
                                       pos_weight=change_pos_w)

    def forward(
        self,
        outputs:       dict,
        fbc_target:    torch.Tensor,
        change_target: torch.Tensor,
    ):
        l_fbc    = self.fbc_loss(outputs["fbc"], fbc_target)
        l_change = self.change_loss(outputs["change"], change_target)
        total    = self.w_fbc * l_fbc + self.w_change * l_change

        breakdown = {
            "total":  total.item(),
            "fbc":    l_fbc.item(),
            "change": l_change.item(),
        }

        # ── Deep supervision aux losses ───────────────────────
        # Only present during training when model outputs aux heads
        if "aux_fbc" in outputs:
            # Aux FBC target: footprint channels only (ch0 of each month)
            # since aux head predicts fbc_ch channels but at coarser scale
            l_aux_fbc = self.fbc_loss(outputs["aux_fbc"], fbc_target)
            total     = total + self.aux_weight * self.w_fbc * l_aux_fbc
            breakdown["aux_fbc"] = l_aux_fbc.item()

        if "aux_change" in outputs:
            l_aux_chg = self.change_loss(outputs["aux_change"], change_target)
            total     = total + self.aux_weight * self.w_change * l_aux_chg
            breakdown["aux_change"] = l_aux_chg.item()

        breakdown["total"] = total.item()   # update after aux
        return total, breakdown


# ── Metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def binary_iou_f1(logits, targets, threshold=0.5, eps=1e-6):
    """IoU and F1 for the positive class. Use footprint channel for FBC."""
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    iou = (tp + eps) / (tp + fp + fn + eps)
    f1  = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return iou.item(), f1.item()


if __name__ == "__main__":
    B, T, fbc_ch = 2, 5, 3
    H, W = 256, 256

    outputs = {
        "fbc":        torch.randn(B, T * fbc_ch, H, W),  # [2,15,256,256]
        "change":     torch.randn(B, 1, H, W),
        "aux_fbc":    torch.randn(B, fbc_ch, H, W),
        "aux_change": torch.randn(B, 1, H, W),
    }
    # fbc target: 3 channels per month
    fbc_t    = torch.zeros(B, T * fbc_ch, H, W)
    fbc_t[:, 0::3] = (torch.rand(B, T, H, W) > 0.80).float()  # footprint
    fbc_t[:, 1::3] = (torch.rand(B, T, H, W) > 0.95).float()  # boundary
    fbc_t[:, 2::3] = (torch.rand(B, T, H, W) > 0.97).float()  # contact

    change_t = (torch.rand(B, 1, H, W) > 0.985).float()

    crit = MultiTaskLoss(fbc_ch=3, n_months=5, w_change=1.0, change_pos_w=11.0)
    total, parts = crit(outputs, fbc_t, change_t)
    print("Loss breakdown:", {k: f"{v:.4f}" for k,v in parts.items()})
    assert torch.isfinite(total), "Loss is not finite!"
    print("OK")