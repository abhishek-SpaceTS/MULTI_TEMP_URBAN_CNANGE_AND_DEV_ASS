"""
segmentation/train.py
=====================

Trains TE_UNet (Temporal EfficientNet U-Net) from scratch.

CHANGES FROM ORIGINAL:
  1. MultiTaskLoss initialised with new defaults (w_change=2.0, pos_w=10.0)
     These flow from config so CLI overrides still work.

  2. Scheduler kept as warmup + cosine decay.

  3. Dynamic w_fbc decay NOT added — w_fbc stays constant throughout.

  4. Threshold sweep added at end of training.

  5. NaN guard added in training loop — skips bad batches before they
     corrupt the model weights.

  6. FIXED: best_change_iou updated BEFORE creating ckpt dict so resume
     always starts from the correct epoch. Previously caused resume to
     repeat 2-3 epochs unnecessarily.

  7. default out_dir changed to segmentation_L/checkpoints

RESUME: pass --resume path/to/last.pt to continue from a checkpoint.
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from rich.text import Text

from config import cfg
from dataset import get_dataloaders, N_MONTHS
from model import TE_UNet
from loss import MultiTaskLoss

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()

IS_KAGGLE = os.path.exists("/kaggle/working")
CKPT_DIR: Path = Path(cfg.out_dir)

FOOTPRINT_IDX = list(range(0, cfg.fbc_ch * N_MONTHS, cfg.fbc_ch))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",            type=int,   default=cfg.epochs)
    ap.add_argument("--batch_size",        type=int,   default=cfg.batch_size)
    ap.add_argument("--lr",                type=float, default=cfg.lr)
    ap.add_argument("--weight_decay",      type=float, default=cfg.weight_decay)
    ap.add_argument("--warmup_epochs",     type=int,   default=cfg.warmup_epochs)
    ap.add_argument("--num_workers",       type=int,   default=cfg.num_workers)
    ap.add_argument("--base",              type=int,   default=cfg.base)
    ap.add_argument("--w_change",          type=float, default=cfg.w_change)
    ap.add_argument("--w_fbc",             type=float, default=cfg.w_fbc)
    ap.add_argument("--change_pos_w",      type=float, default=cfg.change_pos_w)
    ap.add_argument("--samples_per_epoch", type=int,   default=cfg.samples_per_epoch)
    ap.add_argument("--out_dir",           type=str,   default="segmentation_F/checkpoints_v5")
    ap.add_argument("--resume",            type=str,   default=None)
    return ap.parse_args()


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to ~0. Stepped once per iteration."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class MetricAccumulator:
    """Global TP/FP/FN accumulator — correct per-dataset IoU/F1."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5) -> None:
        preds = (torch.sigmoid(logits) > threshold).float()
        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()

    def iou_f1(self, eps: float = 1e-6):
        iou = (self.tp + eps) / (self.tp + self.fp + self.fn + eps)
        f1  = (2 * self.tp + eps) / (2 * self.tp + self.fp + self.fn + eps)
        return iou, f1


# ── Dashboard helpers ────────────────────────────────────────────────────────

def _color(value: float, low: float = 0.3, high: float = 0.6) -> str:
    if value >= high:   return "bold green"
    if value >= low:    return "yellow"
    return "red"


def build_metric_table(epoch, total_epochs, train_loss, val,
                       best_iou, lr) -> Table:
    t = Table(show_header=True, header_style="bold cyan",
              border_style="bright_black", expand=True)
    t.add_column("Metric",  style="dim",     min_width=22)
    t.add_column("Current", justify="right", min_width=10)
    t.add_column("Best",    justify="right", min_width=10)
    chg_iou = val["change_iou"]
    chg_f1  = val["change_f1"]
    seg_f1  = val["seg_f1"]
    t.add_row("Epoch",         f"[bold white]{epoch}[/] / {total_epochs}", "—")
    t.add_row("Learning Rate", f"[cyan]{lr:.2e}[/]", "—")
    t.add_row("Train Loss",    f"[yellow]{train_loss:.4f}[/]", "—")
    t.add_row("Val Loss",      f"[yellow]{val['loss']:.4f}[/]", "—")
    t.add_row("Seg F1",        f"[{_color(seg_f1)}]{seg_f1:.4f}[/]", "—")
    t.add_row("Change F1",     f"[{_color(chg_f1)}]{chg_f1:.4f}[/]", "—")
    t.add_row("[bold]Change IoU  ★",
              f"[{_color(chg_iou)}]{chg_iou:.4f}[/]",
              f"[bold green]{best_iou:.4f}[/]")
    return t


def build_history_table(history: list) -> Table:
    t = Table(show_header=True, header_style="bold magenta",
              border_style="bright_black", expand=True)
    for col in ("Ep", "T-Loss", "V-Loss", "Seg-F1", "Chg-F1", "Chg-IoU"):
        t.add_column(col, justify="right", min_width=9)
    for row in history[-10:]:
        iou = row["change_iou"]
        t.add_row(str(row["epoch"]),
                  f"{row['train_loss']:.4f}", f"{row['val_loss']:.4f}",
                  f"{row['seg_f1']:.4f}",     f"{row['change_f1']:.4f}",
                  f"[{_color(iou)}]{iou:.4f}[/]")
    return t


# ── Threshold sweep ──────────────────────────────────────────────────────────

@torch.no_grad()
def sweep_threshold(model, loader, device, task: str = "change"):
    """Sweep thresholds 0.35-0.65 on val set. Call after training completes."""
    log(f"\n── Threshold sweep ({task}) ──")
    model.eval()
    best_thresh, best_f1 = 0.5, 0.0
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    for thresh in thresholds:
        acc = MetricAccumulator()
        for images, masks, change, _ in loader:
            images = images.to(device)
            masks  = masks.to(device)
            change = change.to(device)
            out    = model(images)
            if task == "change":
                acc.update(out["change"], change, thresh)
            else:
                acc.update(out["fbc"][:, FOOTPRINT_IDX],
                           masks[:, FOOTPRINT_IDX], thresh)
        iou, f1 = acc.iou_f1()
        log(f"  thresh={thresh:.2f}  IoU={iou:.4f}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    log(f"  → Best threshold: {best_thresh}  F1={best_f1:.4f}")
    return best_thresh


# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold,
             batch_progress, val_task):
    model.eval()
    chg_acc, fp_acc = MetricAccumulator(), MetricAccumulator()
    loss_sum, n = 0.0, 0

    batch_progress.reset(val_task, total=len(loader))
    batch_progress.update(val_task, description="[cyan]  Validating  ",
                          visible=True, extra="")

    for images, masks, change, _meta in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        change = change.to(device, non_blocking=True)

        out = model(images)
        loss, _ = criterion(out, masks, change)
        loss_sum += loss.item()
        n        += 1

        chg_acc.update(out["change"], change, threshold)
        fp_acc.update(out["fbc"][:, FOOTPRINT_IDX],
                      masks[:, FOOTPRINT_IDX], threshold)
        batch_progress.advance(val_task)
        batch_progress.update(val_task, extra=f"val_loss={loss_sum/n:.4f}")

    chg_iou, chg_f1 = chg_acc.iou_f1()
    fp_iou,  fp_f1  = fp_acc.iou_f1()
    return {
        "loss":       loss_sum / max(1, n),
        "change_iou": chg_iou,
        "change_f1":  chg_f1,
        "seg_iou":    fp_iou,
        "seg_f1":     fp_f1,
    }


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:d}m{s:02d}s"


def log(msg: str):
    print(msg, flush=True)


def main():
    args = parse_args()

    cfg.epochs            = args.epochs
    cfg.batch_size        = args.batch_size
    cfg.lr                = args.lr
    cfg.weight_decay      = args.weight_decay
    cfg.warmup_epochs     = args.warmup_epochs
    cfg.num_workers       = args.num_workers
    cfg.base              = args.base
    cfg.w_change          = args.w_change
    cfg.w_fbc             = args.w_fbc
    cfg.change_pos_w      = args.change_pos_w
    cfg.samples_per_epoch = args.samples_per_epoch
    cfg.out_dir           = args.out_dir

    global CKPT_DIR
    CKPT_DIR = Path(cfg.out_dir)

    assert cfg.n_months == N_MONTHS, (
        f"config.n_months ({cfg.n_months}) != dataset.N_MONTHS ({N_MONTHS})"
    )

    torch.manual_seed(cfg.seed)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.amp and (device == "cuda")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    loaders      = get_dataloaders(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        n_months=cfg.n_months, samples_per_epoch=cfg.samples_per_epoch,
        fbc_ch=cfg.fbc_ch,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    train_ds     = loaders["train_ds"]

    model = TE_UNet(**cfg.model_kwargs()).to(device)

    criterion = MultiTaskLoss(
        fbc_ch=cfg.fbc_ch,
        n_months=cfg.n_months,
        w_fbc=cfg.w_fbc,
        w_change=cfg.w_change,
        change_pos_w=cfg.change_pos_w,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    scheduler    = make_scheduler(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_change_iou = -1.0
    start_epoch     = 1

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            log(f"Resuming from {resume_path} ...")
            ckpt_r = torch.load(resume_path, map_location=device,
                                weights_only=False)
            model.load_state_dict(ckpt_r["model"])
            optimizer.load_state_dict(ckpt_r["optimizer"])
            if "scheduler" in ckpt_r:
                scheduler.load_state_dict(ckpt_r["scheduler"])
            if "scaler" in ckpt_r and ckpt_r["scaler"] and use_amp:
                scaler.load_state_dict(ckpt_r["scaler"])
            if "best_change_iou" in ckpt_r:
                best_change_iou = ckpt_r["best_change_iou"]
            start_epoch = ckpt_r["epoch"] + 1
            log(f"Resumed: start_epoch={start_epoch}  "
                f"best_change_iou={best_change_iou:.4f}")
        else:
            log(f"WARNING: resume path not found: {resume_path} — starting fresh")

    n_params = sum(p.numel() for p in model.parameters())
    log("=" * 65)
    log(f"  TE-UNet  |  {n_params:,} params")
    log(f"  Device={device}  AMP={use_amp}  base={cfg.base}  "
        f"d_model={cfg.d_model}  heads={cfg.n_heads}  layers={cfg.n_layers}")
    log(f"  Epochs {start_epoch}→{cfg.epochs}  "
        f"batch={cfg.batch_size}  lr={cfg.lr:.1e}  "
        f"wd={cfg.weight_decay:.1e}  warmup={cfg.warmup_epochs}ep")
    log(f"  w_change={cfg.w_change}  w_fbc={cfg.w_fbc}  "
        f"pos_weight={cfg.change_pos_w}  fbc_ch={cfg.fbc_ch}")
    log(f"  Checkpoints → {CKPT_DIR.resolve()}")
    log("=" * 65)

    # ── Progress bars ────────────────────────────────────────────────────────
    epoch_progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]Overall"),
        BarColumn(bar_width=32, style="cyan", complete_style="bold cyan"),
        MofNCompleteColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    )
    batch_progress = Progress(
        SpinnerColumn(style="magenta"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30, style="magenta", complete_style="bold magenta"),
        MofNCompleteColumn(), TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]{task.fields[extra]}"),
        console=console,
    )
    epoch_task = epoch_progress.add_task("", total=cfg.epochs)
    train_task = batch_progress.add_task(
        "[magenta]  Training    ", total=len(train_loader), extra="")
    val_task   = batch_progress.add_task(
        "[cyan]  Validating  ", total=len(val_loader), extra="", visible=False)

    history = []
    _dummy_val = dict(loss=0.0, change_iou=0.0, change_f1=0.0,
                      seg_iou=0.0, seg_f1=0.0)
    current = dict(epoch=0, train_loss=0.0, val=_dummy_val, lr=cfg.lr)

    def make_dashboard() -> Layout:
        c = current
        root = Layout()
        root.split_column(Layout(name="panels", ratio=5),
                          Layout(name="ep_bar",  ratio=1),
                          Layout(name="bat_bar", ratio=1))
        root["panels"].split_column(Layout(name="top",    ratio=2),
                                    Layout(name="bottom", ratio=3))
        root["top"].split_row(Layout(name="metrics", ratio=3),
                              Layout(name="config",  ratio=2))
        root["metrics"].update(Panel(
            build_metric_table(c["epoch"], cfg.epochs, c["train_loss"],
                               c["val"], best_change_iou, c["lr"]),
            title="[bold cyan]  Live Metrics", border_style="cyan",
        ))
        info = Text()
        info.append(f"  batch size   {cfg.batch_size}\n",        style="dim")
        info.append(f"  base LR      {cfg.lr:.1e}\n",            style="dim")
        info.append(f"  weight_decay {cfg.weight_decay:.1e}\n",  style="dim")
        info.append(f"  warmup       {cfg.warmup_epochs} ep\n",  style="dim")
        info.append(f"  w_change     {cfg.w_change}\n",          style="dim")
        info.append(f"  w_fbc        {cfg.w_fbc}  (constant)\n", style="dim")
        info.append(f"  pos_weight   {cfg.change_pos_w}\n",      style="dim")
        info.append(f"\n  last.pt  →  resume training\n",        style="yellow")
        info.append(f"  best.pt  →  inference only\n",           style="bold green")
        root["config"].update(Panel(info, title="[bold]  Run Config",
                                    border_style="bright_black"))
        root["bottom"].update(Panel(
            build_history_table(history),
            title="[bold magenta]  Epoch History  (last 10)",
            border_style="magenta",
        ))
        root["ep_bar"].update(epoch_progress)
        root["bat_bar"].update(batch_progress)
        return root

    class _FakeLive:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    live_ctx = _FakeLive() if IS_KAGGLE else Live(
        make_dashboard(), console=console, refresh_per_second=8, screen=False,
    )

    with live_ctx as live:

        def refresh():
            if not IS_KAGGLE:
                live.update(make_dashboard())

        for epoch in range(start_epoch, cfg.epochs + 1):
            t0 = time.time()
            train_ds.shuffle(seed=epoch)

            model.train()
            running = 0.0

            batch_progress.reset(train_task, total=len(train_loader))
            batch_progress.update(train_task,
                                  description="[magenta]  Training    ",
                                  visible=True, extra="")
            batch_progress.update(val_task, visible=False)

            for it, (images, masks, change, _meta) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)
                change = change.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(images)
                    loss, parts = criterion(out, masks, change)

                # NaN guard — skip bad batches before they corrupt the model
                if not torch.isfinite(loss):
                    log(f"  WARNING: non-finite loss={loss.item()} "
                        f"ep={epoch} batch={it} — skipping")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += parts["total"]

                avg_loss = running / (it + 1)
                lr_live  = scheduler.get_last_lr()[0]
                batch_progress.advance(train_task)
                batch_progress.update(train_task,
                    extra=f"loss={avg_loss:.4f}  lr={lr_live:.2e}")
                current.update(epoch=epoch, train_loss=avg_loss, lr=lr_live)
                refresh()

                if IS_KAGGLE and (it + 1) % 500 == 0:
                    log(f"  [{epoch}/{cfg.epochs}] "
                        f"batch {it+1}/{len(train_loader)} | "
                        f"loss={avg_loss:.4f} | lr={lr_live:.2e}")

            train_loss = running / max(1, len(train_loader))
            lr_now     = scheduler.get_last_lr()[0]

            batch_progress.update(train_task, visible=False)
            val = evaluate(model, val_loader, criterion, device,
                           cfg.threshold, batch_progress, val_task)
            epoch_time = time.time() - t0

            history.append(dict(
                epoch=epoch, train_loss=train_loss,
                val_loss=val["loss"], seg_f1=val["seg_f1"],
                change_f1=val["change_f1"], change_iou=val["change_iou"],
            ))
            current.update(epoch=epoch, train_loss=train_loss,
                           val=val, lr=lr_now)
            epoch_progress.advance(epoch_task)
            refresh()

            log(f"[Epoch {epoch:>3}/{cfg.epochs}] "
                f"time={fmt_time(epoch_time)} | lr={lr_now:.2e} | "
                f"train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | "
                f"seg_f1={val['seg_f1']:.4f} | "
                f"chg_f1={val['change_f1']:.4f} | "
                f"chg_iou={val['change_iou']:.4f}")

            # ── Checkpoint ───────────────────────────────────────────────
            # FIX: update best_change_iou BEFORE creating ckpt dict.
            # Previously updated after, so stored metadata was always one
            # epoch behind — causing resume to repeat 2-3 epochs.
            is_best = val["change_iou"] > best_change_iou
            if is_best:
                best_change_iou = val["change_iou"]

            ckpt = {
                "epoch":           epoch,
                "model":           model.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "scheduler":       scheduler.state_dict(),
                "scaler":          scaler.state_dict() if use_amp else None,
                "best_change_iou": best_change_iou,   # always correct now
                "val":             val,
                "model_cfg":       cfg.model_kwargs(),
                "config":          cfg.as_dict(),
            }
            torch.save(ckpt, CKPT_DIR / "last.pt")
            if is_best:
                torch.save(ckpt, CKPT_DIR / "best.pt")
                log(f"  ★ New best  chg_iou={best_change_iou:.4f}  → best.pt")
            # ─────────────────────────────────────────────────────────────

            # Threshold tuning reminder at epoch 20
            if epoch == 20:
                log("\n── Epoch 20 check ──")
                log(f"  chg_recall < 0.30  → raise change_pos_w to 15-20")
                log(f"  chg_precision < 0.20 → lower change_pos_w to 5-8")
                log(f"  both reasonable    → keep at {cfg.change_pos_w}")

    # ── Post-training threshold sweep ────────────────────────────────────────
    log("\n" + "=" * 65)
    log("  Training complete.")
    log(f"  Best val change IoU = {best_change_iou:.4f}")
    log("  Running threshold sweep on val set...")

    best_seg_thresh    = sweep_threshold(model, val_loader, device, task="fbc")
    best_change_thresh = sweep_threshold(model, val_loader, device, task="change")

    log(f"\n  Use threshold={best_seg_thresh} for segmentation")
    log(f"  Use threshold={best_change_thresh} for change detection")
    log(f"  Checkpoints → {CKPT_DIR.resolve()}")
    log("=" * 65)


if __name__ == "__main__":
    main()