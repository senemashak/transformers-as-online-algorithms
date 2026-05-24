"""Train the CacheEvictionTransformer on Belady labels.

Single-GPU usage (from caching/):
    python3 -m learned_eviction.train --data-dir data/run_20260422 --device cuda

Multi-GPU (DDP) usage:
    CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 \\
        -m learned_eviction.train --data-dir data/run_20260422
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .model import CacheEvictionTransformer


def _init_ddp():
    """Detect torchrun environment and initialise the process group.

    Returns (is_ddp, local_rank, global_rank, world_size).
    """
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, local_rank, rank, world_size


def build_loaders(
    data_dir: Path,
    k: int,
    w: int,
    batch_size: int,
    num_workers: int,
    is_ddp: bool,
    world_size: int,
    rank: int,
    label_mode: str = "all_timesteps",
    max_per_trace: int | None = None,
):
    """Default to LRU+LFU+ARC trace files in data_dir. Trace-level 80/10/10 split."""
    from .dataset import default_split

    trace_files = sorted(data_dir.glob("*_traces.npy"))
    assert trace_files, f"no *_traces.npy under {data_dir}"

    train_ds, val_ds, test_ds = default_split(
        trace_files,
        cache_size=k,
        context_window=w,
        val_frac=0.1,
        test_frac=0.1,
        seed=0,
        cache_root=data_dir.parent / "belady_cache",
        label_mode=label_mode,
        max_per_trace=max_per_trace,
    )
    if rank == 0:
        print(f"train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,}")

    split_info = {
        "files": [p.name for p in trace_files],
        "train": train_ds.row_indices_per_file,
        "val": val_ds.row_indices_per_file,
        "test": test_ds.row_indices_per_file,
    }

    if is_ddp:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    loaders = (
        DataLoader(train_ds, sampler=train_sampler,
                   shuffle=(train_sampler is None), drop_last=True, **common),
        DataLoader(val_ds, sampler=val_sampler, shuffle=False, **common),
        DataLoader(test_ds, sampler=test_sampler, shuffle=False, **common),
    )
    return loaders, split_info, train_sampler


@torch.no_grad()
def evaluate(model, loader, device, is_ddp: bool):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        cache = batch["cache"].to(device)
        seq = batch["seq"].to(device)
        label = batch["label"].to(device)
        logits = model(cache, seq)
        loss_sum += loss_fn(logits, label).item()
        correct += (logits.argmax(-1) == label).sum().item()
        total += label.numel()

    if is_ddp:
        t = torch.tensor([loss_sum, correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_sum, correct, total = t.tolist()

    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args):
    is_ddp, local_rank, rank, world_size = _init_ddp()
    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    is_main = rank == 0

    data_dir = Path(args.data_dir)

    (train_loader, val_loader, _test_loader), split_info, train_sampler = build_loaders(
        data_dir, k=args.k, w=args.context_window,
        batch_size=args.batch_size, num_workers=args.num_workers,
        is_ddp=is_ddp, world_size=world_size, rank=rank,
        label_mode=args.label_mode,
        max_per_trace=args.max_per_trace,
    )

    log_path = Path(args.log_dir)
    if is_main:
        log_path.mkdir(parents=True, exist_ok=True)
        with open(log_path / "split.json", "w") as f:
            json.dump({"data_dir": str(data_dir), **split_info}, f)
        print(f"split saved to {log_path / 'split.json'}")
        print(f"world_size = {world_size}  device = {device}")

    model = CacheEvictionTransformer(
        vocab_size=args.vocab_size,
        cache_size=args.k,
        context_window=args.context_window,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model: {n_params:,} params")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if is_ddp else model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    # Build LR scheduler.
    total_steps = len(train_loader) * args.epochs  # per-rank; SequentialLR will step this many times
    scheduler = None
    if args.lr_schedule == "cosine":
        warmup_steps = max(args.warmup_steps, 1)
        warmup = LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps,
        )
        cos_steps = max(total_steps - warmup_steps, 1)
        cosine = CosineAnnealingLR(opt, T_max=cos_steps, eta_min=args.lr_min)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
        if is_main:
            print(f"lr schedule: cosine  warmup={warmup_steps} steps  "
                  f"peak={args.lr}  min={args.lr_min}  total={total_steps}")
    elif args.lr_schedule == "constant":
        if is_main:
            print(f"lr schedule: constant {args.lr}")
    else:
        raise ValueError(f"unknown lr_schedule: {args.lr_schedule}")

    if is_main:
        train_loss_f = open(log_path / "train_loss.jsonl", "a")
        val_loss_f = open(log_path / "val_loss.jsonl", "a")

    best_val = float("inf")
    step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        for batch in train_loader:
            cache = batch["cache"].to(device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            logits = model(cache, seq)
            loss = loss_fn(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            if is_main and step % args.log_every == 0:
                acc = (logits.argmax(-1) == label).float().mean().item()
                current_lr = opt.param_groups[0]["lr"]
                print(f"epoch {epoch} step {step}  lr {current_lr:.2e}  "
                      f"loss {loss.item():.4f}  acc {acc:.3f}")
                train_loss_f.write(json.dumps({
                    "step": step, "epoch": epoch,
                    "loss": loss.item(), "acc": acc,
                    "lr": current_lr,
                }) + "\n")
                train_loss_f.flush()

        val_loss, val_acc = evaluate(model, val_loader, device, is_ddp=is_ddp)
        dt = time.time() - t0

        if is_main:
            print(f"[epoch {epoch}] val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  ({dt:.1f}s)")
            val_loss_f.write(json.dumps({
                "epoch": epoch, "step": step,
                "val_loss": val_loss, "val_acc": val_acc, "dt_s": dt,
            }) + "\n")
            val_loss_f.flush()

            ckpt = {"model": raw_model.state_dict(), "args": vars(args),
                    "epoch": epoch, "step": step, "val_loss": val_loss}
            torch.save(ckpt, log_path / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, log_path / "best.pt")
                print(f"  ↳ saved new best to {log_path / 'best.pt'}")

    if is_main:
        train_loss_f.close()
        val_loss_f.close()

    if is_ddp:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/run_20260422")
    p.add_argument("--log-dir", type=str, default="learned_eviction/runs/default")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--context-window", type=int, default=1024)
    p.add_argument("--vocab-size", type=int, default=513)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4, help="peak (and constant if no schedule) LR")
    p.add_argument("--lr-schedule", type=str, default="constant",
                   choices=["constant", "cosine"],
                   help="cosine = linear warmup to --lr then cosine decay to --lr-min")
    p.add_argument("--lr-min", type=float, default=1e-5,
                   help="floor LR for cosine schedule; ignored for constant")
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="linear warmup steps for cosine schedule")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--label-mode", type=str, default="all_timesteps",
                   choices=["event", "all_timesteps"],
                   help="event: train only on full-cache miss events (old default). "
                        "all_timesteps: train on every full-cache timestep using "
                        "Belady's hypothetical furthest-future choice.")
    p.add_argument("--max-per-trace", type=int, default=None,
                   help="only used with label-mode=all_timesteps: cap per-trace "
                        "examples via deterministic subsampling. None = all.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
