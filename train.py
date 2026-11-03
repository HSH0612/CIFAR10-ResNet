"""Train a sub-5M-parameter CIFAR ResNet on the permitted competition data."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CompetitionImages, read_rows
from models.resnet import CifarResNet, count_parameters


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PARAMETER_LIMIT = 5_000_000


def make_split(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Reserve 10% of each class for validation; never touch test images."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {label: [] for label in range(10)}
    for index, row in enumerate(rows):
        by_class[int(row["label"])].append(index)
    validation_indices = set()
    for indices in by_class.values():
        rng.shuffle(indices)
        validation_indices.update(indices[: max(1, len(indices) // 10)])
    train_rows = [row for index, row in enumerate(rows) if index not in validation_indices]
    val_rows = [row for index, row in enumerate(rows) if index in validation_indices]
    return train_rows, val_rows


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(1) == labels).sum().item()
        total += batch_size
    return loss_sum / total, correct / total


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        help="New run directory; defaults to a timestamped directory under runs/",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-batches", type=int, help="Short pipeline check only")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers cannot be negative")
    if not (DATA_DIR / "train_labels.csv").is_file():
        parser.error("Competition data is missing; run download_data.py first")
    if args.output is None:
        args.output = (
            args.resume.parent if args.resume
            else PROJECT_DIR / "runs" / f"resnet-w{args.width}-{datetime.now():%Y%m%d-%H%M%S}"
        )
    args.output = args.output.resolve()
    config_path = args.output / "config.json"
    metrics_path = args.output / "metrics.csv"
    validation_ids_path = args.output / "validation_ids.txt"
    if args.resume:
        args.resume = args.resume.resolve()
        if args.resume.parent != args.output:
            parser.error("--resume checkpoint must be inside --output (or omit --output)")
        if not args.resume.is_file() or not config_path.is_file() or not metrics_path.is_file():
            parser.error("Resume requires a checkpoint, config.json, and metrics.csv in the same run directory")
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory already contains files: {args.output}. Choose a new directory or use --resume.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    rows = read_rows(DATA_DIR / "train_labels.csv")
    train_rows, val_rows = make_split(rows, args.seed)
    train_loader = DataLoader(
        CompetitionImages(DATA_DIR, train_rows, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        CompetitionImages(DATA_DIR, val_rows, train=False),
        batch_size=args.batch_size * 2,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = CifarResNet(args.width).to(device, memory_format=torch.channels_last)
    parameter_count = count_parameters(model)
    if parameter_count > PARAMETER_LIMIT:
        parser.error(f"Model has {parameter_count} trainable parameters; limit is {PARAMETER_LIMIT}")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "width": args.width,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "train_size": len(train_rows),
        "validation_size": len(val_rows),
        "trainable_parameters": parameter_count,
        "optimizer": "SGD momentum=0.9 weight_decay=5e-4",
        "scheduler": "cosine annealing",
        "augmentation": "random crop padding=4; random horizontal flip",
        "normalization": "(x - 0.5) / 0.5 per channel",
    }
    validation_ids = "\n".join(row["id"] for row in val_rows) + "\n"
    if args.resume:
        recorded_config = json.loads(config_path.read_text(encoding="utf-8"))
        if recorded_config != config:
            parser.error("Resume arguments do not match the run's config.json")
        if not validation_ids_path.is_file() or validation_ids_path.read_text(encoding="utf-8") != validation_ids:
            parser.error("Resume validation split does not match validation_ids.txt")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        validation_ids_path.write_text(validation_ids, encoding="utf-8")

    start_epoch = 1
    best_accuracy = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["width"] != args.width or checkpoint["seed"] != args.seed:
            parser.error("Resume checkpoint width/seed does not match the current run")
        with metrics_path.open(newline="", encoding="utf-8") as source:
            recorded_epochs = [int(row["epoch"]) for row in csv.DictReader(source)]
        if recorded_epochs != list(range(1, checkpoint["epoch"] + 1)):
            parser.error("Resume checkpoint epoch does not match metrics.csv")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_accuracy"]

    print(
        f"Device: {device}; train: {len(train_rows)}; validation: {len(val_rows)}; "
        f"trainable parameters: {parameter_count:,}",
        flush=True,
    )
    print(f"Run directory: {args.output}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.monotonic()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, args.max_batches
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device, None, None)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        seconds = time.monotonic() - start
        is_best = val_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, val_accuracy)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "width": args.width,
            "seed": args.seed,
            "trainable_parameters": parameter_count,
        }
        save_checkpoint(args.output / "last.pt", state)
        if is_best:
            save_checkpoint(args.output / "best.pt", state)
        header_needed = not metrics_path.exists()
        with metrics_path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            if header_needed:
                writer.writerow(
                    ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr", "seconds"]
                )
            writer.writerow(
                [epoch, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate, seconds]
            )
        print(
            f"epoch {epoch:03d}/{args.epochs}: train {train_accuracy:.2%}, "
            f"val {val_accuracy:.2%}, best {best_accuracy:.2%}, "
            f"loss {val_loss:.4f}, {seconds:.1f}s",
            flush=True,
        )
    print(f"Final best validation accuracy: {best_accuracy:.2%}", flush=True)
    print(f"Trainable parameters: {parameter_count:,}", flush=True)


if __name__ == "__main__":
    main()
