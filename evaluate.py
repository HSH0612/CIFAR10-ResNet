"""Evaluate the best checkpoint on the held-out validation split."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CompetitionImages, read_rows
from models.resnet import CifarResNet, count_parameters
from train import make_split


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results" / "cifar_resnet_w40")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--tta-flip", action="store_true",
        help="Average logits from each image and its horizontal flip",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.run / "best.pt", map_location=device, weights_only=False)
    model = CifarResNet(checkpoint["width"]).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _, val_rows = make_split(read_rows(ROOT / "data" / "train_labels.csv"), checkpoint["seed"])
    loader = DataLoader(CompetitionImages(ROOT / "data", val_rows, train=False), batch_size=args.batch_size, num_workers=0)

    matrix = np.zeros((10, 10), dtype=np.int64)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                if args.tta_flip:
                    flipped_logits = model(images.flip(-1))
                    logits = (logits.float() + flipped_logits.float()) / 2
                predictions = logits.argmax(1).cpu().numpy()
            np.add.at(matrix, (labels.numpy(), predictions), 1)
    accuracy = np.trace(matrix) / matrix.sum()
    if not args.tta_flip and abs(accuracy - checkpoint["best_accuracy"]) > 1e-8:
        raise ValueError("Recomputed validation accuracy does not match the checkpoint")
    if count_parameters(model) > 5_000_000:
        raise ValueError("Model exceeds the parameter limit")

    suffix = "_flip_tta" if args.tta_flip else ""
    per_class_path = args.run / f"per_class{suffix}.csv"
    confusion_path = args.run / f"confusion{suffix}.png"
    with per_class_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("label", "class", "correct", "total", "accuracy"))
        for label, name in enumerate(CLASS_NAMES):
            correct, total = int(matrix[label, label]), int(matrix[label].sum())
            writer.writerow((label, name, correct, total, correct / total))

    fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=int(matrix.max()))
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Validation confusion matrix (accuracy {accuracy:.2%})")
    for row in range(10):
        for column in range(10):
            value = int(matrix[row, column])
            ax.text(column, row, str(value), ha="center", va="center", fontsize=7,
                    color="white" if value > matrix.max() / 2 else "#26384B")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(confusion_path, dpi=170)
    plt.close(fig)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Saved {per_class_path.name} and {confusion_path.name} in {args.run}")


if __name__ == "__main__":
    main()
