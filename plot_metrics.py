"""Save reproducible training curves from a run's metrics.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PROJECT_DIR / "results" / "cifar_resnet_w40")
    args = parser.parse_args()
    with (args.run / "metrics.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("No completed epochs in metrics.csv")
    epochs = [int(row["epoch"]) for row in rows]
    for field_pair, ylabel, filename in (
        (("train_accuracy", "val_accuracy"), "Accuracy", "accuracy.png"),
        (("train_loss", "val_loss"), "Cross-entropy loss", "loss.png"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for field in field_pair:
            axis.plot(epochs, [float(row[field]) for row in rows], label=field.replace("_", " "))
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.savefig(args.run / filename, dpi=160)
        plt.close(fig)
    print(f"Saved accuracy.png and loss.png in {args.run}")


if __name__ == "__main__":
    main()
