"""Create a Kaggle submission.csv from a trained checkpoint."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CompetitionImages, read_rows
from models.resnet import CifarResNet, count_parameters


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "results" / "cifar_resnet_w40" / "best.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--tta-flip", action="store_true",
        help="Average logits from each image and its horizontal flip",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CifarResNet(checkpoint["width"]).to(device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_rows = read_rows(DATA_DIR / "test.csv")
    sample_rows = read_rows(DATA_DIR / "sample_submission.csv")
    if [row["id"] for row in test_rows] != [row["id"] for row in sample_rows]:
        raise SystemExit("Test IDs do not match the sample submission")
    loader = DataLoader(
        CompetitionImages(DATA_DIR, test_rows, train=False),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    with torch.inference_mode():
        for images, ids in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                if args.tta_flip:
                    flipped_logits = model(images.flip(-1))
                    logits = (logits.float() + flipped_logits.float()) / 2
                labels = logits.argmax(1).cpu().tolist()
            predictions.extend(zip(ids.tolist(), labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["id", "label"])
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print(f"Horizontal-flip test-time augmentation: {'on' if args.tta_flip else 'off'}")
    print(f"Checkpoint single-view validation accuracy: {checkpoint['best_accuracy']:.2%}")
    print(f"Trainable parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    main()
