"""Load only the images and labels provided by the course competition."""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TRAIN_MEAN = (0.5, 0.5, 0.5)
TRAIN_STD = (0.5, 0.5, 0.5)


def training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ]
    )


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(TRAIN_MEAN, TRAIN_STD)]
    )


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


class CompetitionImages(Dataset):
    def __init__(self, data_dir: Path, rows: list[dict[str, str]], *, train: bool) -> None:
        self.folder = data_dir / ("train_images" if "label" in rows[0] else "test_images")
        self.rows = rows
        self.transform = training_transform() if train else evaluation_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.folder / row["filename"]) as image:
            tensor = self.transform(image.convert("RGB"))
        if "label" in row:
            return tensor, int(row["label"])
        return tensor, int(row["id"])
