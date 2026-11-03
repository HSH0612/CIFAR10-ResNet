"""Download and unpack this project's Kaggle competition data."""

import argparse
import csv
import os
from pathlib import Path
from zipfile import ZipFile


COMPETITION = "cifar-10-res-net-architecture-challenge-dl"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ENV_FILE = PROJECT_DIR / ".env"
ARCHIVE = DATA_DIR / "project1_competition_data.zip"
PNG_SIGNATURE = bytes.fromhex("89504e470d0a1a0a")
import kagglehub


def load_local_token() -> bool:
    """Load a Kaggle token from .env without displaying it."""
    if not ENV_FILE.is_file():
        return False
    entries = [
        line.strip()
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    token = None
    for entry in entries:
        if entry.startswith("export "):
            entry = entry[len("export ") :].strip()
        if entry.startswith("KAGGLE_API_TOKEN="):
            token = entry.split("=", 1)[1].strip().strip('"\'')
            break
    if token is None and len(entries) == 1 and "=" not in entries[0]:
        token = entries[0].strip('"\'')
    if not token:
        raise SystemExit("Put KAGGLE_API_TOKEN=your_token in .env; keep it private.")
    os.environ["KAGGLE_API_TOKEN"] = token
    return True


def validate_data() -> None:
    """Verify that every image named by the competition CSVs exists as a PNG."""
    total = 0
    for csv_name, folder in (("train_labels.csv", "train_images"), ("test.csv", "test_images")):
        with (DATA_DIR / csv_name).open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                filename = row["filename"]
                if Path(filename).name != filename or not filename.endswith(".png"):
                    raise SystemExit(f"Unexpected filename in {csv_name}")
                image = DATA_DIR / folder / filename
                if not image.is_file():
                    raise SystemExit(f"Missing image: {image}")
                with image.open("rb") as file:
                    if file.read(8) != PNG_SIGNATURE:
                        raise SystemExit(f"Invalid PNG: {image}")
                total += 1
    print(f"Verified {total} competition images and both image indexes.")


def extract_archive(archive_path: Path) -> None:
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit("Archive contains an unsafe path")
        print(f"Extracting {len(archive.namelist())} files from {archive_path.name}...", flush=True)
        archive.extractall(DATA_DIR)
    validate_data()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--login", action="store_true", help="Prompt for a Kaggle token")
    parser.add_argument(
        "--archive", type=Path, help="A competition ZIP downloaded from Kaggle or your own Kaggle Notebook"
    )
    args = parser.parse_args()
    archive_path = args.archive if args.archive is not None else ARCHIVE
    if args.archive is not None and not archive_path.is_file():
        parser.error(f"Archive does not exist: {archive_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    expected = (
        "train_labels.csv",
        "test.csv",
        "sample_submission.csv",
        "train_images",
        "test_images",
    )
    if all((DATA_DIR / name).exists() for name in expected):
        validate_data()
        print(f"Competition data location: {DATA_DIR}")
        return
    if archive_path.is_file():
        extract_archive(archive_path)
        print(f"Competition data location: {DATA_DIR}")
        return

    from_env = load_local_token()
    if args.login:
        kagglehub.login()
    elif from_env:
        print("Using Kaggle API token from local .env file.")

    if any(DATA_DIR.iterdir()):
        raise SystemExit(
            "Partial competition files are present, but the full archive is missing. "
            "Download the full competition ZIP or create one with kaggle_export/export_data.py "
            "in your own Kaggle Notebook, then pass it with --archive."
        )
    try:
        kagglehub.competition_download(COMPETITION, output_dir=str(DATA_DIR))
    except kagglehub.exceptions.UnauthenticatedError as exc:
        raise SystemExit("Kaggle needs an API token in .env or --login.") from exc
    except kagglehub.exceptions.KaggleApiHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            raise SystemExit(
                "Kaggle's bulk archive endpoint returned 404. Run "
                "kaggle_export/export_data.py in your own Kaggle Notebook with "
                "the competition data attached, then pass its ZIP with --archive."
            ) from None
        if status in (401, 403):
            raise SystemExit(
                "Kaggle rejected the request. Check that this token belongs to the "
                "account that joined the competition and accepted its rules."
            ) from None
        raise
    validate_data()
    print(f"Competition data location: {DATA_DIR}")


if __name__ == "__main__":
    main()
