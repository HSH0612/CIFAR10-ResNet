"""Package the course competition's mounted input files for local use."""

from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


INPUT_ROOT = Path("/kaggle/input")
ARCHIVE = Path("/kaggle/working/project1_competition_data.zip")

candidates = [
    path.parent
    for path in INPUT_ROOT.rglob("train_labels.csv")
    if (path.parent / "test.csv").is_file()
    and (path.parent / "sample_submission.csv").is_file()
    and (path.parent / "train_images").is_dir()
    and (path.parent / "test_images").is_dir()
]
if len(candidates) != 1:
    raise RuntimeError(f"Expected one mounted competition dataset; found {len(candidates)}")

source = candidates[0]
files = sorted(path for path in source.rglob("*") if path.is_file())
with ZipFile(ARCHIVE, "w", compression=ZIP_STORED, allowZip64=True) as output:
    for path in files:
        output.write(path, path.relative_to(source).as_posix())

print(f"Packaged {len(files)} competition files into {ARCHIVE.name}")
print(f"Archive size: {ARCHIVE.stat().st_size} bytes")
