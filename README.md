# CIFAR-10 ResNet Project 1

This folder is the Project 1 codebase. It trains a CIFAR-sized residual network
from random initialization using only the course competition images. The model
has **4,368,690 trainable parameters**, below the 5 million limit. Competition
data stays in `data/` and is excluded from Git.

## Project layout

- `models/resnet.py`: compact residual network and parameter count.
- `dataset.py`, `download_data.py`: competition data loading and retrieval.
- `train.py`, `evaluate.py`, `predict.py`: training, held-out validation, and Kaggle CSV generation.
- `plot_metrics.py`, `notebooks/`: plots and executed result checks.
- `results/cifar_resnet_w40/`: training metrics, configuration, validation IDs, and figures. Local checkpoints are excluded from Git.
- `data/`, `tmp/`: local competition files and temporary work; both are excluded from Git.

Dependencies are managed with `pyproject.toml`, `uv.lock`, and `.python-version`.

## Python environment in VS Code

This project uses `.python-version` to select Python 3.12, while `pyproject.toml`
and `uv.lock` define and lock the dependencies. `uv sync --extra notebook`
creates `.venv/`. If imports are underlined in VS Code, run **Python: Select
Interpreter** from the Command Palette and choose `.venv/Scripts/python.exe`
on Windows or `.venv/bin/python` on Linux/macOS, then reload the VS Code window.
VS Code may remember a previously selected interpreter.

## Download the data

1. Join the [Kaggle competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/overview) and accept its rules.
2. Install [uv](https://docs.astral.sh/uv/) and run `uv sync --extra notebook` from this folder. `pyproject.toml` and `uv.lock` pin the dependencies; on Windows/Linux, uv uses the PyTorch CUDA 12.8 index. The tested machine has an RTX 3060 12 GB.
3. Download the competition ZIP from Kaggle after joining, then run
   `uv run --extra notebook python download_data.py --archive path/to/competition.zip`.
   The script unpacks it under `data/` and verifies all 52,000 images against
   the CSV indexes. You can also place the extracted `train_images/`,
   `test_images/`, `train_labels.csv`, `test.csv`, and
   `sample_submission.csv` directly in `data/` and run
   `uv run --extra notebook python download_data.py` to validate them.

If the bulk ZIP endpoint is unavailable, create your own Kaggle Notebook, attach
this competition's data as input, paste and run
`kaggle_export/export_data.py`, and download its
`project1_competition_data.zip` output. Pass that local ZIP to
`download_data.py --archive`. The exporter uses only the competition files
mounted in your Kaggle Notebook.

Alternatively, create an API token in your [Kaggle account settings](https://www.kaggle.com/settings/api),
put `KAGGLE_API_TOKEN=your_token` in a local `.env` file, and run
`uv run --extra notebook python download_data.py` to try Kaggle's archive API.
`.env` is excluded from Git and must not be shared. The prediction CSV
follows the exact columns and row order in `sample_submission.csv`.

## Train and evaluate

```powershell
uv run --extra notebook python train.py --epochs 100 --batch-size 128 --workers 0 --output runs/reproduction-001
uv run --extra notebook python plot_metrics.py --run runs/reproduction-001
uv run --extra notebook python evaluate.py --run runs/reproduction-001
uv run --extra notebook python predict.py --checkpoint runs/reproduction-001/best.pt --output submission_reproduction.csv
uv run --extra notebook python evaluate.py --run runs/reproduction-001 --tta-flip
uv run --extra notebook python predict.py --checkpoint runs/reproduction-001/best.pt --tta-flip --output submission_reproduction_flip_tta.csv
```

Training uses a fixed, class-balanced 45,000/5,000 train/validation split from
`train_labels.csv`. Only training images receive random crop and horizontal flip.
The optimizer is SGD with momentum and weight decay; the learning rate follows
cosine annealing. The commands above write a new reproduction to
`runs/reproduction-001/`; choose another empty output directory for each
new experiment. `train.py` refuses to write a new run into a nonempty directory
unless `--resume` names a checkpoint from that same run. The submitted
experiment's small metrics and figures are
included under `results/cifar_resnet_w40/` for inspection. Checkpoints are local
and excluded from Git. Resume an interrupted reproduction with:

```powershell
uv run --extra notebook python train.py --epochs 100 --batch-size 128 --workers 0 --resume runs/reproduction-001/last.pt
```

`predict.py` writes `submission.csv` with exactly the `id,label` columns from
the sample submission. It prints the **validation accuracy** and parameter
count. The separate Kaggle test accuracy is only known after submitting the
CSV; test labels are never used for training or model selection.

For a second prediction run, `predict.py --tta-flip --output submission_flip_tta.csv`
averages the model logits for each image and its horizontal flip. On the fixed
5,000-image validation split, this increased accuracy from 94.82% to 95.20%.
The first submission scored 83.00% on Kaggle's public split, and the flip-TTA
submission scored 83.60%. Both completed scores and their submission references
are recorded in `results/cifar_resnet_w40/kaggle_submissions.csv`; no test labels are
used locally. The final Kaggle private leaderboard accuracy was **81.30%**,
recorded separately in `kaggle_score.json` without assigning it to either CSV.
Reproduce the augmented validation result with
`uv run --extra notebook python evaluate.py --run runs/reproduction-001 --tta-flip`; it writes separate
`per_class_flip_tta.csv` and `confusion_flip_tta.png` files.

After my first submission, I checked whether a mistake in my prediction pipeline
could explain the gap between validation and the public score. I regenerated
the same CSV from `best.pt` and confirmed that its test IDs matched the sample
submission in order. The training and validation splits shared no IDs or exact
duplicate files, and I used the same image transform for validation and test.
Mean confidence was 97.64% on validation images and 93.80% on test images.
Without the test labels, I could not determine why the accuracy differed.

## Architecture and experiment record

`models/resnet.py` implements a CIFAR-sized residual network inspired by the original
ResNet paper. A 3x3 stem feeds four stages with widths 40, 80, 160, and 320.
Each stage has two residual blocks; the last three stages downsample in their
first block. Global average pooling feeds a 10-class linear classifier. This
architecture has 4,368,690 trainable parameters, counted by `count_parameters`.
All weights start from PyTorch's random initialization; no pretrained model or
external image dataset is loaded.

The run uses seed 42 and a stratified 45,000/5,000 split of the 50,000 labeled
competition images. Training uses 32x32 random crop after four-pixel padding,
random horizontal flip, cross-entropy, SGD (learning rate 0.1, momentum 0.9,
weight decay 0.0005), and 100-epoch cosine learning-rate decay. Validation and
test images receive only tensor conversion and channel normalization.
`results/cifar_resnet_w40/config.json`, `metrics.csv`, and `validation_ids.txt` make the
split and training choices inspectable. The best checkpoint is selected only by
validation accuracy. It is stored locally because model checkpoints are too
large for the source repository.

The two plots are generated from `metrics.csv` with `plot_metrics.py`. The
Kaggle score should be recorded separately from validation accuracy after the
submission is accepted; the public leaderboard may differ from the final
private leaderboard.

`notebooks/Project1_Results.ipynb` is the executed record of the original
`results/cifar_resnet_w40/` run. It contains parameter checks, training curves,
per-class validation results, a confusion matrix, and separate checks for the
single-view and flip-TTA submission files. Open it in Jupyter or VS Code to
view the saved results. To rerun the CSV checks, first regenerate the prediction
files using the commands above.
It prints the final private test accuracy (81.30%) and the trainable parameter
count (4,368,690) separately from the two public scores.

## Reference

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. [Deep Residual Learning
for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016. The
architecture here was implemented in this project rather than copied from an
external code repository.
