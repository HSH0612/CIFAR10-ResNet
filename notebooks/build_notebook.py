"""Execute the Project 1 results cells and save their real outputs in an ipynb."""

import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("Project1_Results.ipynb")


def markdown(source):
    return {"cell_type": "markdown", "id": hashlib.sha1(source.encode()).hexdigest()[:12], "metadata": {}, "source": source}


def code(source, count, namespace):
    text_output = io.StringIO()
    figures = []
    original_show = plt.show

    def capture_show(*_args, **_kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            image = io.BytesIO()
            figure.savefig(image, format="png", dpi=140, bbox_inches="tight")
            figures.append(base64.b64encode(image.getvalue()).decode("ascii"))
            plt.close(figure)

    plt.show = capture_show
    try:
        with redirect_stdout(text_output):
            exec(compile(source, f"notebook cell {count}", "exec"), namespace)
    finally:
        plt.show = original_show
        plt.close("all")
    outputs = []
    if text_output.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": text_output.getvalue()})
    for figure in figures:
        outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": figure}})
    return {"cell_type": "code", "id": hashlib.sha1(source.encode()).hexdigest()[:12], "execution_count": count, "metadata": {}, "source": source, "outputs": outputs}


def main():
    if not (ROOT / "results" / "cifar_resnet_w40" / "metrics.csv").is_file():
        raise SystemExit("Training metrics are missing")
    cells = [
        markdown(
            "# Project 1: compact ResNet on CIFAR-10\n\n"
            "**Author:** ShenghuaHuang. The model is trained from scratch using only the course competition images. "
            "This notebook displays the saved experiment under `results/cifar_resnet_w40/`. "
            "To reproduce a fresh run from the project root, use one empty output directory for every command:\n\n"
            "```powershell\n"
            "uv run --extra notebook python train.py --epochs 100 --batch-size 128 --workers 0 --output runs/reproduction-001\n"
            "uv run --extra notebook python plot_metrics.py --run runs/reproduction-001\n"
            "uv run --extra notebook python evaluate.py --run runs/reproduction-001\n"
            "uv run --extra notebook python predict.py --checkpoint runs/reproduction-001/best.pt --output submission_reproduction.csv\n"
            "uv run --extra notebook python evaluate.py --run runs/reproduction-001 --tta-flip\n"
            "uv run --extra notebook python predict.py --checkpoint runs/reproduction-001/best.pt --tta-flip --output submission_reproduction_flip_tta.csv\n"
            "```\n\n"
            "The source repository excludes raw images, API tokens, prediction CSVs, and large checkpoints."
        ),
        markdown("## 1. Configuration and parameter budget\n\nThe model uses a CIFAR-sized stem and four residual stages. The next cell executes the model code and prints the exact trainable parameter count."),
    ]
    namespace = {"__name__": "__main__", "ROOT": ROOT}
    sources = [
        """import csv, json, sys\nfrom pathlib import Path\nROOT = Path.cwd()\nif not (ROOT / 'models' / 'resnet.py').is_file():\n    ROOT = ROOT.parent\nassert (ROOT / 'models' / 'resnet.py').is_file(), 'Open this notebook from the project root or notebooks folder'\nsys.path.insert(0, str(ROOT))\nfrom models.resnet import CifarResNet, count_parameters\nrun = ROOT / 'results' / 'cifar_resnet_w40'\nconfig = json.loads((run / 'config.json').read_text(encoding='utf-8'))\nmodel = CifarResNet(config['width'])\nparameters = count_parameters(model)\nprint('Architecture: 3x3 stem; 2 residual blocks per stage; widths 40, 80, 160, 320')\nprint(f'Trainable parameters: {parameters:,} / 5,000,000')\nprint(f'Training / validation images: {config[\"train_size\"]:,} / {config[\"validation_size\"]:,}')\nprint(f'Seed: {config[\"seed\"]}; epochs: {config[\"epochs\"]}; batch size: {config[\"batch_size\"]}')\nassert parameters == config['trainable_parameters'] and parameters <= 5_000_000""",
        """import matplotlib.pyplot as plt\nwith (run / 'metrics.csv').open(newline='', encoding='utf-8') as stream:\n    metrics = list(csv.DictReader(stream))\nepochs = [int(row['epoch']) for row in metrics]\ntrain_accuracy = [float(row['train_accuracy']) for row in metrics]\nval_accuracy = [float(row['val_accuracy']) for row in metrics]\nbest_index = max(range(len(metrics)), key=lambda i: val_accuracy[i])\nprint(f'Completed epochs: {len(metrics)} / {config[\"epochs\"]}')\nprint(f'Best held-out validation accuracy: {val_accuracy[best_index]:.2%} at epoch {epochs[best_index]}')\nfig, ax = plt.subplots(figsize=(7, 4))\nax.plot(epochs, train_accuracy, label='train')\nax.plot(epochs, val_accuracy, label='validation')\nax.set(xlabel='Epoch', ylabel='Accuracy', title='Training and validation accuracy')\nax.grid(alpha=.25); ax.legend(); plt.show()""",
        """train_loss = [float(row['train_loss']) for row in metrics]\nval_loss = [float(row['val_loss']) for row in metrics]\nfig, ax = plt.subplots(figsize=(7, 4))\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='validation')\nax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training and validation loss')\nax.grid(alpha=.25); ax.legend(); plt.show()""",
        """with (run / 'per_class.csv').open(newline='', encoding='utf-8') as stream:\n    per_class = list(csv.DictReader(stream))\nprint('Held-out validation accuracy by class:')\nfor row in per_class:\n    print(f'{row[\"class\"]:>10}: {float(row[\"accuracy\"]):6.2%} ({row[\"correct\"]}/{row[\"total\"]})')\nimage = plt.imread(run / 'confusion.png')\nfig, ax = plt.subplots(figsize=(7.2, 5.6))\nax.imshow(image); ax.axis('off'); plt.show()""",
        """with (run / 'kaggle_submissions.csv').open(newline='', encoding='utf-8') as stream:
    submission_records = list(csv.DictReader(stream))
historical_files = {'single_view': 'submission.csv', 'horizontal_flip_tta': 'submission_flip_tta.csv'}
reproduction_files = {
    'single_view': 'submission_reproduction.csv',
    'horizontal_flip_tta': 'submission_reproduction_flip_tta.csv',
}
assert {item['method'] for item in submission_records} == set(historical_files)
sample_path = ROOT / 'data' / 'sample_submission.csv'
if sample_path.is_file():
    with sample_path.open(newline='', encoding='utf-8') as stream:
        sample = list(csv.DictReader(stream))
else:
    sample = None
    print('Competition sample CSV is absent in this checkout; regenerate predictions after obtaining the permitted data.')

def check_predictions(path):
    with path.open(newline='', encoding='utf-8') as stream:
        predictions = list(csv.DictReader(stream))
    assert sample is not None
    assert len(predictions) == len(sample) == 2000
    assert list(predictions[0]) == ['id', 'label']
    assert [row['id'] for row in predictions] == [row['id'] for row in sample]
    assert all(row['label'] in {str(i) for i in range(10)} for row in predictions)
    return len(predictions)

for item in submission_records:
    method = item['method']
    historical_name = item['file_name']
    assert historical_name == historical_files[method]
    historical = ROOT / historical_name
    if historical.is_file() and sample is not None:
        count = check_predictions(historical)
        print(f'Historical {historical_name}: {count} valid predictions; Kaggle submission {item["submission_ref"]}; public score {float(item["public_score"]):.2%}')
    elif not historical.is_file():
        print(f'Historical {historical_name}: omitted from Git; the saved notebook output documents the original check and Kaggle score.')
    else:
        print(f'Historical {historical_name}: present, but sample_submission.csv is needed to recheck its format.')
    fresh = ROOT / reproduction_files[method]
    if fresh.is_file() and sample is not None:
        count = check_predictions(fresh)
        print(f'Fresh reproduction {fresh.name}: {count} valid predictions; no Kaggle score is assigned to this new file.')
    elif not fresh.is_file():
        print(f'Fresh reproduction {fresh.name}: absent; run the prediction command above after obtaining competition data.')
    else:
        print(f'Fresh reproduction {fresh.name}: present, but sample_submission.csv is needed to recheck its format.')""",
        """score_file = ROOT / 'kaggle_score.json'\nif score_file.is_file():\n    score = json.loads(score_file.read_text(encoding='utf-8'))\n    assert score['kind'] in {'public', 'private'} and 0 <= score['accuracy'] <= 1\n    print(f'Kaggle {score[\"kind\"]} leaderboard accuracy: {score[\"accuracy\"]:.2%}')\n    if 'method' in score:\n        print(f'Submission method: {score[\"method\"]}')\nelse:\n    print('Kaggle leaderboard accuracy: pending submission and score')\nprint(f'Trainable parameters: {parameters:,}')""",
    ]
    notes = [
        "## 2. Learning curves\n\nThese cells execute directly from the per-epoch metrics produced by `train.py`. Validation labels were held out before training.",
        "## 3. Validation error analysis\n\nThe confusion matrix is computed by `evaluate.py` from the best checkpoint and the same fixed validation IDs.",
        "## 4. Submission and test result\n\nThe saved outputs below are historical checks performed before publication: the original single-view and flip-TTA CSVs were submitted to Kaggle and received the public scores shown. Prediction CSVs and raw data are omitted from Git. In a fresh checkout, first obtain the permitted competition data and run the commands at the top to create new reproduction CSVs; the cell can validate those new files, but their Kaggle scores are unknown until separately submitted. Validation accuracy is reported only for the held-out training split.",
    ]
    cells.append(code(sources[0], 1, namespace))
    cells.append(markdown(notes[0]))
    cells.append(code(sources[1], 2, namespace))
    cells.append(code(sources[2], 3, namespace))
    cells.append(markdown(notes[1]))
    cells.append(code(sources[3], 4, namespace))
    cells.append(markdown(notes[2]))
    cells.append(code(sources[4], 5, namespace))
    cells.append(code(sources[5], 6, namespace))
    cells.append(markdown("## 5. Public leaderboard comparison\n\nThe second submission changes only inference: it averages logits from each image and its horizontal flip. The method was selected using the held-out validation split. Public leaderboard accuracy is reported separately from validation accuracy; the private score is not yet available."))
    cells.append(code(
        """with (run / 'kaggle_submissions.csv').open(newline='', encoding='utf-8') as stream:\n    submissions = list(csv.DictReader(stream))\nfor item in submissions:\n    print(f'{item[\"method\"]} ({item[\"file_name\"]}): validation {float(item[\"validation_accuracy\"]):.2%}; Kaggle public {float(item[\"public_score\"]):.2%}; submission {item[\"submission_ref\"]}')""",
        7,
        namespace,
    ))
    cells.append(markdown("## Reference\n\nK. He, X. Zhang, S. Ren, and J. Sun, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016. The implementation in this repository was written for this project."))
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (PyTorch)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Executed {len(sources) + 1} cells and saved {OUTPUT}")


if __name__ == "__main__":
    main()
