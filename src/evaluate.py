"""
Evaluation: how good is the model on data it has never seen?

Metrics:
  - accuracy  : share of correct predictions
  - macro-F1  : F1 per class, then the plain average (every class counts equally)
  - classification report : precision / recall / F1 for every class
  - confusion matrix      : which class gets confused with which (saved as PNG)

Run on its own (evaluates the saved model in models/model):
    python -m src.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")  # draw plots to files, no window needed (works on servers/CI)
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)

from src.config import LABELS, REPORTS_DIR


@torch.no_grad()  # no gradients needed for evaluation -> faster, less memory
def predict_loader(model, loader, device):
    """Run the model over a DataLoader. Returns (true labels, predictions, mean loss)."""
    model.eval()  # switch off dropout
    y_true, y_pred, total_loss = [], [], 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)                     # passing 'labels' -> model also returns the loss
        total_loss += out.loss.item()
        y_pred += out.logits.argmax(dim=-1).tolist()  # highest score = predicted class
        y_true += batch["labels"].tolist()
    return y_true, y_pred, total_loss / len(loader)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def save_confusion_matrix(y_true, y_pred, path):
    """Rows = true class, columns = predicted class. Perfect model = only the diagonal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def full_report(y_true, y_pred) -> tuple[dict, str]:
    """Metrics dict + human-readable per-class report."""
    metrics = compute_metrics(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    return metrics, report


def log_test_results(y_true, y_pred, test_loss, mlflow=None) -> dict:
    """Print, save to reports/ and (optionally) log to MLflow."""
    metrics, report = full_report(y_true, y_pred)
    metrics["loss"] = test_loss
    cm_path = save_confusion_matrix(y_true, y_pred, REPORTS_DIR / "confusion_matrix.png")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / "classification_report.txt").write_text(report)

    print("\n=== TEST RESULTS ===")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    print(f"macro-F1 : {metrics['macro_f1']:.4f}")
    print(report)

    if mlflow is not None:
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(str(cm_path))
        mlflow.log_text(report, "classification_report.txt")
    return metrics


def evaluate_saved_model(test_size: int = 1000, max_len: int = 64):
    """Evaluate models/model on the test set WITHOUT training (the 'skip training' path)."""
    # Imported here (not at the top) to avoid a circular import with train.py
    import mlflow
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.config import DATASET_ID, MODEL_DIR, TrainConfig
    from src.data import make_splits, tokenize_splits
    from src.train import get_device, setup_mlflow

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"No model in {MODEL_DIR}. Train first: python -m src.train")

    device = get_device()
    cfg = TrainConfig(train_size=10, val_size=10, test_size=test_size, max_len=max_len)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    splits = make_splits(load_dataset(DATASET_ID), cfg)
    test = tokenize_splits(splits, tokenizer, max_len)["test"]
    y_true, y_pred, loss = predict_loader(model, DataLoader(test, batch_size=64), device)

    setup_mlflow()
    with mlflow.start_run(run_name="evaluate-only"):
        mlflow.log_params({"mode": "evaluate_only", "test_size": test_size, "max_len": max_len})
        return log_test_results(y_true, y_pred, loss, mlflow)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate the saved model on the test set")
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--max-len", type=int, default=64)
    a = p.parse_args()
    evaluate_saved_model(a.test_size, a.max_len)
