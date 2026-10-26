"""
All hyperparameters and paths in ONE place.

Why? If you want to change e.g. the learning rate, you change it here
(or via a CLI flag) instead of hunting through several files.
"""
from dataclasses import dataclass
from pathlib import Path

# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parent.parent   # project root folder
MODEL_DIR = ROOT_DIR / "models" / "model"            # where the fine-tuned model is saved
REPORTS_DIR = ROOT_DIR / "reports"                   # confusion matrix PNGs etc.
# MLflow: run metadata (params, metrics) in a local SQLite file, artifacts
# (model files, PNGs) in ./mlruns. SQLite also enables the Model Registry.
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_DIR = (ROOT_DIR / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "news-classifier"
REGISTERED_MODEL_NAME = "news-classifier"

# ---------- Data ----------
DATASET_ID = "fancyzhx/ag_news"   # Hugging Face dataset id
LABELS = ["World", "Sports", "Business", "Sci/Tech"]  # index 0..3, same order as the dataset
MODEL_NAME = "distilbert-base-uncased"  # pre-trained model we fine-tune


@dataclass
class TrainConfig:
    """Training settings. Defaults = FAST preset (runs on a CPU)."""
    train_size: int = 2000      # how many training examples we use
    val_size: int = 500         # validation examples (used after every epoch)
    test_size: int = 1000       # test examples (used ONCE at the very end)
    epochs: int = 1             # full passes over the training data
    max_len: int = 64           # max tokens per headline (longer = cut off)
    batch_size: int = 16        # examples per gradient update
    lr: float = 5e-5            # learning rate (step size of the optimizer)
    weight_decay: float = 0.01  # small penalty on large weights (regularisation)
    warmup_ratio: float = 0.1   # first 10% of steps: LR rises slowly from 0
    seed: int = 42              # same seed -> same split & same results


# Presets: a quick one for CPUs, a bigger one for GPUs.
PRESETS = {
    "fast": dict(train_size=2000, val_size=500, test_size=1000, epochs=1, max_len=64),
    "full": dict(train_size=20000, val_size=2000, test_size=7600, epochs=2, max_len=128),
}
