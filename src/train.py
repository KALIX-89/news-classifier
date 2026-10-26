"""
Fine-tune DistilBERT on AG News with a hand-written PyTorch training loop.

Examples:
    python -m src.train --preset fast                 # CPU-friendly (~2k samples, 1 epoch)
    python -m src.train --preset full                 # GPU (20k samples, 2 epochs)
    python -m src.train --preset fast --lr 2e-5       # override single values
    python -m src.train --skip-if-exists              # model exists? -> only evaluate
"""
import argparse
import random
import time
from dataclasses import asdict

import mlflow
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.config import (LABELS, MLFLOW_ARTIFACT_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI,
                        MODEL_DIR, MODEL_NAME, PRESETS, REGISTERED_MODEL_NAME, TrainConfig)
from src.data import get_dataloaders
from src.evaluate import compute_metrics, log_test_results, predict_loader


def get_device() -> torch.device:
    """Use the fastest hardware available: NVIDIA GPU > Apple Silicon GPU > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    """Same seed -> same random numbers -> reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_mlflow():
    """Point MLflow at our local store and select (or create) the experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_DIR)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def train_one_epoch(model, loader, optimizer, scheduler, device) -> float:
    """One pass over the training data. Returns the average training loss."""
    model.train()  # switch on dropout
    total_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)     # 1) forward pass: predictions + loss (cross-entropy)
        loss = outputs.loss

        optimizer.zero_grad()        # 2) clear old gradients
        loss.backward()              # 3) backpropagation: compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid huge updates
        optimizer.step()             # 4) update the weights
        scheduler.step()             # 5) adjust the learning rate

        total_loss += loss.item()
        if step % 50 == 0 or step == len(loader):
            print(f"  step {step}/{len(loader)}  loss {total_loss / step:.4f}")
    return total_loss / len(loader)


def train(cfg: TrainConfig, run_name: str | None = None, register: bool = False) -> dict:
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    loaders, tokenizer = get_dataloaders(cfg)

    # Load PRE-TRAINED DistilBERT and add a fresh classification head with 4 outputs.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(loaders["train"]) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),  # LR goes 0 -> lr ...
        num_training_steps=total_steps,                          # ... then linearly back to 0
    )

    setup_mlflow()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**asdict(cfg), "model_name": MODEL_NAME, "device": str(device)})
        start = time.time()

        for epoch in range(1, cfg.epochs + 1):
            print(f"\nEpoch {epoch}/{cfg.epochs}")
            train_loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, device)

            # Validation after every epoch: is the model still improving on unseen data?
            y_true, y_pred, val_loss = predict_loader(model, loaders["validation"], device)
            val = compute_metrics(y_true, y_pred)
            print(f"  train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                  f"val_acc {val['accuracy']:.4f} | val_f1 {val['macro_f1']:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]},
                               step=epoch)

        mlflow.log_metric("train_time_sec", time.time() - start)

        # Final evaluation on the TEST set (only once, at the very end).
        y_true, y_pred, test_loss = predict_loader(model, loaders["test"], device)
        metrics = log_test_results(y_true, y_pred, test_loss, mlflow)

        # Save model + tokenizer so predict.py / the API can load them.
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"\nModel saved to {MODEL_DIR}")

        # Log the model to MLflow (and optionally register it in the Model Registry).
        mlflow.log_artifacts(str(MODEL_DIR), artifact_path="model")
        if register:
            # Point the registry at the folder of model files we just logged.
            model_uri = f"{mlflow.active_run().info.artifact_uri}/model"
            mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
            print(f"Registered model '{REGISTERED_MODEL_NAME}' in the MLflow Model Registry")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on AG News")
    p.add_argument("--preset", choices=PRESETS.keys(), default="fast")
    # Optional overrides (None = use the preset value)
    p.add_argument("--train-size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--max-len", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--run-name", type=str, help="name shown in the MLflow UI")
    p.add_argument("--register", action="store_true", help="register model in MLflow Model Registry")
    p.add_argument("--skip-if-exists", action="store_true",
                   help="if models/model already exists, skip training and only evaluate")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(**PRESETS[args.preset])
    for name in ["train_size", "epochs", "max_len", "batch_size", "lr"]:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    if args.skip_if_exists and (MODEL_DIR / "config.json").exists():
        from src.evaluate import evaluate_saved_model
        print(f"Model found in {MODEL_DIR} -> skipping training, evaluating only.")
        evaluate_saved_model(test_size=cfg.test_size, max_len=cfg.max_len)
        return

    run_name = args.run_name or f"{args.preset}-lr{cfg.lr:g}"
    print(f"Config: {cfg}")
    train(cfg, run_name=run_name, register=args.register)


if __name__ == "__main__":
    main()
