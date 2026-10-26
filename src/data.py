"""
Data pipeline: download -> clean -> split -> tokenize -> DataLoaders.

AG News has 120,000 training and 7,600 test articles in 4 classes.
We only use a (configurable) subset so training stays fast.
"""
import html
import re

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import DATASET_ID, MODEL_NAME, TrainConfig


def clean_text(text: str) -> str:
    """Light cleaning. AG News contains HTML leftovers like '&lt;' or '#39;'
    and backslashes. Transformers don't need heavy preprocessing
    (no stemming, no stop-word removal), so we only fix obvious noise."""
    text = html.unescape(text)                      # '&amp;' -> '&'
    text = re.sub(r"#39;", "'", text)               # broken apostrophes
    text = text.replace("\\", " ")                  # stray backslashes
    text = re.sub(r"\s+", " ", text)                # many spaces -> one
    return text.strip()


def make_splits(raw: DatasetDict, cfg: TrainConfig) -> DatasetDict:
    """Create train / validation / test sets.

    - train + validation both come from the ORIGINAL training split
      (shuffled, then cut into two non-overlapping parts).
    - test comes from the ORIGINAL test split -> never seen during training.
    """
    shuffled = raw["train"].shuffle(seed=cfg.seed)
    needed = cfg.train_size + cfg.val_size
    if needed > len(shuffled):
        raise ValueError(f"Asked for {needed} train+val rows, only {len(shuffled)} available")

    train = shuffled.select(range(cfg.train_size))
    val = shuffled.select(range(cfg.train_size, needed))

    test_n = min(cfg.test_size, len(raw["test"]))
    test = raw["test"].shuffle(seed=cfg.seed).select(range(test_n))

    splits = DatasetDict(train=train, validation=val, test=test)
    return splits.map(lambda row: {"text": clean_text(row["text"])})


def tokenize_splits(splits: DatasetDict, tokenizer, max_len: int) -> DatasetDict:
    """Turn text into token ids the model understands.

    padding='max_length' -> every example has exactly max_len tokens
    truncation=True      -> longer texts are cut off
    """
    def _tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_len)

    tokenized = splits.map(_tok, batched=True)
    tokenized = tokenized.rename_column("label", "labels")   # name the model expects
    # Return PyTorch tensors and only the columns the model needs.
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def get_dataloaders(cfg: TrainConfig):
    """Full pipeline used by train.py. Returns (loaders dict, tokenizer)."""
    raw = load_dataset(DATASET_ID)                 # downloads once, then cached
    splits = make_splits(raw, cfg)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = tokenize_splits(splits, tokenizer, cfg.max_len)

    loaders = {
        "train": DataLoader(tokenized["train"], batch_size=cfg.batch_size, shuffle=True),
        "validation": DataLoader(tokenized["validation"], batch_size=cfg.batch_size * 2),
        "test": DataLoader(tokenized["test"], batch_size=cfg.batch_size * 2),
    }
    return loaders, tokenizer


if __name__ == "__main__":
    # Quick sanity check: python -m src.data
    cfg = TrainConfig()
    raw = load_dataset(DATASET_ID)
    print(raw)
    splits = make_splits(raw, cfg)
    print({k: len(v) for k, v in splits.items()})
    print("Example:", splits["train"][0])
