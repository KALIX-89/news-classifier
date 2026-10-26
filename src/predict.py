"""
Inference: load the saved model once, then classify any text.

    python -m src.predict "Apple unveils new iPhone with faster chip"
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODEL_DIR


class NewsClassifier:
    """Wraps tokenizer + model so callers only deal with plain text."""

    def __init__(self, model_dir: Path | str = MODEL_DIR, max_len: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()                         # inference mode (no dropout)
        self.max_len = max_len
        self.id2label = self.model.config.id2label  # {0: "World", 1: "Sports", ...}

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        logits = self.model(**inputs).logits          # raw scores, one per class
        probs = torch.softmax(logits, dim=-1)[0]      # scores -> probabilities (sum = 1)
        idx = int(probs.argmax())
        return {"label": self.id2label[idx], "confidence": round(float(probs[idx]), 4)}


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "NASA launches new telescope to study distant galaxies"
    print(text, "->", NewsClassifier().predict(text))
