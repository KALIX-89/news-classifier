"""
Tests for data cleaning and splitting. They use a tiny fake dataset built in
memory, so no download is needed (fast + works offline in CI).
"""
import pytest
from datasets import Dataset, DatasetDict

from src.config import TrainConfig
from src.data import clean_text, make_splits


@pytest.fixture
def fake_raw() -> DatasetDict:
    """Looks like AG News (columns 'text' and 'label'), but only 100 + 20 rows."""
    train = Dataset.from_dict({"text": [f"train headline {i}" for i in range(100)],
                               "label": [i % 4 for i in range(100)]})
    test = Dataset.from_dict({"text": [f"test headline {i}" for i in range(20)],
                              "label": [i % 4 for i in range(20)]})
    return DatasetDict(train=train, test=test)


# ---------- clean_text ----------
def test_clean_text_fixes_html_and_spaces():
    assert clean_text("  AT&amp;T   buys\\stake ") == "AT&T buys stake"


def test_clean_text_fixes_broken_apostrophe():
    assert clean_text("Google#39;s profit") == "Google's profit"


def test_clean_text_keeps_normal_text():
    assert clean_text("Bayern wins 2-0") == "Bayern wins 2-0"


# ---------- make_splits ----------
def test_split_sizes(fake_raw):
    cfg = TrainConfig(train_size=60, val_size=20, test_size=10)
    splits = make_splits(fake_raw, cfg)
    assert len(splits["train"]) == 60
    assert len(splits["validation"]) == 20
    assert len(splits["test"]) == 10


def test_train_and_validation_do_not_overlap(fake_raw):
    """Data leakage check: no example may be in train AND validation."""
    splits = make_splits(fake_raw, TrainConfig(train_size=60, val_size=40, test_size=10))
    assert set(splits["train"]["text"]).isdisjoint(splits["validation"]["text"])


def test_test_set_comes_from_original_test_split(fake_raw):
    splits = make_splits(fake_raw, TrainConfig(train_size=10, val_size=10, test_size=5))
    assert all(t.startswith("test") for t in splits["test"]["text"])


def test_split_is_reproducible(fake_raw):
    """Same seed -> exactly the same split (important for comparing experiments)."""
    cfg = TrainConfig(train_size=30, val_size=10, test_size=5, seed=7)
    assert make_splits(fake_raw, cfg)["train"]["text"] == make_splits(fake_raw, cfg)["train"]["text"]


def test_too_large_split_raises(fake_raw):
    with pytest.raises(ValueError):
        make_splits(fake_raw, TrainConfig(train_size=90, val_size=20))
