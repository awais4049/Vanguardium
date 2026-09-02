"""
Vanguardium - CNN Character Tokenizer (shared by training and inference)

Defines the fixed character vocabulary and text->sequence encoding used by
the CNN's raw-text pipeline. This module is the single source of truth for
tokenization - train_cnn.py and the future FastAPI inference endpoint must
both import from here, never reimplement this logic, or training/inference
will silently disagree on how text is encoded.

Vocabulary: fixed printable ASCII (0x20-0x7E, 95 chars), chosen over a
data-driven vocab specifically to avoid the CNN's embedding layer learning
a shortcut off rare characters that happen to appear in only one class due
to how synthetic attack payloads were generated (rather than genuine attack
semantics). Verified 0/3809 rows in http_features.csv contain non-ASCII
characters, so this has zero UNK coverage loss on the current dataset.

PAD = 0 (padding token, also used for truncated positions)
UNK = 1 (any character outside the fixed charset - should not occur on
         this dataset, but kept for robustness against future/live inputs,
         e.g. real attacker traffic containing unicode/emoji)
Printable ASCII chars start at index 2.
"""

import json
import os

import numpy as np

MAX_LEN = 256
PAD_IDX = 0
UNK_IDX = 1

# Fixed charset: printable ASCII, space (0x20) through tilde (0x7E) inclusive.
_CHARSET = [chr(i) for i in range(0x20, 0x7F)]
CHAR_TO_IDX = {ch: i + 2 for i, ch in enumerate(_CHARSET)}  # reserve 0,1
IDX_TO_CHAR = {i + 2: ch for i, ch in enumerate(_CHARSET)}
VOCAB_SIZE = len(_CHARSET) + 2  # +PAD +UNK

VOCAB_PATH = os.path.join("data", "models", "cnn_vocab.json")


def text_to_sequence(text: str, max_len: int = MAX_LEN) -> list:
    """Encode a single string into a fixed-length list of char indices.
    Truncates from the end if longer than max_len, pads with PAD_IDX
    (0) at the end if shorter."""
    text = text if isinstance(text, str) else str(text)
    seq = [CHAR_TO_IDX.get(ch, UNK_IDX) for ch in text[:max_len]]
    if len(seq) < max_len:
        seq = seq + [PAD_IDX] * (max_len - len(seq))
    return seq


def texts_to_matrix(texts, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode an iterable of strings into an (n, max_len) int array,
    suitable as direct input to a Keras Embedding layer."""
    return np.array([text_to_sequence(t, max_len) for t in texts], dtype=np.int32)


def save_vocab(path: str = VOCAB_PATH) -> None:
    """Persist the vocab definition for traceability / so inference code
    can verify it matches what a given model was trained with."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "max_len": MAX_LEN,
                "pad_idx": PAD_IDX,
                "unk_idx": UNK_IDX,
                "vocab_size": VOCAB_SIZE,
                "charset": _CHARSET,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    save_vocab()
    print(f"Vocab size: {VOCAB_SIZE} (charset={len(_CHARSET)} + PAD + UNK)")
    print(f"Max length: {MAX_LEN}")
    print(f"Saved vocab definition to {VOCAB_PATH}")