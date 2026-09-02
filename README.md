# Vanguardium

AI-powered cybersecurity middleware proxy and autonomous defense platform.
Final Year Project — BS Data Science, COMSATS University Islamabad.
Supervisor: Mr. Rizwan Rashid.

## Training Environments

This project splits ML training across two environments depending on
hardware requirements. Do not assume `backend/requirements.txt` alone
reproduces everything needed to train every model.

### Local venv (`backend/venv`)

Used for: data capture (mitmproxy), feature extraction, backend API
(FastAPI), and classical ML training (XGBoost).

Install with:
```
pip install -r backend/requirements.txt
```

This does **not** include TensorFlow/Keras. Training scripts that import
`tensorflow` (e.g. `scripts/train_bilstm.py`) will not run in this venv
without installing it separately.

### Google Colab (GPU runtime)

Used for: deep learning model training (BiLSTM, and CNN once added).
No dedicated local GPU is available (integrated graphics only), so these
models are trained on Colab notebooks using a GPU runtime, and the
resulting artifacts (model weights, scalers, encoders, results JSON) are
committed back to `data/models/` and pulled into the local venv for
inference/integration only — not retrained locally.

Colab-side dependencies (not pinned yet, install as needed):
- `tensorflow`
- `scikit-learn`
- `pandas`, `numpy`, `joblib`

### Why the split

Local hardware (integrated GPU) is not viable for training BiLSTM/CNN in
reasonable time. Classical ML (XGBoost) and non-training tasks run fine
locally. This split is intentional, not a gap — but it means "run it
locally" is only valid for scripts that don't import `tensorflow`.

## Model Status (M2 — AI Threat Detection)

| Model | Status | Environment | Artifacts |
|---|---|---|---|
| XGBoost | Trained, committed (`621e844`) | Local venv | `data/models/` |
| BiLSTM | Trained, committed (`6e2e64a`) | Colab (GPU) | `data/models/` |
| CNN | Not started | Colab (GPU), planned | — |

See project scope document and session notes for full module breakdown
(M1–M8).
