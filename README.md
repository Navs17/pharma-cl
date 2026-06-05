# Continual Learning for Pharmaceutical Defect Detection

MEng thesis project. A two-stage system:

1. **Stage 1 — Continual binary defect detector.** A ResNet-18 classifier that
   says *defective / not defective*, learning one product at a time
   (pill → capsule) **without forgetting** earlier products
   (*domain-incremental* continual learning). This is the research core.
2. **Stage 2 — VLM defect describer.** When Stage 1 flags an item, a frozen
   CLIP model names the defect *type* zero-shot (no training).

## Project layout

```
pharma-cl/
├── config.py                  # all paths + hyper-parameters live here
├── requirements.txt
├── data/
│   ├── raw/                   # put pill.zip + capsule.zip here
│   └── processed/             # generated: binary train/val/test ImageFolders + manifest.csv
├── outputs/                   # checkpoints, CSV logs, plots
└── src/
    ├── data/prepare_data.py   # MVTec -> binary good/defective splits  (RUN FIRST)
    ├── data/datasets.py       # transforms + ImageFolder loaders
    ├── models/model.py        # ResNet-18 + binary head + freezing
    ├── train_baseline.py      # Stage 0: static baseline / joint upper bound
    ├── continual/run_continual.py  # Stage 1: Avalanche domain-incremental runner
    └── vlm/describe.py        # Stage 2: CLIP zero-shot defect describer
```

## A. Run locally in VS Code (CPU — fine for data prep + small tests)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# 1. Put pill.zip and capsule.zip in data/raw/  then:
python -m src.data.prepare_data --input data/raw --out data/processed

# 2. Sanity-check the pipeline (slow on CPU; use few epochs)
python -m src.train_baseline --product pill --epochs 2
```

Training the CL methods on CPU is too slow — use Colab for that (below). Use VS
Code to write/edit code and run data prep, then push to GitHub and pull on Colab.

## B. Run experiments on Google Colab (free T4 GPU)

1. Upload `pill.zip` and `capsule.zip` to your Google Drive once (e.g.
   `MyDrive/pharma-data/`).
2. Push this repo to GitHub, then in a Colab notebook:

```python
# Cell 1 — GPU + Drive
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi -L

# Cell 2 — code + deps
!git clone https://github.com/<you>/pharma-cl.git
%cd pharma-cl
!pip install -q -r requirements.txt

# Cell 3 — data prep (reads zips from Drive, writes to local fast disk)
!mkdir -p data/raw
!cp "/content/drive/MyDrive/pharma-data/pill.zip" data/raw/
!cp "/content/drive/MyDrive/pharma-data/capsule.zip" data/raw/
!python -m src.data.prepare_data --input data/raw --out data/processed

# Cell 4 — baseline upper bound
!python -m src.train_baseline --product joint

# Cell 5 — the CL comparison (the main result)
!python -m src.continual.run_continual --strategy all

# Cell 6 — Stage 2 VLM describer
!python -m src.vlm.describe --product pill
```

**Checkpoint to Drive** between long runs (free Colab disconnects after ~12 h):
point `config.OUTPUT_DIR` at a Drive path, or copy `outputs/` to Drive after each
strategy.

## Experiment plan

| Step | What | Why |
|------|------|-----|
| 0 | Baseline (`pill`, then `joint`) | proves the pipeline; `joint` = upper bound |
| 1 | `--strategy naive` | demonstrates catastrophic forgetting (motivation) |
| 2 | `ewc`, `lwf`, `replay` | the CL comparison |
| 3 | (optional) add DER++ | stronger hero method |
| 4 | `src/vlm/describe.py` | zero-shot defect-type description |

Report **average stream accuracy** and **StreamForgetting (BWT)** per strategy.

## Data note

All defect images in MVTec live under `test/`; `train/` is good-only. `prepare_data.py`
pools good = train/good + test/good, defective = all other test folders, then makes
a stratified split. The dataset is small (~440 pill / ~350 capsule images) — this is
a *few-shot-style* continual setting; lean on augmentation and say so in the thesis.
MVTec AD is CC BY-NC-SA 4.0 (Bergmann et al., CVPR 2019) — cite it.
