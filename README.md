# ICU Remaining Length-of-Stay Prediction

Multimodal deep learning pipeline for predicting a patient's **remaining time-to-discharge** (RLOS) from the ICU using three data modalities from MIMIC-IV:

|Modality|Source table|Model branch|
|---|---|---|
|Time series|`MIMIC-IV-time_series`|LSTM|
|Demographics|`MIMIC-IV-static`|Linear encoder|
|Radiology notes|`MIMIC-IV-text`|Bio_ClinicalBERT → Linear encoder|

---

## Project structure

```
project/
├── main.py                          ← pipeline entry point
├── requirements.txt
├── data/
│   └── origin/                      ← place the three raw CSVs here
│       ├── MIMIC-IV-static(Group Assignment).csv
│       ├── MIMIC-IV-text(Group Assignment).csv
│       └── MIMIC-IV-time_series(Group Assignment).csv
└── src/
    ├── data/
    │   ├── loader.py                ← load & filter (icu_death_flag == 0)
    │   ├── splitter.py              ← 8:1:1 stay_id split
    │   ├── static_preprocessor.py  ← StaticPreprocessor
    │   ├── ts_preprocessor.py      ← TimeSeriesPreprocessor
    │   ├── text_preprocessor.py    ← TextPreprocessor (ClinicalBERT + cache)
    │   └── dataset.py              ← ICUDataset + build_dataloaders
    ├── models/
    │   └── multimodal.py           ← MultimodalICUModel
    ├── training/
    │   └── trainer.py              ← Trainer (early stopping, checkpointing)
    └── utils/
        ├── constants.py            ← all config, clip ranges, mappings
        ├── metrics.py              ← MAE, RMSE, MedAE, R²
        └── persistence.py          ← save / load fitted preprocessors
```

---

## Setup

You can choose to use Conda 🛠️ or uv 🛠️ to manage your Python environment. (Python Version 3.11 ✅)

- Option A: Conda

```bash

# Create and activate the environment

conda create -n sph6004_env python=3.11

conda activate sph6004_env

# Install dependencies

pip install -r requirements.txt

```

- Option B: UV (recommended, faster)

```bash

# Create a virtual environment and synchronize dependencies

uv venv -p 3.11

source .venv/bin/activate

uv pip install -r requirements.txt

```
---

## Run

**Full pipeline** (fit preprocessors + train + evaluate):

```bash
python main.py
```

**Resume** (skip fitting; reuse saved preprocessors from a previous run):

```bash
python main.py --resume
```

**Common options:**

```
--epochs       Max training epochs          (default: 100)
--batch_size   Mini-batch size              (default: 64)
--lr           Learning rate               (default: 1e-3)
--patience     Early-stopping patience     (default: 10)
--ckpt_dir     Checkpoint directory        (default: checkpoints/)
```

---

## Data pipeline details
### Quick View
Since the original MIMIC-IV dataset is extremely large, a sampling script is provided to facilitate quick code debugging and feature inspection.

1. Ensure the original data is stored in the `data/origin/` directory.
2. Run the script to extract the first 2000 lines of each file:
    ```bash
    python data/quick_viewer.py
    ```
3. Once finished, the sampled lightweight files will be generated in the `data/processed/` directory.

### Cohort filter

Only ICU stays where the patient was **discharged alive** (`icu_death_flag == 0`) are kept. Dead patients are excluded because the modelling target is time-to-discharge, not time-to-death.

### Train / val / test split

Shuffled at the **stay_id level** (8 : 1 : 1). No patient appears in more than one split.

### Time-series preprocessing

Missing rates are computed on training data only. Each feature is handled according to its missing rate:

|Rate|Strategy|
|---|---|
|< 15 %|Clip → LOCF → head-fill with train median + observation mask|
|15 – 90 %|Same as above|
|≥ 90 %|`ever_measured[t]` binary (was this feature ever observed up to time t?)|

### Windowing

For each time step `t`, a **fixed-length window of W = 24 hours** is fed to the LSTM. Steps before the first recorded hour are zero-padded; `ts_length` tells the LSTM where real data ends.

**Label:** `RLOS(t) = icu_los_hours − elapsed_hours(t)`, clamped to ≥ 0.

### Text (radiology notes)

Notes are encoded once with **Bio_ClinicalBERT** and cached to `data/cache/text_embeddings.pkl`. At each time step `t`, only notes whose inferred timestamp ≤ `t` are used (no future leakage). Multiple valid notes are combined with **time-aware weighted pooling** (`weight ∝ exp(−Δt)`).

When no notes are yet available, a learnable `NO_NOTE` embedding is used instead of silence.

### Static features

Demographic features are broadcast to every windowed sample from the same stay (global context). The age `StandardScaler` is fit on training data only.

---

## Model

```
Time series   →  LSTM (64 hidden, 2 layers)     →  h_ts     (64,)
Static        →  Linear(22→32) + ReLU           →  h_static (32,)
Text          →  [NO_NOTE swap] + Linear(768→64) →  h_text   (64,)
                                                     │ concat
                                            Linear(160→128) + ReLU
                                            Linear(128→1)
                                            ↓
                                        RLOS prediction (hours)
```

---

## Extending the code

|What you want to change|File to edit|
|---|---|
|Feature selection / cleaning|`src/data/static_preprocessor.py` or `src/data/ts_preprocessor.py`|
|Clip ranges for clinical features|`src/utils/constants.py` → `CLINICAL_CLIP_RANGES`|
|Categorical mappings (race, insurance …)|`src/utils/constants.py`|
|Window size|`src/utils/constants.py` → `WINDOW_SIZE`|
|BERT model|`src/utils/constants.py` → `BERT_MODEL_NAME`|
|Model architecture|`src/models/multimodal.py`|
|Training hyperparameters|CLI flags or `src/training/trainer.py`|
|Evaluation metrics|`src/utils/metrics.py`|
