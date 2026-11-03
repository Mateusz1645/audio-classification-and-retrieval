# Audio Genre Classification & "Sounds-Like" Similarity Search

An end-to-end audio machine learning project for music genre classification and content-based audio retrieval using the **FMA-Small** dataset.

---

## Status

Project infrastructure is set up with pinned dependencies, Ruff linting/formatting checks, and automated pytest workflows. Data acquisition and preprocessing pipelines are implemented, and modeling experiments are in active development.

---

## Deliverables & Course Coverage

This project satisfies the requirements for the **Analysis of Unstructured Data** course:

- **Unstructured Data Pipeline:** Raw audio files resampled and converted into 2D Log-Mel Spectrograms ($22.05\text{ kHz}$ mono, 3-second chunks, 128 Mel bins).
- **Three Modeling Approaches:**
  1. **Classical ML Baseline:** Tabular audio features (20 MFCCs, Spectral Centroid, Zero-Crossing Rate) evaluated with a Random Forest classifier.
  2. **Custom Audio CNN:** A 4-block 2D Convolutional Neural Network trained on Log-Mel Spectrograms.
  3. **Fine-Tuned ResNet-18:** Pretrained ResNet-18 backbone adapted for 1-channel spectrograms to learn deep acoustic representations.
- **Content-Based Similarity Search:** 512-dimensional latent embedding extraction indexed with **FAISS** (Inner Product / Cosine Distance) for real-time nearest-neighbor retrieval.
- **Experiment Tracking:** Logging all hyperparameters, metrics (Macro-F1, per-class accuracy, loss curves), and model checkpoints using **MLflow**.
- **Interpretability & Error Analysis:** Visual explanations using **Grad-CAM heatmaps** on spectrograms, $t$-SNE/UMAP projections of FAISS embeddings, and confusion matrix analysis.
- **Interactive Application:** A Streamlit web interface allowing users to upload audio clips, visualize spectrograms, predict genres, and play recommended "sounds-like" tracks.

---

## Repository Structure

```text
audio-classification-and-retrieval/
├── app/                         # Streamlit web application
├── configs/                     # Experiment configurations (YAML/JSON)
├── data/                        # Local datasets (excluded from Git)
│   ├── raw/                     # Downloaded fma_small/ and fma_metadata/
│   ├── interim/                 # Track splits and metadata DataFrames
│   └── processed/               # Extracted features and spectrogram tensors
├── notebooks/                   # Sequential project notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_features_and_baseline.ipynb
│   ├── 03_deep_models.ipynb
│   ├── 04_evaluation_and_interpretability.ipynb
│   └── 05_final_report.ipynb
├── src/                         # Reusable project package
│   └── audio_classification/
│       ├── __init__.py
│       ├── data.py              # Metadata parsing, splits, PyTorch Dataset
│       ├── download_data.py     # Reproducible dataset downloader
│       ├── features.py          # Classical features & Log-Mel Spectrograms
│       ├── models.py            # CNN architectures & feature extractors
│       └── utils.py             # Seed setting and helper utilities
├── tests/                       # Automated unit and integration tests
├── models/                      # Local model checkpoints (excluded from Git)
├── reports/                     # Figures and final presentation slides
│   └── figures/
├── pyproject.toml               # Package metadata, Ruff, and Pytest configs
├── requirements.txt             # Primary dependencies
├── requirements-dev.txt         # Development tools
├── README.md
└── .gitignore

```

---

## Development Setup

This project requires **Python 3.11.9**.

### 1. Create Virtual Environment & Install Dependencies

Using `uv` (Recommended for fast resolution):

```bash
# Verify Python version
python3.11 --version

# Create and activate environment
uv venv --python 3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install requirements
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .

```

Using standard `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .

```

### 2. Register Jupyter Kernel

To select your `.venv` environment inside JupyterLab or VS Code:

```bash
python -m ipykernel install --user --name=audio-retrieval-venv --display-name "Python (.venv)"
python -m jupyterlab

```

---

## Data Acquisition & Reproducibility

The project uses **FMA-Small** (8,000 tracks, 8 genres, 30s clips, ~7.2 GB).

### Downloading the Dataset

Run the data acquisition script from the project root:

```bash
python src/audio_classification/download_data.py

```

This fetches `fma_small.zip` and `fma_metadata.zip` from Switch Cloud servers and extracts them into `data/raw/`.

### Split & Leakage Protection Protocol

To prevent **artist leakage**, models are evaluated using FMA's official **artist-filtered train/validation/test splits**. Tracks from the same artist/album are strictly kept within the same split partition.

### Experiment Tracking

Launch MLflow locally to view tracked runs, hyperparameters, and metric curves:

```bash
mlflow ui

```

Navigate to `http://127.0.0.1:5000` in your browser.

---

## Quality Checks

Run local linting, formatting, and unit tests prior to committing code:

```bash
# Code checks and linting
ruff check .
ruff format --check .

# Automated testing
pytest

```

---

## License

Project code is distributed under the **MIT License** in `LICENSE`. Dataset files and audio clips are subject to their respective Creative Commons licenses as specified in the FMA repository.
