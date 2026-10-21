# Audio Genre Classification & Sounds-Like Similarity Search

An audio machine learning project for genre classification and content-based music retrieval using FMA-Small.

## Status

Project setup is in place: pinned direct dependencies, Ruff checks, and a pytest package import test. Python modules and notebooks remain placeholders. Training, evaluation, and the application are not implemented yet.

## Planned application

- Upload an audio recording.
- Predict its music genre.
- Display its spectrogram.
- Retrieve similar tracks using audio embeddings and FAISS.
- Play the query and retrieved audio in a Streamlit interface.

## Planned approaches

1. Classical baseline: MFCC features with a classical classifier.
2. A small CNN trained on log-Mel spectrograms.
3. A pretrained ResNet-18 fine-tuned on log-Mel spectrograms.

Compare classification results using macro-F1, per-class metrics, and confusion matrices. Evaluate retrieval separately and measure inference time. Include embedding visualizations, prediction explanations, and error analysis.

## Repository structure

```text
app/                         Streamlit application
configs/                     Experiment configuration
data/                        Local datasets (excluded from Git)
data/raw/                    Original audio and metadata
data/interim/                Intermediate audio processing outputs
data/processed/              Features, spectrograms, and embeddings
notebooks/                   Exploration, experiments, and final report
src/audio_classification/    Reusable project code
tests/                       Automated tests
models/                      Local model artifacts (excluded from Git)
reports/figures/              Evaluation figures
requirements.txt             Audio processing and data exploration dependencies
requirements-dev.txt         Development and notebook tools
pyproject.toml               Ruff and pytest configuration
.github/workflows/ci.yml     Pull request checks
```

## Development setup

Use **Python 3.11.9**. From the repository root, create a virtual environment using that Python installation.

Windows PowerShell:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python.exe -m pip check
```

Confirm that the version command reports `3.11.9` before creating the environment.

Linux / macOS:

```bash
python3.11 --version
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pip check
```

Again, verify that the selected interpreter is Python 3.11.9. Direct dependencies are pinned; transitive dependencies are not locked yet.

To open the notebooks on Windows:

```powershell
.venv\Scripts\python.exe -m jupyterlab
```

On Linux / macOS, use `.venv/bin/python -m jupyterlab`.

## Quality checks

Run the same checks used by CI from the repository root. On Windows:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m pytest
```

On Linux / macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

GitHub Actions runs these checks on pull requests, pushes to `main` and `develop`, and manual workflow runs. Ruff checks code, import ordering, and formatting without modifying files. The current test only verifies that the project package can be imported; functional tests will accompany implementation.

To block merging when checks fail, configure branch protection on GitHub to require the `Ruff` and `Tests` checks. The workflow alone does not enforce merge restrictions.

## Data and reproducibility

FMA-Small is the planned dataset. Download instructions, source and license details will be added when data acquisition is implemented. Audio recordings, generated features, and model checkpoints must not be committed to Git.

Verify the official split and artist separation before training. Keep all segments of a track in the same partition. Record fixed train/validation/test splits, random seeds, environment versions, and preprocessing settings. Fit preprocessing on training data only.

Track experiments with MLflow or Weights & Biases; the choice is pending. Document how to download or reproduce model artifacts before publishing a runnable demo.

## Course deliverables

- Raw audio exploration and preprocessing.
- Three modeling approaches; confirm course-block coverage with the instructor.
- Model comparison, interpretability, and error analysis.
- Reproducible experiments and experiment tracking.
- A final notebook that runs from top to bottom.
- Team contribution history and reviewed pull requests.
- Presentation slides and a rehearsed demo, with materials submitted 48 hours before the presentation.

## Initial milestone

Verify the shared environment on each team member's machine, document data acquisition and licensing, explore the dataset, and persist the evaluation splits before implementing the baseline.

## License

Project code is distributed under the MIT license in `LICENSE`. Dataset and audio licenses are separate.
