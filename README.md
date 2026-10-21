# Audio Genre Classification & Sounds-Like Similarity Search

An audio machine learning project for genre classification and content-based music retrieval using FMA-Small.

## Status

Repository scaffold only. Python modules and the dependency file are empty placeholders; notebooks are valid empty Jupyter notebooks. Training, evaluation, and the application are not implemented yet.

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
requirements.txt             Dependencies (to be defined)
```

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

Document data acquisition and licensing, define the shared environment, explore the dataset, and persist the evaluation splits before implementing the baseline.

## License

Project code is distributed under the MIT license in `LICENSE`. Dataset and audio licenses are separate.
