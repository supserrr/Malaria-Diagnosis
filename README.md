# Malaria Diagnosis — CNN Benchmark Study

A group project comparing five deep-learning approaches for automated malaria detection from blood-smear cell images. Each model is owned by one team member and evaluated under a shared, reproducible experimental framework.

## Problem

Malaria is diagnosed by manually inspecting stained blood smears under a microscope — a process that is slow, requires a trained expert, and is often unavailable in the regions that need it most. This project automates the binary classification of single-cell images as **Parasitized** or **Uninfected**.

## Dataset

**NIH Malaria Cell Images** — downloaded automatically from the official NIH URL:

```
https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
```

- 27,558 PNG cell images, evenly balanced between the two classes
- Split 70 / 15 / 15 into train / val / test (fixed seed = 42, shared across all models)
- Input size: 128 × 128 × 3

## Notebooks

| Notebook | Model | Owner |
|---|---|---|
| [Baseline CNN.ipynb](Baseline%20CNN.ipynb) | Baseline CNN | Dan Paul Dushime |
| [Malaria_Improved_CNN_Apoh.ipynb](Malaria_Improved_CNN_Apoh.ipynb) | Improved CNN | Apoh Prince Eldrige |
| [Malaria_VGG16_Delucie (2).ipynb](<Malaria_VGG16_Delucie%20(2).ipynb>) | VGG16 (transfer learning) | Delucie Rurangwa |
| [Malaria_ResNet50_Sheilla.ipynb](Malaria_ResNet50_Sheilla.ipynb) | ResNet50 (transfer learning) | Sheilla Keza Ruvugabigwi |

## Models

### 1. Baseline CNN — Dan Paul Dushime
Two Conv+Pool blocks, no batch normalisation, no augmentation. Deliberately kept simple to provide an honest lower bound. Seven experiments vary the optimiser, learning rate, filter width, dense width, and dropout one variable at a time.

### 2. Improved CNN — Apoh Prince Eldrige
Extends the baseline with a third convolution block, batch normalisation, dropout, and data augmentation. Experiments are structured as a ladder — depth first, then batch-norm, then augmentation, then regularisation tuning — so each ingredient's contribution is measurable.

### 3. VGG16 — Delucie Rurangwa
VGG16 backbone pre-trained on ImageNet, with the classification head replaced by `GlobalAveragePooling → Dropout → Dense → sigmoid`. Experiments progress from frozen feature extraction through fine-tuning of up to the last 50 layers at learning rates of 1e-4 and 1e-5.

### 4. ResNet50 — Sheilla Keza Ruvugabigwi
ResNet50 backbone with the same custom head. Focuses on training stability — skip connections make the depth viable, but fine-tuning deep networks requires careful learning rate control. Best result: **F1 = 0.9627, accuracy = 96.3%** (E6: fine-tune last 50 layers, LR 1e-5).

## Shared Experimental Framework

Every model runs through 7 experiments under the same conditions:

- **tf.data pipeline** with `cache()` + `prefetch()` (one build, reused across all 35 experiments)
- **EarlyStopping** (`patience=5`, restores best weights) + **ReduceLROnPlateau** (`factor=0.3`, `patience=2`)
- **Mixed precision** (`float16`) for GPU efficiency
- **Metrics logged per run:** accuracy, precision, recall, F1, train/val accuracy, overfit gap, epochs run, wall-clock training time
- **Resumable checkpointing:** results are written to a per-model CSV after each experiment; re-running the notebook skips already-finished experiments

**Augmentation** (when enabled): horizontal+vertical flip, ±10° rotation, ±10% zoom, ±10% translation — applied as Keras layers inside the model.

## Key Results (best experiment per model)

| Model | Best Config | Accuracy | F1 |
|---|---|---|---|
| Baseline CNN | E7 dense256 | — | — |
| Improved CNN | E7 wider256 + aug | — | — |
| VGG16 | E6 fine-tune last 50, LR 1e-5 | 96.18% | 0.9609 |
| ResNet50 | E6 fine-tune last 50, LR 1e-5 | **96.30%** | **0.9627** |

> Baseline and Improved CNN results depend on the run environment; fill in from your `results_Baseline.csv` / `results_ImprovedCNN.csv` after training.

## Running the Notebooks

1. Open any notebook in **Google Colab** with a GPU runtime (*Runtime → Change runtime type → GPU*).
2. Set `USE_DRIVE_CKPT = True` to persist results to Google Drive (survives runtime disconnects).
3. Run all cells — the dataset is downloaded automatically from NIH (~337 MB).
4. To restart cleanly, set `RESET_RESULTS = True` once, then flip it back to `False`.

**Dependencies** (pre-installed on Colab):

```
tensorflow >= 2.x
numpy, pandas, matplotlib, scikit-learn
```

## Repository Structure

```
.
├── Baseline CNN.ipynb                  # Model 1 — Dan Paul Dushime
├── Malaria_Improved_CNN_Apoh.ipynb    # Model 2 — Apoh Prince Eldrige
├── Malaria_VGG16_Delucie (2).ipynb    # Model 3 — Delucie Rurangwa
└── Malaria_ResNet50_Sheilla.ipynb     # Model 4 — Sheilla Keza Ruvugabigwi
```

## Team

| Name | Role |
|---|---|
| Dan Paul Dushime | Team lead, Baseline CNN |
| Apoh Prince Eldrige | Improved CNN |
| Delucie Rurangwa | VGG16 transfer learning |
| Sheilla Keza Ruvugabigwi | ResNet50 transfer learning |
