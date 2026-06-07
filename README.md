# Malaria Diagnosis — CNN Benchmark Study

A group project (Team 17) comparing six deep-learning approaches for automated malaria detection from blood-smear cell images. Models range from a simple two-block CNN baseline up to fine-tuned ImageNet backbones, all evaluated under a shared, reproducible experimental framework across 42 total experiments.

## Problem

Malaria is diagnosed by manually inspecting stained blood smears under a microscope — a process that is slow, requires a trained expert, and is often unavailable in the regions that need it most. This project automates the binary classification of single-cell images as **Parasitized** or **Uninfected**.

## Dataset

**NIH Malaria Cell Images** — downloaded automatically from the official NIH URL at runtime:

```
https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
```

| | |
|---|---|
| Total images | 27,558 PNG cell images |
| Class balance | Perfectly balanced (13,780 Parasitized / 13,780 Uninfected) |
| Split | 70 / 15 / 15 → Train 19,290 / Val 4,132 / Test 4,136 |
| Input size | 128 × 128 × 3 |
| Random seed | 42, fixed and shared across all models |

## Notebooks

| Notebook | Model | Owner |
|---|---|---|
| [Malaria_Baseline_CNN_Group.ipynb](Malaria_Baseline_CNN_Group.ipynb) | Baseline CNN | Team 17 (collective) |
| [Malaria_Improved_CNN_Apoh.ipynb](Malaria_Improved_CNN_Apoh.ipynb) | Improved CNN | Apoh Prince Eldrige |
| [Malaria_VGG16_Delucie (2).ipynb](<Malaria_VGG16_Delucie%20(2).ipynb>) | VGG16 | Delucie Rurangwa |
| [Malaria_ResNet50_Sheilla.ipynb](Malaria_ResNet50_Sheilla.ipynb) | ResNet50 | Sheilla Keza Ruvugabigwi |
| [Malaria_DenseNet121_Dan.ipynb](Malaria_DenseNet121_Dan.ipynb) | DenseNet121 | Dan Paul Dushime |
| [Malaria_MobileNetV2_Shima.ipynb](Malaria_MobileNetV2_Shima.ipynb) | MobileNetV2 | Serein Shima Byiringiro |

## Models

### 1. Baseline CNN — Team 17 (collective)
Two Conv+Pool blocks, no batch normalisation, no augmentation. Built collectively as an honest reference point — deliberately kept simple so every later improvement is legible against it. Seven experiments vary the optimiser, learning rate, filter width, dense width, and dropout one variable at a time. Best result with RMSprop lr=1e-3.

### 2. Improved CNN — Apoh Prince Eldrige
Extends the baseline with a third convolution block, batch normalisation, dropout, and data augmentation. Experiments are structured as a ladder — depth first, then batch-norm, then augmentation, then regularisation — so each ingredient's contribution is isolated and measurable.

### 3. VGG16 — Delucie Rurangwa
VGG16 backbone pre-trained on ImageNet, with the top replaced by `GlobalAveragePooling → Dropout → Dense(128) → sigmoid`. Experiments progress from frozen feature extraction through fine-tuning of up to the last 50 layers at LR 1e-5. VGG16 is the most parameter-heavy backbone but also achieves the highest AUC (0.9934).

### 4. ResNet50 — Sheilla Keza Ruvugabigwi
ResNet50 backbone with the same custom head. Focuses on training stability — skip connections make the depth trainable, but fine-tuning requires careful learning rate control (1e-5) to avoid divergence. Ranks 1st overall by F1 and accuracy in the group leaderboard.

### 5. DenseNet121 — Dan Paul Dushime
DenseNet121 uses dense connectivity (each layer receives feature maps from all preceding layers), making it more parameter-efficient than VGG16 or ResNet50. Ranks 2nd overall. Best result achieved with fine-tuning the last 30 layers at LR 1e-4 — notably, this is the only model whose best experiment showed a negative overfit gap (generalising slightly better on val than train).

### 6. MobileNetV2 — Serein Shima Byiringiro
The lightweight backbone, designed for edge and mobile deployment. MobileNetV2 uses depthwise separable convolutions to minimise parameters. The central question for this model is whether a network small enough to run on constrained hardware can keep pace with the heavier backbones — and it largely does, finishing 4th with F1 = 0.9600 while training in a fraction of VGG16's time.

## Shared Experimental Framework

Every model runs through **7 experiments** under identical conditions so results are directly comparable:

- **tf.data pipeline** — `image_dataset_from_directory` with `cache()` + `prefetch(AUTOTUNE)` built once and reused across all 42 experiments
- **Callbacks** — `EarlyStopping(patience=5, restore_best_weights=True)` + `ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6)`
- **Mixed precision** — `float16` on GPU for faster training and lower memory usage
- **Metrics per run** — accuracy, precision, recall, F1, train/val accuracy, overfit gap (`train_acc − val_acc`), epochs run, wall-clock training time
- **Resumable checkpointing** — each experiment's result is appended to a per-model CSV on Google Drive the moment it finishes; re-running the notebook skips already-completed experiments

**Data augmentation** (applied as Keras layers inside the model, active only during training):
horizontal + vertical flip · ±10° rotation · ±10% zoom · ±10% translation

**Transfer-learning head** (VGG16, ResNet50, DenseNet121, MobileNetV2):
`GlobalAveragePooling2D → Dropout → Dense(128 or 256, relu) → Dropout → Dense(1, sigmoid)`

## Results

### Group Leaderboard — best experiment per model

| Rank | Model | Owner | Best Experiment | Accuracy | Precision | Recall | F1 | AUC | Train time (s) |
|:---:|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | ResNet50 | Sheilla Keza Ruvugabigwi | E4 fine-tune last 30, LR 1e-4 | 96.23% | 0.9747 | 0.9492 | **0.9618** | 0.9927 | 965 |
| 🥈 2 | DenseNet121 | Dan Paul Dushime | E4 fine-tune last 30, LR 1e-4 | 96.16% | 0.9686 | 0.9541 | 0.9613 | 0.9921 | 839 |
| 🥉 3 | VGG16 | Delucie Rurangwa | E6 fine-tune last 50, LR 1e-5 | 96.18% | 0.9848 | 0.9381 | 0.9609 | **0.9934** | 1923 |
| 4 | MobileNetV2 | Serein Shima Byiringiro | E4 fine-tune last 30, LR 1e-4 | 96.03% | 0.9680 | 0.9521 | 0.9600 | 0.9908 | **434** |
| 5 | Improved CNN | Apoh Prince Eldrige | E6 +Aug, LR 1e-4, dense 256 | 95.79% | 0.9697 | 0.9454 | 0.9574 | — | 710 |
| 6 | Baseline CNN | Team 17 (collective) | E3 RMSprop lr 1e-3 | 94.22% | 0.9654 | 0.9173 | 0.9407 | 0.9767 | 66 |

### All experiments per model

<details>
<summary><strong>Baseline CNN</strong> — 7 experiments</summary>

| Experiment | Config | Accuracy | F1 | Overfit Gap |
|---|---|:---:|:---:|:---:|
| E1 Adam lr1e-3 | Adam 1e-3, 16/32 filters, dense 64 | 93.45% | 0.9329 | +0.043 |
| E2 Adam lr1e-4 | Adam 1e-4 (lower LR) | 91.73% | 0.9153 | +0.039 |
| **E3 RMSprop lr1e-3** ⭐ | RMSprop 1e-3 | **94.22%** | **0.9407** | +0.046 |
| E4 SGD-momentum | SGD 1e-2, momentum 0.9 | 93.93% | 0.9396 | +0.010 |
| E5 wider filters | 32/64 filters, dense 128 | 93.81% | 0.9372 | +0.050 |
| E6 +dropout0.5 | Wider filters + dropout 0.5 | 93.35% | 0.9333 | +0.020 |
| E7 dense256 | Dense 256 + dropout 0.3 | 94.08% | 0.9396 | +0.039 |

</details>

<details>
<summary><strong>VGG16</strong> — 7 experiments</summary>

| Experiment | Config | Accuracy | F1 | Overfit Gap |
|---|---|:---:|:---:|:---:|
| E1 frozen lr1e-3 | Frozen backbone, LR 1e-3 | 94.66% | 0.9462 | −0.017 |
| E2 frozen +Aug | Frozen + augmentation | 93.74% | 0.9357 | −0.021 |
| E3 frozen drop0.5 | Frozen, dropout 0.5, dense 256 | 92.24% | 0.9219 | −0.042 |
| E4 fine-tune lr1e-4 | Fine-tune last 30, LR 1e-4 | 50.00% | 0.6667 | −0.005 |
| E5 fine-tune lr1e-5 | Fine-tune last 30, LR 1e-5 | 95.77% | 0.9565 | +0.005 |
| **E6 fine-tune unf50** ⭐ | Fine-tune last 50, LR 1e-5 | **96.18%** | **0.9609** | +0.005 |
| E7 fine-tune wide256 | Fine-tune last 30, dense 256 | 95.99% | 0.9587 | +0.002 |

> Note: E4 collapsed to majority-class prediction — LR 1e-4 is too aggressive for VGG16 fine-tuning.

</details>

<details>
<summary><strong>ResNet50</strong> — 7 experiments</summary>

| Experiment | Config | Accuracy | F1 | Overfit Gap |
|---|---|:---:|:---:|:---:|
| E1 frozen lr1e-3 | Frozen backbone, LR 1e-3 | 93.96% | 0.9387 | −0.012 |
| E2 frozen +Aug | Frozen + augmentation | 91.30% | 0.9078 | +0.003 |
| E3 frozen drop0.5 | Frozen, dropout 0.5, dense 256 | 90.04% | 0.8935 | −0.012 |
| **E4 fine-tune lr1e-4** ⭐ | Fine-tune last 30, LR 1e-4 | **95.91%** | **0.9586** | +0.015 |
| E5 fine-tune lr1e-5 | Fine-tune last 30, LR 1e-5 | 95.19% | 0.9512 | +0.004 |
| E6 fine-tune unf50 | Fine-tune last 50, LR 1e-5 | 96.30% | 0.9627 | +0.002 |
| E7 fine-tune wide256 | Fine-tune last 30, dense 256 | 95.58% | 0.9548 | +0.004 |

</details>

<details>
<summary><strong>DenseNet121</strong> — 7 experiments</summary>

| Experiment | Config | Accuracy | F1 | Overfit Gap |
|---|---|:---:|:---:|:---:|
| E1 frozen lr1e-3 | Frozen backbone, LR 1e-3 | 93.11% | 0.9299 | −0.029 |
| E2 frozen +Aug | Frozen + augmentation | 91.03% | 0.9056 | −0.011 |
| E3 frozen drop0.5 | Frozen, dropout 0.5, dense 256 | 89.80% | 0.8921 | −0.041 |
| **E4 fine-tune lr1e-4** ⭐ | Fine-tune last 30, LR 1e-4 | **96.16%** | **0.9613** | −0.005 |
| E5 fine-tune lr1e-5 | Fine-tune last 30, LR 1e-5 | 94.80% | 0.9473 | −0.017 |
| E6 fine-tune unf50 | Fine-tune last 50, LR 1e-5 | 94.97% | 0.9493 | −0.017 |
| E7 fine-tune wide256 | Fine-tune last 30, dense 256 | 94.75% | 0.9462 | −0.014 |

</details>

<details>
<summary><strong>MobileNetV2</strong> — 7 experiments</summary>

| Experiment | Config | Accuracy | F1 | Overfit Gap |
|---|---|:---:|:---:|:---:|
| E1 frozen lr1e-3 | Frozen backbone, LR 1e-3 | 93.67% | 0.9356 | −0.019 |
| E2 frozen +Aug | Frozen + augmentation | 90.74% | — | — |
| E3 frozen drop0.5 | Frozen, dropout 0.5, dense 256 | 89.63% | — | — |
| **E4 fine-tune lr1e-4** ⭐ | Fine-tune last 30, LR 1e-4 | **96.03%** | **0.9600** | +0.008 |
| E5 fine-tune lr1e-5 | Fine-tune last 30, LR 1e-5 | 95.07% | — | — |
| E6 fine-tune unf50 | Fine-tune last 50, LR 1e-5 | 95.99% | — | — |
| E7 fine-tune wide256 | Fine-tune last 30, dense 256 | 95.09% | — | — |

</details>

## Key Takeaways

- **All four transfer-learning backbones exceeded 96% accuracy**, confirming that ImageNet features transfer well to cell-image texture classification.
- **Fine-tuning at LR 1e-5 is critical for VGG16** — E4 (LR 1e-4) collapsed to majority-class prediction, while E5/E6 (LR 1e-5) recovered to 95–96%.
- **DenseNet121's best result shows a negative overfit gap** (−0.005), meaning it generalised slightly better on validation than training — a unique result among all 42 experiments.
- **MobileNetV2 is the efficiency winner**: matches the accuracy of VGG16 and ResNet50 within 0.2% while training in under 7 minutes (vs. 32 minutes for VGG16 fine-tuning).
- **The Baseline CNN's overfit gap (+0.046) is the largest in the study**, confirming it is memorising training cells rather than learning generalisable features — exactly the reference behaviour the baseline was designed to expose.

## Running the Notebooks

1. Open any notebook in **Google Colab** and set the runtime to GPU (*Runtime → Change runtime type → GPU*).
2. Set `USE_DRIVE_CKPT = True` to persist per-experiment results to Google Drive — this survives Colab runtime disconnects.
3. Run all cells. The NIH dataset (~337 MB) is downloaded and extracted automatically.
4. To start a clean run, set `RESET_RESULTS = True` once, then revert it to `False`.

**Dependencies** (pre-installed on Colab):
```
tensorflow >= 2.20
numpy  pandas  matplotlib  scikit-learn
```

## Repository Structure

```
.
├── Malaria_Baseline_CNN_Group.ipynb     # Baseline CNN — Team 17 (collective)
├── Malaria_Improved_CNN_Apoh.ipynb     # Improved CNN — Apoh Prince Eldrige
├── Malaria_VGG16_Delucie (2).ipynb     # VGG16         — Delucie Rurangwa
├── Malaria_ResNet50_Sheilla.ipynb      # ResNet50      — Sheilla Keza Ruvugabigwi
├── Malaria_DenseNet121_Dan.ipynb       # DenseNet121   — Dan Paul Dushime
└── Malaria_MobileNetV2_Shima.ipynb    # MobileNetV2   — Serein Shima Byiringiro
```

## Team 17

| Name | Role |
|---|---|
| Dan Paul Dushime | Team lead · shared data pipeline · DenseNet121 |
| Apoh Prince Eldrige | Improved CNN |
| Delucie Rurangwa | VGG16 |
| Sheilla Keza Ruvugabigwi | ResNet50 |
| Serein Shima Byiringiro | MobileNetV2 |
