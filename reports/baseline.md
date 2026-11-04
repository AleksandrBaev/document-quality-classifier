# Task 2 — MobileNetV2 Baseline (frozen base)

## Setup

- Model: MobileNetV2 (ImageNet weights, base frozen), head GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, sigmoid)
- Input 224×224, `preprocess_input` (mode "tf"), class_mode binary, threshold 0.5
- Adam lr = 0.001, binary_crossentropy, batch 32, up to 50 epochs, EarlyStopping(patience=5, restore_best_weights=True), ModelCheckpoint(save_best_only=True)
- Seed 2473057 (student ID), device CPU
- Split by source image: train 1290 / validation 264 / test 246 (71.7 / 14.7 / 13.7%), good and bad balanced in every split

## Test results (test split only)

| Metric | Value |
|---|---|
| Accuracy | 0.9959 |
| Macro precision / recall / F1 | 0.9960 / 0.9959 / 0.9959 |
| Bad: precision / recall / F1 | 1.0000 / 0.9919 / 0.9959 |
| Good: precision / recall / F1 | 0.9919 / 1.0000 / 0.9960 |
| Confusion matrix (rows true bad, good) | [[122, 1], [0, 123]] |
| Epochs run | 28 of 50 (best epoch 23) |
| Training time | 192 s |
| Latency | 51.9 ms per image (20 warm-up, 100 timed) |
| Model size / parameters | 11.06 MB / 2,422,081 |

## Interpretation

1. The training and validation curves follow each other closely and both flatten out after about epoch 18. Validation loss is noisier and has several spikes (for example around epochs 10, 15, 17 and 26), and validation accuracy dips to 0.97 at those points; this is typical for a validation set of only 264 images, where one or two images move the value. The lowest validation loss (about 0.014) is at epoch 23, so early stopping ended the run at epoch 28. There is no sign of overfitting (train 0.995, validation 0.992 at the end) and no underfitting, since both losses are low.
2. The test accuracy of 0.996 is higher than the validation accuracy. With only 246 test images one error changes the result by 0.4 percentage points, so the difference is noise and not a real gain.
3. The bad class is the harder one: its recall is 0.992 (one bad image was accepted as good), while the recall of good is 1.0. This is the more expensive error for the pipeline, because a bad photo passes the quality gate.
4. The single error is a passport image with glare (`bad_passport_glare_202_v1.jpg`). The glare is a white ellipse over the lower part of the page; it hides only part of the address lines, while the name, the date of birth and the photo are still readable. So this "bad" label is borderline, and the model's decision is understandable. Glare is also the defect that the thesis found hardest (F1 = 0.92), and it matches the noisy-label risk from the dataset audit.
5. Such a high score on a small synthetic dataset should not be read as the real-world accuracy: the thesis reports a synthetic-to-real accuracy drop of 2.8–5.8%.
6. Against the Task 1 acceptance criteria this baseline passes accuracy (≥ 0.92), macro F1 (≥ 0.92), recall of bad (≥ 0.90), latency (≤ 100 ms) and size (≤ 50 MB). The per-defect recall for glare was not measured here.
