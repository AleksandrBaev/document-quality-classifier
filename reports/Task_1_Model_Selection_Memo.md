# Task 1 — Model Selection Memo

Document photo quality gate for the Telegram bot (Russian INN, passport, SNILS)

Student ID / seed: 2473057

## 1. Purpose and pipeline context

The bot receives a photo of a Russian document (INN certificate, passport page, SNILS) from the user. The quality model is stage 1 of the pipeline, the first gate: it decides whether the photo is good enough to be passed on. The full pipeline is: upload → quality check → classification → perspective correction → OCR → JSON → operator (Lecture 4 [6]), so every later stage depends on this decision. If the photo is bad (blur, glare, skew, occlusion, exposure), the bot must immediately ask the user to retake it. The user is waiting in the chat during this step, so the model has to answer fast, and it must be cheap enough to run on a CPU server now and on a mobile or edge device later.

## 2. Data

The dataset has 1800 images (640×640): 900 good and 900 bad, split evenly over three document classes (600 per class). Bad images are split evenly over five defects: blur, glare, skew, occlusion and exposure (180 each). Good images include mild rotations and 90/180/270° rotations. The split is 1290 / 264 / 246 images (train / validation / test), with good and bad exactly balanced inside every split. All variants of one source photo (300 sources, 3 variants each) are placed in the same split, which I verified in metadata.jsonl, so there is no leakage between train and test through sibling images.

The dataset is small and synthetic: the defects are generated artificially. The thesis reports a synthetic-to-real accuracy drop of 2.8–5.8% (thesis Table 7, as cited in the course task requirements and in Lecture 3 [6]), so test numbers on this dataset are optimistic for real mobile photos.

The audit of the previous build of this dataset (Dataset Audit Report, lab 4 [7]) points out further limits that I keep in mind for the acceptance criteria. Some bad images, especially mild skew, glare, occlusion and exposure changes on the white INN, are still readable, so part of the bad labels is noisy. About 44% of the good images are rotated by 90, 180 or 270°, so a model could learn orientation instead of quality. Every image carries the same red SYNTHETIC TRAINING SAMPLE line and each class has its own background colour, which can work as a shortcut. There are only 100 source documents per class, each used three times.

## 3. Comparison of candidate architectures

| Model (family) | Native input | Params / size (approx.) | Strength | Specific risk for our project |
| --- | --- | --- | --- | --- |
| **MobileNetV2 (depthwise-separable CNN with inverted residuals) [1]** | **224×224** | **3.5M; ~14 MB; ImageNet top-1 ~71%** | **Very small and fast on CPU; good baseline for transfer learning; easy export to TFLite** | **Low capacity: may miss subtle glare and slight blur; weaker than larger nets on fine texture** |
| **YOLOv8-cls n / s (CSP-style CNN, classification head) [4]** | **224×224 (trained here at 640)** | **n: 2.7M, s: 6.4M; file ~6 / ~13 MB (estimate); ImageNet top-1 ~67–69% (n), ~72–74% (s)** | **Compact, simple training and export pipeline (ONNX, TFLite); input matches our 640×640 images** | **Input 640 costs about 8× more pixels than 224, so latency may grow; class names come from folder names, so folders must stay good / bad** |
| **EfficientNet-B0 (compound-scaled MBConv CNN) [3]** | **224×224** | **5.3M; ~29 MB; ImageNet top-1 ~77%** | **Best accuracy per parameter in this list; still under the size limit** | **Requires its own preprocessing (torch mode); slower than MobileNetV2 on CPU; fine-tuning is sensitive to learning rate** |
| **ResNet50 (residual CNN) [2]** | **224×224** | **25.6M; ~98 MB; ImageNet top-1 ~75%** | **Strong, well-studied baseline with stable training** | **Size is about twice our limit and latency is higher; overfits easily on 1290 training images** |

Parameter counts, sizes and ImageNet accuracy are approximate values from the Keras Applications and Ultralytics documentation [4][5]; I will replace them by measured values in Tasks 2–4.

## 4. Transfer learning and the two training regimes

Training a CNN from scratch needs a lot of labelled data. We have only 1290 training images, so transfer learning is mandatory: I start from a network pre-trained on ImageNet and reuse what it has already learned. The early layers of such a network learn general visual features: edges, corners, textures, simple shapes. Deeper layers combine them into more specific, object-level features.

**Regime A — frozen base.** All pre-trained weights are kept fixed and only a new small head (global average pooling, dense layer, dropout, one sigmoid output) is trained. The model does not learn new visual features. It only learns how to map the existing ImageNet features to the labels good / bad. It is fast, has few trainable parameters and little risk of overfitting, but it cannot adapt the features to document-specific defects such as glare or a skewed text line.

**Regime B — full fine-tuning.** All layers are unfrozen and trained with a low learning rate (for example 1e-5 to 1e-4), so the pre-trained weights are adjusted slightly and not destroyed. The model can adapt its features to our defects and usually gets higher accuracy, but it is slower, needs more memory and overfits more easily on a small synthetic dataset.

Plan: run the frozen regime first as the baseline, then compare it with fine-tuning if time allows.

## 5. Constraint analysis

1. **Stage 1 gate → latency.** The user waits for feedback right after sending the photo, so the budget is 100 ms per image for the model on the course CPU machine. This favours MobileNetV2, YOLOv8n-cls and EfficientNet-B0; ResNet50 is the heaviest candidate.
2. **Small synthetic dataset → transfer learning.** With 1290 training images and artificial defects, a large model such as ResNet50 overfits. Pre-trained small models with a frozen or lightly fine-tuned base are the safer choice.
3. **CPU server now, mobile later → model size.** A size limit of 50 MB keeps the model deployable through TFLite or ONNX on a phone. ResNet50 (~98 MB) fails this limit.
4. **Accuracy and the synthetic-to-real gap.** The most expensive error is letting a bad photo through, because later stages then fail or return wrong data. Recall on the bad class matters more than overall accuracy, and glare is the expected hardest defect (F1 = 0.92 in the thesis, Lecture 4 [6]).

## 6. Acceptance criteria (fixed before any experiment)

The user needs real photos to pass the gate with at least 0.85 accuracy. Because synthetic results drop by up to 5.8% on real photos, the threshold on the synthetic test split has to be higher: 0.85 + 0.058 = 0.908, rounded up to 0.92 for a safety margin. For reference, the thesis YOLOv8 classifier reached 94.5% accuracy on synthetic data with 2.1 ms inference and a 2.8 MB model (Lecture 4 [6]), so 0.92 is demanding but realistic. All thresholds are measured on the test split only, once, after training is finished.

| Criterion | Threshold |
| --- | --- |
| **Accuracy on test split** | **≥ 0.92** |
| **Macro F1 on test split** | **≥ 0.92** |
| **Recall of class bad** | **≥ 0.90** |
| **Recall for glare images (per defect)** | **≥ 0.85** |
| **Latency per image on course CPU (20 warm-up + 100 timed predictions)** | **≤ 100 ms** |
| **Model size (saved weights file)** | **≤ 50 MB** |

Decision rule: among models that meet every threshold, the one with the lowest latency is chosen, with macro F1 as the tie-breaker. If no model meets a threshold, this is reported in the decision memo and not hidden by changing the number afterwards. Fixed training parameters for all models: epochs 50, early stopping patience 5, batch 32, same split, seed 2473057.

## 7. Shortlist for Tasks 2–4

Task 2: MobileNetV2 with a frozen base. It is the smallest and fastest reasonable model, so it is the baseline against which everything else is measured. Task 3: YOLOv8n-cls and YOLOv8s-cls. They have a different architecture and a different input size (640), so the comparison shows what higher resolution costs in latency and what it gains for small defects such as glare. Task 4: EfficientNet-B0. It has the best accuracy per parameter among the candidates and still fits the 50 MB and latency limits, so it tests whether extra accuracy is worth extra time. ResNet50 is rejected because of its size (about twice the limit), higher latency and overfitting risk on a small dataset.

## References

[1] Sandler M., Howard A., Zhu M., Zhmoginov A., Chen L.-C. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR, 2018.

[2] He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.

[3] Tan M., Le Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.

[4] Ultralytics. YOLOv8 Classification (Classify) documentation. docs.ultralytics.com/tasks/classify/

[5] Keras Applications documentation. keras.io/api/applications/

[6] Course lectures: Lecture 3 (synthetic-to-real gap in the thesis) and Lecture 4 (pipeline, thesis results, glare F1); thesis Table 7 as cited in the Task 1 and Task 4 requirements.

[7] Dataset Audit Report, lab 4 (audit of the previous dataset build).
