# Week 8 Decision Memo

## Decision

Use **YOLOv8s-cls** for the document quality gate in a CPU pilot. The gate is the first step in the pipeline. It checks whether a photo is clear enough for the next step. Start with operator review during the pilot, then review the results on real photos before wider use.

## Evidence

All four models used the same 246 synthetic test images. CPU latency was measured with the same protocol, using 20 warm-up images and 100 timed images. The time includes reading and preparing each image and predicting its class.

| Model | Test accuracy | Macro F1 | CPU latency (ms/image) | Checkpoint size (MiB) |
|---|---:|---:|---:|---:|
| MobileNetV2 | 99.59% | 0.9959 | 57.47 | 11.06 |
| YOLOv8n-cls | 98.78% | 0.9878 | 16.75 | 2.83 |
| **YOLOv8s-cls** | **99.59%** | **0.9959** | **32.74** | **9.78** |
| EfficientNet-B0 | 97.97% | 0.9797 | 18.53 | 15.56 |

YOLOv8s-cls tied MobileNetV2 for the highest accuracy and macro F1. It was 24.73 ms faster per image and its checkpoint was 1.28 MiB smaller. Its mean latency of 32.74 ms is below the 100 ms limit for the first gate, with 67.26 ms to spare in this benchmark. This timing covers the quality check only.

[Model comparison](comparison.md) · [CPU latency measurements](task4_cpu_latency.json)

## Rejected Alternatives

- **MobileNetV2** also passed the time limit and made one test error. YOLOv8s-cls had the same scores with lower latency and a smaller checkpoint.
- **YOLOv8n-cls** was the fastest and smallest option. It passed three bad photos as good, while YOLOv8s-cls passed one. YOLOv8s-cls still fits the time limit, so the extra time buys better test accuracy and F1.
- **EfficientNet-B0** was faster than YOLOv8s-cls, but it had the lowest accuracy and macro F1, the largest checkpoint, and five bad photos passed as good.

## Risks and Mitigation

- **Glare.** YOLOv8s-cls passed one of the 26 bad photos with glare as good. Add stronger glare augmentation during training. Vary the position, size, and brightness of reflections, and include real photos taken under lamps and in sunlight.
- **Synthetic-to-real gap.** Thesis Table 7, cited in the task brief, reports a gap of 2.8–5.8%. The 99.59% accuracy in this memo was measured on synthetic images. Collect and label real photos from different phones and lighting conditions. Use a real validation set to tune the score threshold for accepting a photo. Then check the final setting on a separate real test set.
- **Operator verification.** Send uncertain photos to an operator during the pilot. The operator can accept the photo or ask for a new one. Log corrections and review how often bad photos pass and good photos are rejected.

[Test error analysis](error_analysis.md)

## Deployment Note

Run the saved YOLOv8s-cls checkpoint on the CPU server first. Keep the 640 × 640 input and the same image preparation as in the experiment. Before wider use, check the quality gate on the target server under normal load and confirm it meets the 100 ms limit. For a later mobile or edge version, export the model to ONNX or TFLite and measure accuracy and latency again on the target device.
