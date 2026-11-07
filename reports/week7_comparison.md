# Task 3 YOLOv8 Classification Comparison

All three models used the same test set of 246 images, with 123 images in each class. The existing split contains 1,290 training images, 264 validation images, and 246 test images. The actual split is about 71.7% / 14.7% / 13.7%, while the assignment gives 70% / 15% / 15%. The original split was kept for Task 3. The metadata contains 300 source IDs, and each ID appears in only one split.

## Test results

| Model | Input | Accuracy | Macro precision | Macro recall | Macro F1 | Original latency (ms/image) | Size (MiB) | Parameters | Training time (s) | Epochs run | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 frozen | 224 × 224 | 0.9959 | 0.9960 | 0.9959 | 0.9959 | 51.919 | 11.06 | 2,422,081 | 192.00 | 28 | 1 |
| YOLOv8n-cls | 640 × 640 | 0.9878 | 0.9881 | 0.9878 | 0.9878 | 14.849 | 2.83 | 1,440,850 | 1,401.84 | 14 | 3 |
| YOLOv8s-cls | 640 × 640 | 0.9959 | 0.9960 | 0.9959 | 0.9959 | 30.315 | 9.78 | 5,083,298 | 4,459.13 | 21 | 1 |

The table uses the three original `results.json` files. File sizes use 1 MiB = 1,048,576 bytes. The validation logs show the best top-1 accuracy at epoch 9 for YOLOv8n-cls and epoch 16 for YOLOv8s-cls. Final YOLO predictions were checked again on all 246 test images with the saved checkpoints and gave the same confusion matrices and error files.

## Comparison

YOLOv8n-cls had 98.78% test accuracy, which was 0.81 percentage points below MobileNetV2. YOLOv8s-cls and MobileNetV2 both reached 99.59% accuracy, but they missed different photos. The YOLOv8n-cls checkpoint was 8.23 MiB smaller than the MobileNetV2 checkpoint, and the YOLOv8s-cls checkpoint was 1.28 MiB smaller. In the original runs, the recorded latency was 37.07 ms lower for YOLOv8n-cls and 21.60 ms lower for YOLOv8s-cls than for MobileNetV2. The original timing included reading and preparing a photo for YOLO, while MobileNetV2 received an array that was already prepared. MobileNetV2 used a frozen feature extractor and a custom head with one sigmoid output. Both YOLOv8 models used their two-class classification head and full fine-tuning. The YOLO models used 640 × 640 inputs, while MobileNetV2 used 224 × 224 inputs. Input size, network size, and the prediction software all affect latency, so this experiment cannot show how much each factor contributed.

## CPU latency check

The original scripts both use 20 warm-up predictions and 100 timed predictions. The YOLO script repeats one test image, while the MobileNetV2 script uses prepared arrays from several test images.

In a separate check, all three models used the same 120 JPEG files from the test split on the same CPU. The files were selected by sorting the test paths and sampling with seed `2473057`. The first 20 images were used for warm-up and the next 100 were timed. Each timed prediction starts with a file path and ends when the predicted class is available. The timing includes reading and preparing the image. The models were loaded before timing.

| Model | Repeat latency (ms/image) | Difference from MobileNetV2 (ms/image) |
|---|---:|---:|
| MobileNetV2 frozen | 57.47 | 0.00 |
| YOLOv8n-cls | 16.75 | −40.72 |
| YOLOv8s-cls | 32.74 | −24.73 |

The repeat check used Ultralytics 8.4.202 and TensorFlow 2.21.0 with Keras 3.12.4. The MobileNetV2 checkpoint was saved with Keras 3.14.0. For this check, the loader skipped only unsupported options for disabled batch renormalization and unused quantization. The saved checkpoint was used without changing its weights or file. The original JSON files still contain the original measurements.

## Reproducing Task 3

From the project root, run the command below to reproduce the original training and test evaluation. The experiment used an Intel Core Ultra 5 225H CPU, seed `2473057`, 50 planned epochs, patience 5, batch size 32, and image size 640.

```bat
python -u scripts\train_yolo_quality.py
```

The [YOLOv8n-cls test matrix](../results/week7_yolov8n_cls/confusion_matrix.png) and [YOLOv8s-cls test matrix](../results/week7_yolov8s_cls/confusion_matrix.png) use true labels as rows and predicted labels as columns, in the order `bad`, `good`. Their counts are `[[120, 3], [0, 123]]` and `[[122, 1], [0, 123]]`. The Ultralytics training plots and validation confusion matrices are saved separately in the run folders.
