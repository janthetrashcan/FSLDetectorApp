# FSL Alphabet Detector – Android Application

A research-grade Android app for evaluating YOLOv8, YOLO-NAS, and YOLO11 TFLite models
on the **Filipino Sign Language (FSL) alphabet** (26 English letters + Ñ = **27 classes**).

---

## Features

| Feature | Details |
|---|---|
| **Model switching** | Tap a chip to switch between YOLOv8, YOLO-NAS, YOLO11 instantly |
| **Backend selection** | CPU via XNNPACK delegate · GPU via OpenGL delegate |
| **Confidence / IoU sliders** | Adjust thresholds without restarting |
| **Single image mode** | Pick any image from gallery; see bounding box overlay + timing breakdown |
| **Batch mode** | Point to a folder; the app processes all images and computes full metrics |
| **Metrics computed** | Precision · Recall · Accuracy · F1-score · mAP@50 · Inference speed stats |
| **Per-class breakdown** | TP / FP / FN / P / R / F1 / AP@50 for all 27 classes in a scrollable table + grouped bar chart |
| **Normalized confusion** | Per-class AP accounts for class imbalance |
| **Warm-up** | 10 warm-up frames run automatically before batch timing starts |

---

## Quick Setup

### 1 · Place TFLite model files in assets

```
app/src/main/assets/
    yolov8_fsl.tflite      ← YOLOv8 export
    yolonas_fsl.tflite     ← YOLO-NAS export
    yolo11_fsl.tflite      ← YOLO11 export
```

Export commands (Python, run in your training environment):

```python
# YOLOv8 / YOLO11
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="tflite", imgsz=640, int8=False)   # → best_float32.tflite

# YOLO-NAS (super-gradients)
from super_gradients.training import models
net = models.get("yolo_nas_s", num_classes=27, checkpoint_path="best.pth")
net.export("yolonas_fsl.tflite", output_predictions_format="FLAT_FORMAT", input_image_shape=[640, 640])
```

### 2 · Label files for batch evaluation

For each image `sign_001.jpg`, create a paired `sign_001.txt` in **YOLO format**:
```
<class_index> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1]. Class indices follow:
```
0=A  1=B  2=C  3=D  4=E  5=F  6=G  7=H  8=I  9=J  10=K  11=L  12=M
13=N  14=Ñ  15=O  16=P  17=Q  18=R  19=S  20=T  21=U  22=V  23=W  24=X  25=Y  26=Z
```

Images without a paired `.txt` are still processed but contribute 0 GT entries (all detections count as FP).

### 3 · Build and run

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Minimum SDK: **API 26 (Android 8.0)**. Target SDK: **34**.

---

## Output format model requirements

The detector auto-detects the output tensor shape on model load:

| Model | Expected output shape | Layout |
|---|---|---|
| YOLOv8 | `[1, 31, 8400]` | cols-first `[batch, 4+classes, anchors]` |
| YOLO11  | `[1, 31, 8400]` | cols-first (same as YOLOv8) |
| YOLO-NAS | `[1, 8400, 31]` | anchors-first `[batch, anchors, 4+classes]` |

If your TFLite export uses a different number of anchors (e.g., 3549 for 416px input), update
`INPUT_SIZE` in `YOLODetector.kt` accordingly.

---

## Metrics — Formulas

All formulas match those in the study document:

### Precision
```
P = TP / (TP + FP)
```
High precision → low false positive rate; model is trustworthy.

### Recall
```
R = TP / (TP + FN)
```
High recall → low false negative rate; app detects the sign when it happens.

### Accuracy
```
A = (TP + TN) / (TP + FP + TN + FN)
```
Overall correctness. Interpreted alongside P and R due to potential class imbalance.

### F1-Score
```
F1 = 2 × (P × R) / (P + R)
```
Harmonic mean of precision and recall. More robust than accuracy alone.

### mAP@50
```
mAP@50 = (1/N) × Σ AP_i      where IoU threshold = 0.50
```
AP per class computed with the **11-point interpolated** method.
An IoU ≥ 0.50 match is required to count a detection as TP.

### Inference Speed
- Measured in **milliseconds (ms)**
- Lower is better (less latency, more real-time capable)
- Reported: **mean · min · max · std dev** across all batch images
- 10 warm-up iterations are discarded before measurement begins

---

## Project structure

```
app/src/main/
├── assets/                          ← Place .tflite models here
├── java/com/fsl/detector/
│   ├── detector/
│   │   ├── YOLODetector.kt          ← Core TFLite inference engine (all 3 models)
│   │   └── DetectionResult.kt       ← Data classes: DetectionResult, InferenceOutput, GroundTruth
│   ├── metrics/
│   │   └── MetricsCalculator.kt     ← P / R / Acc / F1 / mAP@50 / timing stats
│   ├── ui/
│   │   ├── MainActivity.kt          ← Model + backend selector, image/batch pickers
│   │   ├── SingleImageActivity.kt   ← Single-image inference + overlay
│   │   ├── ResultsActivity.kt       ← Batch metrics summary + chart + table
│   │   ├── BoundingBoxOverlayView.kt← Custom view for drawing detections
│   │   ├── DetectorViewModel.kt     ← MVVM ViewModel with coroutine-based inference
│   │   └── PerClassAdapter.kt       ← RecyclerView adapter for per-class table
│   └── utils/
│       └── LabelUtils.kt            ← YOLO label parsing, bitmap decoding, folder scanning
└── res/
    ├── layout/
    │   ├── activity_main.xml
    │   ├── activity_single_image.xml
    │   ├── activity_results.xml
    │   └── item_per_class.xml
    └── values/
        ├── strings.xml
        └── themes.xml
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `tensorflow-lite:2.14.0` | TFLite runtime |
| `tensorflow-lite-gpu` | GPU delegate (OpenGL) |
| `tensorflow-lite-support` | Buffer utilities |
| `MPAndroidChart v3.1.0` | Per-class bar chart |
| `kotlinx-coroutines` | Async inference without blocking the UI |
| Material3 | Chips, sliders, cards |

Add JitPack to your `settings.gradle` for MPAndroidChart:
```gradle
maven { url 'https://jitpack.io' }
```

---

## Notes for the study

- Inference time is measured **from frame capture to bounding box output**, averaged across all images
- **Warm-up**: 10 iterations on the first image are discarded
- CPU backend uses the XNNPACK delegate; GPU backend uses the OpenGL delegate
- Results are reported separately per backend (switch the chip and re-run)
- The **normalized confusion matrix** is generated from per-class TP / FP / FN counts
- F1 standard deviation across all 27 classes is displayed to quantify **classification consistency**
