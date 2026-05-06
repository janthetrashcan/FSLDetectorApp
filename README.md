#  FSLDetectorApp
FSLDetectorApp is a proof-of-concept application that can load YOLO models in TensorFlow Lite format to generate inferences from images. At its core, the application is merely a YOLO interpreter that can detect and label any objects from images.  For the purpose of detecting the Filipino Sign Language (FSL) alphabet, it is adapted to detect 27 classes, with each class representing a letter in the FSL alphabet.

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z


## Application Features
FSLDetectorApp contains several rudimentary features that were essential in achieving the objectives of the study. Features of the application include:
### Model Switching
Users are allowed to switch between multiple models.
### Backend Selection
Users can select between CPU inference via XNNPack delegate and GPU inference via OpenGL delegate.
### Confidence / IoU Sliders
Users can adjust model thresholds without restarting the application.
### Single Image Mode
A single image can be selected from the gallery. Once processed, a bounding box overlay with a confidence score and timing breakdown is shown.
### Batch Mode
A folder can be selected, where the app processes all images and label files and computes performance metrics.
### Metrics Computation
Performance metrics such as precision, recall, accuracy, F1-score, mAP@50, and inference speed statistics are computed.
### Per-class Breakdown
True positives, false positives, false negatives, precision, recall, F1-score, and AP@50 for all classes are shown in a scrollable table. A normalized confusion matrix is also shown at the end.
### Warm-up
10 warm-up frames are run automatically before batch timing starts to simulate the slight delay in opening the camera.
### Results Export
For batch edits, results are exported as three (3) files: normalized confusion matrix (CSV), per-class summary (CSV), and summary of batch metrics (JSON).

As of the current version of the application, it does not support live, real-time inferences directly from a camera feed. However, this may be implemented in a future iteration of the app.
