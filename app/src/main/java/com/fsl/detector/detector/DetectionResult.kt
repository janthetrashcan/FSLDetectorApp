package com.fsl.detector.detector

import android.graphics.RectF
import android.net.Uri

/**
 * Represents a single detection bounding box with its associated class and confidence.
 */
data class DetectionResult(
    val boundingBox: RectF,       // normalized [0,1] coordinates (xMin, yMin, xMax, yMax)
    val classIndex: Int,
    val className: String,
    val confidence: Float
)

/**
 * Holds the full output of running inference on one image.
 */
data class InferenceOutput(
    val detections: List<DetectionResult>,
    val inferenceTimeMs: Long,       // raw model inference duration
    val preprocessTimeMs: Long,      // resize + normalize
    val postprocessTimeMs: Long      // NMS + decode
) {
    val totalTimeMs: Long get() = inferenceTimeMs + preprocessTimeMs + postprocessTimeMs
}

/**
 * Ground-truth annotation for a single image, loaded from YOLO-format .txt files.
 */
data class GroundTruth(
    val classIndex: Int,
    val boundingBox: RectF          // normalized [0,1] xCenter,yCenter,w,h → converted to xMin,yMin,xMax,yMax
)

data class ModelConfig(
    val displayName: String,
    val uri: Uri
)

enum class BackendType(val displayName: String) {
    CPU("CPU (XNNPACK)"),
    GPU("GPU (OpenGL)")
}
