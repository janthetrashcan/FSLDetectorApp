package com.fsl.detector.metrics

import android.graphics.RectF
import android.util.Log
import com.fsl.detector.detector.DetectionResult
import com.fsl.detector.detector.GroundTruth
import com.fsl.detector.detector.YOLODetector
import kotlin.math.max

/**
 * Computes detection metrics: Precision, Recall, Accuracy, F1, mAP@50,
 * and per-class breakdowns, matching the formulas described in the study.
 */
object MetricsCalculator {

    data class ImageMetricsInput(
        val detections: List<DetectionResult>,
        val groundTruths: List<GroundTruth>,
        val inferenceTimeMs: Long,
        val debugTag: String = ""
    )

    data class PerClassStats(
        val className: String,
        val tp: Int,
        val fp: Int,
        val fn: Int,
        val tn: Int,
        val precision: Float,
        val recall: Float,
        val f1: Float,
        val ap50: Float
    )

    data class AggregateMetrics(
        val precision: Float,
        val recall: Float,
        val accuracy: Float,
        val f1: Float,
        val mAP50: Float,
        val meanInferenceMs: Float,
        val minInferenceMs: Long,
        val maxInferenceMs: Long,
        val stdDevInferenceMs: Float,
        val perClassStats: List<PerClassStats>,
        val f1StdDev: Float,
        val totalImages: Int,
        val totalDetections: Int,
        val totalGroundTruths: Int,
        val confusionMatrix: Array<FloatArray>  // [gtClass][predClass], row-normalized
    )

    private const val IOU_THRESHOLD_50 = 0.50f

    /**
     * Compute IoU between two bounding boxes.
     */
    fun iou(a: RectF, b: RectF): Float {
        val interLeft   = max(a.left, b.left)
        val interTop    = max(a.top, b.top)
        val interRight  = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)
        if (interRight <= interLeft || interBottom <= interTop) return 0f
        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val unionArea = a.width() * a.height() + b.width() * b.height() - interArea
        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    /**
     * Match detections to ground truths for a single image at IoU ≥ 0.50.
     * Returns a list of (detection, matched_gt_or_null) pairs.
     */
    fun matchDetectionsToGroundTruths(
        detections: List<DetectionResult>,
        groundTruths: List<GroundTruth>,
        iouThreshold: Float = IOU_THRESHOLD_50,
        debugTag: String = ""
    ): Pair<List<Pair<DetectionResult, GroundTruth?>>, List<GroundTruth>> {
        val sortedDets = detections.sortedByDescending { it.confidence }
        val matchedGTs = mutableSetOf<Int>()
        val results    = mutableListOf<Pair<DetectionResult, GroundTruth?>>()

        // ── Debug: log first image's boxes ──────────────────────────────
        if (debugTag.isNotEmpty()) {
            Log.d("MetricsDebug", "=== $debugTag ===")
            Log.d("MetricsDebug", "  Detections: ${detections.size}, GTs: ${groundTruths.size}")
            detections.take(3).forEach { d ->
                Log.d("MetricsDebug", "  DET cls=${d.classIndex}(${d.className}) " +
                        "conf=${"%.2f".format(d.confidence)} " +
                        "box=[L=${"%.1f".format(d.boundingBox.left)} " +
                        "T=${"%.1f".format(d.boundingBox.top)} " +
                        "R=${"%.1f".format(d.boundingBox.right)} " +
                        "B=${"%.1f".format(d.boundingBox.bottom)}]")
            }
            groundTruths.take(3).forEach { g ->
                Log.d("MetricsDebug", "  GT  cls=${g.classIndex} " +
                        "box=[L=${"%.1f".format(g.boundingBox.left)} " +
                        "T=${"%.1f".format(g.boundingBox.top)} " +
                        "R=${"%.1f".format(g.boundingBox.right)} " +
                        "B=${"%.1f".format(g.boundingBox.bottom)}]")
            }
            // Cross-check every detection vs every GT
            for (d in detections.take(3)) {
                for (g in groundTruths.take(3)) {
                    val iouVal = iou(d.boundingBox, g.boundingBox)
                    Log.d("MetricsDebug", "  IoU(det_cls=${d.classIndex}, gt_cls=${g.classIndex}) = ${"%.4f".format(iouVal)}")
                }
            }
        }
        // ── End debug ────────────────────────────────────────────────────

        for (det in sortedDets) {
            var bestIoU  = iouThreshold
            var bestGtIdx = -1
            for ((idx, gt) in groundTruths.withIndex()) {
                if (idx in matchedGTs) continue
                if (gt.classIndex != det.classIndex) continue
                val iouVal = iou(det.boundingBox, gt.boundingBox)
                if (iouVal >= bestIoU) { bestIoU = iouVal; bestGtIdx = idx }
            }
            if (bestGtIdx >= 0) {
                matchedGTs.add(bestGtIdx)
                results.add(det to groundTruths[bestGtIdx])
            } else {
                results.add(det to null)
            }
        }
        val unmatchedGTs = groundTruths.filterIndexed { idx, _ -> idx !in matchedGTs }
        return results to unmatchedGTs
    }

    /**
     * Compute Average Precision at IoU=0.50 for a single class using
     * the 11-point interpolated AP method.
     */
    private fun computeAP(
        detections: List<Pair<Float, Boolean>>,  // (confidence, isTP)
        numGT: Int
    ): Float {
        if (numGT == 0) return 0f
        val sorted = detections.sortedByDescending { it.first }
        var tp = 0; var fp = 0
        val precisions = mutableListOf<Float>()
        val recalls = mutableListOf<Float>()
        for ((_, isTP) in sorted) {
            if (isTP) tp++ else fp++
            precisions.add(tp.toFloat() / (tp + fp))
            recalls.add(tp.toFloat() / numGT)
        }
        // 11-point interpolation
        var ap = 0f
        val recallThresholds = (0..10).map { it / 10f }
        for (thresh in recallThresholds) {
            val maxP = precisions.zip(recalls)
                .filter { (_, r) -> r >= thresh }
                .maxOfOrNull { (p, _) -> p } ?: 0f
            ap += maxP
        }
        return ap / 11f
    }

    /**
     * Compute all metrics across a batch of image results.
     *
     * @param results  List of per-image inference inputs (detections + ground truths + timing)
     */
    fun computeAggregateMetrics(results: List<ImageMetricsInput>): AggregateMetrics {
        val numClasses = YOLODetector.NUM_CLASSES
        val classNames = YOLODetector.FSL_CLASSES

        val classDetections = Array(numClasses) { mutableListOf<Pair<Float, Boolean>>() }
        val classGTCounts   = IntArray(numClasses)
        val rawMatrix       = Array(numClasses) { IntArray(numClasses) }  // [gtClass][predClass]

        var globalTP = 0; var globalFP = 0; var globalFN = 0; var globalTN = 0
        val inferenceTimes = results.map { it.inferenceTimeMs }

        var debugFired = false
        for (result in results) {
            val tag = if (!debugFired && result.debugTag.isNotEmpty()
                && result.detections.isNotEmpty() && result.groundTruths.isNotEmpty()) {
                debugFired = true
                result.debugTag
            } else ""

            // Single call — reuse matchedPairs for both metrics AND confusion matrix
            val (matchedPairs, unmatchedGTs) = matchDetectionsToGroundTruths(
                result.detections, result.groundTruths, debugTag = tag
            )

            for (gt in result.groundTruths) {
                classGTCounts[gt.classIndex]++
            }

            for ((det, gt) in matchedPairs) {
                if (gt != null) {
                    globalTP++
                    classDetections[det.classIndex].add(det.confidence to true)
                    // TP: row = actual class, col = predicted class
                    rawMatrix[gt.classIndex][det.classIndex]++
                } else {
                    globalFP++
                    classDetections[det.classIndex].add(det.confidence to false)
                }
            }

            globalFN += unmatchedGTs.size
            for (gt in unmatchedGTs) {
                classDetections[gt.classIndex].add(0f to false)
                // FN: predicted nothing — counts against the GT class row but no column to assign
                // Row sum deficit handles this automatically during normalization
            }

            val detectedClasses = result.detections.map { it.classIndex }.toSet()
            val gtClasses       = result.groundTruths.map { it.classIndex }.toSet()
            for (c in 0 until numClasses) {
                if (c !in gtClasses && c !in detectedClasses) globalTN++
            }
        }

        // Row-normalize: divide each cell by the total GT count for that class
        // Using classGTCounts as the denominator instead of rowSum so FNs are included
        val confusionMatrix = Array(numClasses) { r ->
            val total = classGTCounts[r]
            if (total == 0) FloatArray(numClasses)
            else FloatArray(numClasses) { c -> rawMatrix[r][c].toFloat() / total }
        }

        val totalPredictions = globalTP + globalFP + globalTN + globalFN
        val precision = if (globalTP + globalFP > 0) globalTP.toFloat() / (globalTP + globalFP) else 0f
        val recall    = if (globalTP + globalFN > 0) globalTP.toFloat() / (globalTP + globalFN) else 0f
        val accuracy  = if (totalPredictions > 0) (globalTP + globalTN).toFloat() / totalPredictions else 0f
        val f1        = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0f

        val perClassStats = (0 until numClasses).map { c ->
            val gtCount = classGTCounts[c]
            val dets    = classDetections[c]
            val tp  = dets.count { it.second }
            val fp  = dets.count { !it.second && it.first > 0f }
            val fn  = gtCount - tp
            val p   = if (tp + fp > 0) tp.toFloat() / (tp + fp) else 0f
            val r   = if (tp + fn > 0) tp.toFloat() / (tp + fn) else 0f
            val f1c = if (p + r > 0) 2 * p * r / (p + r) else 0f
            val ap  = computeAP(dets, gtCount)
            PerClassStats(classNames[c], tp, fp, fn, 0, p, r, f1c, ap)
        }

        val mAP50 = perClassStats
            .filterIndexed { i, _ -> classGTCounts[i] > 0 }
            .map { it.ap50 }
            .average()
            .toFloat()

        val classF1s = perClassStats
            .filterIndexed { i, _ -> classGTCounts[i] > 0 }
            .map { it.f1 }
        val meanF1   = if (classF1s.isNotEmpty()) classF1s.average().toFloat() else 0f
        val f1StdDev = if (classF1s.size > 1)
            kotlin.math.sqrt(classF1s.map { (it - meanF1) * (it - meanF1) }.average()).toFloat()
        else 0f

        val meanInference   = if (inferenceTimes.isNotEmpty()) inferenceTimes.average().toFloat() else 0f
        val minInference    = inferenceTimes.minOrNull() ?: 0L
        val maxInference    = inferenceTimes.maxOrNull() ?: 0L
        val stdDevInference = if (inferenceTimes.size > 1)
            kotlin.math.sqrt(inferenceTimes.map { (it - meanInference) * (it - meanInference) }.average()).toFloat()
        else 0f

        return AggregateMetrics(
            precision         = precision,
            recall            = recall,
            accuracy          = accuracy,
            f1                = f1,
            mAP50             = mAP50,
            meanInferenceMs   = meanInference,
            minInferenceMs    = minInference,
            maxInferenceMs    = maxInference,
            stdDevInferenceMs = stdDevInference,
            perClassStats     = perClassStats,
            f1StdDev          = f1StdDev,
            totalImages       = results.size,
            totalDetections   = results.sumOf { it.detections.size },
            totalGroundTruths = results.sumOf { it.groundTruths.size },
            confusionMatrix   = confusionMatrix
        )
    }
}
