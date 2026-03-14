package com.fsl.detector.metrics

import android.graphics.RectF
import android.util.Log
import com.fsl.detector.detector.DetectionResult
import com.fsl.detector.detector.GroundTruth
import com.fsl.detector.detector.YOLODetector
import kotlin.math.max

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
        // (n+1)×(n+1): rows=actual class, cols=predicted class
        // last row  (index n) = FP  — detection with no matching GT
        // last col  (index n) = FN  — GT that was never detected
        // off-diagonal (r!=c, both <n) = misclassification
        val confusionMatrix: Array<FloatArray>,  // row-normalised
        val rawConfusionMatrix: Array<IntArray>  // raw counts
    )

    private const val IOU_THRESHOLD_50 = 0.50f

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

    // ── Same-class matching — used for TP/FP/FN/AP metrics ──────────

    fun matchDetectionsToGroundTruths(
        detections: List<DetectionResult>,
        groundTruths: List<GroundTruth>,
        iouThreshold: Float = IOU_THRESHOLD_50,
        debugTag: String = ""
    ): Pair<List<Pair<DetectionResult, GroundTruth?>>, List<GroundTruth>> {
        val sortedDets = detections.sortedByDescending { it.confidence }
        val matchedGTs = mutableSetOf<Int>()
        val results    = mutableListOf<Pair<DetectionResult, GroundTruth?>>()

        if (debugTag.isNotEmpty()) {
            Log.d("MetricsDebug", "=== $debugTag ===")
            Log.d("MetricsDebug", "  Detections: ${detections.size}, GTs: ${groundTruths.size}")
            detections.take(5).forEach { d ->
                Log.d("MetricsDebug", "  DET cls=${d.classIndex}(${d.className}) " +
                        "conf=${"%.3f".format(d.confidence)} " +
                        "box=[L=${"%.1f".format(d.boundingBox.left)} " +
                        "T=${"%.1f".format(d.boundingBox.top)} " +
                        "R=${"%.1f".format(d.boundingBox.right)} " +
                        "B=${"%.1f".format(d.boundingBox.bottom)}]")
            }
            groundTruths.take(5).forEach { g ->
                Log.d("MetricsDebug", "  GT  cls=${g.classIndex} " +
                        "box=[L=${"%.1f".format(g.boundingBox.left)} " +
                        "T=${"%.1f".format(g.boundingBox.top)} " +
                        "R=${"%.1f".format(g.boundingBox.right)} " +
                        "B=${"%.1f".format(g.boundingBox.bottom)}]")
            }
            detections.take(3).forEach { d ->
                groundTruths.take(3).forEach { g ->
                    val iouVal = iou(d.boundingBox, g.boundingBox)
                    Log.d("MetricsDebug",
                        "  IoU(det_cls=${d.classIndex}, gt_cls=${g.classIndex}) " +
                                "= ${"%.4f".format(iouVal)}" +
                                if (d.classIndex != g.classIndex) " ← CLASS MISMATCH" else "")
                }
            }
        }

        for (det in sortedDets) {
            var bestIoU   = iouThreshold
            var bestGtIdx = -1
            for ((idx, gt) in groundTruths.withIndex()) {
                if (idx in matchedGTs) continue
                if (gt.classIndex != det.classIndex) continue  // same-class only
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

    // ── IoU-only matching — used exclusively for confusion matrix ────
    // Matches each GT to the highest-IoU detection regardless of class.
    // Returns list of (gtClassIndex, predClassIndex or null if FN).
    private fun matchForConfusionMatrix(
        detections: List<DetectionResult>,
        groundTruths: List<GroundTruth>,
        iouThreshold: Float = IOU_THRESHOLD_50
    ): List<Pair<Int?, Int?>> {
        val sortedDets  = detections.sortedByDescending { it.confidence }
        val matchedDets = mutableSetOf<Int>()
        val results     = mutableListOf<Pair<Int?, Int?>>()  // gtClass → predClass|null

        for (gt in groundTruths) {
            var bestIoU   = iouThreshold
            var bestDetIdx = -1
            for ((idx, det) in sortedDets.withIndex()) {
                if (idx in matchedDets) continue
                val iouVal = iou(gt.boundingBox, det.boundingBox)
                if (iouVal >= bestIoU) { bestIoU = iouVal; bestDetIdx = idx }
            }
            if (bestDetIdx >= 0) {
                matchedDets.add(bestDetIdx)
                // gt.classIndex → det.classIndex (diagonal=TP, off-diagonal=misclassification)
                results.add(gt.classIndex to sortedDets[bestDetIdx].classIndex)
            } else {
                // FN: GT was never detected
                results.add(gt.classIndex to null)
            }
        }

        // FPs: detections not matched to any GT
        for ((idx, det) in sortedDets.withIndex()) {
            if (idx !in matchedDets) {
                results.add(Pair<Int?, Int?>(null, det.classIndex))
            }
        }

        return results
    }

    // ── AP computation ───────────────────────────────────────────────

    private fun computeAP(
        detections: List<Pair<Float, Boolean>>,
        numGT: Int
    ): Float {
        if (numGT == 0) return 0f
        val sorted = detections.sortedByDescending { it.first }
        var tp = 0; var fp = 0
        val precisions = mutableListOf<Float>()
        val recalls    = mutableListOf<Float>()
        for ((_, isTP) in sorted) {
            if (isTP) tp++ else fp++
            precisions.add(tp.toFloat() / (tp + fp))
            recalls.add(tp.toFloat() / numGT)
        }
        var ap = 0f
        for (thresh in (0..10).map { it / 10f }) {
            val maxP = precisions.zip(recalls)
                .filter { (_, r) -> r >= thresh }
                .maxOfOrNull { (p, _) -> p } ?: 0f
            ap += maxP
        }
        return ap / 11f
    }

    // ── Main aggregation ─────────────────────────────────────────────

    fun computeAggregateMetrics(results: List<ImageMetricsInput>): AggregateMetrics {
        val numClasses = YOLODetector.NUM_CLASSES
        val classNames = YOLODetector.FSL_CLASSES
        val bg         = numClasses  // background index in (n+1)×(n+1) matrix

        val classDetections = Array(numClasses) { mutableListOf<Pair<Float, Boolean>>() }
        val classGTCounts   = IntArray(numClasses)

        // (n+1)×(n+1): [gtClass][predClass], bg=last index
        val rawMatrix = Array(numClasses + 1) { IntArray(numClasses + 1) }

        var globalTP = 0; var globalFP = 0; var globalFN = 0; var globalTN = 0
        val inferenceTimes = results.map { it.inferenceTimeMs }

        var debugFired = false
        for (result in results) {
            val tag = if (!debugFired && result.debugTag.isNotEmpty()
                && result.detections.isNotEmpty() && result.groundTruths.isNotEmpty()) {
                debugFired = true
                result.debugTag
            } else ""

            // ── Pass 1: same-class matching for TP/FP/FN/AP ─────
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
                } else {
                    globalFP++
                    classDetections[det.classIndex].add(det.confidence to false)
                }
            }

            globalFN += unmatchedGTs.size
            for (gt in unmatchedGTs) {
                classDetections[gt.classIndex].add(0f to false)
            }

            val detectedClasses = result.detections.map { it.classIndex }.toSet()
            val gtClasses       = result.groundTruths.map { it.classIndex }.toSet()
            for (c in 0 until numClasses) {
                if (c !in gtClasses && c !in detectedClasses) globalTN++
            }

            // ── Pass 2: IoU-only matching for confusion matrix ───
            val cmMatches = matchForConfusionMatrix(result.detections, result.groundTruths)
            for ((gtCls, predCls) in cmMatches) {
                when {
                    gtCls != null && predCls != null -> rawMatrix[gtCls][predCls]++
                    gtCls != null && predCls == null -> rawMatrix[gtCls][bg]++
                    gtCls == null && predCls != null -> rawMatrix[bg][predCls]++
                }
            }
        }

        // Row-normalise: each actual-class row divides by GT count,
        // BG row (FPs) divides by total FP count
        val confusionMatrix = Array(numClasses + 1) { r ->
            if (r < numClasses) {
                val total = classGTCounts[r]
                if (total == 0) FloatArray(numClasses + 1)
                else FloatArray(numClasses + 1) { c -> rawMatrix[r][c].toFloat() / total }
            } else {
                val total = rawMatrix[bg].sum()
                if (total == 0) FloatArray(numClasses + 1)
                else FloatArray(numClasses + 1) { c -> rawMatrix[bg][c].toFloat() / total }
            }
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
            precision           = precision,
            recall              = recall,
            accuracy            = accuracy,
            f1                  = f1,
            mAP50               = mAP50,
            meanInferenceMs     = meanInference,
            minInferenceMs      = minInference,
            maxInferenceMs      = maxInference,
            stdDevInferenceMs   = stdDevInference,
            perClassStats       = perClassStats,
            f1StdDev            = f1StdDev,
            totalImages         = results.size,
            totalDetections     = results.sumOf { it.detections.size },
            totalGroundTruths   = results.sumOf { it.groundTruths.size },
            confusionMatrix     = confusionMatrix,
            rawConfusionMatrix  = rawMatrix
        )
    }
}