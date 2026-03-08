package com.fsl.detector.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.RectF
import android.net.Uri
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class YOLODetector(
    private val context: Context,
    private val modelConfig: ModelConfig,
    private val backendType: BackendType = BackendType.CPU,
    private val confidenceThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.45f
) {
    companion object {
        const val NUM_CLASSES = 27
        val FSL_CLASSES = listOf(
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Ñ"
        )
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    // Input properties — read from model at load time
    private var inputSize     = 640
    private var isQuantized   = false
    private var inputByteSize = 0

    // Output properties — read from model at load time
    private var isSplitOutput        = false
    private var outputIsAnchorsFirst = false
    private var numAnchors           = 8400

    private data class LetterboxInfo(val scale: Float, val padLeft: Int, val padTop: Int)

    init { loadModel() }

    // ── Model loading ────────────────────────────────────────────────

    private fun loadModel() {
        val options = Interpreter.Options().apply {
            numThreads = 4
            when (backendType) {
                BackendType.GPU -> {
                    var gpuLoaded = false
                    try {
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate!!)
                        gpuLoaded = true
                    } catch (e: Exception) { gpuDelegate = null }
                    catch (e: Error)     { gpuDelegate = null }
                    if (!gpuLoaded) useXNNPACK = true
                }
                BackendType.CPU -> useXNNPACK = true
            }
        }

        val model = loadModelFromUri(modelConfig.uri)
        interpreter = Interpreter(model, options)

        // ── Input tensor ─────────────────────────────────────────
        val inputTensor = interpreter!!.getInputTensor(0)
        val inputShape  = inputTensor.shape()   // [1, H, W, C]
        inputSize       = inputShape[1].coerceAtLeast(inputShape[2])
        isQuantized     = inputTensor.dataType() == org.tensorflow.lite.DataType.UINT8
        inputByteSize   = inputTensor.numBytes()

        android.util.Log.d("YOLODetector",
            "Model: ${modelConfig.displayName} | " +
                    "inputSize=$inputSize | isQuantized=$isQuantized | " +
                    "inputBytes=$inputByteSize | outputTensors=${interpreter!!.outputTensorCount}")

        // ── Output tensors — check count FIRST ───────────────────
        val numOutputs = interpreter!!.outputTensorCount

        if (numOutputs >= 2) {
            // YOLO-NAS: output_0 = [1, anchors, 4]  output_1 = [1, anchors, num_classes]
            isSplitOutput        = true
            outputIsAnchorsFirst = true
            val boxShape = interpreter!!.getOutputTensor(0).shape()
            numAnchors   = boxShape[1]

            android.util.Log.d("YOLODetector",
                "Split output | boxes=${boxShape.toList()} | " +
                        "scores=${interpreter!!.getOutputTensor(1).shape().toList()}")
        } else {
            // YOLOv8 / YOLO11: single tensor [1, 31, 8400] or [1, 8400, 31]
            isSplitOutput = false
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            if (outputShape.size == 3) {
                outputIsAnchorsFirst = outputShape[2] == (NUM_CLASSES + 4)
                numAnchors = if (outputIsAnchorsFirst) outputShape[1] else outputShape[2]
            }

            android.util.Log.d("YOLODetector",
                "Single output | shape=${outputShape.toList()} | " +
                        "anchorsFirst=$outputIsAnchorsFirst | numAnchors=$numAnchors")
        }
    }

    private fun loadModelFromUri(uri: Uri): MappedByteBuffer {
        val pfd = context.contentResolver.openFileDescriptor(uri, "r")
            ?: throw IllegalArgumentException("Cannot open model file: $uri")
        val fileChannel = FileInputStream(pfd.fileDescriptor).channel
        val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size())
        fileChannel.close()
        pfd.close()
        return buffer
    }

    // ── Public API ───────────────────────────────────────────────────

    fun detect(bitmap: Bitmap): InferenceOutput {
        val t0 = System.currentTimeMillis()
        val (inputBuffer, lb) = preprocessBitmap(bitmap)
        val t1 = System.currentTimeMillis()

        return if (isSplitOutput) {
            // YOLO-NAS: two output tensors
            val boxArray   = Array(1) { Array(numAnchors) { FloatArray(4) } }
            val scoreArray = Array(1) { Array(numAnchors) { FloatArray(NUM_CLASSES) } }
            val outputs    = mapOf(0 to boxArray, 1 to scoreArray)
            interpreter?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

            // ── Raw logit diagnostic (first inference only) ───────
            val rawScores = scoreArray[0]
            var rawMin = Float.MAX_VALUE
            var rawMax = -Float.MAX_VALUE
            var rawSum = 0.0
            val sampleAnchors = minOf(100, numAnchors)
            for (i in 0 until sampleAnchors) {
                for (c in 0 until NUM_CLASSES) {
                    val v = rawScores[i][c]
                    if (v < rawMin) rawMin = v
                    if (v > rawMax) rawMax = v
                    rawSum += v
                }
            }
            android.util.Log.d("YOLODetector",
                "Raw logits (first $sampleAnchors anchors) | " +
                        "min=${"%.4f".format(rawMin)} " +
                        "max=${"%.4f".format(rawMax)} " +
                        "mean=${"%.4f".format(rawSum / (sampleAnchors * NUM_CLASSES))}")
            // ── End diagnostic ────────────────────────────────────

            val t2   = System.currentTimeMillis()
            val dets = parseAndNMSSplit(boxArray, scoreArray, lb)
            val t3   = System.currentTimeMillis()
            InferenceOutput(dets, t2 - t1, t1 - t0, t3 - t2)
        } else {
            // YOLOv8 / YOLO11: single output tensor
            val outputArray = if (outputIsAnchorsFirst)
                Array(1) { Array(numAnchors) { FloatArray(NUM_CLASSES + 4) } }
            else
                Array(1) { Array(NUM_CLASSES + 4) { FloatArray(numAnchors) } }
            interpreter?.run(inputBuffer, outputArray)
            val t2   = System.currentTimeMillis()
            val dets = parseAndNMS(outputArray, lb)
            val t3   = System.currentTimeMillis()
            InferenceOutput(dets, t2 - t1, t1 - t0, t3 - t2)
        }
    }

    // ── Pre-processing ───────────────────────────────────────────────

    private fun preprocessBitmap(bitmap: Bitmap): Pair<ByteBuffer, LetterboxInfo> {
        val scale   = minOf(inputSize.toFloat() / bitmap.width, inputSize.toFloat() / bitmap.height)
        val scaledW = (bitmap.width  * scale).toInt()
        val scaledH = (bitmap.height * scale).toInt()
        val padLeft = (inputSize - scaledW) / 2
        val padTop  = (inputSize - scaledH) / 2

        val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxed)
        canvas.drawColor(Color.argb(255, 114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        canvas.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), null)
        scaled.recycle()

        val buf = ByteBuffer.allocateDirect(inputByteSize)
        buf.order(ByteOrder.nativeOrder())

        val argbBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 4)
        letterboxed.copyPixelsToBuffer(argbBuffer)
        letterboxed.recycle()
        argbBuffer.rewind()

        while (argbBuffer.hasRemaining()) {
            val r = argbBuffer.get().toInt() and 0xFF
            val g = argbBuffer.get().toInt() and 0xFF
            val b = argbBuffer.get().toInt() and 0xFF
            argbBuffer.get() // skip alpha

            if (isQuantized) {
                // INT8 quantized: pass raw bytes
                buf.put(r.toByte())
                buf.put(g.toByte())
                buf.put(b.toByte())
            } else if (isSplitOutput) {
                // YOLO-NAS: normalization is baked into TFLite graph — pass raw [0, 255] floats.
                // If max logit is still ~0.017 after this, try BGR order instead:
                //   buf.putFloat(b.toFloat())
                //   buf.putFloat(g.toFloat())
                //   buf.putFloat(r.toFloat())
                buf.putFloat(r.toFloat())
                buf.putFloat(g.toFloat())
                buf.putFloat(b.toFloat())
            } else {
                // YOLOv8 / YOLO11: simple [0, 1] normalization
                buf.putFloat(r / 255f)
                buf.putFloat(g / 255f)
                buf.putFloat(b / 255f)
            }
        }
        buf.rewind()
        return buf to LetterboxInfo(scale, padLeft, padTop)
    }

    // ── Post-processing: single tensor (YOLOv8 / YOLO11) ────────────

    private fun parseAndNMS(
        rawOutput: Array<Array<FloatArray>>,
        lb: LetterboxInfo
    ): List<DetectionResult> {
        val candidates = mutableListOf<DetectionResult>()
        if (outputIsAnchorsFirst) {
            for (anchor in rawOutput[0]) {
                val cx = anchor[0]; val cy = anchor[1]
                val w  = anchor[2]; val h  = anchor[3]
                var maxScore = -Float.MAX_VALUE; var classIdx = 0
                for (c in 0 until NUM_CLASSES) {
                    if (anchor[4 + c] > maxScore) { maxScore = anchor[4 + c]; classIdx = c }
                }
                if (maxScore >= confidenceThreshold)
                    candidates.add(DetectionResult(
                        yoloBoxToRect(cx, cy, w, h, lb), classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        } else {
            val data = rawOutput[0]
            for (i in 0 until numAnchors) {
                var maxScore = -Float.MAX_VALUE; var classIdx = 0
                for (c in 0 until NUM_CLASSES) {
                    if (data[4 + c][i] > maxScore) { maxScore = data[4 + c][i]; classIdx = c }
                }
                if (maxScore >= confidenceThreshold)
                    candidates.add(DetectionResult(
                        yoloBoxToRect(data[0][i], data[1][i], data[2][i], data[3][i], lb),
                        classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        }
        return nonMaxSuppression(candidates)
    }

    // ── Post-processing: split tensors (YOLO-NAS) ────────────────────

    private fun parseAndNMSSplit(
        boxArray:   Array<Array<FloatArray>>,
        scoreArray: Array<Array<FloatArray>>,
        lb: LetterboxInfo
    ): List<DetectionResult> {
        val candidates = mutableListOf<DetectionResult>()
        val boxes  = boxArray[0]
        val scores = scoreArray[0]

        // ── Diagnostic ────────────────────────────────────────────────
        var maxScoreOverall = -Float.MAX_VALUE
        var maxScoreAnchor  = 0
        var maxScoreClass   = 0
        var aboveThreshold  = 0
        for (i in 0 until numAnchors) {
            for (c in 0 until NUM_CLASSES) {
                val prob = scores[i][c]  // already a probability — no sigmoid
                if (prob > maxScoreOverall) {
                    maxScoreOverall = prob
                    maxScoreAnchor  = i
                    maxScoreClass   = c
                }
                if (prob >= confidenceThreshold) aboveThreshold++
            }
        }
        android.util.Log.d("YOLODetector",
            "YOLO-NAS scores (raw) | max=${"%.4f".format(maxScoreOverall)} " +
                    "at anchor=$maxScoreAnchor cls=$maxScoreClass(${FSL_CLASSES[maxScoreClass]}) | " +
                    "above threshold($confidenceThreshold): $aboveThreshold")

        data class AS(val idx: Int, val cls: Int, val prob: Float)
        val top5 = mutableListOf<AS>()
        for (i in 0 until numAnchors) {
            var best = -Float.MAX_VALUE; var bestC = 0
            for (c in 0 until NUM_CLASSES) {
                val p = scores[i][c]
                if (p > best) { best = p; bestC = c }
            }
            top5.add(AS(i, bestC, best))
        }
        top5.sortByDescending { it.prob }
        top5.take(5).forEach {
            val b = boxes[it.idx]
            android.util.Log.d("YOLODetector",
                "  top anchor[${it.idx}] cls=${it.cls}(${FSL_CLASSES[it.cls]}) " +
                        "prob=${"%.4f".format(it.prob)} " +
                        "box=[${b[0]}, ${b[1]}, ${b[2]}, ${b[3]}]")
        }
        // ── End diagnostic ─────────────────────────────────────────────

        for (i in 0 until numAnchors) {
            var maxScore = -Float.MAX_VALUE
            var classIdx = 0
            for (c in 0 until NUM_CLASSES) {
                val prob = scores[i][c]  // raw probability directly
                if (prob > maxScore) { maxScore = prob; classIdx = c }
            }
            if (maxScore >= confidenceThreshold) {
                val b = boxes[i]
                val rect = RectF(
                    (b[0] - lb.padLeft) / lb.scale,
                    (b[1] - lb.padTop)  / lb.scale,
                    (b[2] - lb.padLeft) / lb.scale,
                    (b[3] - lb.padTop)  / lb.scale
                )
                candidates.add(DetectionResult(rect, classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        }
        return nonMaxSuppression(candidates)
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private fun sigmoid(x: Float): Float =
        1f / (1f + Math.exp(-x.toDouble())).toFloat()

    // YOLOv8/YOLO11: normalized cxcywh → original image pixel xyxy via letterbox inversion
    private fun yoloBoxToRect(
        cx: Float, cy: Float, w: Float, h: Float,
        lb: LetterboxInfo
    ): RectF {
        val cxPx = cx * inputSize
        val cyPx = cy * inputSize
        val wPx  = w  * inputSize
        val hPx  = h  * inputSize
        return RectF(
            ((cxPx - wPx / 2f) - lb.padLeft) / lb.scale,
            ((cyPx - hPx / 2f) - lb.padTop)  / lb.scale,
            ((cxPx + wPx / 2f) - lb.padLeft) / lb.scale,
            ((cyPx + hPx / 2f) - lb.padTop)  / lb.scale
        )
    }

    // ── NMS ──────────────────────────────────────────────────────────

    private fun nonMaxSuppression(detections: List<DetectionResult>): List<DetectionResult> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept   = mutableListOf<DetectionResult>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(best.boundingBox, it.boundingBox) > iouThreshold }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val iL = maxOf(a.left, b.left); val iT = maxOf(a.top, b.top)
        val iR = minOf(a.right, b.right); val iB = minOf(a.bottom, b.bottom)
        if (iR <= iL || iB <= iT) return 0f
        val inter = (iR - iL) * (iB - iT)
        return inter / (a.width() * a.height() + b.width() * b.height() - inter)
    }

    // ── Lifecycle ────────────────────────────────────────────────────

    fun close() {
        interpreter?.close(); gpuDelegate?.close()
        interpreter = null;   gpuDelegate = null
    }
}