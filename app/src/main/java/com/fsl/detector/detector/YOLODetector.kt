package com.fsl.detector.detector

import ai.onnxruntime.*
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
import java.nio.FloatBuffer
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
        private const val YOLO_NAS_SCORE_THRESHOLD  = 0.01f
        private const val YOLO_NAS_NMS_IOU_THRESHOLD = 0.7f
    }

    // ── TFLite fields ─────────────────────────────────────────────────
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    // ── ONNX Runtime fields ───────────────────────────────────────────
    private var ortEnv:     OrtEnvironment? = null
    private var ortSession: OrtSession?     = null

    // ── Common model properties ───────────────────────────────────────
    private var inputSize            = 640
    private var isQuantized          = false
    private var inputByteSize        = 0
    private var isSplitOutput        = false
    private var outputIsAnchorsFirst = false
    private var numAnchors           = 8400
    private var isOnnx               = false
    // ONNX-specific: NCHW vs NHWC and output tensor names
    private var onnxInputName        = "images"
    private var onnxBoxOutputName    = ""
    private var onnxScoreOutputName  = ""
    private var onnxSingleOutputName = ""

    private data class LetterboxInfo(val scale: Float, val padLeft: Int, val padTop: Int)

    init { loadModel() }

    // ── Model loading ────────────────────────────────────────────────

    private fun loadModel() {
        val name = modelConfig.displayName.lowercase()
        isOnnx   = name.endsWith(".onnx")

        if (isOnnx) loadOnnxModel()
        else        loadTfliteModel()
    }

    // ── ONNX loading ─────────────────────────────────────────────────

    private fun loadOnnxModel() {
        val modelBytes = context.contentResolver.openInputStream(modelConfig.uri)
            ?.use { it.readBytes() }
            ?: throw IllegalArgumentException("Cannot read ONNX model: ${modelConfig.uri}")

        ortEnv = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setInterOpNumThreads(2)
            when (backendType) {
                BackendType.GPU -> {
                    try {
                        // NNAPI is the Android GPU path for ONNX Runtime
                        addNnapi()
                        android.util.Log.d("YOLODetector", "ONNX: NNAPI delegate enabled")
                    } catch (e: Exception) {
                        android.util.Log.w("YOLODetector", "ONNX: NNAPI unavailable, using CPU: ${e.message}")
                    }
                }
                BackendType.CPU -> { /* default CPU */ }
            }
        }
        ortSession = ortEnv!!.createSession(modelBytes, opts)

        // ── Inspect inputs ────────────────────────────────────────
        val inputInfo = ortSession!!.inputInfo
        onnxInputName = inputInfo.keys.first()
        val inputShape = (inputInfo[onnxInputName]?.info as? TensorInfo)?.shape
            ?: longArrayOf(1, 3, 640, 640)
        // ONNX from PyTorch is NCHW: [1, C, H, W]
        inputSize = inputShape[2].toInt().coerceAtLeast(inputShape[3].toInt())

        android.util.Log.d("YOLODetector",
            "ONNX model: ${modelConfig.displayName} | " +
                    "input=$onnxInputName shape=${inputShape.toList()} | " +
                    "outputTensors=${ortSession!!.outputInfo.size}")

        // ── Inspect outputs ───────────────────────────────────────
        val outputNames = ortSession!!.outputNames.toList()
        android.util.Log.d("YOLODetector", "ONNX outputs: $outputNames")

        if (outputNames.size >= 2) {
            // YOLO-NAS: two outputs — boxes + scores
            isSplitOutput     = true
            onnxBoxOutputName   = outputNames[0]
            onnxScoreOutputName = outputNames[1]
            val boxShape   = (ortSession!!.outputInfo[onnxBoxOutputName]?.info as? TensorInfo)?.shape
            numAnchors     = boxShape?.get(1)?.toInt() ?: 8400
            android.util.Log.d("YOLODetector",
                "ONNX split output | boxes=$onnxBoxOutputName " +
                        "scores=$onnxScoreOutputName numAnchors=$numAnchors")
        } else {
            // YOLOv8 / YOLO11: single output
            isSplitOutput        = false
            onnxSingleOutputName = outputNames[0]
            val outShape = (ortSession!!.outputInfo[onnxSingleOutputName]?.info as? TensorInfo)?.shape
            if (outShape != null && outShape.size == 3) {
                outputIsAnchorsFirst = outShape[2] == (NUM_CLASSES + 4).toLong()
                numAnchors = if (outputIsAnchorsFirst) outShape[1].toInt() else outShape[2].toInt()
            }
            android.util.Log.d("YOLODetector",
                "ONNX single output | name=$onnxSingleOutputName " +
                        "shape=${outShape?.toList()} anchorsFirst=$outputIsAnchorsFirst")
        }

        isQuantized   = false  // ONNX models are always float
        inputByteSize = 1 * 3 * inputSize * inputSize * 4  // NCHW float32
    }

    // ── TFLite loading ────────────────────────────────────────────────

    private fun loadTfliteModel() {
        val model = loadModelFromUri(modelConfig.uri)
        interpreter = tryLoadWithGpu(model) ?: tryLoadWithCpu(model)

        val inputTensor = interpreter!!.getInputTensor(0)
        val inputShape  = inputTensor.shape()
        inputSize       = inputShape[1].coerceAtLeast(inputShape[2])
        isQuantized     = inputTensor.dataType() == org.tensorflow.lite.DataType.UINT8
        inputByteSize   = inputTensor.numBytes()

        android.util.Log.d("YOLODetector",
            "TFLite model: ${modelConfig.displayName} | " +
                    "inputSize=$inputSize | isQuantized=$isQuantized | " +
                    "inputBytes=$inputByteSize | outputTensors=${interpreter!!.outputTensorCount}")

        val numOutputs = interpreter!!.outputTensorCount
        if (numOutputs >= 2) {
            isSplitOutput        = true
            outputIsAnchorsFirst = true
            val boxShape = interpreter!!.getOutputTensor(0).shape()
            numAnchors   = boxShape[1]
            android.util.Log.d("YOLODetector",
                "TFLite split output | boxes=${boxShape.toList()} | " +
                        "scores=${interpreter!!.getOutputTensor(1).shape().toList()}")
        } else {
            isSplitOutput = false
            val outputShape = interpreter!!.getOutputTensor(0).shape()
            if (outputShape.size == 3) {
                outputIsAnchorsFirst = outputShape[2] == (NUM_CLASSES + 4)
                numAnchors = if (outputIsAnchorsFirst) outputShape[1] else outputShape[2]
            }
            android.util.Log.d("YOLODetector",
                "TFLite single output | shape=${outputShape.toList()} | " +
                        "anchorsFirst=$outputIsAnchorsFirst | numAnchors=$numAnchors")
        }
    }

    private fun tryLoadWithGpu(model: MappedByteBuffer): Interpreter? {
        if (backendType != BackendType.GPU) return null
        return try {
            val delegate = GpuDelegate()
            val interp   = Interpreter(model, Interpreter.Options().apply {
                numThreads = 4
                addDelegate(delegate)
            })
            gpuDelegate = delegate
            android.util.Log.d("YOLODetector", "TFLite GPU delegate loaded")
            interp
        } catch (e: Exception) {
            android.util.Log.w("YOLODetector", "TFLite GPU failed, falling back: ${e.message}")
            gpuDelegate?.close(); gpuDelegate = null; null
        } catch (e: Error) {
            android.util.Log.w("YOLODetector", "TFLite GPU error, falling back: ${e.message}")
            gpuDelegate?.close(); gpuDelegate = null; null
        }
    }

    private fun tryLoadWithCpu(model: MappedByteBuffer): Interpreter {
        android.util.Log.d("YOLODetector", "TFLite CPU/XNNPACK")
        return Interpreter(model, Interpreter.Options().apply {
            numThreads  = 4
            useXNNPACK = true
        })
    }

    private fun loadModelFromUri(uri: Uri): MappedByteBuffer {
        val pfd = context.contentResolver.openFileDescriptor(uri, "r")
            ?: throw IllegalArgumentException("Cannot open model: $uri")
        val fc  = FileInputStream(pfd.fileDescriptor).channel
        val buf = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size())
        fc.close(); pfd.close()
        return buf
    }

    // ── Public API ───────────────────────────────────────────────────

    fun detect(bitmap: Bitmap): InferenceOutput {
        return if (isOnnx) detectOnnx(bitmap) else detectTflite(bitmap)
    }

    // ── ONNX inference ───────────────────────────────────────────────

    private fun detectOnnx(bitmap: Bitmap): InferenceOutput {
        val t0 = System.currentTimeMillis()
        val (floatArray, lb) = preprocessBitmapToNCHW(bitmap)
        val t1 = System.currentTimeMillis()

        val env     = ortEnv!!
        val session = ortSession!!

        // Shape: [1, 3, H, W] for ONNX (NCHW)
        val shape  = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArray), shape)

        val outputs = session.run(mapOf(onnxInputName to tensor))
        val t2 = System.currentTimeMillis()

        val dets = if (isSplitOutput) {
            parseOnnxSplit(outputs, lb)
        } else {
            parseOnnxSingle(outputs, lb)
        }
        val t3 = System.currentTimeMillis()

        tensor.close(); outputs.close()
        return InferenceOutput(dets, t2 - t1, t1 - t0, t3 - t2)
    }

    private fun parseOnnxSplit(outputs: OrtSession.Result, lb: LetterboxInfo): List<DetectionResult> {
        // YOLO-NAS: output_0=boxes [1,anchors,4], output_1=scores [1,anchors,classes]
        @Suppress("UNCHECKED_CAST")
        val boxTensor   = (outputs[onnxBoxOutputName].get().value   as Array<Array<FloatArray>>)[0]
        @Suppress("UNCHECKED_CAST")
        val scoreTensor = (outputs[onnxScoreOutputName].get().value as Array<Array<FloatArray>>)[0]

        // ── Diagnostic ────────────────────────────────────────────
        var rawMax  = -Float.MAX_VALUE
        var rawMin  = Float.MAX_VALUE
        var rawSum  = 0.0
        val sample  = minOf(100, numAnchors)
        for (i in 0 until sample) {
            for (c in 0 until NUM_CLASSES) {
                val v = scoreTensor[i][c]
                if (v > rawMax) rawMax = v
                if (v < rawMin) rawMin = v
                rawSum += v
            }
        }
        android.util.Log.d("YOLODetector",
            "ONNX-NAS raw scores (first $sample anchors) | " +
                    "min=${"%.4f".format(rawMin)} max=${"%.4f".format(rawMax)} " +
                    "mean=${"%.4f".format(rawSum / (sample * NUM_CLASSES))}")
        // Decide whether to apply sigmoid based on score range
        // If max > 1.0 → raw logits, apply sigmoid
        // If max <= 1.0 → already probabilities, use directly
        val needsSigmoid = rawMax > 1.0f
        android.util.Log.d("YOLODetector", "ONNX-NAS needsSigmoid=$needsSigmoid")
        // ── End diagnostic ────────────────────────────────────────

        val candidates = mutableListOf<DetectionResult>()
        val imgW = (inputSize - 2 * lb.padLeft) / lb.scale
        val imgH = (inputSize - 2 * lb.padTop)  / lb.scale

        for (i in 0 until numAnchors) {
            var maxScore = -Float.MAX_VALUE
            var classIdx = 0
            for (c in 0 until NUM_CLASSES) {
                val prob = if (needsSigmoid) sigmoid(scoreTensor[i][c]) else scoreTensor[i][c]
                if (prob > maxScore) { maxScore = prob; classIdx = c }
            }
            if (maxScore >= YOLO_NAS_SCORE_THRESHOLD) {
                val b    = boxTensor[i]
                val rect = RectF(
                    ((b[0] - lb.padLeft) / lb.scale).coerceIn(0f, imgW),
                    ((b[1] - lb.padTop)  / lb.scale).coerceIn(0f, imgH),
                    ((b[2] - lb.padLeft) / lb.scale).coerceIn(0f, imgW),
                    ((b[3] - lb.padTop)  / lb.scale).coerceIn(0f, imgH)
                )
                candidates.add(DetectionResult(rect, classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        }
        return nonMaxSuppression(candidates, YOLO_NAS_NMS_IOU_THRESHOLD)
            .filter { it.confidence >= confidenceThreshold }
    }

    private fun parseOnnxSingle(outputs: OrtSession.Result, lb: LetterboxInfo): List<DetectionResult> {
        @Suppress("UNCHECKED_CAST")
        val raw = (outputs[onnxSingleOutputName].get().value as Array<Array<FloatArray>>)[0]
        val candidates = mutableListOf<DetectionResult>()
        val imgW = (inputSize - 2 * lb.padLeft) / lb.scale
        val imgH = (inputSize - 2 * lb.padTop)  / lb.scale

        if (outputIsAnchorsFirst) {
            for (anchor in raw) {
                val cx = anchor[0]; val cy = anchor[1]
                val w  = anchor[2]; val h  = anchor[3]
                var maxScore = -Float.MAX_VALUE; var classIdx = 0
                for (c in 0 until NUM_CLASSES) {
                    if (anchor[4 + c] > maxScore) { maxScore = anchor[4 + c]; classIdx = c }
                }
                if (maxScore >= confidenceThreshold)
                    candidates.add(DetectionResult(
                        yoloBoxToRect(cx, cy, w, h, lb, imgW, imgH),
                        classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        } else {
            for (i in 0 until numAnchors) {
                var maxScore = -Float.MAX_VALUE; var classIdx = 0
                for (c in 0 until NUM_CLASSES) {
                    if (raw[4 + c][i] > maxScore) { maxScore = raw[4 + c][i]; classIdx = c }
                }
                if (maxScore >= confidenceThreshold)
                    candidates.add(DetectionResult(
                        yoloBoxToRect(raw[0][i], raw[1][i], raw[2][i], raw[3][i], lb, imgW, imgH),
                        classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        }
        return nonMaxSuppression(candidates, iouThreshold)
    }

    // ── TFLite inference ─────────────────────────────────────────────

    private fun detectTflite(bitmap: Bitmap): InferenceOutput {
        val t0 = System.currentTimeMillis()
        val (inputBuffer, lb) = preprocessBitmapToNHWC(bitmap)
        val t1 = System.currentTimeMillis()

        return if (isSplitOutput) {
            val boxArray   = Array(1) { Array(numAnchors) { FloatArray(4) } }
            val scoreArray = Array(1) { Array(numAnchors) { FloatArray(NUM_CLASSES) } }
            interpreter?.runForMultipleInputsOutputs(
                arrayOf(inputBuffer),
                mapOf(0 to boxArray, 1 to scoreArray)
            )

            // Diagnostic
            val raw = scoreArray[0]
            var rMin = Float.MAX_VALUE; var rMax = -Float.MAX_VALUE; var rSum = 0.0
            val s = minOf(100, numAnchors)
            for (i in 0 until s) for (c in 0 until NUM_CLASSES) {
                val v = raw[i][c]; if (v < rMin) rMin = v; if (v > rMax) rMax = v; rSum += v
            }
            android.util.Log.d("YOLODetector",
                "TFLite-NAS raw scores | min=${"%.4f".format(rMin)} " +
                        "max=${"%.4f".format(rMax)} mean=${"%.4f".format(rSum / (s * NUM_CLASSES))}")

            val t2   = System.currentTimeMillis()
            val dets = parseTfliteSplit(boxArray, scoreArray, lb)
            val t3   = System.currentTimeMillis()
            InferenceOutput(dets, t2 - t1, t1 - t0, t3 - t2)
        } else {
            val outputArray = if (outputIsAnchorsFirst)
                Array(1) { Array(numAnchors) { FloatArray(NUM_CLASSES + 4) } }
            else
                Array(1) { Array(NUM_CLASSES + 4) { FloatArray(numAnchors) } }
            interpreter?.run(inputBuffer, outputArray)
            val t2   = System.currentTimeMillis()
            val dets = parseTfliteSingle(outputArray, lb)
            val t3   = System.currentTimeMillis()
            InferenceOutput(dets, t2 - t1, t1 - t0, t3 - t2)
        }
    }

    private fun parseTfliteSplit(
        boxArray: Array<Array<FloatArray>>,
        scoreArray: Array<Array<FloatArray>>,
        lb: LetterboxInfo
    ): List<DetectionResult> {
        val boxes  = boxArray[0]
        val scores = scoreArray[0]
        val candidates = mutableListOf<DetectionResult>()
        val imgW = (inputSize - 2 * lb.padLeft) / lb.scale
        val imgH = (inputSize - 2 * lb.padTop)  / lb.scale

        for (i in 0 until numAnchors) {
            var maxScore = -Float.MAX_VALUE; var classIdx = 0
            for (c in 0 until NUM_CLASSES) {
                val p = scores[i][c]  // already probabilities from TFLite graph
                if (p > maxScore) { maxScore = p; classIdx = c }
            }
            if (maxScore >= YOLO_NAS_SCORE_THRESHOLD) {
                val b = boxes[i]
                val rect = RectF(
                    ((b[0] - lb.padLeft) / lb.scale).coerceIn(0f, imgW),
                    ((b[1] - lb.padTop)  / lb.scale).coerceIn(0f, imgH),
                    ((b[2] - lb.padLeft) / lb.scale).coerceIn(0f, imgW),
                    ((b[3] - lb.padTop)  / lb.scale).coerceIn(0f, imgH)
                )
                candidates.add(DetectionResult(rect, classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        }
        return nonMaxSuppression(candidates, YOLO_NAS_NMS_IOU_THRESHOLD)
            .filter { it.confidence >= confidenceThreshold }
    }

    private fun parseTfliteSingle(
        rawOutput: Array<Array<FloatArray>>,
        lb: LetterboxInfo
    ): List<DetectionResult> {
        val candidates = mutableListOf<DetectionResult>()
        val imgW = (inputSize - 2 * lb.padLeft) / lb.scale
        val imgH = (inputSize - 2 * lb.padTop)  / lb.scale

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
                        yoloBoxToRect(cx, cy, w, h, lb, imgW, imgH),
                        classIdx, FSL_CLASSES[classIdx], maxScore))
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
                        yoloBoxToRect(data[0][i], data[1][i], data[2][i], data[3][i], lb, imgW, imgH),
                        classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        }
        return nonMaxSuppression(candidates, iouThreshold)
    }

    // ── Pre-processing ───────────────────────────────────────────────

    // ONNX: NCHW [1, 3, H, W] — returns FloatArray
    private fun preprocessBitmapToNCHW(bitmap: Bitmap): Pair<FloatArray, LetterboxInfo> {
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

        val pixels = IntArray(inputSize * inputSize)
        letterboxed.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        letterboxed.recycle()

        // NCHW: channel-first layout [C, H, W] flattened
        val result = FloatArray(3 * inputSize * inputSize)
        val hw     = inputSize * inputSize
        for (i in pixels.indices) {
            val px = pixels[i]
            // YOLO-NAS: normalization baked in graph — pass raw [0, 255]
            result[i]          = ((px shr 16) and 0xFF).toFloat()  // R channel
            result[hw + i]     = ((px shr 8)  and 0xFF).toFloat()  // G channel
            result[2 * hw + i] = (px           and 0xFF).toFloat()  // B channel
        }

        return result to LetterboxInfo(scale, padLeft, padTop)
    }

    // TFLite: NHWC [1, H, W, 3] — returns ByteBuffer
    private fun preprocessBitmapToNHWC(bitmap: Bitmap): Pair<ByteBuffer, LetterboxInfo> {
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

        val buf = ByteBuffer.allocateDirect(inputByteSize).order(ByteOrder.nativeOrder())
        val argbBuf = ByteBuffer.allocateDirect(inputSize * inputSize * 4)
        letterboxed.copyPixelsToBuffer(argbBuf)
        letterboxed.recycle()
        argbBuf.rewind()

        while (argbBuf.hasRemaining()) {
            val r = argbBuf.get().toInt() and 0xFF
            val g = argbBuf.get().toInt() and 0xFF
            val b = argbBuf.get().toInt() and 0xFF
            argbBuf.get() // skip alpha
            if (isQuantized) {
                buf.put(r.toByte()); buf.put(g.toByte()); buf.put(b.toByte())
            } else if (isSplitOutput) {
                // YOLO-NAS TFLite: normalization baked in — raw [0, 255]
                buf.putFloat(r.toFloat()); buf.putFloat(g.toFloat()); buf.putFloat(b.toFloat())
            } else {
                // YOLOv8 / YOLO11: [0, 1]
                buf.putFloat(r / 255f); buf.putFloat(g / 255f); buf.putFloat(b / 255f)
            }
        }
        buf.rewind()
        return buf to LetterboxInfo(scale, padLeft, padTop)
    }

    // ── Shared helpers ───────────────────────────────────────────────

    private fun yoloBoxToRect(
        cx: Float, cy: Float, w: Float, h: Float,
        lb: LetterboxInfo, imgW: Float, imgH: Float
    ): RectF {
        val cxPx = cx * inputSize; val cyPx = cy * inputSize
        val wPx  = w  * inputSize; val hPx  = h  * inputSize
        return RectF(
            (((cxPx - wPx / 2f) - lb.padLeft) / lb.scale).coerceIn(0f, imgW),
            (((cyPx - hPx / 2f) - lb.padTop)  / lb.scale).coerceIn(0f, imgH),
            (((cxPx + wPx / 2f) - lb.padLeft) / lb.scale).coerceIn(0f, imgW),
            (((cyPx + hPx / 2f) - lb.padTop)  / lb.scale).coerceIn(0f, imgH)
        )
    }

    private fun sigmoid(x: Float): Float =
        1f / (1f + Math.exp(-x.toDouble())).toFloat()

    private fun nonMaxSuppression(
        detections: List<DetectionResult>,
        nmsIouThreshold: Float
    ): List<DetectionResult> {
        return detections
            .groupBy { it.classIndex }
            .values
            .flatMap { cls ->
                val sorted = cls.sortedByDescending { it.confidence }.toMutableList()
                val kept   = mutableListOf<DetectionResult>()
                while (sorted.isNotEmpty()) {
                    val best = sorted.removeAt(0)
                    kept.add(best)
                    sorted.removeAll { iou(best.boundingBox, it.boundingBox) > nmsIouThreshold }
                }
                kept
            }
    }

    private fun iou(a: RectF, b: RectF): Float {
        val iL = maxOf(a.left, b.left);  val iT = maxOf(a.top, b.top)
        val iR = minOf(a.right, b.right); val iB = minOf(a.bottom, b.bottom)
        if (iR <= iL || iB <= iT) return 0f
        val inter = (iR - iL) * (iB - iT)
        return inter / (a.width() * a.height() + b.width() * b.height() - inter)
    }

    // ── Lifecycle ────────────────────────────────────────────────────

    fun close() {
        interpreter?.close();  gpuDelegate?.close()
        ortSession?.close();   ortEnv?.close()
        interpreter = null;    gpuDelegate = null
        ortSession  = null;    ortEnv      = null
    }
}