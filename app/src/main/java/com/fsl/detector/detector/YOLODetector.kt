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
        const val INPUT_SIZE  = 640
        const val NUM_CLASSES = 27
        val FSL_CLASSES = listOf(
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Ñ"
        )
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var outputIsAnchorsFirst = false
    private var numAnchors = 8400

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

        val outputShape = interpreter!!.getOutputTensor(0).shape()
        if (outputShape.size == 3) {
            outputIsAnchorsFirst = outputShape[2] == (NUM_CLASSES + 4)
            numAnchors = if (outputIsAnchorsFirst) outputShape[1] else outputShape[2]
        }
    }

    private fun loadModelFromUri(uri: Uri): MappedByteBuffer {
        // Works for both content:// Uris (SAF) and file:// Uris
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
        val preprocessStart = System.currentTimeMillis()
        val (inputBuffer, lb) = preprocessBitmap(bitmap)
        val preprocessTime = System.currentTimeMillis() - preprocessStart

        val outputArray = allocateOutputArray()

        val inferenceStart = System.currentTimeMillis()
        interpreter?.run(inputBuffer, outputArray)
        val inferenceTime = System.currentTimeMillis() - inferenceStart

        val postprocessStart = System.currentTimeMillis()
        val detections = parseAndNMS(outputArray, lb)
        val postprocessTime = System.currentTimeMillis() - postprocessStart

        return InferenceOutput(detections, inferenceTime, preprocessTime, postprocessTime)
    }

    // ── Pre-processing ───────────────────────────────────────────────

    private val reusableInputBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .apply { order(ByteOrder.nativeOrder()) }
    }

    private fun preprocessBitmap(bitmap: Bitmap): Pair<ByteBuffer, LetterboxInfo> {
        val scale   = minOf(INPUT_SIZE.toFloat() / bitmap.width, INPUT_SIZE.toFloat() / bitmap.height)
        val scaledW = (bitmap.width  * scale).toInt()
        val scaledH = (bitmap.height * scale).toInt()
        val padLeft = (INPUT_SIZE - scaledW) / 2
        val padTop  = (INPUT_SIZE - scaledH) / 2

        val letterboxed = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxed)
        canvas.drawColor(Color.argb(255, 114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        canvas.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), null)
        scaled.recycle()

        reusableInputBuffer.rewind()
        val argbBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 4)
        letterboxed.copyPixelsToBuffer(argbBuffer)
        letterboxed.recycle()
        argbBuffer.rewind()

        while (argbBuffer.hasRemaining()) {
            val r = (argbBuffer.get().toInt() and 0xFF) / 255f
            val g = (argbBuffer.get().toInt() and 0xFF) / 255f
            val b = (argbBuffer.get().toInt() and 0xFF) / 255f
            argbBuffer.get() // skip alpha
            reusableInputBuffer.putFloat(r)
            reusableInputBuffer.putFloat(g)
            reusableInputBuffer.putFloat(b)
        }
        reusableInputBuffer.rewind()
        return reusableInputBuffer to LetterboxInfo(scale, padLeft, padTop)
    }

    // ── Output allocation ────────────────────────────────────────────

    private fun allocateOutputArray(): Array<Array<FloatArray>> =
        if (outputIsAnchorsFirst)
            Array(1) { Array(numAnchors) { FloatArray(NUM_CLASSES + 4) } }
        else
            Array(1) { Array(NUM_CLASSES + 4) { FloatArray(numAnchors) } }

    // ── Post-processing ──────────────────────────────────────────────

    private fun parseAndNMS(rawOutput: Array<Array<FloatArray>>, lb: LetterboxInfo): List<DetectionResult> {
        val candidates = mutableListOf<DetectionResult>()
        if (outputIsAnchorsFirst) {
            for (anchor in rawOutput[0]) {
                val cx = anchor[0]; val cy = anchor[1]; val w = anchor[2]; val h = anchor[3]
                val classScores = anchor.copyOfRange(4, 4 + NUM_CLASSES)
                val maxScore = classScores.max()
                val classIdx = classScores.indexOfFirst { it == maxScore }
                if (maxScore >= confidenceThreshold)
                    candidates.add(DetectionResult(yoloBoxToRect(cx, cy, w, h, lb), classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        } else {
            val data = rawOutput[0]
            for (i in 0 until numAnchors) {
                var maxScore = -Float.MAX_VALUE; var classIdx = 0
                for (c in 0 until NUM_CLASSES) {
                    val score = data[4 + c][i]
                    if (score > maxScore) { maxScore = score; classIdx = c }
                }
                if (maxScore >= confidenceThreshold)
                    candidates.add(DetectionResult(yoloBoxToRect(data[0][i], data[1][i], data[2][i], data[3][i], lb), classIdx, FSL_CLASSES[classIdx], maxScore))
            }
        }
        return nonMaxSuppression(candidates)
    }

    private fun yoloBoxToRect(cx: Float, cy: Float, w: Float, h: Float, lb: LetterboxInfo): RectF {
        val cxPx = cx * INPUT_SIZE; val cyPx = cy * INPUT_SIZE
        val wPx  = w  * INPUT_SIZE; val hPx  = h  * INPUT_SIZE
        return RectF(
            ((cxPx - wPx / 2f) - lb.padLeft) / lb.scale,
            ((cyPx - hPx / 2f) - lb.padTop)  / lb.scale,
            ((cxPx + wPx / 2f) - lb.padLeft) / lb.scale,
            ((cyPx + hPx / 2f) - lb.padTop)  / lb.scale
        )
    }

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

    fun close() {
        interpreter?.close(); gpuDelegate?.close()
        interpreter = null;   gpuDelegate = null
    }
}