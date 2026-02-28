package com.fsl.detector.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import androidx.core.graphics.scale
import android.graphics.Canvas
import android.graphics.Color

/**
 * Unified TFLite detector for YOLOv8, YOLO-NAS, and YOLO11 models.
 *
 * Expected TFLite export shapes:
 *  - YOLOv8 / YOLO11: input [1,640,640,3], output [1, num_classes+4, 8400]
 *    where rows 0-3 are (cx,cy,w,h) and rows 4-N are class scores.
 *  - YOLO-NAS:        input [1,640,640,3], output [1, 8400, num_classes+4]
 *    (transposed format — row-major anchors)
 *
 * If your export differs (e.g., already has sigmoid applied or different anchor count),
 * adjust parseOutputTensor() accordingly.
 */
class YOLODetector(
    private val context: Context,
    private val modelType: ModelType,
    private val backendType: BackendType = BackendType.CPU,
    private val confidenceThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.45f
) {
    private data class LetterboxInfo(val scale: Float, val padLeft: Int, val padTop: Int)

    companion object {
        const val INPUT_SIZE = 640
        const val NUM_CLASSES = 27          // A–Z + Ñ
        val FSL_CLASSES = listOf(
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Ñ"
        )
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    // Detected output layout after loading the model
    private var outputIsAnchorsFirst = false   // true = [1, anchors, coords+classes]
    private var numAnchors = 8400

    init {
        loadModel()
    }

    private val reusableInputBuffer: ByteBuffer by lazy {
        ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
    }


    // ─────────────────────────── Model loading ────────────────────────────

    private fun loadModel() {
        val options = Interpreter.Options().apply {
            numThreads = 4
            when (backendType) {
                BackendType.GPU -> {
                    try {
                        gpuDelegate = GpuDelegate()
                        addDelegate(gpuDelegate!!)
                    } catch (e: Exception) {
                        gpuDelegate = null
                        useXNNPACK = true
                    } catch (e: Error) {
                        // Catches NoClassDefFoundError / UnsatisfiedLinkError
                        gpuDelegate = null
                        useXNNPACK = true
                    }
                }
                BackendType.CPU -> {
                    useXNNPACK = true
                }
            }
        }

        val model = loadModelFile(modelType.assetFileName)
        interpreter = Interpreter(model, options)

        // Inspect output tensor shape to determine layout
        val outputShape = interpreter!!.getOutputTensor(0).shape()
        // outputShape is one of:
        //   [1, 31, 8400]  → columns-first (YOLOv8/YOLO11 default)
        //   [1, 8400, 31]  → anchors-first (YOLO-NAS / some transposed exports)
        if (outputShape.size == 3) {
            outputIsAnchorsFirst = outputShape[2] == (NUM_CLASSES + 4)
            numAnchors = if (outputIsAnchorsFirst) outputShape[1] else outputShape[2]
        }
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val mappedBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileChannel.close()
        fileInputStream.close()
        assetFileDescriptor.close()
        return mappedBuffer
    }

    // ─────────────────────────── Public API ──────────────────────────────

    /** Run inference on a single bitmap and return detections + timing. */
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

    // ─────────────────────────── Pre-processing ──────────────────────────

    private fun preprocessBitmap(bitmap: Bitmap): Pair<ByteBuffer, LetterboxInfo> {
        val scale  = minOf(INPUT_SIZE.toFloat() / bitmap.width, INPUT_SIZE.toFloat() / bitmap.height)
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

        // Reuse buffer — just rewind and overwrite
        reusableInputBuffer.rewind()

        // copyPixelsToBuffer gives ARGB packed ints — extract channels directly
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

    // ─────────────────────────── Output allocation ───────────────────────

    private fun allocateOutputArray(): Array<Array<FloatArray>> {
        return if (outputIsAnchorsFirst) {
            // [1, numAnchors, NUM_CLASSES+4]
            Array(1) { Array(numAnchors) { FloatArray(NUM_CLASSES + 4) } }
        } else {
            // [1, NUM_CLASSES+4, numAnchors]
            Array(1) { Array(NUM_CLASSES + 4) { FloatArray(numAnchors) } }
        }
    }

    // ─────────────────────────── Post-processing ─────────────────────────

    private fun parseAndNMS(
        rawOutput: Array<Array<FloatArray>>,
        lb: LetterboxInfo
    ): List<DetectionResult> {
        val candidates = mutableListOf<DetectionResult>()

        if (outputIsAnchorsFirst) {
            val anchors = rawOutput[0]
            for (anchor in anchors) {
                val cx = anchor[0]; val cy = anchor[1]
                val w  = anchor[2]; val h  = anchor[3]
                val classScores = anchor.copyOfRange(4, 4 + NUM_CLASSES)
                val maxScore = classScores.max()
                val classIdx = classScores.indexOfFirst { it == maxScore }
                if (maxScore >= confidenceThreshold) {
                    val rect = yoloBoxToRect(cx, cy, w, h, lb)
                    candidates.add(DetectionResult(rect, classIdx, FSL_CLASSES[classIdx], maxScore))
                }
            }
        } else {
            val data   = rawOutput[0]
            val cxRow  = data[0]; val cyRow = data[1]
            val wRow   = data[2]; val hRow  = data[3]
            for (i in 0 until numAnchors) {
                var maxScore = -Float.MAX_VALUE
                var classIdx = 0
                for (c in 0 until NUM_CLASSES) {
                    val score = data[4 + c][i]
                    if (score > maxScore) { maxScore = score; classIdx = c }
                }
                if (maxScore >= confidenceThreshold) {
                    val rect = yoloBoxToRect(cxRow[i], cyRow[i], wRow[i], hRow[i], lb)
                    candidates.add(DetectionResult(rect, classIdx, FSL_CLASSES[classIdx], maxScore))
                }
            }
        }

        return nonMaxSuppression(candidates)
    }

    private fun yoloBoxToRect(
        cx: Float, cy: Float, w: Float, h: Float,
        lb: LetterboxInfo
    ): RectF {
        // Model outputs are normalized [0,1] relative to INPUT_SIZE — convert to pixels first
        val cxPx = cx * INPUT_SIZE
        val cyPx = cy * INPUT_SIZE
        val wPx  = w  * INPUT_SIZE
        val hPx  = h  * INPUT_SIZE

        // Undo letterbox: remove padding then undo scale
        val left   = ((cxPx - wPx / 2f) - lb.padLeft) / lb.scale
        val top    = ((cyPx - hPx / 2f) - lb.padTop)  / lb.scale
        val right  = ((cxPx + wPx / 2f) - lb.padLeft) / lb.scale
        val bottom = ((cyPx + hPx / 2f) - lb.padTop)  / lb.scale
        return RectF(left, top, right, bottom)
    }

    // ─────────────────────────── NMS ─────────────────────────────────────

    private fun nonMaxSuppression(detections: List<DetectionResult>): List<DetectionResult> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<DetectionResult>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(best.boundingBox, it.boundingBox) > iouThreshold }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft   = maxOf(a.left, b.left)
        val interTop    = maxOf(a.top, b.top)
        val interRight  = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)
        if (interRight <= interLeft || interBottom <= interTop) return 0f
        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val aArea = a.width() * a.height()
        val bArea = b.width() * b.height()
        return interArea / (aArea + bArea - interArea)
    }

    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
    }
}
