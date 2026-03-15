package com.fsl.detector.ui

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import androidx.documentfile.provider.DocumentFile
import com.fsl.detector.detector.BackendType
import com.fsl.detector.detector.ModelConfig
import com.fsl.detector.detector.YOLODetector
import com.fsl.detector.metrics.MetricsCalculator
import com.fsl.detector.utils.LabelUtils
import com.fsl.detector.utils.MetricsCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class DetectorViewModel(application: Application) : AndroidViewModel(application) {

    var selectedModel: ModelConfig? = null
    var selectedBackend: BackendType = BackendType.CPU
    var confidenceThreshold: Float = 0.25f
    var iouThreshold: Float = 0.45f

    private var detector: YOLODetector? = null

    val availableModels   = MutableLiveData<List<ModelConfig>>(emptyList())
    val singleImageResult = MutableLiveData<com.fsl.detector.detector.InferenceOutput?>()
    val batchProgress     = MutableLiveData<Pair<Int, Int>>()
    val batchMetrics      = MutableLiveData<MetricsCalculator.AggregateMetrics?>()
    val isLoading         = MutableLiveData(false)
    val errorMessage      = MutableLiveData<String?>()

    // ── Model discovery ──────────────────────────────────────────────

    fun scanForModels(folderUri: Uri) {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val models = withContext(Dispatchers.IO) {
                val found = mutableListOf<ModelConfig>()
                val dir = DocumentFile.fromTreeUri(context, folderUri)
                    ?: return@withContext found

                android.util.Log.d("DetectorViewModel",
                    "Scanning folder: ${dir.name} | total files: ${dir.listFiles().size}")

                dir.listFiles().forEach { file ->
                    when {
                        file.isFile && (
                                file.name?.endsWith(".tflite", ignoreCase = true) == true ||
                                        file.name?.endsWith(".onnx",   ignoreCase = true) == true
                                ) -> {
                            found.add(ModelConfig(
                                displayName = file.name!!,
                                uri         = file.uri
                            ))
                            android.util.Log.d("DetectorViewModel", "Found model: ${file.name}")
                        }
                        file.isDirectory -> {
                            file.listFiles().forEach { sub ->
                                if (sub.isFile && (
                                            sub.name?.endsWith(".tflite", ignoreCase = true) == true ||
                                                    sub.name?.endsWith(".onnx",   ignoreCase = true) == true
                                            )) {
                                    found.add(ModelConfig(
                                        displayName = sub.name!!,
                                        uri         = sub.uri
                                    ))
                                    android.util.Log.d("DetectorViewModel",
                                        "Found model (subdir): ${sub.name}")
                                }
                            }
                        }
                    }
                }
                found.sortedBy { it.displayName }
            }

            android.util.Log.d("DetectorViewModel", "Total models found: ${models.size}")
            availableModels.value = models
            if (models.isNotEmpty() && selectedModel == null) {
                selectedModel = models.first()
            }
        }
    }

    // ── Detector lifecycle ───────────────────────────────────────────

    private fun getOrCreateDetector(): YOLODetector {
        val current = detector
        if (current != null) return current
        val model = selectedModel ?: throw IllegalStateException("No model selected")
        val d = YOLODetector(
            getApplication(), model, selectedBackend,
            confidenceThreshold, iouThreshold
        )
        detector = d
        return d
    }

    fun resetDetector() {
        detector?.close()
        detector = null
    }

    // ── Single image ─────────────────────────────────────────────────

    fun runSingleImage(bitmap: Bitmap) {
        viewModelScope.launch {
            isLoading.value = true
            errorMessage.value = null
            try {
                val result = withContext(Dispatchers.Default) {
                    getOrCreateDetector().detect(bitmap)
                }
                singleImageResult.value = result
            } catch (e: Exception) {
                errorMessage.value = "Inference failed: ${e.message}"
                singleImageResult.value = null
            } finally {
                isLoading.value = false
            }
        }
    }

    // ── Batch processing ─────────────────────────────────────────────

    fun runBatchProcessing(folderUri: Uri) {
        viewModelScope.launch {
            isLoading.value = true
            batchMetrics.value  = null
            errorMessage.value  = null
            try {
                val metrics = withContext(Dispatchers.Default) {
                    val context    = getApplication<Application>()
                    val imageFiles = LabelUtils.getImageDocuments(folderUri, context)
                    if (imageFiles.isEmpty())
                        throw IllegalArgumentException("No images found in selected folder")

                    val dirIndex = LabelUtils.buildDirectoryIndex(folderUri, context)
                    val det      = getOrCreateDetector()

                    // ONNX Runtime needs more warmup iterations for JIT graph optimization
                    val isOnnxModel = selectedModel?.displayName
                        ?.endsWith(".onnx", ignoreCase = true) == true
                    val warmupCount = if (isOnnxModel) 20 else 10
                    android.util.Log.d("DetectorViewModel",
                        "Warming up model (${if (isOnnxModel) "ONNX" else "TFLite"}) " +
                                "with $warmupCount iterations")

                    val firstBitmap = LabelUtils.decodeBitmapFromDocument(
                        imageFiles.first(), context)
                    if (firstBitmap != null) {
                        repeat(warmupCount) { det.detect(firstBitmap) }
                        firstBitmap.recycle()
                    }

                    data class DecodedItem(
                        val bitmap: Bitmap,
                        val gts: List<com.fsl.detector.detector.GroundTruth>,
                        val index: Int,
                        val name: String
                    )

                    val channel  = Channel<DecodedItem>(capacity = 3)
                    val producer = launch(Dispatchers.IO) {
                        imageFiles.forEachIndexed { idx, docFile ->
                            val bmp = LabelUtils.decodeBitmapFromDocument(docFile, context)
                                ?: return@forEachIndexed
                            val gts = LabelUtils.loadGroundTruthsFromIndex(
                                docFile, dirIndex, context)
                            channel.send(DecodedItem(bmp, gts, idx, docFile.name ?: ""))
                        }
                        channel.close()
                    }

                    val results = mutableListOf<MetricsCalculator.ImageMetricsInput>()
                    for (item in channel) {
                        val output    = det.detect(item.bitmap)
                        val scaledGTs = item.gts.map { gt ->
                            gt.copy(boundingBox = android.graphics.RectF(
                                gt.boundingBox.left   * item.bitmap.width,
                                gt.boundingBox.top    * item.bitmap.height,
                                gt.boundingBox.right  * item.bitmap.width,
                                gt.boundingBox.bottom * item.bitmap.height
                            ))
                        }
                        val debugTag = if (item.index == 0) item.name else ""
                        results.add(MetricsCalculator.ImageMetricsInput(
                            output.detections, scaledGTs,
                            output.inferenceTimeMs, debugTag
                        ))
                        item.bitmap.recycle()
                        withContext(Dispatchers.Main) {
                            batchProgress.value = (item.index + 1) to imageFiles.size
                        }
                    }
                    producer.join()
                    MetricsCalculator.computeAggregateMetrics(results)
                }

                MetricsCache.lastConfidenceThreshold = confidenceThreshold
                MetricsCache.lastIouThreshold        = iouThreshold
                MetricsCache.lastMetrics             = metrics
                MetricsCache.lastModelName           = selectedModel?.displayName ?: ""
                MetricsCache.lastBackend             = selectedBackend.displayName
                batchMetrics.value = metrics

            } catch (e: Exception) {
                errorMessage.value = "Batch processing failed: ${e.message}"
            } finally {
                isLoading.value = false
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        detector?.close()
    }
}