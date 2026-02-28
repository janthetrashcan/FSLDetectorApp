package com.fsl.detector.ui

import android.app.Application
import android.content.ContentResolver
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.fsl.detector.detector.BackendType
import com.fsl.detector.detector.InferenceOutput
import com.fsl.detector.detector.ModelType
import com.fsl.detector.detector.YOLODetector
import com.fsl.detector.metrics.MetricsCalculator
import com.fsl.detector.utils.LabelUtils
import com.fsl.detector.utils.MetricsCache.lastMetrics
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import android.util.Log
import com.fsl.detector.detector.GroundTruth
import com.fsl.detector.utils.MetricsCache
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

class DetectorViewModel(application: Application) : AndroidViewModel(application) {

    var selectedModel: ModelType = ModelType.YOLOV8
    var selectedBackend: BackendType = BackendType.CPU
    var confidenceThreshold: Float = 0.25f
    var iouThreshold: Float = 0.45f

    private var detector: YOLODetector? = null

    val singleImageResult = MutableLiveData<InferenceOutput?>()
    val batchProgress     = MutableLiveData<Pair<Int, Int>>()  // processed / total
    val batchMetrics      = MutableLiveData<MetricsCalculator.AggregateMetrics?>()
    val isLoading         = MutableLiveData(false)
    val errorMessage      = MutableLiveData<String?>()

    private fun getOrCreateDetector(): YOLODetector {
        val current = detector
        if (current != null) return current
        val d = YOLODetector(
            getApplication(),
            selectedModel,
            selectedBackend,
            confidenceThreshold,
            iouThreshold
        )
        detector = d
        return d
    }

    fun resetDetector() {
        detector?.close()
        detector = null
    }

    /** Run inference on a single bitmap. */
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

    /**
     * Process all images in a directory.
     * Expects paired YOLO-format .txt label files for metrics computation.
     * Runs 10 warm-up frames before measuring, averaged over all images.
     */
    fun runBatchProcessing(folderUri: Uri) {
        viewModelScope.launch {
            isLoading.value = true
            batchMetrics.value = null
            errorMessage.value = null
            try {
                val metrics = withContext(Dispatchers.Default) {
                    val context    = getApplication<Application>()
                    val imageFiles = LabelUtils.getImageDocuments(folderUri, context)
                    if (imageFiles.isEmpty()) throw IllegalArgumentException("No images found in selected folder")

                    val dirIndex = LabelUtils.buildDirectoryIndex(folderUri, context)
                    val det      = getOrCreateDetector()

                    // Warm-up
                    val firstBitmap = LabelUtils.decodeBitmapFromDocument(imageFiles.first(), context)
                    if (firstBitmap != null) {
                        repeat(10) { det.detect(firstBitmap) }
                        firstBitmap.recycle()
                    }

                    // Pipeline: Channel buffers up to 3 decoded bitmaps ahead
                    // so IO and inference overlap instead of running sequentially
                    data class DecodedItem(
                        val bitmap: Bitmap,
                        val gts: List<GroundTruth>,
                        val index: Int,
                        val name: String
                    )

                    val channel = Channel<DecodedItem>(capacity = 3)

                    // Producer coroutine: decode on IO thread
                    val producer = launch(Dispatchers.IO) {
                        imageFiles.forEachIndexed { idx, docFile ->
                            val bmp = LabelUtils.decodeBitmapFromDocument(docFile, context) ?: return@forEachIndexed
                            val gts = LabelUtils.loadGroundTruthsFromIndex(docFile, dirIndex, context)
                            channel.send(DecodedItem(bmp, gts, idx, docFile.name ?: ""))
                        }
                        channel.close()
                    }

                    // Consumer: inference runs on Default thread while producer decodes next image
                    val results = mutableListOf<MetricsCalculator.ImageMetricsInput>()
                    for (item in channel) {
                        val output = det.detect(item.bitmap)

                        val scaledGTs = item.gts.map { gt ->
                            gt.copy(
                                boundingBox = android.graphics.RectF(
                                    gt.boundingBox.left   * item.bitmap.width,
                                    gt.boundingBox.top    * item.bitmap.height,
                                    gt.boundingBox.right  * item.bitmap.width,
                                    gt.boundingBox.bottom * item.bitmap.height
                                )
                            )
                        }

                        val debugTag = if (item.index == 0) item.name else ""
                        results.add(MetricsCalculator.ImageMetricsInput(
                            output.detections, scaledGTs, output.inferenceTimeMs, debugTag
                        ))

                        item.bitmap.recycle()

                        withContext(Dispatchers.Main) {
                            batchProgress.value = (item.index + 1) to imageFiles.size
                        }
                    }

                    producer.join()
                    MetricsCalculator.computeAggregateMetrics(results)
                }
                MetricsCache.lastMetrics    = metrics
                MetricsCache.lastModelName  = selectedModel.displayName
                MetricsCache.lastBackend    = selectedBackend.displayName
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
