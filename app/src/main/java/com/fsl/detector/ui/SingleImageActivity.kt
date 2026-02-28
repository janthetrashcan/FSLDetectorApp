package com.fsl.detector.ui

import android.net.Uri
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.fsl.detector.databinding.ActivitySingleImageBinding
import com.fsl.detector.detector.BackendType
import com.fsl.detector.detector.ModelType
import com.fsl.detector.detector.YOLODetector
import com.fsl.detector.utils.LabelUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SingleImageActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_IMAGE_URI   = "image_uri"
        const val EXTRA_MODEL       = "model"
        const val EXTRA_BACKEND     = "backend"
        const val EXTRA_CONFIDENCE  = "confidence"
        const val EXTRA_IOU         = "iou"
    }

    private lateinit var binding: ActivitySingleImageBinding
    private var detector: YOLODetector? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySingleImageBinding.inflate(layoutInflater)
        setContentView(binding.root)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        title = "Single Image Inference"

        val uriStr    = intent.getStringExtra(EXTRA_IMAGE_URI) ?: return
        val modelName = intent.getStringExtra(EXTRA_MODEL) ?: ModelType.YOLOV8.name
        val backendName = intent.getStringExtra(EXTRA_BACKEND) ?: BackendType.CPU.name
        val confidence  = intent.getFloatExtra(EXTRA_CONFIDENCE, 0.25f)
        val iou         = intent.getFloatExtra(EXTRA_IOU, 0.45f)

        val modelType   = ModelType.valueOf(modelName)
        val backendType = BackendType.valueOf(backendName)
        val imageUri    = Uri.parse(uriStr)

        runInference(imageUri, modelType, backendType, confidence, iou)
    }

    private fun runInference(
        uri: Uri,
        modelType: ModelType,
        backendType: BackendType,
        confidence: Float,
        iou: Float
    ) {
        binding.progressBar.visibility = View.VISIBLE
        binding.groupResults.visibility = View.GONE

        lifecycleScope.launch {
            try {
                val (bitmap, output) = withContext(Dispatchers.Default) {
                    val bmp = LabelUtils.decodeBitmapFromUri(this@SingleImageActivity, uri)
                        ?: throw Exception("Could not decode image")
                    val det = YOLODetector(
                        this@SingleImageActivity, modelType, backendType, confidence, iou
                    )
                    detector = det
                    bmp to det.detect(bmp)
                }

                // Show image with overlay
                binding.imageView.setImageBitmap(bitmap)
                binding.overlayView.apply {
                    originalImageWidth  = bitmap.width
                    originalImageHeight = bitmap.height
                    detections = output.detections
                }

                // Display timing info
                binding.tvModel.text    = "Model: ${modelType.displayName} | Backend: ${backendType.displayName}"
                binding.tvDetections.text = "Detections found: ${output.detections.size}"
                binding.tvPreprocess.text = "Preprocess: ${output.preprocessTimeMs} ms"
                binding.tvInference.text  = "Inference:  ${output.inferenceTimeMs} ms"
                binding.tvPostprocess.text = "Postprocess: ${output.postprocessTimeMs} ms"
                binding.tvTotal.text   = "Total:      ${output.totalTimeMs} ms"

                // Build detection list text
                val detText = if (output.detections.isEmpty()) "No detections above threshold."
                else output.detections.joinToString("\n") { det ->
                    "  • ${det.className}: ${"%.1f".format(det.confidence * 100)}%  " +
                    "Box: [${det.boundingBox.left.toInt()}, ${det.boundingBox.top.toInt()}, " +
                    "${det.boundingBox.right.toInt()}, ${det.boundingBox.bottom.toInt()}]"
                }
                binding.tvDetectionList.text = detText

                binding.progressBar.visibility = View.GONE
                binding.groupResults.visibility = View.VISIBLE

            } catch (e: Exception) {
                binding.progressBar.visibility = View.GONE
                binding.tvDetectionList.text = "Error: ${e.message}"
                binding.groupResults.visibility = View.VISIBLE
            }
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        finish(); return true
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
    }
}
