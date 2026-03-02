package com.fsl.detector.ui

import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.chip.Chip
import com.fsl.detector.databinding.ActivityCameraBinding
import com.fsl.detector.detector.BackendType
import com.fsl.detector.detector.ModelConfig
import com.fsl.detector.detector.YOLODetector
import com.fsl.detector.utils.MetricsCache
import com.fsl.detector.utils.SystemStatsMonitor
import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_BACKEND    = "backend"
        const val EXTRA_CONFIDENCE = "confidence"
        const val EXTRA_IOU        = "iou"
    }

    private lateinit var binding: ActivityCameraBinding
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    private var detector: YOLODetector? = null
    private var selectedModel: ModelConfig? = null
    private var backendType  = BackendType.CPU
    private var confidence   = 0.25f
    private var iou          = 0.45f

    // FPS tracking
    private var frameTimestamps = ArrayDeque<Long>(30)

    // Stats polling job
    private var statsJob: Job? = null

    // Inference lock — skip frame if previous still running
    @Volatile private var inferenceRunning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)
        supportActionBar?.hide()

        backendType = BackendType.valueOf(
            intent.getStringExtra(EXTRA_BACKEND) ?: BackendType.CPU.name
        )
        confidence = intent.getFloatExtra(EXTRA_CONFIDENCE, 0.25f)
        iou        = intent.getFloatExtra(EXTRA_IOU, 0.45f)

        // Populate model chips from MetricsCache (reuse already-scanned models)
        val models = MetricsCache.availableModels
        if (models.isEmpty()) {
            Toast.makeText(this, "No models loaded — scan from main screen first", Toast.LENGTH_LONG).show()
            finish(); return
        }
        buildModelChips(models)
        selectModel(models.first())
        binding.tvActiveModel.text = "Model: ${models.first().displayName}"

        binding.btnFlipCamera.setOnClickListener { flipCamera() }
        binding.btnClose.setOnClickListener      { finish() }

        startCamera()
        startStatsPolling()
    }

    // ── Model chips ──────────────────────────────────────────────────

    private fun buildModelChips(models: List<ModelConfig>) {
        binding.chipGroupModels.removeAllViews()
        models.forEachIndexed { idx, model ->
            val chip = Chip(this).apply {
                id          = View.generateViewId()
                text        = model.displayName
                isCheckable = true
                isChecked   = (idx == 0)

                // Explicit checked/unchecked colors so selection is always visible
                val states = arrayOf(
                    intArrayOf( android.R.attr.state_checked),
                    intArrayOf(-android.R.attr.state_checked)
                )
                val colors = intArrayOf(
                    android.graphics.Color.parseColor("#4ECDC4"),  // checked — teal
                    android.graphics.Color.parseColor("#444444")   // unchecked — dark gray
                )
                chipBackgroundColor = android.content.res.ColorStateList(states, colors)
                setTextColor(android.graphics.Color.WHITE)
                chipStrokeWidth = 0f

                setOnCheckedChangeListener { _, checked ->
                    if (checked) selectModel(model)
                }
            }
            binding.chipGroupModels.addView(chip)
        }
    }

    // Replace selectModel — guards against concurrent switches crashing the detector
    private fun selectModel(model: ModelConfig) {
        if (model == selectedModel) return      // already active, skip
        selectedModel = model
        binding.tvActiveModel.text = "Model: ${model.displayName}"

        // Block inference while swapping detector
        inferenceRunning = true
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                detector?.close()
                detector = null
                detector = YOLODetector(this@CameraActivity, model, backendType, confidence, iou)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@CameraActivity, "Failed to load ${model.displayName}", Toast.LENGTH_SHORT).show()
                }
            } finally {
                inferenceRunning = false
            }
        }
    }

    // ── Camera ───────────────────────────────────────────────────────

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(binding.previewView.surfaceProvider)
        }

        val rotation = binding.previewView.display?.rotation ?: android.view.Surface.ROTATION_0

        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(640, 640))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setTargetRotation(rotation)
            .build()
            .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

        val selector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, selector, preview, analysis)
        } catch (e: Exception) {
            Log.e("CameraActivity", "Camera bind failed", e)
        }
    }

    private fun flipCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
            CameraSelector.LENS_FACING_FRONT
        else
            CameraSelector.LENS_FACING_BACK
        bindCamera()
    }

    // ── Frame analysis ───────────────────────────────────────────────

    private fun analyzeFrame(imageProxy: ImageProxy) {
        if (inferenceRunning) { imageProxy.close(); return }
        val det = detector ?: run { imageProxy.close(); return }

        inferenceRunning = true
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val rawBitmap = imageProxy.toBitmap()
        imageProxy.close()

        // Rotate bitmap to upright orientation before inference
        val bitmap = if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            val rotated = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
            rawBitmap.recycle()
            rotated
        } else rawBitmap

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val output = det.detect(bitmap)
                bitmap.recycle()

                // FPS
                val now = System.currentTimeMillis()
                frameTimestamps.addLast(now)
                while (frameTimestamps.size > 30) frameTimestamps.removeFirst()
                val fps = if (frameTimestamps.size >= 2) {
                    val span = (frameTimestamps.last() - frameTimestamps.first()) / 1000f
                    if (span > 0f) (frameTimestamps.size - 1) / span else 0f
                } else 0f

                withContext(Dispatchers.Main) {
                    binding.overlayView.apply {
                        originalImageWidth  = bitmap.width
                        originalImageHeight = bitmap.height
                        detections = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                            output.detections.map { d ->
                                d.copy(
                                    boundingBox = RectF(
                                        previewWidth - d.boundingBox.right,
                                        d.boundingBox.top,
                                        previewWidth - d.boundingBox.left,
                                        d.boundingBox.bottom
                                    )
                                )
                            }
                        } else output.detections
                    }
                    binding.tvFps.text        = "FPS: ${"%.1f".format(fps)}"
                    binding.tvInferenceMs.text = "${output.inferenceTimeMs}ms"
                }
            } catch (e: Exception) {
                Log.e("CameraActivity", "Inference error", e)
            } finally {
                inferenceRunning = false
            }
        }
    }

    // ── System stats polling ─────────────────────────────────────────

    private fun startStatsPolling() {
        statsJob = lifecycleScope.launch {
            while (isActive) {
                val stats = withContext(Dispatchers.IO) {
                    SystemStatsMonitor.getStats(this@CameraActivity)
                }
                val cpuText = if (stats.cpuPercent >= 0f)
                    "CPU: ${"%.1f".format(stats.cpuPercent)}%"
                else "CPU: N/A"

                val ramText = "RAM: ${stats.ramUsedMb}/${stats.ramTotalMb} MB"

                val gpuText = if (stats.gpuPercent >= 0f)
                    "${stats.gpuLabel}: ${"%.1f".format(stats.gpuPercent)}%"
                else "GPU: N/A"

                binding.tvCpu.text = cpuText
                binding.tvRam.text = ramText
                binding.tvGpu.text = gpuText

                delay(1000L)
            }
        }
    }

    // ── Lifecycle ────────────────────────────────────────────────────

    override fun onDestroy() {
        super.onDestroy()
        statsJob?.cancel()
        cameraExecutor.shutdown()
        detector?.close()
    }
}