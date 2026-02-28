package com.fsl.detector.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.fsl.detector.databinding.ActivityMainBinding
import com.fsl.detector.detector.BackendType
import com.fsl.detector.detector.ModelType
import com.fsl.detector.utils.LabelUtils
import com.fsl.detector.utils.MetricsCache
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: DetectorViewModel by viewModels()

    // ─── Launchers ───────────────────────────────────────────────────────

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { launchSingleImageActivity(it) }
    }

    private val folderPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        uri?.let {
            contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            viewModel.runBatchProcessing(it)
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results.values.all { it }) {
            pendingAction?.invoke()
        } else {
            Toast.makeText(this, "Storage permission is required to process images", Toast.LENGTH_LONG).show()
        }
        pendingAction = null
    }

    private var pendingAction: (() -> Unit)? = null

    // ─── Lifecycle ───────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupModelSelector()
        setupBackendSelector()
        setupSliders()
        setupButtons()
        observeViewModel()
    }

    // ─── Setup ───────────────────────────────────────────────────────────

    private fun setupModelSelector() {
        binding.chipGroupModel.setOnCheckedStateChangeListener { _, checkedIds ->
            val model = when (checkedIds.firstOrNull()) {
                binding.chipYolov8.id  -> ModelType.YOLOV8
                binding.chipYoloNas.id -> ModelType.YOLO_NAS
                binding.chipYolo11.id  -> ModelType.YOLO11
                else -> return@setOnCheckedStateChangeListener
            }
            viewModel.selectedModel = model
            viewModel.resetDetector()
            binding.tvActiveModel.text = "Active: ${model.displayName} (${model.assetFileName})"
        }
        binding.chipYolov8.isChecked = true
        binding.tvActiveModel.text = "Active: ${ModelType.YOLOV8.displayName} (${ModelType.YOLOV8.assetFileName})"
    }

    private fun setupBackendSelector() {
        binding.chipGroupBackend.setOnCheckedStateChangeListener { _, checkedIds ->
            val backend = when (checkedIds.firstOrNull()) {
                binding.chipCpu.id -> BackendType.CPU
                binding.chipGpu.id -> BackendType.GPU
                else -> return@setOnCheckedStateChangeListener
            }
            viewModel.selectedBackend = backend
            viewModel.resetDetector()
        }
        binding.chipCpu.isChecked = true
    }

    private fun setupSliders() {
        binding.sliderConfidence.apply {
            value = viewModel.confidenceThreshold
            addOnChangeListener { _, value, _ ->
                viewModel.confidenceThreshold = value
                viewModel.resetDetector()
                binding.tvConfidence.text = "Confidence threshold: ${"%.2f".format(value)}"
            }
        }
        binding.sliderIou.apply {
            value = viewModel.iouThreshold
            addOnChangeListener { _, value, _ ->
                viewModel.iouThreshold = value
                viewModel.resetDetector()
                binding.tvIou.text = "IoU (NMS) threshold: ${"%.2f".format(value)}"
            }
        }
        binding.tvConfidence.text = "Confidence threshold: ${"%.2f".format(viewModel.confidenceThreshold)}"
        binding.tvIou.text = "IoU (NMS) threshold: ${"%.2f".format(viewModel.iouThreshold)}"
    }

    private fun setupButtons() {
        binding.btnSingleImage.setOnClickListener {
            withPermission { imagePickerLauncher.launch("image/*") }
        }
        binding.btnBatchFolder.setOnClickListener {
            withPermission { folderPickerLauncher.launch(null) }
        }
    }

    private fun observeViewModel() {
        viewModel.isLoading.observe(this) { loading ->
            binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
            binding.btnSingleImage.isEnabled = !loading
            binding.btnBatchFolder.isEnabled = !loading
        }

        viewModel.batchProgress.observe(this) { (done, total) ->
            binding.tvBatchProgress.visibility = View.VISIBLE
            binding.tvBatchProgress.text = "Processing: $done / $total images"
        }

        viewModel.batchMetrics.observe(this) { metrics ->
            metrics ?: return@observe
            binding.tvBatchProgress.visibility = View.GONE
            MetricsCache.lastModelName = viewModel.selectedModel.displayName
            MetricsCache.lastBackend = viewModel.selectedBackend.displayName
            startActivity(Intent(this, ResultsActivity::class.java))
        }

        viewModel.errorMessage.observe(this) { msg ->
            msg ?: return@observe
            binding.tvBatchProgress.visibility = View.GONE
            AlertDialog.Builder(this)
                .setTitle("Error")
                .setMessage(msg)
                .setPositiveButton("OK", null)
                .show()
        }
    }

    // ─── Navigation ──────────────────────────────────────────────────────

    private fun launchSingleImageActivity(uri: Uri) {
        val intent = Intent(this, SingleImageActivity::class.java).apply {
            putExtra(SingleImageActivity.EXTRA_IMAGE_URI, uri.toString())
            putExtra(SingleImageActivity.EXTRA_MODEL, viewModel.selectedModel.name)
            putExtra(SingleImageActivity.EXTRA_BACKEND, viewModel.selectedBackend.name)
            putExtra(SingleImageActivity.EXTRA_CONFIDENCE, viewModel.confidenceThreshold)
            putExtra(SingleImageActivity.EXTRA_IOU, viewModel.iouThreshold)
        }
        startActivity(intent)
    }

    // ─── Permissions ─────────────────────────────────────────────────────

    private fun withPermission(action: () -> Unit) {
        val perms = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        val allGranted = perms.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
        if (allGranted) action() else {
            pendingAction = action
            permissionLauncher.launch(perms)
        }
    }
}
