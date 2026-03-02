package com.fsl.detector.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.chip.Chip
import com.fsl.detector.databinding.ActivityMainBinding
import com.fsl.detector.detector.BackendType
import com.fsl.detector.detector.ModelConfig
import com.fsl.detector.utils.MetricsCache

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: DetectorViewModel by viewModels()
    private var pendingAction: (() -> Unit)? = null

    // ── Launchers ────────────────────────────────────────────────────

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { launchSingleImageActivity(it) } }

    private val batchFolderLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        uri?.let {
            contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            viewModel.runBatchProcessing(it)
        }
    }

    private val modelScanLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        uri?.let {
            contentResolver.takePersistableUriPermission(it, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            viewModel.scanForModels(it)
        }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results.values.all { it }) pendingAction?.invoke()
        else Toast.makeText(this, "Storage permission required", Toast.LENGTH_LONG).show()
        pendingAction = null
    }

    // ── Lifecycle ────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupBackendSelector()
        setupSliders()
        setupButtons()
        observeViewModel()
    }

    // ── Setup ────────────────────────────────────────────────────────

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
            addOnChangeListener { _, v, _ ->
                viewModel.confidenceThreshold = v
                viewModel.resetDetector()
                binding.tvConfidence.text = "Confidence threshold: ${"%.2f".format(v)}"
            }
        }
        binding.sliderIou.apply {
            value = viewModel.iouThreshold
            addOnChangeListener { _, v, _ ->
                viewModel.iouThreshold = v
                viewModel.resetDetector()
                binding.tvIou.text = "IoU (NMS) threshold: ${"%.2f".format(v)}"
            }
        }
        binding.tvConfidence.text = "Confidence threshold: ${"%.2f".format(viewModel.confidenceThreshold)}"
        binding.tvIou.text        = "IoU (NMS) threshold: ${"%.2f".format(viewModel.iouThreshold)}"
    }

    private fun setupButtons() {
        binding.btnScanModels.setOnClickListener {
            withPermission { modelScanLauncher.launch(null) }
        }
        binding.btnSingleImage.setOnClickListener {
            if (viewModel.selectedModel == null) {
                Toast.makeText(this, "Scan for models first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            withPermission { imagePickerLauncher.launch("image/*") }
        }
        binding.btnBatchFolder.setOnClickListener {
            if (viewModel.selectedModel == null) {
                Toast.makeText(this, "Scan for models first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            withPermission { batchFolderLauncher.launch(null) }
        }
        binding.btnLiveCamera.setOnClickListener {
            if (MetricsCache.availableModels.isEmpty()) {
                Toast.makeText(this, "Scan for models first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val intent = Intent(this, CameraActivity::class.java).apply {
                putExtra(CameraActivity.EXTRA_BACKEND,    viewModel.selectedBackend.name)
                putExtra(CameraActivity.EXTRA_CONFIDENCE, viewModel.confidenceThreshold)
                putExtra(CameraActivity.EXTRA_IOU,        viewModel.iouThreshold)
            }
            startActivity(intent)
        }
    }

    private fun observeViewModel() {
        // Rebuild model chips whenever the scanned list changes
        viewModel.availableModels.observe(this) { models ->
            binding.chipGroupModel.removeAllViews()
            if (models.isEmpty()) {
                binding.tvActiveModel.text = "No .tflite files found — tap Scan"
                return@observe
            }
            models.forEachIndexed { idx, model ->
                val chip = Chip(this).apply {
                    id   = View.generateViewId()
                    text = model.displayName
                    isCheckable  = true
                    isChecked    = (idx == 0)
                    setOnCheckedChangeListener { _, checked ->
                        if (checked) {
                            viewModel.selectedModel = model
                            viewModel.resetDetector()
                            binding.tvActiveModel.text = "Active: ${model.displayName}"
                        }
                    }
                }
                binding.chipGroupModel.addView(chip)
            }
            // Select first by default
            viewModel.selectedModel = models.first()
            binding.tvActiveModel.text = "Active: ${models.first().displayName}"
        }

        viewModel.isLoading.observe(this) { loading ->
            binding.progressBar.visibility  = if (loading) View.VISIBLE else View.GONE
            binding.btnSingleImage.isEnabled = !loading
            binding.btnBatchFolder.isEnabled = !loading
            binding.btnScanModels.isEnabled  = !loading
        }

        viewModel.batchProgress.observe(this) { (done, total) ->
            binding.tvBatchProgress.visibility = View.VISIBLE
            binding.tvBatchProgress.text = "Processing: $done / $total images"
        }

        viewModel.batchMetrics.observe(this) { metrics ->
            metrics ?: return@observe
            binding.tvBatchProgress.visibility = View.GONE
            startActivity(Intent(this, ResultsActivity::class.java))
        }

        viewModel.errorMessage.observe(this) { msg ->
            msg ?: return@observe
            binding.tvBatchProgress.visibility = View.GONE
            AlertDialog.Builder(this).setTitle("Error").setMessage(msg)
                .setPositiveButton("OK", null).show()
        }
    }

    // ── Navigation ───────────────────────────────────────────────────

    private fun launchSingleImageActivity(uri: Uri) {
        val intent = Intent(this, SingleImageActivity::class.java).apply {
            putExtra(SingleImageActivity.EXTRA_IMAGE_URI,   uri.toString())
            putExtra(SingleImageActivity.EXTRA_MODEL_URI,   viewModel.selectedModel!!.uri.toString())
            putExtra(SingleImageActivity.EXTRA_MODEL_NAME,  viewModel.selectedModel!!.displayName)
            putExtra(SingleImageActivity.EXTRA_BACKEND,     viewModel.selectedBackend.name)
            putExtra(SingleImageActivity.EXTRA_CONFIDENCE,  viewModel.confidenceThreshold)
            putExtra(SingleImageActivity.EXTRA_IOU,         viewModel.iouThreshold)
        }
        startActivity(intent)
    }

    // ── Permissions ──────────────────────────────────────────────────

    private fun withPermission(action: () -> Unit) {
        val perms = buildList {
            add(Manifest.permission.CAMERA)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
                add(Manifest.permission.READ_MEDIA_IMAGES)
            else
                add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }.toTypedArray()

        if (perms.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED })
            action()
        else { pendingAction = action; permissionLauncher.launch(perms) }
    }
}