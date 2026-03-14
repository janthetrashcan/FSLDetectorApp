package com.fsl.detector.ui

import android.content.ContentValues
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.fsl.detector.databinding.ActivityResultsBinding
import com.fsl.detector.detector.YOLODetector
import com.fsl.detector.metrics.MetricsCalculator
import com.fsl.detector.utils.MetricsCache
import com.github.mikephil.charting.data.BarData
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import org.json.JSONObject
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ResultsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val metrics   = MetricsCache.lastMetrics   ?: return
        val modelName = MetricsCache.lastModelName ?: "Unknown"
        val backend   = MetricsCache.lastBackend   ?: "CPU"

        setupSummary(metrics, modelName, backend)
        setupBarChart(metrics)
        setupPerClassTable(metrics)
        setupConfusionMatrix(metrics)
        setupExport(metrics, modelName, backend)
    }

    // ── Summary ──────────────────────────────────────────────────────

    private fun setupSummary(
        m: MetricsCalculator.AggregateMetrics,
        model: String,
        backend: String
    ) {
        binding.tvSummaryHeader.text = "$model  ·  $backend"
        binding.tvTotalImages.text   = "Images evaluated   : ${m.totalImages}"
        binding.tvTotalDets.text     = "Total detections   : ${m.totalDetections}"
        binding.tvTotalGTs.text      = "Total ground truths: ${m.totalGroundTruths}"
        binding.tvPrecision.text     = "Precision  : ${"%.4f".format(m.precision)}"
        binding.tvRecall.text        = "Recall     : ${"%.4f".format(m.recall)}"
        binding.tvAccuracy.text      = "Accuracy   : ${"%.4f".format(m.accuracy)}"
        binding.tvF1.text            = "F1-Score   : ${"%.4f".format(m.f1)}  (±${"%.4f".format(m.f1StdDev)})"
        binding.tvMap50.text         = "mAP@50     : ${"%.4f".format(m.mAP50)}"
        binding.tvInfMean.text       = "Inference mean : ${"%.1f".format(m.meanInferenceMs)} ms"
        binding.tvInfMin.text        = "Inference min  : ${m.minInferenceMs} ms"
        binding.tvInfMax.text        = "Inference max  : ${m.maxInferenceMs} ms"
        binding.tvInfStd.text        = "Inference σ    : ${"%.1f".format(m.stdDevInferenceMs)} ms"
    }

    // ── Bar chart ────────────────────────────────────────────────────

    private fun setupBarChart(m: MetricsCalculator.AggregateMetrics) {
        val entries = m.perClassStats.mapIndexed { i, s -> BarEntry(i.toFloat(), s.f1) }
        val dataSet = BarDataSet(entries, "F1 per class").apply {
            colors = m.perClassStats.map { s ->
                when {
                    s.f1 >= 0.8f -> android.graphics.Color.parseColor("#27AE60")
                    s.f1 >= 0.5f -> android.graphics.Color.parseColor("#F39C12")
                    else         -> android.graphics.Color.parseColor("#E74C3C")
                }
            }
            setDrawValues(false)
        }

        val textColor = resolveAttrColor(android.R.attr.textColorPrimary)
        binding.barChart.apply {
            data = BarData(dataSet).apply { barWidth = 0.85f }
            setBackgroundColor(android.graphics.Color.TRANSPARENT)
            description.isEnabled = false
            legend.isEnabled      = false
            xAxis.apply {
                valueFormatter     = IndexAxisValueFormatter(YOLODetector.FSL_CLASSES)
                granularity        = 1f
                labelRotationAngle = -45f
                this.textColor     = textColor
                setDrawGridLines(false)
            }
            axisLeft.apply {
                axisMinimum    = 0f
                axisMaximum    = 1f
                this.textColor = textColor
            }
            axisRight.isEnabled = false
            animateY(600)
            invalidate()
        }
    }

    // ── Per-class table ──────────────────────────────────────────────

    private fun setupPerClassTable(m: MetricsCalculator.AggregateMetrics) {
        binding.rvPerClass.apply {
            layoutManager = LinearLayoutManager(this@ResultsActivity)
            adapter       = PerClassAdapter(m.perClassStats)
        }
    }

    // ── Confusion matrix ─────────────────────────────────────────────

    private fun setupConfusionMatrix(m: MetricsCalculator.AggregateMetrics) {
        binding.confusionMatrixView.matrix = m.confusionMatrix
    }

    // ── Export ───────────────────────────────────────────────────────

    private fun setupExport(
        m: MetricsCalculator.AggregateMetrics,
        modelName: String,
        backend: String
    ) {
        binding.btnExport.setOnClickListener {
            binding.btnExport.isEnabled       = false
            binding.tvExportStatus.visibility = View.VISIBLE
            binding.tvExportStatus.text       = "Exporting…"

            try {
                val stamp    = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val safeName = modelName.replace(Regex("[^A-Za-z0-9_\\-]"), "_")

                val jsonName = "${safeName}_${backend}_${stamp}_summary.json"
                val csvName  = "${safeName}_${backend}_${stamp}_per_class.csv"
                val cmName   = "${safeName}_${backend}_${stamp}_confusion_matrix.csv"

                writeToDownloads(jsonName, "application/json") { buildSummaryJson(m, modelName, backend) }
                writeToDownloads(csvName,  "text/csv")         { buildPerClassCsv(m) }
                writeToDownloads(cmName,   "text/csv")         { buildConfusionMatrixCsv(m) }

                binding.tvExportStatus.text =
                    "✓ Saved to Downloads:\n• $jsonName\n• $csvName\n• $cmName"
            } catch (e: Exception) {
                binding.tvExportStatus.text = "✗ Export failed: ${e.message}"
            } finally {
                binding.btnExport.isEnabled = true
            }
        }
    }

    private fun writeToDownloads(fileName: String, mimeType: String, content: () -> String) {
        val stream: OutputStream = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            }
            val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("MediaStore insert failed for $fileName")
            contentResolver.openOutputStream(uri)
                ?: throw IllegalStateException("Cannot open stream for $fileName")
        } else {
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            dir.mkdirs()
            java.io.FileOutputStream(java.io.File(dir, fileName))
        }
        stream.use { it.write(content().toByteArray(Charsets.UTF_8)) }
    }

    // ── Content builders ─────────────────────────────────────────────

    private fun buildSummaryJson(
        m: MetricsCalculator.AggregateMetrics,
        modelName: String,
        backend: String
    ): String = JSONObject().apply {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

        put("model_name",             modelName)
        put("backend",                backend)
        put("executed_at",            sdf.format(Date()))
        put("confidence_threshold",   MetricsCache.lastConfidenceThreshold.toDouble())
        put("iou_nms_threshold",      MetricsCache.lastIouThreshold.toDouble())
        put("images_evaluated",       m.totalImages)
        put("total_detections",       m.totalDetections)
        put("total_ground_truths",    m.totalGroundTruths)
        put("precision",              m.precision.toDouble())
        put("recall",                 m.recall.toDouble())
        put("accuracy",               m.accuracy.toDouble())
        put("f1_score",               m.f1.toDouble())
        put("f1_std_dev",             m.f1StdDev.toDouble())
        put("mAP50",                  m.mAP50.toDouble())
        put("inference_mean_ms",      m.meanInferenceMs.toDouble())
        put("inference_min_ms",       m.minInferenceMs)
        put("inference_max_ms",       m.maxInferenceMs)
        put("inference_std_ms",       m.stdDevInferenceMs.toDouble())
        put("exported_at",            sdf.format(Date()))
    }.toString(2)

    private fun buildPerClassCsv(m: MetricsCalculator.AggregateMetrics): String {
        val sb = StringBuilder()
        sb.appendLine("class,tp,fp,fn,precision,recall,f1,ap50")
        for (s in m.perClassStats) {
            sb.appendLine(
                "${s.className},${s.tp},${s.fp},${s.fn}," +
                        "${"%.6f".format(s.precision)},${"%.6f".format(s.recall)}," +
                        "${"%.6f".format(s.f1)},${"%.6f".format(s.ap50)}"
            )
        }
        return sb.toString()
    }

    private fun buildConfusionMatrixCsv(m: MetricsCalculator.AggregateMetrics): String {
        val baseLabels = YOLODetector.FSL_CLASSES
        val n          = m.rawConfusionMatrix.size  // n+1 if BG row included
        val hasBG      = n == baseLabels.size + 1
        val labels     = if (hasBG) baseLabels + listOf("BG") else baseLabels
        val sb         = StringBuilder()

        // ── Normalized matrix ─────────────────────────────────────
        sb.appendLine("NORMALIZED CONFUSION MATRIX")
        sb.appendLine("rows=Actual (BG row=FPs)  cols=Predicted (BG col=FNs)")
        sb.append("actual\\predicted")
        labels.forEach { sb.append(",$it") }
        sb.appendLine()
        for (r in 0 until n) {
            sb.append(labels.getOrElse(r) { "BG" })
            for (c in 0 until n) {
                sb.append(",${"%.6f".format(m.confusionMatrix[r][c])}")
            }
            sb.appendLine()
        }

        sb.appendLine()
        sb.appendLine()

        // ── Raw matrix ────────────────────────────────────────────
        sb.appendLine("RAW CONFUSION MATRIX (counts)")
        sb.appendLine("rows=Actual (BG row=FPs)  cols=Predicted (BG col=FNs)")
        sb.append("actual\\predicted")
        labels.forEach { sb.append(",$it") }
        sb.appendLine()
        for (r in 0 until n) {
            sb.append(labels.getOrElse(r) { "BG" })
            for (c in 0 until n) {
                sb.append(",${m.rawConfusionMatrix[r][c]}")
            }
            sb.appendLine()
        }

        return sb.toString()
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private fun resolveAttrColor(attr: Int): Int {
        val ta    = theme.obtainStyledAttributes(intArrayOf(attr))
        val color = ta.getColor(0, android.graphics.Color.BLACK)
        ta.recycle()
        return color
    }
}