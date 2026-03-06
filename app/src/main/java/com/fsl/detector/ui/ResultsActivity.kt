package com.fsl.detector.ui

import android.graphics.Color
import android.os.Bundle
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.fsl.detector.databinding.ActivityResultsBinding
import com.fsl.detector.metrics.MetricsCalculator
import com.fsl.detector.utils.MetricsCache
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter

class ResultsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        title = "Batch Evaluation Results"

        val metrics   = MetricsCache.lastMetrics ?: run { finish(); return }
        val modelName = MetricsCache.lastModelName
        val backend   = MetricsCache.lastBackend

        populateSummary(metrics, modelName, backend)
        setupBarChart(metrics)
        setupPerClassTable(metrics)
        setupConfusionMatrix(metrics)
    }

    private fun populateSummary(m: MetricsCalculator.AggregateMetrics, model: String, backend: String) {
        binding.tvSummaryHeader.text  = "Model: $model  |  Backend: $backend"
        binding.tvTotalImages.text    = "Images evaluated: ${m.totalImages}"
        binding.tvTotalDets.text      = "Total detections: ${m.totalDetections}"
        binding.tvTotalGTs.text       = "Total ground truths: ${m.totalGroundTruths}"

        binding.tvPrecision.text  = "Precision:  ${"%.4f".format(m.precision)}  (${pct(m.precision)})"
        binding.tvRecall.text     = "Recall:     ${"%.4f".format(m.recall)}  (${pct(m.recall)})"
        binding.tvAccuracy.text   = "Accuracy:   ${"%.4f".format(m.accuracy)}  (${pct(m.accuracy)})"
        binding.tvF1.text         = "F1-Score:   ${"%.4f".format(m.f1)}  (${pct(m.f1)})  ± ${"%.4f".format(m.f1StdDev)} std dev"
        binding.tvMap50.text      = "mAP@50:     ${"%.4f".format(m.mAP50)}  (${pct(m.mAP50)})"

        binding.tvInfMean.text = "Mean inference:   ${"%.1f".format(m.meanInferenceMs)} ms"
        binding.tvInfMin.text  = "Min  inference:   ${m.minInferenceMs} ms"
        binding.tvInfMax.text  = "Max  inference:   ${m.maxInferenceMs} ms"
        binding.tvInfStd.text  = "Std dev:          ${"%.2f".format(m.stdDevInferenceMs)} ms"
    }

    private fun pct(v: Float) = "${"%.1f".format(v * 100)}%"

    private fun setupBarChart(m: MetricsCalculator.AggregateMetrics) {
        val labels = m.perClassStats.map { it.className }

        val precVals: List<BarEntry> = m.perClassStats.mapIndexed { i, s -> BarEntry(i.toFloat(), s.precision) }
        val recVals:  List<BarEntry> = m.perClassStats.mapIndexed { i, s -> BarEntry(i.toFloat(), s.recall) }
        val f1Vals:   List<BarEntry> = m.perClassStats.mapIndexed { i, s -> BarEntry(i.toFloat(), s.f1) }
        val apVals:   List<BarEntry> = m.perClassStats.mapIndexed { i, s -> BarEntry(i.toFloat(), s.ap50) }

        val dsPrecision = BarDataSet(precVals, "Precision").apply { color = Color.parseColor("#4ECDC4") }
        val dsRecall    = BarDataSet(recVals,  "Recall").apply    { color = Color.parseColor("#FF6B6B") }
        val dsF1        = BarDataSet(f1Vals,   "F1").apply        { color = Color.parseColor("#45B7D1") }
        val dsAP        = BarDataSet(apVals,   "AP@50").apply     { color = Color.parseColor("#96CEB4") }

        val groupSpace   = 0.1f
        val barSpace     = 0.02f
        val computedBarW = (1f - groupSpace - 4 * barSpace) / 4

        val barData = BarData(dsPrecision, dsRecall, dsF1, dsAP)
        barData.barWidth = computedBarW

        binding.barChart.apply {
            data = barData
            groupBars(0f, groupSpace, barSpace)
            xAxis.apply {
                position           = XAxis.XAxisPosition.BOTTOM
                granularity        = 1f
                valueFormatter     = IndexAxisValueFormatter(labels)
                labelRotationAngle = -45f
                setDrawGridLines(false)
            }
            axisLeft.apply { axisMinimum = 0f; axisMaximum = 1f }
            axisRight.isEnabled   = false
            description.isEnabled = false
            legend.isEnabled      = true
            animateY(800)
            invalidate()
        }
    }

    private fun setupPerClassTable(m: MetricsCalculator.AggregateMetrics) {
        binding.rvPerClass.layoutManager = LinearLayoutManager(this)
        binding.rvPerClass.adapter = PerClassAdapter(m.perClassStats)
    }

    private fun setupConfusionMatrix(m: MetricsCalculator.AggregateMetrics) {
        binding.confusionMatrixView.matrix = m.confusionMatrix
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) { finish(); return true }
        return super.onOptionsItemSelected(item)
    }
}