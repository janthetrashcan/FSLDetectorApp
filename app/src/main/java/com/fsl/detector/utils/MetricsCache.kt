package com.fsl.detector.utils

import com.fsl.detector.detector.ModelConfig
import com.fsl.detector.metrics.MetricsCalculator

object MetricsCache {
    var lastMetrics: MetricsCalculator.AggregateMetrics? = null
    var lastModelName: String = ""
    var lastBackend: String = ""

    // Shared model list so CameraActivity can access scanned models
    var availableModels: List<ModelConfig> = emptyList()
}