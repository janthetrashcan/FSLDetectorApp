package com.fsl.detector.utils

import com.fsl.detector.metrics.MetricsCalculator

object MetricsCache {
    var lastMetrics:            MetricsCalculator.AggregateMetrics? = null
    var lastModelName:          String? = null
    var lastBackend:            String? = null
    var availableModels:        List<String> = emptyList()
    var lastConfidenceThreshold: Float = 0.25f
    var lastIouThreshold:        Float = 0.45f
}