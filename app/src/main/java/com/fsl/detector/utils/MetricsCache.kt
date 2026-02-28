package com.fsl.detector.utils

import com.fsl.detector.metrics.MetricsCalculator

object MetricsCache {
    var lastMetrics: MetricsCalculator.AggregateMetrics? = null
    var lastModelName: String = ""
    var lastBackend: String = ""
}