package com.fsl.detector.utils

import android.app.ActivityManager
import android.content.Context
import java.io.File
import java.io.RandomAccessFile

object SystemStatsMonitor {

    data class SystemStats(
        val cpuPercent: Float,
        val ramUsedMb: Long,
        val ramTotalMb: Long,
        val gpuPercent: Float,   // -1f if unavailable
        val gpuLabel: String
    )

    // ── CPU ──────────────────────────────────────────────────────────
    private var lastCpuIdle: Long = 0L
    private var lastCpuTotal: Long = 0L

    fun getCpuPercent(): Float {
        return try {
            val lines = RandomAccessFile("/proc/stat", "r").use { it.readLine() }
            val toks  = lines.trim().split("\\s+".toRegex()).drop(1).map { it.toLong() }
            // user, nice, system, idle, iowait, irq, softirq, steal ...
            val idle  = toks[3] + toks[4]          // idle + iowait
            val total = toks.take(8).sum()

            val diffIdle  = idle  - lastCpuIdle
            val diffTotal = total - lastCpuTotal

            lastCpuIdle  = idle
            lastCpuTotal = total

            if (diffTotal == 0L) 0f
            else ((diffTotal - diffIdle).toFloat() / diffTotal * 100f).coerceIn(0f, 100f)
        } catch (e: Exception) { -1f }
    }

    // ── RAM ──────────────────────────────────────────────────────────
    fun getRamStats(context: Context): Pair<Long, Long> {
        val am   = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        val totalMb = info.totalMem  / 1_048_576L
        val usedMb  = (info.totalMem - info.availMem) / 1_048_576L
        return usedMb to totalMb
    }

    // ── GPU (Qualcomm / Mali / generic sysfs) ────────────────────────
    private val gpuBusyPaths = listOf(
        // Qualcomm Adreno
        "/sys/class/kgsl/kgsl-3d0/gpubusy",
        "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
        // Mali
        "/sys/class/misc/mali0/device/utilization",
        "/sys/kernel/gpu/gpu_busy"
    )

    fun getGpuStats(): Pair<Float, String> {
        for (path in gpuBusyPaths) {
            val file = File(path)
            if (!file.exists() || !file.canRead()) continue
            return try {
                val raw = file.readText().trim()
                when {
                    // Qualcomm "gpubusy": "numerator denominator"
                    raw.contains(" ") -> {
                        val (num, den) = raw.split(" ").map { it.toLong() }
                        if (den == 0L) -1f to "N/A"
                        else (num.toFloat() / den * 100f).coerceIn(0f, 100f) to "Adreno"
                    }
                    // Percentage directly
                    else -> raw.trimEnd('%').toFloatOrNull()?.coerceIn(0f, 100f)?.let {
                        it to "GPU"
                    } ?: (-1f to "N/A")
                }
            } catch (e: Exception) { -1f to "N/A" }
        }
        return -1f to "N/A"
    }

    fun getStats(context: Context): SystemStats {
        val cpu           = getCpuPercent()
        val (ram, ramMax) = getRamStats(context)
        val (gpu, gpuLbl) = getGpuStats()
        return SystemStats(cpu, ram, ramMax, gpu, gpuLbl)
    }
}