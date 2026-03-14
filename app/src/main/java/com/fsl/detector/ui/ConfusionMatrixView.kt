package com.fsl.detector.ui

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import androidx.core.graphics.ColorUtils
import com.fsl.detector.detector.YOLODetector
import kotlin.math.min

class ConfusionMatrixView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    var matrix: Array<FloatArray> = emptyArray()
        set(value) { field = value; invalidate() }

    private val baseLabels = YOLODetector.FSL_CLASSES
    private val n          get() = matrix.size
    private val labels     get() = if (n == baseLabels.size + 1)
        baseLabels + listOf("BG") else baseLabels

    // Fixed heatmap colors — do NOT use colorSurface as COLOR_LOW
    // because it's identical to the card background, making cells invisible
    private val COLOR_EMPTY = Color.parseColor("#E8EAF6")  // very light indigo
    private val COLOR_LOW   = Color.parseColor("#C5CAE9")  // light indigo — visible on any bg
    private val COLOR_HIGH  = Color.parseColor("#1565C0")  // deep blue
    private val COLOR_DIAG  = Color.parseColor("#0D47A1")  // deeper blue for TP diagonal
    private val COLOR_FP    = Color.parseColor("#B71C1C")  // dark red  — BG row
    private val COLOR_FN    = Color.parseColor("#4A148C")  // dark purple — BG col

    private val COLOR_EMPTY_DARK = Color.parseColor("#1A1C2E")
    private val COLOR_LOW_DARK   = Color.parseColor("#1E2454")

    private val cellPaint        = Paint(Paint.ANTI_ALIAS_FLAG)
    private val borderPaint      = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = 0.5f
    }
    private val textPaint        = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.DEFAULT_BOLD
    }
    private val labelPaint       = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        typeface = Typeface.DEFAULT
    }
    private val axisPaint        = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize  = 11f
        typeface  = Typeface.DEFAULT_BOLD
        textAlign = Paint.Align.CENTER
    }
    private val legendLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize  = 10f
        textAlign = Paint.Align.LEFT
    }
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    // Pan & zoom
    private var scaleFactor = 1f
    private var offsetX     = 0f
    private var offsetY     = 0f
    private val minScale    = 0.3f
    private val maxScale    = 4f

    private val scaleDetector = ScaleGestureDetector(context,
        object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                scaleFactor = (scaleFactor * detector.scaleFactor).coerceIn(minScale, maxScale)
                invalidate()
                return true
            }
        })

    private val gestureDetector = GestureDetector(context,
        object : GestureDetector.SimpleOnGestureListener() {
            override fun onScroll(
                e1: MotionEvent?, e2: MotionEvent,
                distanceX: Float, distanceY: Float
            ): Boolean {
                offsetX -= distanceX
                offsetY -= distanceY
                invalidate()
                return true
            }
        })

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        return true
    }

    private fun isDarkMode(): Boolean {
        val uiMode = context.resources.configuration.uiMode and
                android.content.res.Configuration.UI_MODE_NIGHT_MASK
        return uiMode == android.content.res.Configuration.UI_MODE_NIGHT_YES
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (n == 0) return

        val dark          = isDarkMode()
        val currentLabels = labels
        val bgRow         = n - 1
        val hasBG         = n == baseLabels.size + 1

        // Resolve theme-dependent colors once per draw
        val colorLabel    = resolveThemeColor(android.R.attr.textColorPrimary)
        val colorLabelSec = resolveThemeColor(android.R.attr.textColorSecondary)
        val colorBorder   = resolveMaterialColor("colorOutline", Color.GRAY)
        val colorSurface  = resolveMaterialColor("colorSurface", if (dark) Color.BLACK else Color.WHITE)

        val emptyCell = if (dark) COLOR_EMPTY_DARK else COLOR_EMPTY
        val lowCell   = if (dark) COLOR_LOW_DARK   else COLOR_LOW

        bgPaint.color          = colorSurface
        borderPaint.color      = colorBorder
        labelPaint.color       = colorLabel
        axisPaint.color        = colorLabel
        legendLabelPaint.color = colorLabelSec

        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), bgPaint)

        val labelMargin  = 52f
        val legendW      = 24f
        val legendPad    = 8f

        val availW       = width  - labelMargin - legendW - legendPad * 2
        val availH       = height - labelMargin
        val baseCellSize = min(availW, availH) / n
        val cellSize     = baseCellSize * scaleFactor

        textPaint.textSize  = (cellSize * 0.38f).coerceIn(7f, 16f)
        labelPaint.textSize = (baseCellSize * 0.42f).coerceIn(8f, 13f)

        val matrixLeft = labelMargin + offsetX
        val matrixTop  = labelMargin + offsetY

        // ── Draw cells ───────────────────────────────────────────
        for (r in 0 until n) {
            for (c in 0 until n) {
                val value  = matrix[r][c]
                val left   = matrixLeft + c * cellSize
                val top    = matrixTop  + r * cellSize
                val right  = left + cellSize
                val bottom = top  + cellSize

                if (right < 0 || left > width || bottom < 0 || top > height) continue

                val isBgRow = hasBG && r == bgRow
                val isBgCol = hasBG && c == n - 1
                val isBgCorner = isBgRow && isBgCol

                val cellColor = when {
                    isBgCorner ->
                        emptyCell
                    isBgRow ->
                        // FP row: empty → red
                        if (value <= 0f) emptyCell
                        else ColorUtils.blendARGB(emptyCell, COLOR_FP, value)
                    isBgCol ->
                        // FN col: empty → purple
                        if (value <= 0f) emptyCell
                        else ColorUtils.blendARGB(emptyCell, COLOR_FN, value)
                    value <= 0f ->
                        emptyCell
                    r == c ->
                        // TP diagonal: low → deep blue
                        ColorUtils.blendARGB(lowCell, COLOR_DIAG, value)
                    else ->
                        // Misclassification: low → high blue
                        ColorUtils.blendARGB(lowCell, COLOR_HIGH, value)
                }

                cellPaint.color = cellColor
                canvas.drawRect(left, top, right, bottom, cellPaint)
                canvas.drawRect(left, top, right, bottom, borderPaint)

                // Draw value text — always show if cell has any value
                if (cellSize >= 16f && value > 0.001f) {
                    val txt = if (value >= 0.995f) "1.0"
                    else ".${"%.0f".format(value * 100).padStart(2, '0')}"
                    val tW  = textPaint.measureText(txt)
                    val tH  = textPaint.textSize

                    // Compute luminance of cell to pick contrasting text color
                    val lum = ColorUtils.calculateLuminance(cellColor)
                    textPaint.color = if (lum < 0.35) Color.WHITE else Color.BLACK

                    canvas.drawText(
                        txt,
                        left + (cellSize - tW) / 2f,
                        top  + (cellSize + tH) / 2f - 2f,
                        textPaint
                    )
                }
            }
        }

        // ── Row labels (actual) ──────────────────────────────────
        labelPaint.textAlign = Paint.Align.RIGHT
        for (r in 0 until n) {
            val top = matrixTop + r * cellSize
            if (top + cellSize < 0 || top > height) continue
            val lbl = currentLabels.getOrElse(r) { "?" }
            // BG row label in red, all others in normal theme color
            labelPaint.color = if (hasBG && r == bgRow) COLOR_FP else colorLabel
            canvas.drawText(
                lbl,
                labelMargin - 4f,
                top + (cellSize + labelPaint.textSize) / 2f - 2f,
                labelPaint
            )
        }

        // ── Column labels (predicted) ────────────────────────────
        labelPaint.textAlign = Paint.Align.LEFT
        for (c in 0 until n) {
            val left = matrixLeft + c * cellSize
            if (left + cellSize < 0 || left > width) continue
            val lbl = currentLabels.getOrElse(c) { "?" }
            // BG col label in purple, all others in normal theme color
            labelPaint.color = if (hasBG && c == n - 1) COLOR_FN else colorLabel
            canvas.save()
            canvas.rotate(-90f, left + cellSize / 2f, labelMargin - 4f)
            canvas.drawText(
                lbl,
                left + cellSize / 2f - labelPaint.measureText(lbl) / 2f,
                labelMargin - 4f,
                labelPaint
            )
            canvas.restore()
        }

        // Reset label paint color after loops
        labelPaint.color = colorLabel

        // ── Axis labels ──────────────────────────────────────────
        canvas.drawText(
            "Predicted",
            matrixLeft + n * cellSize / 2f,
            14f,
            axisPaint
        )
        canvas.save()
        canvas.rotate(-90f, 10f, matrixTop + n * cellSize / 2f)
        canvas.drawText("Actual", 10f, matrixTop + n * cellSize / 2f, axisPaint)
        canvas.restore()

        // ── Legend ───────────────────────────────────────────────
        val legendLeft   = width  - legendW - legendPad
        val legendTop    = matrixTop
        val legendHeight = n * cellSize
        if (legendHeight > 0) {
            val legendShader = LinearGradient(
                legendLeft, legendTop + legendHeight,
                legendLeft, legendTop,
                lowCell, COLOR_HIGH,
                Shader.TileMode.CLAMP
            )
            cellPaint.shader = legendShader
            canvas.drawRect(
                legendLeft, legendTop,
                legendLeft + legendW, legendTop + legendHeight,
                cellPaint
            )
            cellPaint.shader = null

            canvas.drawText("1.0", legendLeft + legendW + 2f, legendTop + 10f,               legendLabelPaint)
            canvas.drawText("0.5", legendLeft + legendW + 2f, legendTop + legendHeight / 2f, legendLabelPaint)
            canvas.drawText("0.0", legendLeft + legendW + 2f, legendTop + legendHeight,      legendLabelPaint)
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private fun resolveThemeColor(attrResId: Int): Int {
        val tv = android.util.TypedValue()
        context.theme.resolveAttribute(attrResId, tv, true)
        return tv.data
    }

    private fun resolveMaterialColor(attrName: String, fallback: Int): Int {
        val attrId = context.resources.getIdentifier(attrName, "attr", context.packageName)
        if (attrId == 0) return fallback
        val tv = android.util.TypedValue()
        return if (context.theme.resolveAttribute(attrId, tv, true)) tv.data else fallback
    }
}