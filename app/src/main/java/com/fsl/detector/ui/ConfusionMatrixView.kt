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

    // Automatically append "BG" label when matrix is (n+1)×(n+1)
    private val baseLabels = YOLODetector.FSL_CLASSES
    private val n get() = matrix.size
    private val labels get() = if (n == baseLabels.size + 1)
        baseLabels + listOf("BG") else baseLabels

    private val COLOR_HIGH = Color.parseColor("#1565C0")
    private val COLOR_DIAG = Color.parseColor("#0D47A1")
    private val COLOR_FP   = Color.parseColor("#B71C1C")  // dark red for BG row (FPs)
    private val COLOR_FN   = Color.parseColor("#4A148C")  // dark purple for BG col (FNs)

    private val COLOR_LOW       get() = resolveMaterialColor("colorSurface",        Color.WHITE)
    private val COLOR_EMPTY     get() = resolveMaterialColor("colorSurfaceVariant", Color.LTGRAY)
    private val COLOR_LABEL     get() = resolveThemeColor(android.R.attr.textColorPrimary)
    private val COLOR_LABEL_SEC get() = resolveThemeColor(android.R.attr.textColorSecondary)
    private val COLOR_BORDER    get() = resolveMaterialColor("colorOutline",         Color.GRAY)
    private val COLOR_BG        get() = resolveMaterialColor("colorSurface",        Color.WHITE)

    private val cellPaint        = Paint(Paint.ANTI_ALIAS_FLAG)
    private val borderPaint      = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = 0.5f
    }
    private val textPaint        = Paint(Paint.ANTI_ALIAS_FLAG).apply { typeface = Typeface.DEFAULT_BOLD }
    private val labelPaint       = Paint(Paint.ANTI_ALIAS_FLAG).apply { typeface = Typeface.DEFAULT }
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

    private fun resolveThemeColor(attrResId: Int): Int {
        val typedValue = android.util.TypedValue()
        context.theme.resolveAttribute(attrResId, typedValue, true)
        return typedValue.data
    }

    private fun resolveMaterialColor(attrName: String, fallback: Int): Int {
        val pkg     = context.packageName
        val attrId  = context.resources.getIdentifier(attrName, "attr", pkg)
        if (attrId == 0) return fallback
        val tv = android.util.TypedValue()
        return if (context.theme.resolveAttribute(attrId, tv, true)) tv.data else fallback
    }

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

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (n == 0) return

        val currentLabels = labels
        val bgRow = n - 1  // background row index (FPs) when matrix is n+1
        val hasBG = n == baseLabels.size + 1

        bgPaint.color          = COLOR_BG
        borderPaint.color      = COLOR_BORDER
        labelPaint.color       = COLOR_LABEL
        axisPaint.color        = COLOR_LABEL
        legendLabelPaint.color = COLOR_LABEL_SEC

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

                cellPaint.color = when {
                    value <= 0f             -> COLOR_EMPTY
                    hasBG && r == bgRow && c < n - 1 ->
                        // FP cells: BG row, actual class cols
                        ColorUtils.blendARGB(COLOR_EMPTY, COLOR_FP, value)
                    hasBG && c == n - 1 && r < n - 1 ->
                        // FN cells: actual class rows, BG col
                        ColorUtils.blendARGB(COLOR_EMPTY, COLOR_FN, value)
                    hasBG && r == bgRow && c == n - 1 ->
                        // BG×BG corner — unused, draw as empty
                        COLOR_EMPTY
                    r == c                  -> COLOR_DIAG
                    else                    -> ColorUtils.blendARGB(COLOR_LOW, COLOR_HIGH, value)
                }
                canvas.drawRect(left, top, right, bottom, cellPaint)
                canvas.drawRect(left, top, right, bottom, borderPaint)

                if (cellSize >= 18f && value > 0.001f) {
                    val txt = if (value >= 0.995f) "1.0"
                    else ".${"%.0f".format(value * 100).padStart(2, '0')}"
                    val tW  = textPaint.measureText(txt)
                    val tH  = textPaint.textSize
                    textPaint.color = if (value > 0.55f) Color.WHITE else COLOR_LABEL
                    canvas.drawText(txt,
                        left + (cellSize - tW) / 2f,
                        top  + (cellSize + tH) / 2f - 2f,
                        textPaint)
                }
            }
        }

        // ── Row labels (actual) ──────────────────────────────────
        labelPaint.textAlign = Paint.Align.RIGHT
        for (r in 0 until n) {
            val top = matrixTop + r * cellSize
            if (top + cellSize < 0 || top > height) continue
            // Highlight BG label in red
            labelPaint.color = if (hasBG && r == bgRow) COLOR_FP else COLOR_LABEL
            canvas.drawText(
                currentLabels.getOrElse(r) { "?" },
                labelMargin - 4f,
                top + (cellSize + labelPaint.textSize) / 2f - 2f,
                labelPaint
            )
        }
        labelPaint.color = COLOR_LABEL

        // ── Column labels (predicted) ────────────────────────────
        labelPaint.textAlign = Paint.Align.LEFT
        for (c in 0 until n) {
            val left = matrixLeft + c * cellSize
            if (left + cellSize < 0 || left > width) continue
            labelPaint.color = if (hasBG && c == n - 1) COLOR_FN else COLOR_LABEL
            val lbl = currentLabels.getOrElse(c) { "?" }
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
        labelPaint.color = COLOR_LABEL

        // ── Axis labels ──────────────────────────────────────────
        canvas.drawText("Predicted", matrixLeft + n * cellSize / 2f, 14f, axisPaint)
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
                COLOR_LOW, COLOR_HIGH,
                Shader.TileMode.CLAMP
            )
            cellPaint.shader = legendShader
            canvas.drawRect(legendLeft, legendTop, legendLeft + legendW, legendTop + legendHeight, cellPaint)
            cellPaint.shader = null

            canvas.drawText("1.0", legendLeft + legendW + 2f, legendTop + 10f,               legendLabelPaint)
            canvas.drawText("0.5", legendLeft + legendW + 2f, legendTop + legendHeight / 2f, legendLabelPaint)
            canvas.drawText("0.0", legendLeft + legendW + 2f, legendTop + legendHeight,      legendLabelPaint)
        }
    }

    private fun resolveAttrColor(attr: Int): Int {
        val ta = context.obtainStyledAttributes(intArrayOf(attr))
        val color = ta.getColor(0, Color.BLACK)
        ta.recycle()
        return color
    }
}