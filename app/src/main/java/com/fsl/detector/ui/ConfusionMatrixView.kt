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
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class ConfusionMatrixView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    var matrix: Array<FloatArray> = emptyArray()
        set(value) { field = value; invalidate() }

    private val labels = YOLODetector.FSL_CLASSES
    private val n get() = matrix.size

    // Colors: white (0) → deep blue (1.0)
    private val COLOR_LOW  = Color.parseColor("#FFFFFF")
    private val COLOR_HIGH = Color.parseColor("#1565C0")
    private val COLOR_DIAG = Color.parseColor("#0D47A1")

    private val cellPaint   = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint   = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.BLACK
        typeface = Typeface.DEFAULT_BOLD
    }
    private val labelPaint  = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.parseColor("#37474F")
        typeface = Typeface.DEFAULT
    }
    private val whitePaint  = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color     = Color.parseColor("#E0E0E0")
        style     = Paint.Style.STROKE
        strokeWidth = 0.5f
    }

    // Pan & zoom
    private var scaleFactor = 1f
    private var offsetX     = 0f
    private var offsetY     = 0f
    private val minScale    = 0.4f
    private val maxScale    = 3f

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

        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), whitePaint)

        val labelMargin = 52f
        val legendW     = 24f
        val legendPad   = 8f

        val availW = width  - labelMargin - legendW - legendPad * 2
        val availH = height - labelMargin
        val baseCellSize = min(availW, availH) / n

        val cellSize = baseCellSize * scaleFactor
        textPaint.textSize  = (cellSize * 0.38f).coerceIn(8f, 18f)
        labelPaint.textSize = (baseCellSize * 0.42f).coerceIn(9f, 14f)

        val matrixLeft = labelMargin + offsetX
        val matrixTop  = labelMargin + offsetY

        // ── Draw cells ──────────────────────────────────────────────
        for (r in 0 until n) {
            for (c in 0 until n) {
                val value = matrix[r][c]
                val left  = matrixLeft + c * cellSize
                val top   = matrixTop  + r * cellSize
                val right  = left + cellSize
                val bottom = top  + cellSize

                // Clip to visible area
                if (right < 0 || left > width || bottom < 0 || top > height) continue

                val blended = ColorUtils.blendARGB(COLOR_LOW, COLOR_HIGH, value)
                cellPaint.color = if (r == c && value > 0f) COLOR_DIAG else blended
                canvas.drawRect(left, top, right, bottom, cellPaint)
                canvas.drawRect(left, top, right, bottom, borderPaint)

                // Cell value text (only if cell is large enough to read)
                if (cellSize >= 20f && value > 0.001f) {
                    val txt = if (value >= 0.995f) "1.0"
                    else ".${"%.0f".format(value * 100).padStart(2, '0')}"
                    val tW  = textPaint.measureText(txt)
                    val tH  = textPaint.textSize
                    textPaint.color = if (value > 0.55f) Color.WHITE else Color.BLACK
                    canvas.drawText(txt, left + (cellSize - tW) / 2f, top + (cellSize + tH) / 2f - 2f, textPaint)
                }
            }
        }

        // ── Row labels (ground truth) ─────────────────────────────
        labelPaint.textAlign = Paint.Align.RIGHT
        for (r in 0 until n) {
            val top = matrixTop + r * cellSize
            if (top + cellSize < 0 || top > height) continue
            canvas.drawText(
                labels[r],
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
            canvas.save()
            canvas.rotate(-90f, left + cellSize / 2f, labelMargin - 4f)
            canvas.drawText(
                labels[c],
                left + cellSize / 2f - labelPaint.measureText(labels[c]) / 2f,
                labelMargin - 4f,
                labelPaint
            )
            canvas.restore()
        }

        // ── Axis labels ──────────────────────────────────────────
        val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color     = Color.parseColor("#546E7A")
            textSize  = 11f
            typeface  = Typeface.DEFAULT_BOLD
            textAlign = Paint.Align.CENTER
        }
        canvas.drawText("Predicted", matrixLeft + n * cellSize / 2f, 14f, axisPaint)
        canvas.save()
        canvas.rotate(-90f, 10f, matrixTop + n * cellSize / 2f)
        canvas.drawText("Actual", 10f, matrixTop + n * cellSize / 2f, axisPaint)
        canvas.restore()

        // ── Color legend (right side) ────────────────────────────
        val legendLeft   = width - legendW - legendPad
        val legendTop    = matrixTop
        val legendHeight = n * cellSize
        if (legendHeight > 0) {
            val legendRect = RectF(legendLeft, legendTop, legendLeft + legendW, legendTop + legendHeight)
            val legendShader = LinearGradient(
                legendLeft, legendTop + legendHeight,
                legendLeft, legendTop,
                COLOR_LOW, COLOR_HIGH,
                Shader.TileMode.CLAMP
            )
            cellPaint.shader = legendShader
            canvas.drawRect(legendRect, cellPaint)
            cellPaint.shader = null

            val legendLabelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color     = Color.parseColor("#37474F")
                textSize  = 10f
                textAlign = Paint.Align.LEFT
            }
            canvas.drawText("1.0", legendLeft + legendW + 2f, legendTop + 10f, legendLabelPaint)
            canvas.drawText("0.5", legendLeft + legendW + 2f, legendTop + legendHeight / 2f, legendLabelPaint)
            canvas.drawText("0.0", legendLeft + legendW + 2f, legendTop + legendHeight, legendLabelPaint)
        }
    }
}