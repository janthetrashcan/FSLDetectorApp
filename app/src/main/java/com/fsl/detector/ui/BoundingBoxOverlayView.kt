package com.fsl.detector.ui

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.fsl.detector.detector.DetectionResult
import com.fsl.detector.detector.YOLODetector
import androidx.core.graphics.toColorInt

/**
 * Custom View that draws bounding boxes and labels over a displayed image.
 * Coordinates in DetectionResult are in *original image* pixel space.
 * This view scales them to match the displayed image bounds.
 */
class BoundingBoxOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val COLORS = listOf(
        "#FF6B6B".toColorInt(), "#4ECDC4".toColorInt(),
        "#45B7D1".toColorInt(), "#96CEB4".toColorInt(),
        "#FFEAA7".toColorInt(), "#DDA0DD".toColorInt(),
        "#98D8C8".toColorInt(), "#F7DC6F".toColorInt(),
        "#BB8FCE".toColorInt(), "#85C1E9".toColorInt()
    )

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }

    var detections: List<DetectionResult> = emptyList()
        set(value) { field = value; invalidate() }

    var originalImageWidth: Int = 1
    var originalImageHeight: Int = 1

    @SuppressLint("DrawAllocation")
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val scaleX = width.toFloat() / originalImageWidth
        val scaleY = height.toFloat() / originalImageHeight

        for (det in detections) {
            val color = COLORS[det.classIndex % COLORS.size]
            boxPaint.color = color
            textBgPaint.color = color

            val scaledBox = RectF(
                det.boundingBox.left   * scaleX,
                det.boundingBox.top    * scaleY,
                det.boundingBox.right  * scaleX,
                det.boundingBox.bottom * scaleY
            )

            canvas.drawRoundRect(scaledBox, 8f, 8f, boxPaint)

            val label = "${det.className} ${"%.0f".format(det.confidence * 100)}%"
            val textW = textPaint.measureText(label)
            val textH = textPaint.textSize
            val labelRect = RectF(
                scaledBox.left,
                scaledBox.top - textH - 8f,
                scaledBox.left + textW + 16f,
                scaledBox.top
            )
            canvas.drawRoundRect(labelRect, 4f, 4f, textBgPaint)
            canvas.drawText(label, scaledBox.left + 8f, scaledBox.top - 6f, textPaint)
        }
    }
}
