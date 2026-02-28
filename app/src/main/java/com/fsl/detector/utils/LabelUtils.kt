package com.fsl.detector.utils

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import com.fsl.detector.detector.GroundTruth

object LabelUtils {

    private val IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "bmp", "webp")

    /** List all image DocumentFiles in a SAF tree Uri folder. */
    fun getImageDocuments(folderUri: Uri, context: Context): List<DocumentFile> {
        val dir = DocumentFile.fromTreeUri(context, folderUri) ?: return emptyList()
        return dir.listFiles()
            .filter { it.isFile && it.name?.substringAfterLast('.')?.lowercase() in IMAGE_EXTENSIONS }
            .sortedBy { it.name }
    }

    /** Decode a bitmap from a DocumentFile via ContentResolver. */
    fun decodeBitmapFromDocument(doc: DocumentFile, context: Context): Bitmap? {
        return try {
            context.contentResolver.openInputStream(doc.uri)?.use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Load ground truth labels for an image DocumentFile.
     * Looks for a sibling .txt file with the same base name in the same folder.
     */
    fun loadGroundTruthsFromDocument(
        imageDoc: DocumentFile,
        folderUri: Uri,
        context: Context
    ): List<GroundTruth> {
        val baseName = imageDoc.name?.substringBeforeLast('.') ?: return emptyList()
        val dir = DocumentFile.fromTreeUri(context, folderUri) ?: return emptyList()
        val labelDoc = dir.findFile("$baseName.txt") ?: return emptyList()
        return try {
            context.contentResolver.openInputStream(labelDoc.uri)?.use { stream ->
                stream.bufferedReader().readLines().mapNotNull { line ->
                    parseLabelLine(line)
                }
            } ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun parseLabelLine(line: String): GroundTruth? {
        val parts = line.trim().split("\\s+".toRegex())
        if (parts.size < 5) return null
        return try {
            val classIdx = parts[0].toInt()
            val xc = parts[1].toFloat()
            val yc = parts[2].toFloat()
            val w  = parts[3].toFloat()
            val h  = parts[4].toFloat()
            val rect = RectF(xc - w / 2f, yc - h / 2f, xc + w / 2f, yc + h / 2f)
            GroundTruth(classIdx, rect)
        } catch (e: NumberFormatException) {
            null
        }
    }

    /** Decode a bitmap from a content/file Uri. */
    fun decodeBitmapFromUri(context: Context, uri: Uri): Bitmap? {
        return try {
            context.contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it)
            }
        } catch (e: Exception) {
            null
        }
    }

    fun buildDirectoryIndex(folderUri: Uri, context: Context): Map<String, DocumentFile> {
        val dir = DocumentFile.fromTreeUri(context, folderUri) ?: return emptyMap()
        return dir.listFiles().associateBy { it.name ?: "" }
    }

    fun loadGroundTruthsFromIndex(
        imageDoc: DocumentFile,
        index: Map<String, DocumentFile>,
        context: Context
    ): List<GroundTruth> {
        val baseName  = imageDoc.name?.substringBeforeLast('.') ?: return emptyList()
        val labelDoc  = index["$baseName.txt"] ?: return emptyList()
        return try {
            context.contentResolver.openInputStream(labelDoc.uri)?.use { stream ->
                stream.bufferedReader().readLines().mapNotNull { parseLabelLine(it) }
            } ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun decodeBitmapFromDocument(doc: DocumentFile, context: Context, targetSize: Int = 640): Bitmap? {
        return try {
            // First pass: read dimensions only
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.contentResolver.openInputStream(doc.uri)?.use { BitmapFactory.decodeStream(it, null, opts) }

            // Calculate sample size: largest power-of-2 that keeps both dims above targetSize
            var sampleSize = 1
            val (w, h) = opts.outWidth to opts.outHeight
            while ((w / (sampleSize * 2)) >= targetSize && (h / (sampleSize * 2)) >= targetSize) {
                sampleSize *= 2
            }

            // Second pass: decode at reduced resolution
            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize        = sampleSize
                inPreferredConfig   = Bitmap.Config.ARGB_8888
            }
            context.contentResolver.openInputStream(doc.uri)?.use { BitmapFactory.decodeStream(it, null, decodeOpts) }
        } catch (e: Exception) {
            null
        }
    }

}