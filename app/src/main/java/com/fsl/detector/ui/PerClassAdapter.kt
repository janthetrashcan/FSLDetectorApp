package com.fsl.detector.ui

import android.annotation.SuppressLint
import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.fsl.detector.R
import com.fsl.detector.metrics.MetricsCalculator
import androidx.core.graphics.toColorInt

class PerClassAdapter(
    private val data: List<MetricsCalculator.PerClassStats>
) : RecyclerView.Adapter<PerClassAdapter.VH>() {

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val tvClass:     TextView = view.findViewById(R.id.tvClass)
        val tvTP:        TextView = view.findViewById(R.id.tvTP)
        val tvFP:        TextView = view.findViewById(R.id.tvFP)
        val tvFN:        TextView = view.findViewById(R.id.tvFN)
        val tvPrecision: TextView = view.findViewById(R.id.tvPrecision)
        val tvRecall:    TextView = view.findViewById(R.id.tvRecall)
        val tvF1:        TextView = view.findViewById(R.id.tvF1)
        val tvAP:        TextView = view.findViewById(R.id.tvAP)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
        VH(LayoutInflater.from(parent.context).inflate(R.layout.item_per_class, parent, false))

    override fun getItemCount() = data.size

    @SuppressLint("SetTextI18n")
    override fun onBindViewHolder(holder: VH, position: Int) {
        val s = data[position]
        holder.tvClass.text     = s.className
        holder.tvTP.text        = "${s.tp}"
        holder.tvFP.text        = "${s.fp}"
        holder.tvFN.text        = "${s.fn}"
        holder.tvPrecision.text = "${"%.3f".format(s.precision)}"
        holder.tvRecall.text    = "${"%.3f".format(s.recall)}"
        holder.tvF1.text        = "${"%.3f".format(s.f1)}"
        holder.tvAP.text        = "${"%.3f".format(s.ap50)}"

        // Color-code F1: green > 0.8, yellow > 0.5, red otherwise
        holder.tvF1.setTextColor(
            when {
                s.f1 >= 0.8f -> "#27AE60".toColorInt()
                s.f1 >= 0.5f -> "#F39C12".toColorInt()
                else         -> "#E74C3C".toColorInt()
            }
        )
    }
}
