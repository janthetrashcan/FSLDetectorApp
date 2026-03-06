package com.fsl.detector.ui

import android.annotation.SuppressLint
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.graphics.toColorInt
import androidx.recyclerview.widget.RecyclerView
import com.fsl.detector.R
import com.fsl.detector.metrics.MetricsCalculator

class PerClassAdapter(
    private val data: List<MetricsCalculator.PerClassStats>
) : RecyclerView.Adapter<PerClassAdapter.VH>() {

    class VH(view: View) : RecyclerView.ViewHolder(view) {
        val tvClass: TextView = view.findViewById(R.id.tvClass)
        val tvTp:    TextView = view.findViewById(R.id.tvTp)
        val tvFp:    TextView = view.findViewById(R.id.tvFp)
        val tvFn:    TextView = view.findViewById(R.id.tvFn)
        val tvP:     TextView = view.findViewById(R.id.tvP)
        val tvR:     TextView = view.findViewById(R.id.tvR)
        val tvF1:    TextView = view.findViewById(R.id.tvF1)
        val tvAp:    TextView = view.findViewById(R.id.tvAp)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
        VH(LayoutInflater.from(parent.context).inflate(R.layout.item_per_class, parent, false))

    override fun getItemCount() = data.size

    @SuppressLint("SetTextI18n")
    override fun onBindViewHolder(holder: VH, position: Int) {
        val s = data[position]
        holder.tvClass.text = s.className
        holder.tvTp.text    = "${s.tp}"
        holder.tvFp.text    = "${s.fp}"
        holder.tvFn.text    = "${s.fn}"
        holder.tvP.text     = "%.3f".format(s.precision)
        holder.tvR.text     = "%.3f".format(s.recall)
        holder.tvF1.text    = "%.3f".format(s.f1)
        holder.tvAp.text    = "%.3f".format(s.ap50)

        holder.tvF1.setTextColor(
            when {
                s.f1 >= 0.8f -> "#27AE60".toColorInt()
                s.f1 >= 0.5f -> "#F39C12".toColorInt()
                else         -> "#E74C3C".toColorInt()
            }
        )
    }
}