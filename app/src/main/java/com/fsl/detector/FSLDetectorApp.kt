package com.fsl.detector

import android.app.Application
import androidx.appcompat.app.AppCompatDelegate

class FSLDetectorApp : Application() {
    override fun onCreate() {
        super.onCreate()
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM)
    }
}