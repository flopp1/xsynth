use super::ConfigPath;
use serde::{Deserialize, Serialize};
use std::{ops::RangeInclusive, path::PathBuf};
use xsynth_core::channel::ChannelInitOptions;
use xsynth_realtime::{SynthFormat, ThreadCount, XSynthRealtimeConfig};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    // Channel options
    layers: Option<usize>,
    fade_out_killing: bool,

    // Realtime synth options
    render_window_ms: f64,
    multithreading: ThreadCount,
    ignore_range: RangeInclusive<u8>,

    // Spectral pipeline options
    enable_spectral: bool,
    spectral_fft_size: usize,
    spectral_max_peaks_per_frame: usize,
    fft_step: usize,
}

impl Default for Settings {
    fn default() -> Self {
        let chandef = ChannelInitOptions::default();

        Self {
            layers: Some(4),
            fade_out_killing: chandef.fade_out_killing,
            render_window_ms: 10.0,
            multithreading: ThreadCount::None,
            ignore_range: 0..=0,
            enable_spectral: true,
            spectral_fft_size: 8192,
            spectral_max_peaks_per_frame: 32,
            fft_step: 2048,
        }
    }
}

impl Settings {
    pub fn get_layers(&self) -> Option<usize> {
        self.layers
    }

    pub fn get_synth_config(&self) -> XSynthRealtimeConfig {
        let spectral_config = if self.enable_spectral {
            Some(xsynth_core::spectral::SpectralConfig {
                fft_size: self.spectral_fft_size,
                fft_step: self.fft_step,
                max_voices: self.layers,
                enable_phase_fade_out: self.fade_out_killing,
                max_peaks_per_frame: self.spectral_max_peaks_per_frame,
            })
        } else {
            None
        };
        XSynthRealtimeConfig {
            channel_init_options: ChannelInitOptions {
                fade_out_killing: self.fade_out_killing,
            },
            render_window_ms: self.render_window_ms,
            format: SynthFormat::Midi,
            multithreading: self.multithreading,
            ignore_range: self.ignore_range.clone(),
            spectral_config,
        }
    }
}

impl ConfigPath for Settings {
    fn filename() -> PathBuf {
        "settings.json".into()
    }
}
