use std::sync::Arc;
pub mod analyser;
pub mod fft;
pub mod pipeline;
pub mod voice;

pub use analyser::{analyze_pcm_sample, AnalyzedSample};
pub use fft::SpectralPlans;
pub use pipeline::SpectralPipeline;
pub use voice::SpectralVoice;

//fundemental object representing one selected frequency from the peak frequencies in the sample
#[derive(Clone, Debug)]
pub struct HarmonicTrack {
    pub frequency: f32,
    pub phase_at_origin: f32,
    pub magnitude_curve: Arc<[f32]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct SpectralConfig {
    pub fft_size: usize,
    pub fft_step: usize,
    pub max_voices: Option<usize>,
    pub enable_phase_fade_out: bool,
    pub max_peaks_per_frame: usize, 
    pub magnitude_res: usize
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            fft_size: 8192,
            fft_step: 2048, // 75% overlap for optimal OLA or something idk
            max_voices: Some(4 * 512),
            enable_phase_fade_out: true,
            max_peaks_per_frame: 64,
            magnitude_res: 48
        }
    }
}
