pub mod analyser;
pub mod pipeline;
pub mod voice;
pub mod fft;

// Re-export key structures to keep imports clean across the rest of the engine
pub use analyser::{AnalyzedSample, analyze_pcm_sample};
pub use pipeline::{SpectralPipeline, SpectralConfig};
pub use fft::SpectralPlans;
pub use voice::SpectralVoice;

/// A helper struct that holds a single spectral channel's data frame.
/// This matches the structure being read inside `SpectralVoice::accumulate_bins`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ComplexBin {
    pub re: f32,
    pub im: f32,
}

impl ComplexBin {
    /// Creates a new complex frequency bin representation
    #[inline(always)]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Computes the absolute magnitude of the frequency bin
    #[inline(always)]
    pub fn magnitude(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Computes the phase angle of the frequency bin in radians
    #[inline(always)]
    pub fn phase(&self) -> f32 {
        self.im.atan2(self.re)
    }
}