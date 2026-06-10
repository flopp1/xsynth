pub mod analyser;
pub mod pipeline;
pub mod voice;
pub mod fft;

// Re-export key structures to keep imports clean across the rest of the engine
pub use analyser::{AnalyzedSample, analyze_pcm_sample};
pub use pipeline::{SpectralPipeline, SpectralConfig};
pub use fft::SpectralPlans;
pub use voice::SpectralVoice;

/// A helper struct that holds a single spectral bin's data in polar coordinates, including frequency, magnitude, and phase.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bin {
    pub frequency: f32,
    pub magnitude: f32,   
    pub phase: f32,
}


impl Bin {
    /// Creates a new polar representation of a complex frequency bin
    #[inline(always)]
    pub fn new(frequency: f32, magnitude: f32, phase: f32) -> Self {
        Self { frequency: frequency, magnitude: magnitude, phase: phase }
    }
}