pub mod analyser;
pub mod fft;
pub mod pipeline;
pub mod voice;

// Re-export key structures to keep imports clean across the rest of the engine
pub use analyser::{analyze_pcm_sample, AnalyzedSample};
pub use fft::SpectralPlans;
pub use pipeline::{SpectralConfig, SpectralPipeline};
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
        Self {
            frequency: frequency,
            magnitude: magnitude,
            phase: phase,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::spectral::SpectralConfig;
    // test 1
    #[test]
    fn test_frequency_to_bin_mapping_consistency() {
        let sample_rate = 44100.0_f32;
        let config = SpectralConfig::default();
        let fft_size = config.fft_size;
        let bin_count = fft_size / 2 + 1;

        // analysis-side delta_f (from analyser.rs)
        let analysis_delta_f = sample_rate / fft_size as f32;

        // pipeline-side output_bin_hz (from voice.rs spectral_generate_template)
        let output_bin_hz = sample_rate / fft_size as f32;

        // These MUST be equal for unity pitch ratio to be a no-op mapping.
        assert_eq!(
            analysis_delta_f, output_bin_hz,
            "analysis delta_f and pipeline output_bin_hz disagree — frequency mapping is scrambled"
        );

        // --- Bass-specific check ---
        // Test a range of frequencies including low bass notes, and verify
        // round-trip bin mapping at unity pitch ratio is identity (or off by
        // at most rounding error of 1 bin).
        let test_freqs = [
            27.5,  // A0 - lowest piano note
            55.0,  // A1
            110.0, // A2
            220.0, // A3
            440.0, // A4
            880.0, // A5
        ];

        for &freq in &test_freqs {
            // Simulate analysis: which bin would this frequency's energy land in
            // (before parabolic interpolation, i.e. the raw bin index)?
            let analysis_bin = (freq / analysis_delta_f).round() as usize;

            // After parabolic interpolation, peak.frequency is a continuous value
            // near (but not necessarily exactly) analysis_bin * analysis_delta_f.
            // Use the exact bin-center frequency as the "peak.frequency" for this test.
            let peak_freq = analysis_bin as f32 * analysis_delta_f;

            // total_pitch_ratio = 1.0 (unity pitch, same sample rate)
            let total_pitch_ratio = 1.0_f32;
            let shifted_freq = peak_freq * total_pitch_ratio;

            // Pipeline-side mapping (from spectral_generate_template):
            let target_bin_exact = shifted_freq / output_bin_hz;
            let target_bin = target_bin_exact.round() as usize;

            println!(
                "freq={}Hz -> analysis_bin={} (peak_freq={}Hz) -> target_bin={}",
                freq, analysis_bin, peak_freq, target_bin
            );

            assert_eq!(
                analysis_bin, target_bin,
                "Frequency {}Hz: analysis bin {} != pipeline target bin {} — mapping is inconsistent",
                freq, analysis_bin, target_bin
            );

            assert!(
                target_bin < bin_count,
                "target_bin {} out of range (bin_count={})",
                target_bin,
                bin_count
            );
        }

        // --- Low-frequency resolution check ---
        // For bass notes, delta_f determines how many distinct bins exist below,
        // say, 200Hz. If fft_size is too small, multiple distinct bass harmonics
        // collapse onto the SAME bin, which would explain "a lot of bass is lost"
        // — they're not lost, they're being summed/aliased onto one bin and the
        // semitone+bin-spacing filter in analysis may then discard all but one.
        let bins_below_200hz = (200.0 / analysis_delta_f).floor() as usize;
        println!(
            "bins available below 200Hz: {} (delta_f={}Hz)",
            bins_below_200hz, analysis_delta_f
        );

        // A1 (55Hz) to A2 (110Hz) to A3 (220Hz) -- with fft_size=2048 at 44.1kHz,
        // delta_f ~= 43Hz, so there are only ~4-5 bins below 200Hz. Many bass
        // harmonics will collapse onto the same bin or fail the 1.5*delta_f
        // spacing check in peak selection, explaining bass loss.
        assert!(
            bins_below_200hz >= 8,
            "Only {} bins below 200Hz (delta_f={}Hz) — fft_size={} is too small for bass \
            resolution; harmonics below ~{}Hz will collapse onto shared bins or fail \
            the minimum-spacing peak selection filter. Consider fft_size=4096 or larger.",
            bins_below_200hz,
            analysis_delta_f,
            fft_size,
            bins_below_200hz as f32 * analysis_delta_f
        );
    }
}
