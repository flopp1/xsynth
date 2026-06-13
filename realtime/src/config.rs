use std::ops::RangeInclusive;
pub use xsynth_core::{
    channel::ChannelInitOptions,
    channel_group::{SynthFormat, ThreadCount},
    spectral::SpectralConfig,
};

/// Options for initializing a new RealtimeSynth.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct XSynthRealtimeConfig {
    /// Channel initialization options (same for all channels).
    /// See the `ChannelInitOptions` documentation for more information.
    pub channel_init_options: ChannelInitOptions,

    /// The length of the buffer reader in ms.
    ///
    /// Default: `10.0`
    pub render_window_ms: f64,

    /// Defines the format that the synthesizer will use. See the `SynthFormat`
    /// documentation for more information.
    ///
    /// Default: `SynthFormat::Midi`
    pub format: SynthFormat,

    /// Controls the multithreading used for rendering per-voice audio for all
    /// the voices stored in a key for a channel. See the `ThreadCount` documentation
    /// for the available options.
    ///
    /// Default: `ThreadCount::None`
    pub multithreading: ThreadCount,

    /// A range of velocities that will not be played.
    ///
    /// Default: `0..=0`
    pub ignore_range: RangeInclusive<u8>,

    /// Configuration parameters for the frequency-domain spectral engine.
    /// If `None`, the synthesizer defaults to standard time-domain rendering.
    ///
    /// Default: `None`
    pub spectral_config: Option<SpectralConfig>,
}

impl Default for XSynthRealtimeConfig {
    fn default() -> Self {
        Self {
            channel_init_options: Default::default(),
            render_window_ms: 10.0,
            format: Default::default(),
            multithreading: ThreadCount::None,
            ignore_range: 0..=0,
            spectral_config: Some(xsynth_core::spectral::SpectralConfig {
                fft_size: 8192,         // Fix property name (window_size instead of fft_size)
                max_voices: Some(4096), // Set high voice ceiling to support intensive benchmark stress loops
                enable_phase_fade_out: true,
                max_peaks_per_frame: 32, // Default value, can be adjusted as needed
                fft_step: 2048,          // Default FFT step size
                magnitude_res: 48,
            }),
        }
    }
}
