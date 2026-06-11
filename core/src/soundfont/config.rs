use crate::spectral::SpectralConfig;

/// Type of the audio sample interpolation algorithm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
pub enum Interpolator {
    /// Nearest neighbor interpolation
    Nearest,

    /// Linear interpolation
    Linear,
}

/// Type of curve to be used in certain envelope stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
pub enum EnvelopeCurveType {
    Linear,
    Exponential,
}

/// Options for the curves of a specific envelope.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct EnvelopeOptions {
    pub attack_curve: EnvelopeCurveType,
    pub decay_curve: EnvelopeCurveType,
    pub release_curve: EnvelopeCurveType,
}

impl Default for EnvelopeOptions {
    fn default() -> Self {
        Self {
            attack_curve: EnvelopeCurveType::Exponential,
            decay_curve: EnvelopeCurveType::Linear,
            release_curve: EnvelopeCurveType::Linear,
        }
    }
}

/// Options for initializing/loading a new sample soundfont.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct SoundfontInitOptions {
    pub bank: Option<u8>,
    pub preset: Option<u8>,
    pub vol_envelope_options: EnvelopeOptions,
    pub use_effects: bool,
    pub interpolator: Interpolator,

    /// NEW: Optional settings parameters enabling load-time spectral data generation.
    /// If `Some`, the parser handles offline analysis matrices during asset loading phases.
    pub spectral_config: Option<SpectralConfig>,
}

impl Default for SoundfontInitOptions {
    fn default() -> Self {
        Self {
            bank: None,
            preset: None,
            vol_envelope_options: Default::default(),
            use_effects: true,
            interpolator: Interpolator::Nearest,
            spectral_config: None, // Defaults to standard legacy time-domain execution
        }
    }
}
