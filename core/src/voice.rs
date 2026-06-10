#![allow(dead_code)]
#![allow(non_camel_case_types)] // For the SIMD library

mod envelopes;
pub(crate) use envelopes::*;

mod simd;
pub(crate) use simd::*;

mod simdvoice;
pub(crate) use simdvoice::*;

mod base;
pub(crate) use base::*;

mod squarewave;
#[allow(unused_imports)]
pub(crate) use squarewave::*;

mod channels;
#[allow(unused_imports)]
pub(crate) use channels::*;

mod constant;
pub(crate) use constant::*;

mod sampler;
pub(crate) use sampler::*;

mod control;
pub(crate) use control::*;

mod cutoff;
pub(crate) use cutoff::*;

use rustfft::num_complex::Complex;
use std::hash::Hash;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpectralGroupKey {
    pub sample_data_ptr: usize,
    pub root_note: u8,
    pub trigger_note: u8,
    pub current_frame_bits: u32,
    pub current_pitch_bend_bits: u32,
    pub phase_signature: u64,
}

pub struct SpectralStateSnapshot {
    pub current_frame: f32,
    pub previous_phases: Vec<f32>,
}

/// Common interface for spectral voice generators.
pub trait SpectralVoiceSampleGenerator {
    fn fft_size(&self) -> usize;
    fn fft_step(&self) -> usize;
    fn update_host_sample_rate(&mut self, new_rate: f32);
    /*fn accumulate_bins(
        &mut self,
        shared_bins: &mut [Complex<f32>],
        velocity: u8,
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
    );*/

    fn spectral_group_key(&self) -> SpectralGroupKey;
    fn spectral_generate_template(
        &mut self,
        template_bins: &mut [Complex<f32>],
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
    );
    fn spectral_copy_state_from(&mut self, source: &dyn SpectralVoiceSampleGenerator);
    fn spectral_state_snapshot(&self) -> SpectralStateSnapshot;
    fn spectral_apply_state_snapshot(&mut self, snapshot: &SpectralStateSnapshot);
    fn spectral_advance_gain(&mut self, velocity: u8, fft_step: usize) -> f32;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Options to modify the envelope of a voice.
#[derive(Copy, Clone)]
pub struct EnvelopeControlData {
    /// Controls the attack. Can take values from 0 to 128
    /// according to the MIDI CC spec.
    pub attack: Option<u8>,

    /// Controls the release. Can take values from 0 to 128
    /// according to the MIDI CC spec.
    pub release: Option<u8>,
}

/// How a voice should be released.
#[derive(Copy, Clone, PartialEq)]
pub enum ReleaseType {
    /// Standard release. Uses the voice's envelope.
    Standard,

    /// Kills the voice with a fadeout of 1ms.
    Kill,
}

/// Options to control the parameters of a voice.
#[derive(Copy, Clone)]
pub struct VoiceControlData {
    /// Pitch multiplier
    pub voice_pitch_multiplier: f32,

    /// Envelope control
    pub envelope: EnvelopeControlData,
}

impl VoiceControlData {
    pub fn new_defaults() -> Self {
        VoiceControlData {
            voice_pitch_multiplier: 1.0,
            envelope: EnvelopeControlData {
                attack: None,
                release: None,
            },
        }
    }
}

pub trait VoiceGeneratorBase: Sync + Send {
    fn ended(&self) -> bool;
    fn signal_release(&mut self, rel_type: ReleaseType);
    fn process_controls(&mut self, control: &VoiceControlData);
}

pub trait VoiceSampleGenerator: VoiceGeneratorBase {
    fn render_to(&mut self, buffer: &mut [f32]);
    
    fn as_spectral_voice_mut(&mut self) -> Option<&mut dyn SpectralVoiceSampleGenerator> {
        None
    }
}

pub trait Voice: VoiceSampleGenerator + Send + Sync {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn is_releasing(&self) -> bool;
    fn is_killed(&self) -> bool;

    fn velocity(&self) -> u8;
    fn exclusive_class(&self) -> Option<u8>;
}
