use rustfft::num_complex::Complex;
use simdeez::prelude::*;
use std::sync::Arc;

use super::{AnalyzedSample};
use crate::voice::{
    ReleaseType, 
    SIMDVoiceEnvelope, 
    SIMDVoiceGenerator,
    SpectralVoiceSampleGenerator, 
    VoiceControlData, 
    VoiceGeneratorBase, 
    VoiceSampleGenerator,
};

pub struct SpectralVoice<T: Simd> {
    sample_data: Arc<AnalyzedSample>,
    envelope: SIMDVoiceEnvelope<T>,
    current_frame: f32,
    root_note: u8,
    trigger_note: u8,
    sample_rate: f32,
    current_volume: f32,
    current_pitch_bend: f32,
    last_pitch_ratio: f32,
}

impl<T: Simd> SpectralVoice<T> {
    pub fn new(
        sample_data: Arc<AnalyzedSample>,
        envelope: SIMDVoiceEnvelope<T>,
        root_note: u8,
        trigger_note: u8,
        sample_rate: f32,
    ) -> Self {
        let note_diff = trigger_note - root_note;
        let base_pitch_ratio = (note_diff as f32 / 12.0).exp2();
        let sample_rate_scaling = sample_data.original_sample_rate as f32 / sample_rate;
        let initial_pitch_ratio = base_pitch_ratio * 1.0 * sample_rate_scaling; /* current_pitch_bend starts at 1.0 */ 
        Self {
            sample_data,
            envelope,
            current_frame: 0.0,
            root_note,
            trigger_note,
            sample_rate,
            current_volume: 1.0,
            current_pitch_bend: 1.0,
            last_pitch_ratio: initial_pitch_ratio,
        }
    }

    pub fn fft_size(&self) -> usize {
        self.sample_data.fft_size
    }

    pub fn fft_step(&self) -> usize {
        self.sample_data.fft_step
    }

    pub fn update_host_sample_rate(&mut self, new_rate: f32) {
        self.sample_rate = new_rate;
    }

    pub fn spectral_process_voice(
        &mut self,
        scratch: &mut Vec<(usize, Complex<f32>)>,
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
        magnitude_res: usize,
    ) {
        scratch.clear();

        let frame_idx = self.current_frame as usize;
        if frame_idx >= self.sample_data.total_goertzel_samples {
            return;
        }

        let note_diff = self.trigger_note as f32 - self.root_note as f32;
        let base_pitch_ratio = (note_diff / 12.0_f32).exp2();
        let sample_rate_scaling = self.sample_data.original_sample_rate as f32 / self.sample_rate;
        let total_pitch_ratio = base_pitch_ratio * self.current_pitch_bend * sample_rate_scaling;
        self.last_pitch_ratio = total_pitch_ratio;
        let output_bin_hz = self.sample_rate / fft_size as f32;

        for (_, harmonic) in self.sample_data.harmonics.iter().enumerate() {
            let magnitude = harmonic.magnitude_curve[frame_idx];
            // frequency == 0.0 marks an inert/padding harmonic
            if harmonic.frequency <= 0.0 {
                continue;
            } 
            if magnitude <= 0.0 {
                continue;
            }
            if frame_idx >= harmonic.magnitude_curve.len() {
                continue;
            }
            let shifted_freq = harmonic.frequency * total_pitch_ratio;
 
            // Skip near-DC pile-up from heavily downward-shifted low-frequency
            // content (the bass-floor fix from earlier).
            //if shifted_freq < output_bin_hz * 0.5 {
            //    continue;
            //}
 
            let target_bin = shifted_freq / output_bin_hz;
            let lo_f = target_bin.floor();
            if lo_f.is_nan() {
                continue;
            }
            let frac = target_bin - lo_f;
            let lo = lo_f as isize;
            let hi = lo + 1;
 
            let current_time_samples = frame_idx * magnitude_res;
            let phase = harmonic.phase_at_origin + 2.0 * std::f32::consts::PI * shifted_freq * (current_time_samples as f32 / self.sample_rate);
            let (sin_p, cos_p) = phase.sin_cos();
            let contrib = Complex::new(magnitude * cos_p, magnitude * sin_p);
 
            if lo >= 0 && (lo as usize) < bin_count {
                scratch.push((lo as usize, contrib * (1.0 - frac)));
            }
            if hi >= 0 && (hi as usize) < bin_count {
                scratch.push((hi as usize, contrib * frac));
            }
        }
        self.current_frame += fft_step as f32 / magnitude_res as f32;
    }

    pub fn get_spectral_gain(&mut self, velocity: u8, fft_step: usize) -> f32 {
        let g_start = self.envelope.get_value_at_current_time();
        //let envelope_steps = (fft_step as f32 * self.last_pitch_ratio).round().max(0.0) as usize;
        for _ in 0..fft_step {
            let _ = self.envelope.next_sample();
        }
        let g_end = self.envelope.get_value_at_current_time();
        ((g_start + g_end) * 0.5) * self.current_volume * (velocity as f32 / 127.0)
    }
}

impl<T: Simd> VoiceGeneratorBase for SpectralVoice<T> {
    #[inline(always)]
    fn ended(&self) -> bool {
        self.envelope.ended() || (self.current_frame as usize) >= self.sample_data.total_goertzel_samples
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.envelope.signal_release(rel_type);
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.current_pitch_bend = control.voice_pitch_multiplier;
        self.envelope.process_controls(control);
    }
}

impl<T: Simd> VoiceSampleGenerator for SpectralVoice<T> {
    #[inline(always)]
    fn render_to(&mut self, _buffer: &mut [f32]) {}

    #[inline(always)]
    fn as_spectral_voice_mut(&mut self) -> Option<&mut dyn SpectralVoiceSampleGenerator> {
        Some(self)
    }
}

impl<T: Simd> SpectralVoiceSampleGenerator for SpectralVoice<T> {
    #[inline(always)]
    fn fft_size(&self) -> usize {
        self.fft_size()
    }

    #[inline(always)]
    fn fft_step(&self) -> usize {
        self.fft_step()
    }

    #[inline(always)]
    fn update_host_sample_rate(&mut self, new_rate: f32) {
        self.update_host_sample_rate(new_rate);
    }

    #[inline(always)]
    fn spectral_process_voice(
        &mut self,
        scratch: &mut Vec<(usize, rustfft::num_complex::Complex<f32>)>,
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
        magnitude_res: usize,
    ) {
        self.spectral_process_voice(scratch, fft_size, fft_step, bin_count, magnitude_res);
    }

    #[inline(always)]
    fn get_spectral_gain(&mut self, velocity: u8, fft_step: usize) -> f32 {
        self.get_spectral_gain(velocity, fft_step)
    }

    #[inline(always)]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline(always)]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
