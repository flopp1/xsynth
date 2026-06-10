use std::hash::{Hash, Hasher};
use std::sync::Arc;
use rustfft::num_complex::Complex;
use simdeez::prelude::*;

use crate::voice::{ReleaseType, SpectralGroupKey, SpectralStateSnapshot, VoiceControlData, VoiceGeneratorBase, VoiceSampleGenerator, SIMDVoiceEnvelope, SIMDVoiceGenerator, SpectralVoiceSampleGenerator};
use super::{AnalyzedSample, ComplexBin};

pub struct SpectralVoice<T: Simd> {
    sample_data: Arc<AnalyzedSample>,
    envelope: SIMDVoiceEnvelope<T>,
    current_frame: f32,
    root_note: u8,
    trigger_note: u8,
    sample_rate: f32,
    current_volume: f32,
    current_pitch_bend: f32,
    
    /// Historical accumulator tracking phase angles for phase-locked interpolation
    previous_phases: Vec<f32>,
}

impl<T: Simd> SpectralVoice<T> {
    pub fn new(
        sample_data: Arc<AnalyzedSample>,
        envelope: SIMDVoiceEnvelope<T>,
        root_note: u8,
        trigger_note: u8,
        sample_rate: f32,
    ) -> Self {
        let bin_count = sample_data.fft_size / 2 + 1;
        Self {
            sample_data,
            envelope,
            current_frame: 0.0,
            root_note,
            trigger_note,
            sample_rate,
            current_volume: 1.0,
            current_pitch_bend: 1.0,
            previous_phases: vec![0.0; bin_count],
        }
    }

    pub fn fft_size(&self) -> usize { self.sample_data.fft_size }

    pub fn fft_step(&self) -> usize { self.sample_data.fft_step }

    pub fn update_host_sample_rate(&mut self, new_rate: f32) {
        self.sample_rate = new_rate;
    }

    pub fn spectral_group_key(&self) -> SpectralGroupKey {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let phase_count = self.previous_phases.len().min(4);
        for phase in self.previous_phases.iter().take(phase_count) {
            phase.to_bits().hash(&mut hasher);
        }

        SpectralGroupKey {
            sample_data_ptr: Arc::as_ptr(&self.sample_data) as usize,
            root_note: self.root_note,
            trigger_note: self.trigger_note,
            current_frame_bits: self.current_frame.to_bits(),
            current_pitch_bend_bits: self.current_pitch_bend.to_bits(),
            phase_signature: hasher.finish(),
        }
    }

    pub fn spectral_generate_template(
        &mut self,
        template_bins: &mut [Complex<f32>],
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
    ) {
        let frame_idx = self.current_frame as usize;
        if frame_idx >= self.sample_data.total_frames {
            template_bins.fill(Complex::new(0.0, 0.0));
            return;
        }

        let current_frame_slice = match self.sample_data.get_frame_slice(0, frame_idx) {
            Some(slice) => slice,
            None => {
                template_bins.fill(Complex::new(0.0, 0.0));
                return;
            }
        };

        let note_diff = self.trigger_note as f32 - self.root_note as f32;
        let base_pitch_ratio = (note_diff / 12.0_f32).exp2();
        let sample_rate_scaling = self.sample_data.original_sample_rate as f32 / self.sample_rate;
        let total_pitch_ratio = base_pitch_ratio * self.current_pitch_bend * sample_rate_scaling;

        let expected_phase_step = (fft_step as f32 * 2.0 * std::f32::consts::PI) / fft_size as f32;

        for target_bin in 0..bin_count {
            let source_bin_exact = target_bin as f32 / total_pitch_ratio;
            let source_bin_floor = source_bin_exact as usize;

            if source_bin_floor + 1 >= current_frame_slice.len() {
                template_bins[target_bin] = Complex::new(0.0, 0.0);
                continue;
            }

            let frac = source_bin_exact - source_bin_floor as f32;

            let c_left = current_frame_slice[source_bin_floor];
            let c_right = current_frame_slice[source_bin_floor + 1];

            let c = Complex::new(
                (1.0 - frac) * c_left.re + frac * c_right.re,
                (1.0 - frac) * c_left.im + frac * c_right.im,
            );

            let raw_phase = c.im.atan2(c.re);
            let expected = self.previous_phases[target_bin]
                + target_bin as f32 * expected_phase_step;
            let delta = raw_phase
                - (self.previous_phases[target_bin]
                    - target_bin as f32 * expected_phase_step * (frame_idx as f32 - 1.0).max(0.0));
            let delta = delta
                - (delta / (2.0 * std::f32::consts::PI)).round()
                    * 2.0
                    * std::f32::consts::PI;
            let true_phase = expected + delta;
            self.previous_phases[target_bin] = true_phase;

            let (sin_p, cos_p) = true_phase.sin_cos();
            template_bins[target_bin] = Complex::new(c.norm() * cos_p, c.norm() * sin_p);
        }

        self.current_frame += total_pitch_ratio;
    }

    pub fn copy_spectral_state_from(&mut self, source: &Self) {
        self.current_frame = source.current_frame;
        self.previous_phases.copy_from_slice(&source.previous_phases);
    }

    pub fn spectral_state_snapshot(&self) -> SpectralStateSnapshot {
        SpectralStateSnapshot {
            current_frame: self.current_frame,
            previous_phases: self.previous_phases.clone(),
        }
    }

    pub fn spectral_apply_state_snapshot(&mut self, snapshot: &SpectralStateSnapshot) {
        self.current_frame = snapshot.current_frame;
        self.previous_phases.copy_from_slice(&snapshot.previous_phases);
    }

    pub fn spectral_advance_gain(&mut self, velocity: u8, fft_step: usize) -> f32 {
        let average_gain = self.envelope.average_gain_over(fft_step);
        average_gain * self.current_volume * (velocity as f32 / 127.0)
    }

    pub fn accumulate_bins(
        &mut self,
        shared_bins: &mut [Complex<f32>],
        velocity: u8,
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
    ) {
        let frame_idx = self.current_frame as usize;
        if frame_idx >= self.sample_data.total_frames {
            return;
        }
        let current_frame_slice = match self.sample_data.get_frame_slice(0, frame_idx) {
            Some(slice) => slice,
            None => return,
        };

        // 1. Envelope gain — sample midpoint of the hop for this frame
        let g_start = self.envelope.get_value_at_current_time();
        for _ in 0..fft_step {
            let _ = self.envelope.next_sample();
        }
        let g_end = self.envelope.get_value_at_current_time();
        let total_gain = ((g_start + g_end) * 0.5) * self.current_volume * (velocity as f32 / 127.0);

        // 2. Pitch ratio — note interval + pitch bend + sample rate compensation
        let note_diff = self.trigger_note as f32 - self.root_note as f32;
        let base_pitch_ratio = (note_diff / 12.0_f32).exp2();
        let sample_rate_scaling = self.sample_data.original_sample_rate as f32 / self.sample_rate;
        let total_pitch_ratio = base_pitch_ratio * self.current_pitch_bend * sample_rate_scaling;

        // Expected phase advance per bin per hop under the current pitch ratio
        let expected_phase_step = (fft_step as f32 * 2.0 * std::f32::consts::PI) / fft_size as f32;

        // 3. Accumulate into shared complex bins
        for target_bin in 0..bin_count {
            let source_bin_exact = target_bin as f32 / total_pitch_ratio;
            let source_bin_floor = source_bin_exact as usize;

            if source_bin_floor + 1 >= current_frame_slice.len() {
                continue;
            }

            let frac = source_bin_exact - source_bin_floor as f32;

            let c_left = current_frame_slice[source_bin_floor];
            let c_right = current_frame_slice[source_bin_floor + 1];

            // Linearly interpolate the complex bins directly and derive magnitude/phase once.
            let c = ComplexBin::new(
                (1.0 - frac) * c_left.re + frac * c_right.re,
                (1.0 - frac) * c_left.im + frac * c_right.im,
            );

            let mag = c.magnitude() * total_gain;
            let raw_phase = c.phase();

            // Phase vocoder unwrapping
            let expected = self.previous_phases[target_bin] + target_bin as f32 * expected_phase_step;
            let delta = raw_phase - (self.previous_phases[target_bin] - target_bin as f32 * expected_phase_step * (frame_idx as f32 - 1.0).max(0.0));
            let delta = delta - (delta / (2.0 * std::f32::consts::PI)).round() * 2.0 * std::f32::consts::PI;
            let true_phase = expected + delta;
            self.previous_phases[target_bin] = true_phase;

            let (sin_p, cos_p) = true_phase.sin_cos();
            shared_bins[target_bin] += Complex::new(mag * cos_p, mag * sin_p);
        }

        // Advance frame position by pitch ratio so playback speed tracks pitch
        self.current_frame += total_pitch_ratio;
    }
}

impl<T: Simd> VoiceGeneratorBase for SpectralVoice<T> {
    #[inline(always)]
    fn ended(&self) -> bool {
        self.envelope.ended() || (self.current_frame as usize) >= self.sample_data.total_frames
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
    fn accumulate_bins(
        &mut self,
        shared_bins: &mut [rustfft::num_complex::Complex<f32>],
        velocity: u8,
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
    ) {
        self.accumulate_bins(shared_bins, velocity, fft_size, fft_step, bin_count);
    }

    #[inline(always)]
    fn spectral_group_key(&self) -> SpectralGroupKey {
        SpectralVoice::spectral_group_key(self)
    }

    #[inline(always)]
    fn spectral_generate_template(
        &mut self,
        template_bins: &mut [rustfft::num_complex::Complex<f32>],
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
    ) {
        self.spectral_generate_template(template_bins, fft_size, fft_step, bin_count);
    }

    #[inline(always)]
    fn spectral_copy_state_from(&mut self, source: &dyn SpectralVoiceSampleGenerator) {
        if let Some(source) = source.as_any().downcast_ref::<SpectralVoice<T>>() {
            self.copy_spectral_state_from(source);
        }
    }

    #[inline(always)]
    fn spectral_state_snapshot(&self) -> SpectralStateSnapshot {
        SpectralVoice::spectral_state_snapshot(self)
    }

    #[inline(always)]
    fn spectral_apply_state_snapshot(&mut self, snapshot: &SpectralStateSnapshot) {
        SpectralVoice::spectral_apply_state_snapshot(self, snapshot);
    }

    #[inline(always)]
    fn spectral_advance_gain(&mut self, velocity: u8, fft_step: usize) -> f32 {
        self.spectral_advance_gain(velocity, fft_step)
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
