use std::hash::{Hash, Hasher};
use std::sync::Arc;
use rustfft::num_complex::Complex;
use simdeez::prelude::*;

use crate::voice::{ReleaseType, SpectralGroupKey, SpectralStateSnapshot, VoiceControlData, VoiceGeneratorBase, VoiceSampleGenerator, SIMDVoiceEnvelope, SIMDVoiceGenerator, SpectralVoiceSampleGenerator};
use super::AnalyzedSample;

pub struct SpectralVoice<T: Simd> {
    sample_data: Arc<AnalyzedSample>,
    envelope: SIMDVoiceEnvelope<T>,
    current_frame: f32,
    root_note: u8,
    trigger_note: u8,
    sample_rate: f32,
    current_volume: f32,
    current_pitch_bend: f32,

    /// Phase accumulator state, indexed by PEAK index within a frame (0..peaks_per_frame),
    /// not by FFT bin index. Since peaks_per_frame (e.g. 64) is fixed and far smaller than
    /// bin_count (e.g. 513/2049), this is a small fixed-size vector regardless of FFT size.
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
        // Sized to peaks_per_frame, not fft_size/2+1 — phase tracking is per selected
        // peak, not per dense FFT bin.
        let peak_count = sample_data.peaks_per_frame;
        Self {
            sample_data,
            envelope,
            current_frame: 0.0,
            root_note,
            trigger_note,
            sample_rate,
            current_volume: 1.0,
            current_pitch_bend: 1.0,
            previous_phases: vec![0.0; peak_count],
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

    /// Builds the per-block complex spectrum template from this voice's currently
    /// selected analysis peaks.
    ///
    /// Unlike the previous dense implementation, this iterates ONLY the peaks stored
    /// for the current frame (typically ~64) rather than every FFT bin (typically
    /// 513-2049). Each peak carries its own `frequency` (in Hz, computed at analysis
    /// time), which is converted directly to a target bin index after applying the
    /// pitch ratio — there is no "source bin index" lookup, because the peak array is
    /// sparse and not addressable by bin index.
    pub fn spectral_generate_template(
        &mut self,
        template_bins: &mut [Complex<f32>],
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
    ) {
        // Clear only once — peaks write sparsely, so anything not touched must be zero.
        template_bins.fill(Complex::new(0.0, 0.0));

        let frame_idx = self.current_frame as usize;
        if frame_idx >= self.sample_data.total_frames {
            return;
        }

        let current_frame_peaks = match self.sample_data.get_frame_slice(0, frame_idx) {
            Some(slice) => slice,
            None => return,
        };

        let note_diff = self.trigger_note as f32 - self.root_note as f32;
        let base_pitch_ratio = (note_diff / 12.0_f32).exp2();
        let sample_rate_scaling = self.sample_data.original_sample_rate as f32 / self.sample_rate;
        let total_pitch_ratio = base_pitch_ratio * self.current_pitch_bend * sample_rate_scaling;

        // Hz-per-bin in the OUTPUT spectrum (pipeline's fft_size/sample_rate), used to
        // convert a peak's (pitch-shifted) frequency into a target bin index.
        let output_bin_hz = self.sample_rate / fft_size as f32;

        // Expected phase advance per hop for a sinusoid at a given frequency:
        // delta_phase = 2*pi * freq * (fft_step / sample_rate)
        let hop_seconds = fft_step as f32 / self.sample_rate;

        // previous_phases is indexed by PEAK SLOT (0..peaks_per_frame), not by bin.
        // If the analysis stored fewer peaks for this frame than peaks_per_frame
        // (e.g. a quiet frame with few bins above the noise floor), the unused
        // slots simply retain stale phase values that are never read.
        for (peak_idx, peak) in current_frame_peaks.iter().enumerate() {
            // A zero-magnitude peak means this slot was unused for this frame
            // (analysis pads to peaks_per_frame with empty Bins). Skip it —
            // writing a zero-magnitude contribution is harmless but wasted work.
            if peak.magnitude <= 0.0 {
                continue;
            }

            // Pitch-shift the peak's frequency directly.
            let shifted_freq = peak.frequency * total_pitch_ratio;

            // Map to target bin in the OUTPUT spectrum.
            let target_bin_exact = shifted_freq / output_bin_hz;
            let target_bin = target_bin_exact.round() as usize;

            if target_bin >= bin_count {
                continue;
            }

            // Phase vocoder: predict phase advance for this peak's (shifted) frequency
            // over one hop, then unwrap against the previous true phase for this peak slot.
            let expected_phase_step = 2.0 * std::f32::consts::PI * shifted_freq * hop_seconds;
            let expected = self.previous_phases[peak_idx] + expected_phase_step;

            let raw_phase = peak.phase;
            let delta = raw_phase - (self.previous_phases[peak_idx]
                - expected_phase_step * (frame_idx as f32 - 1.0).max(0.0));
            let delta = delta - (delta / (2.0 * std::f32::consts::PI)).round() * 2.0 * std::f32::consts::PI;
            let true_phase = expected + delta;
            self.previous_phases[peak_idx] = true_phase;

            let (sin_p, cos_p) = true_phase.sin_cos();

            // Multiple peaks (from different voices in a group, or even from the same
            // voice if pitch-shifting causes two source peaks to collide on one output
            // bin) can target the same bin — accumulate rather than overwrite.
            template_bins[target_bin] += Complex::new(peak.magnitude * cos_p, peak.magnitude * sin_p);
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
        let g_start = self.envelope.get_value_at_current_time();
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