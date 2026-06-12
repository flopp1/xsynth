use rustfft::num_complex::Complex;
use simdeez::prelude::*;
use std::sync::Arc;

use super::AnalyzedSample;
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
    previous_phases: Vec<f32>, // Phase accumulator state, indexed by peak index within a frame
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
        let peak_count = sample_data.peaks_per_frame;
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
            previous_phases: vec![0.0; peak_count],
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
    ) {
        scratch.clear();

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
        self.last_pitch_ratio = total_pitch_ratio;
        let output_bin_hz = self.sample_rate / fft_size as f32;

        // Expected phase advance per hop for a sinusoid at a given frequency:
        // delta_phase = 2*pi * freq * (fft_step / sample_rate)
        //let hop_seconds = fft_step as f32 / self.sample_rate;

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
            if shifted_freq < output_bin_hz * 0.5 {
                continue;
            }
            // Map to target bin in the OUTPUT spectrum.
            let target_bin_exact = shifted_freq / output_bin_hz;
            // fractional interpolation across neighboring bins
            let lo_f = target_bin_exact.floor();
            let frac = target_bin_exact - lo_f;
            if lo_f.is_nan() {
                continue;
            }
            let lo = lo_f as isize;
            let hi = lo + 1;
            /*let target_bin = target_bin_exact.round() as usize;

            if target_bin >= bin_count {
                continue;
            }*/

            // Phase vocoder: predict phase advance for this peak's (shifted) frequency
            // over one hop, then unwrap against the previous true phase for this peak slot.
            //let expected_phase_step = 2.0 * std::f32::consts::PI * shifted_freq * hop_seconds;
            //let expected = self.previous_phases[peak_idx] + expected_phase_step;

            //let raw_phase = peak.phase;
            //let delta = raw_phase - (self.previous_phases[peak_idx]
            //    - expected_phase_step * (frame_idx as f32 - 1.0).max(0.0));
            //let delta = delta - (delta / (2.0 * std::f32::consts::PI)).round() * 2.0 * std::f32::consts::PI;
            //let true_phase = expected + delta;
            //self.previous_phases[peak_idx] = true_phase;
            //end of phase vocoder

            //direct phase tracking without explicit unwrapping — relies on high overlap
            //to keep phase changes between frames small and consistent.
            //This is more robust to inharmonicity and non-linear pitch shifts,
            //at the cost of potentially less accurate phase tracking at low FFT sizes and low frequencies.
            // In SpectralVoice, replace previous_phases usage with per-peak running phase
            // driven by the SHIFTED frequency, advanced by one hop's worth of phase per block:
            let phase_increment =
                2.0 * std::f32::consts::PI * shifted_freq * (fft_step as f32 / self.sample_rate);
            self.previous_phases[peak_idx] =
                (self.previous_phases[peak_idx] + phase_increment) % (2.0 * std::f32::consts::PI);
            let true_phase = self.previous_phases[peak_idx];

            let (sin_p, cos_p) = true_phase.sin_cos();

            let contrib = Complex::new(peak.magnitude * cos_p, peak.magnitude * sin_p);

            // distribute to lo and hi bins proportionally
            if lo >= 0 && (lo as usize) < bin_count {
                let c = contrib * (1.0 - frac);
                scratch.push((lo as usize, c));
            }
            if hi >= 0 && (hi as usize) < bin_count {
                let c = contrib * frac;
                scratch.push((hi as usize, c));
            }
        }

        self.current_frame += 1.0;
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
    fn spectral_process_voice(
        &mut self,
        scratch: &mut Vec<(usize, rustfft::num_complex::Complex<f32>)>,
        fft_size: usize,
        fft_step: usize,
        bin_count: usize,
    ) {
        self.spectral_process_voice(scratch, fft_size, fft_step, bin_count);
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
