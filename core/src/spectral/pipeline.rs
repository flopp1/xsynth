use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::Arc;
use rustfft::num_complex::Complex;
use simdeez::prelude::*;

use crate::voice::{SpectralGroupKey, Voice};
use super::SpectralPlans;

/// Configuration profile defining the structural dimensions and behavior of the spectral engine
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct SpectralConfig {
    pub fft_size: usize,
    pub fft_step: usize,
    /// Maximum number of simultaneous spectral voices.
    /// `None` means unlimited voices are accepted.
    pub max_voices: Option<usize>,
    pub enable_phase_fade_out: bool,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            fft_step: 256,
            max_voices: Some(4 * 512),
            enable_phase_fade_out: true,
        }
    }
}

pub struct SpectralPipeline<T: Simd> {
    config: SpectralConfig,
    pub sample_rate: f32,
    fft_plans: Arc<SpectralPlans>,

    // Per-block accumulation in complex domain — summed across voices before IFFT
    complex_accumulator: Vec<Complex<f32>>,

    // Reusable IFFT input/output scratch — avoids per-block heap allocation
    ifft_buffer: Vec<Complex<f32>>,
    time_domain_scratch: Vec<f32>,
    template_bins: Vec<Complex<f32>>,

    // OLA state: fft_size long, shifted by fft_step each block
    overlap_buffer: Vec<f32>,

    // Ring buffer for decoupling block size from audio callback size
    // VecDeque gives O(1) front drain vs Vec's O(n)
    ring_buffer: VecDeque<f32>,

    _marker: PhantomData<T>,
}

impl<T: Simd> SpectralPipeline<T> {
    /// Construct with a shared SpectralPlans — call sites should Arc::clone the existing
    /// plan rather than constructing a new one, so FFT butterfly tables aren't duplicated.
    pub fn new(config: SpectralConfig, sample_rate: f32, fft_plans: Arc<SpectralPlans>) -> Self {
        let fft_size = config.fft_size;
        let bin_count = fft_size / 2 + 1;

        Self {
            fft_plans,
            complex_accumulator: vec![Complex::new(0.0, 0.0); bin_count],
            ifft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            time_domain_scratch: vec![0.0; fft_size],
            template_bins: vec![Complex::new(0.0, 0.0); bin_count],
            // OLA buffer is fft_size, not fft_size*2 — the second half was never
            // written in the old code, only zero-padded, so it served no purpose
            overlap_buffer: vec![0.0; fft_size],
            config,
            sample_rate,
            ring_buffer: VecDeque::new(),
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn config(&self) -> &SpectralConfig {
        &self.config
    }

    /// Drains up to `out_buffer.len()` samples from the ring buffer into `out_buffer`,
    /// generating new spectral blocks as needed.
    pub fn drain_into(&mut self, out_buffer: &mut [f32], active_voices: &mut [&mut dyn Voice]) {
        let mut filled = 0;
        while filled < out_buffer.len() {
            if self.ring_buffer.is_empty() {
                self.process_spectral_block(active_voices);
            }

            let needed = out_buffer.len() - filled;
            let n = self.ring_buffer.len().min(needed);

            // VecDeque::as_slices gives up to two contiguous slices (head/tail wrap).
            // Copy each slice in turn to avoid a collect() or element-wise loop.
            let (a, b) = self.ring_buffer.as_slices();
            let first = n.min(a.len());
            out_buffer[filled..filled + first].copy_from_slice(&a[..first]);
            if first < n {
                let second = n - first;
                out_buffer[filled + first..filled + n].copy_from_slice(&b[..second]);
            }
            self.ring_buffer.drain(..n);
            filled += n;
        }
    }

    /// Generates one hop-length worth of output samples via spectral accumulation + IFFT + OLA.
    pub fn process_spectral_block(&mut self, active_voices: &mut [&mut dyn Voice]) {
        let fft_size = self.config.fft_size;
        let fft_step = self.config.fft_step;
        let bin_count = fft_size / 2 + 1;

        // 1. Reset complex accumulator
        self.complex_accumulator.fill(Complex::new(0.0, 0.0));

        // 2. Accumulate complex bins from spectral voices using grouping.
        //    Voices with identical spectral state, pitch and frame position can share a
        //    single template computation, making accumulation scale with active groups
        //    rather than with raw voice count.
        let mut groups: HashMap<SpectralGroupKey, Vec<&mut dyn Voice>> = HashMap::with_capacity(active_voices.len());
        for voice_ref in active_voices.iter_mut() {
            if let Some(spectral_voice) = voice_ref.as_spectral_voice_mut() {
                debug_assert_eq!(
                    spectral_voice.fft_size(), self.config.fft_size,
                    "Voice FFT size {} does not match pipeline {}",
                    spectral_voice.fft_size(), self.config.fft_size
                );
                debug_assert_eq!(
                    spectral_voice.fft_step(), self.config.fft_step,
                    "Voice FFT step {} does not match pipeline {}",
                    spectral_voice.fft_step(), self.config.fft_step
                );

                spectral_voice.update_host_sample_rate(self.sample_rate);
                let key = spectral_voice.spectral_group_key();
                groups.entry(key).or_default().push(*voice_ref);
            }
        }

        self.template_bins.fill(Complex::new(0.0, 0.0));
        for voices in groups.values_mut() {
            let mut total_gain = 0.0;
            let rep_voice = voices.get_mut(0).unwrap();
            let rep_spectral = (*rep_voice).as_spectral_voice_mut().unwrap();

            rep_spectral.spectral_generate_template(&mut self.template_bins, fft_size, fft_step, bin_count);
            let snapshot = rep_spectral.spectral_state_snapshot();

            for voice_slot in voices.iter_mut().skip(1) {
                let voice_spectral = (*voice_slot).as_spectral_voice_mut().unwrap();
                voice_spectral.spectral_apply_state_snapshot(&snapshot);
            }

            for voice_slot in voices.iter_mut() {
                let velocity = voice_slot.velocity();
                let voice_spectral = (*voice_slot).as_spectral_voice_mut().unwrap();
                total_gain += voice_spectral.spectral_advance_gain(velocity, fft_step);
            }

            if total_gain != 0.0 {
                for bin in 0..bin_count {
                    self.complex_accumulator[bin] += self.template_bins[bin] * total_gain;
                }
            }
        }

        // 3. Build the full symmetric IFFT input buffer from the accumulated complex spectrum.
        //    Bins 0..bin_count are the unique positive-frequency bins; the upper half is the
        //    conjugate mirror required by a real-valued IFFT.
        self.ifft_buffer[0] = self.complex_accumulator[0];
        for bin in 1..bin_count - 1 {
            let c = self.complex_accumulator[bin];
            self.ifft_buffer[bin] = c;
            self.ifft_buffer[fft_size - bin] = c.conj();
        }
        self.ifft_buffer[bin_count - 1] = self.complex_accumulator[bin_count - 1];

        // 4. In-place IFFT using the pre-built plan — no allocation
        self.fft_plans.execute_inverse(&mut self.ifft_buffer);

        // 5. Normalize (rustfft IFFT is unscaled) and write to time-domain scratch
        let scale = 1.0 / fft_size as f32;
        for i in 0..fft_size {
            self.time_domain_scratch[i] = self.ifft_buffer[i].re * scale;
        }

        // 6. Apply synthesis window
        self.fft_plans.apply_synthesis_window(&mut self.time_domain_scratch);

        // 7. Overlap-add: accumulate the windowed frame into the overlap buffer
        for i in 0..fft_size {
            self.overlap_buffer[i] += self.time_domain_scratch[i];
        }

        // 8. Push exactly fft_step samples to the ring buffer — NOT fft_size.
        //    The remaining (fft_size - fft_step) samples stay in overlap_buffer to be
        //    summed with the next frame. This is what makes OLA reconstruct correctly.
        self.ring_buffer.extend(self.overlap_buffer[..fft_step].iter().copied());

        // 9. Shift overlap buffer left by fft_step, zero the vacated tail.
        //    copy_within is safe for overlapping ranges within the same slice.
        self.overlap_buffer.copy_within(fft_step..fft_size, 0);
        self.overlap_buffer[fft_size - fft_step..].fill(0.0);
    }
}