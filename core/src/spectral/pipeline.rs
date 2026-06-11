use rustfft::num_complex::Complex;
use simdeez::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::Arc;

use super::SpectralPlans;
use crate::voice::{SpectralGroupKey, Voice};

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
    pub max_peaks_per_frame: usize, // New parameter to control the number of peaks retained per frame
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            fft_size: 8192,
            fft_step: 2048, // 75% overlap is common for good time-frequency resolution balance
            max_voices: Some(4 * 512),
            enable_phase_fade_out: true,
            max_peaks_per_frame: 64,
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

    /// Generates one step-length worth of output samples via spectral accumulation + IFFT + OLA.
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
        let mut groups: HashMap<SpectralGroupKey, Vec<&mut dyn Voice>> =
            HashMap::with_capacity(active_voices.len());
        for voice_ref in active_voices.iter_mut() {
            if let Some(spectral_voice) = voice_ref.as_spectral_voice_mut() {
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

            rep_spectral.spectral_generate_template(
                &mut self.template_bins,
                fft_size,
                fft_step,
                bin_count,
            );
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
        self.fft_plans
            .apply_synthesis_window(&mut self.time_domain_scratch);

        // 7. Overlap-add: accumulate the windowed frame into the overlap buffer
        for i in 0..fft_size {
            self.overlap_buffer[i] += self.time_domain_scratch[i];
        }

        // 8. Push exactly fft_step samples to the ring buffer.
        //    The remaining (fft_size - fft_step) samples stay in overlap_buffer to be
        //    summed with the next frame for correct OLA reconstruction.
        self.ring_buffer
            .extend(self.overlap_buffer[..fft_step].iter().copied());

        // 9. Shift overlap buffer left by fft_step, zero the vacated tail.
        //    copy_within is safe for overlapping ranges within the same slice.
        self.overlap_buffer.copy_within(fft_step..fft_size, 0);
        self.overlap_buffer[fft_size - fft_step..].fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::{AnalyzedSample, Bin, SpectralVoice};
    use crate::voice::{make_sustained_envelope, VoiceBase};
    use simdeez::scalar::Scalar;
    use std::sync::Arc;
    //test 2
    #[test]
    fn test_sine_through_pipeline_ola() {
        let sample_rate = 44100.0_f32;
        let config = SpectralConfig::default(); // fft_size=8192, overlap=0.75 -> fft_step=256
        let fft_size = config.fft_size;
        //let bin_count = fft_size / 2 + 1;
        let delta_f = sample_rate / fft_size as f32;

        // Pick a frequency that lands close to bin center to start (e.g. bin 10 -> ~430.6 Hz)
        let test_bin = 10;
        let test_freq = test_bin as f32 * delta_f;

        // Construct a synthetic AnalyzedSample with ONE peak per frame, constant across
        // many frames (simulating a sustained tone). Use enough frames to get several
        // seconds of output and look for periodic discontinuities.
        let total_frames = 200;
        let peaks_per_frame = 1;
        let mut flat_peaks = vec![Bin::default(); total_frames * peaks_per_frame];
        for f in 0..total_frames {
            let window_sum: f32 = {
                let plans = SpectralPlans::new(fft_size, config.fft_step);
                plans.window().iter().sum()
            };
            let expected_peak_magnitude = 1.0 * window_sum / 2.0;

            flat_peaks[f] = Bin::new(test_freq, expected_peak_magnitude, 0.0);
        }

        let analyzed = AnalyzedSample {
            flat_peaks: Arc::from(flat_peaks),
            total_frames,
            peaks_per_frame,
            channels_count: 1,
            original_sample_rate: sample_rate as u32,
            fft_size,
            fft_step: config.fft_step,
        };

        let fft_plans = Arc::new(SpectralPlans::new(fft_size, config.fft_step));
        let mut pipeline: SpectralPipeline<Scalar> =
            SpectralPipeline::new(config, sample_rate, fft_plans.clone());

        // Construct one voice at unity pitch (root_note == trigger_note)
        let spectral_voice = SpectralVoice::<Scalar>::new(
            Arc::new(analyzed),
            /* envelope */
            make_sustained_envelope(sample_rate as u32), // envelope that stays at 1.0
            /* root_note */ 60,
            /* trigger_note */ 60,
            sample_rate,
        );
        let mut voice = VoiceBase::new(127, None, spectral_voice);
        // Render ~1 second of audio
        let mut output = vec![0.0f32; sample_rate as usize];
        let mut voices: Vec<&mut dyn Voice> = vec![&mut voice];
        pipeline.drain_into(&mut output, &mut voices);

        // --- Assertions ---

        // 1. No clipping / blowup — amplitude should be bounded and reasonable.
        //    For a single unity-magnitude peak with Hann window + 75% OLA, the
        //    reconstructed sine amplitude should be roughly O(1), NOT orders of
        //    magnitude larger or smaller. If this fails, it's a normalization bug.
        let max_abs = output.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        println!("max abs amplitude: {}", max_abs);
        assert!(
            max_abs > 0.01,
            "Output is near-silent — check IFFT/OLA scale factor"
        );
        assert!(
            max_abs < 100.0,
            "Output is blown up — check IFFT/OLA scale factor"
        );

        // 2. Steady-state amplitude consistency — skip the first fft_size samples
        //    (OLA ramp-up) and check that peak amplitude in each fft_step-sized
        //    chunk is roughly constant. Large variation = OLA discontinuity =
        //    "buzzy/metallic" artifact.
        let step = config.fft_step;
        let skip = fft_size; // skip startup transient
        let mut chunk_peaks = Vec::new();
        let mut i = skip;
        while i + step <= output.len() {
            let chunk_max = output[i..i + step]
                .iter()
                .fold(0.0f32, |m, &x| m.max(x.abs()));
            chunk_peaks.push(chunk_max);
            i += step;
        }

        let avg = chunk_peaks.iter().sum::<f32>() / chunk_peaks.len() as f32;
        let max_dev = chunk_peaks
            .iter()
            .map(|&p| (p - avg).abs())
            .fold(0.0, f32::max);
        let rel_dev = max_dev / avg.max(1e-6);

        println!(
            "steady-state chunk peaks: avg={}, max_dev={}, rel_dev={}",
            avg, max_dev, rel_dev
        );

        // For a perfect OLA reconstruction of a sustained sine, chunk-to-chunk peak
        // amplitude should vary by only a few percent (the sine just isn't aligned
        // to chunk boundaries). Large rel_dev (say >30%) indicates OLA isn't
        // summing correctly across hops -- the smoking gun for "buzzy/metallic".
        assert!(
            rel_dev < 0.3,
            "Chunk peak amplitude varies too much ({}%) — OLA discontinuity suspected",
            rel_dev * 100.0
        );

        // 3. Spectral check: FFT the steady-state portion and confirm energy is
        //    concentrated at test_freq, not spread across many bins (which would
        //    indicate phase discontinuities introducing sidebands/noise).
        let analysis_fft_size = 4096;
        let mut check_buf: Vec<Complex<f32>> = output[skip..skip + analysis_fft_size]
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft_forward(analysis_fft_size);
        fft.process(&mut check_buf);

        let check_bin_count = analysis_fft_size / 2 + 1;
        let check_delta_f = sample_rate / analysis_fft_size as f32;
        let mags: Vec<f32> = (0..check_bin_count)
            .map(|b| (check_buf[b].re.powi(2) + check_buf[b].im.powi(2)).sqrt())
            .collect();

        let total_energy: f32 = mags.iter().map(|m| m * m).sum();
        let expected_bin = (test_freq / check_delta_f).round() as usize;

        // Energy in a window around the expected bin
        let window_radius = 3;
        let lo = expected_bin.saturating_sub(window_radius);
        let hi = (expected_bin + window_radius).min(check_bin_count - 1);
        let peak_energy: f32 = mags[lo..=hi].iter().map(|m| m * m).sum();

        let concentration = peak_energy / total_energy.max(1e-9);
        println!(
            "spectral concentration around {}Hz: {}",
            test_freq, concentration
        );

        // For a clean sine, >95% of energy should be within a few bins of the
        // fundamental. Significant energy elsewhere = noise/distortion being
        // introduced by the OLA process itself.
        assert!(
            concentration > 0.9,
            "Energy spread across spectrum ({}% concentrated) — OLA introducing distortion",
            concentration * 100.0
        );
    }
}
