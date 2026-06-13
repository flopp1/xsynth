use rustfft::num_complex::Complex;
use simdeez::prelude::*;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;
use super::{SpectralConfig, SpectralPlans};
use crate::voice::Voice;

pub struct SpectralPipeline<T: Simd> {
    config: SpectralConfig,
    pub sample_rate: f32,
    fft_plans: Arc<SpectralPlans>,
    complex_accumulator: Vec<Complex<f32>>,
    ifft_buffer: Vec<Complex<f32>>,
    template_scratch: Vec<(usize, Complex<f32>)>,
    overlap_buffer: Vec<f32>,
    ring_buffer: VecDeque<f32>,
    _marker: PhantomData<T>,
}

impl<T: Simd> SpectralPipeline<T> {
    pub fn new(config: SpectralConfig, sample_rate: f32, fft_plans: Arc<SpectralPlans>) -> Self {
        let fft_size = config.fft_size;
        let bin_count = fft_size / 2 + 1;

        Self {
            fft_plans,
            complex_accumulator: vec![Complex::new(0.0, 0.0); bin_count],
            ifft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            template_scratch: Vec::with_capacity(config.max_peaks_per_frame * 2),
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

    pub fn drain_into_pipeline(&mut self, out_buffer: &mut [f32], active_voices: &mut [&mut dyn Voice]) {
        let mut filled = 0;
        while filled < out_buffer.len() {
            if self.ring_buffer.is_empty() {
                self.process_spectral_block(active_voices);
            }

            let needed = out_buffer.len() - filled;
            let n = self.ring_buffer.len().min(needed);

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

    // Generates one step-length worth of output samples via spectral accumulation + IFFT + OLA.
    pub fn process_spectral_block(&mut self, active_voices: &mut [&mut dyn Voice]) {
        let fft_size = self.config.fft_size;
        let fft_step = self.config.fft_step;
        let magnitude_res = self.config.magnitude_res;
        let bin_count = fft_size / 2 + 1;

        // Reset complex accumulator
        self.complex_accumulator.fill(Complex::new(0.0, 0.0));

        // Accumulate complex bins from spectral voices
        for voice_ref in active_voices.iter_mut() {
            let velocity = voice_ref.velocity();
            if let Some(spectral_voice) = voice_ref.as_spectral_voice_mut() {
                spectral_voice.update_host_sample_rate(self.sample_rate);
                
                let gain = spectral_voice.get_spectral_gain(velocity, fft_step);
                if gain == 0.0 { continue; }
                spectral_voice.spectral_process_voice(
                    &mut self.template_scratch,
                    fft_size,
                    fft_step,
                    bin_count,
                    magnitude_res,
                );
                for (bin_idx, c) in self.template_scratch.iter() {
                    let scaled = Complex::new(c.re * gain, c.im * gain);
                    self.complex_accumulator[*bin_idx] += scaled;
                }
            }
        }

        // Build the full symmetric IFFT input buffer from the accumulated complex spectrum.
        // Bins 0..bin_count are the unique positive-frequency bins; the upper half is the
        // conjugate mirror required by a real-valued IFFT.
        self.ifft_buffer[0] = self.complex_accumulator[0];
        for bin in 1..bin_count - 1 {
            let c = self.complex_accumulator[bin];
            self.ifft_buffer[bin] = c;
            self.ifft_buffer[fft_size - bin] = c.conj();
        }
        self.ifft_buffer[bin_count - 1] = self.complex_accumulator[bin_count - 1];

        self.fft_plans.execute_inverse(&mut self.ifft_buffer);

        // Fused Extraction, Windowing, and Overlap-Add
        self.fft_plans.window_and_overlap_add(&self.ifft_buffer, &mut self.overlap_buffer);

        // Push exactly fft_step samples to the ring buffer.
        // The remaining (fft_size - fft_step) samples stay in overlap_buffer to be
        // summed with the next frame for correct OLA reconstruction.
        self.ring_buffer.extend(self.overlap_buffer[..fft_step].iter().copied());

        // Shift overlap buffer left by fft_step, zero the vacated tail
        self.overlap_buffer.copy_within(fft_step..fft_size, 0);
        self.overlap_buffer[fft_size - fft_step..].fill(0.0);
    }
}
/*
[cfg(test)]
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
        pipeline.drain_into_pipeline(&mut output, &mut voices);

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

    #[test]
    fn test_dense_fft_ifft_roundtrip() {
        use hound::{WavReader, WavWriter, WavSpec, SampleFormat};
        use rustfft::num_complex::Complex;
        use std::fs;

        // Load a mono WAV
        let mut reader = WavReader::open("C:\\Users\\ethen\\Downloads\\xsynth\\target\\release-with-debug\\input.wav").unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1, "use mono samples for test");

        let samples: Vec<f32> = match spec.sample_format {
            SampleFormat::Int => {
                match spec.bits_per_sample {
                    16 => reader.samples::<i16>()
                        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
                        .collect(),
                    24 | 32 => reader.samples::<i32>()
                        .map(|s| s.unwrap() as f32 / i32::MAX as f32)
                        .collect(),
                    _ => panic!("unsupported int bit depth: {}", spec.bits_per_sample),
                }
            }
            SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap()).collect()
            }
        };


        // FFT setup
        let fft_size = 32768;
        let fft_step = 2048;
        let plans = crate::spectral::SpectralPlans::new(fft_size, fft_step);

        // Output buffer
        let mut output = vec![0.0f32; samples.len()];
        let mut fft_buffer = vec![Complex::new(0.0, 0.0); fft_size];

        // Frame loop: full FFT/IFFT, no peak selection
        for frame_idx in 0..((samples.len() - fft_size) / fft_step) {
            let start = frame_idx * fft_step;
            for i in 0..fft_size {
                fft_buffer[i].re = samples[start + i] * plans.window()[i];
                fft_buffer[i].im = 0.0;
            }

            plans.execute_forward(&mut fft_buffer);

            // Mirror for real IFFT
            for bin in 1..fft_size/2 {
                fft_buffer[fft_size - bin] = fft_buffer[bin].conj();
            }

            plans.execute_inverse(&mut fft_buffer);

            let scale = 1.0 / fft_size as f32;
            for i in 0..fft_size {
                let pos = start + i;
                if pos < output.len() {
                    output[pos] += fft_buffer[i].re * scale;
                }
            }
        }

        // Write out reconstructed WAV
        let out_spec = WavSpec {
            channels: 1,
            sample_rate: spec.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create("C:\\Users\\ethen\\Downloads\\xsynth\\target\\release-with-debug\\output.wav", out_spec).unwrap();
        for s in output {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(v).unwrap();
        }
        writer.finalize().unwrap();

        // Manual check: listen to dense_roundtrip.wav — should sound identical to input.wav
        assert!(fs::metadata("C:\\Users\\ethen\\Downloads\\xsynth\\target\\release-with-debug\\output.wav").is_ok());
    }

}
*/