use std::sync::Arc;
use rustfft::num_complex::Complex;
use super::{Bin, SpectralConfig, SpectralPlans};

/// Represents an analyzed frequency snapshot for a voice asset, stored as a sparse
/// set of the most prominent spectral peaks per frame rather than the full dense
/// FFT bin range.
///
/// Memory layout: each frame contributes exactly `peaks_per_frame` entries
/// (zero-padded if fewer real peaks were found), packed as:
/// [Channel 0 Frame 0 (peaks_per_frame entries), Channel 0 Frame 1, ..., Channel 1 Frame 0, ...]
#[derive(Clone, Debug)]
pub struct AnalyzedSample {
    pub flat_peaks: Arc<[Bin]>,
    pub total_frames: usize,
    pub peaks_per_frame: usize,
    pub channels_count: usize,
    pub original_sample_rate: u32,
    pub fft_size: usize,
    pub fft_step: usize,
}

impl AnalyzedSample {
    /// Returns the slice of (up to) `peaks_per_frame` peaks for the given channel/frame.
    /// Entries with `magnitude == 0.0` are padding and represent "no peak in this slot".
    #[inline(always)]
    pub fn get_frame_slice(&self, channel: usize, frame_idx: usize) -> Option<&[Bin]> {
        if channel >= self.channels_count || frame_idx >= self.total_frames {
            return None;
        }

        let channel_stride = self.total_frames * self.peaks_per_frame;
        let frame_stride = frame_idx * self.peaks_per_frame;

        let start = (channel * channel_stride) + frame_stride;
        let end = start + self.peaks_per_frame;

        Some(&self.flat_peaks[start..end])
    }

    pub fn duration_seconds(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        let total_samples = (self.total_frames - 1) * self.fft_step + self.fft_size;
        total_samples as f32 / self.original_sample_rate as f32
    }
}

/// Consumes raw resampled PCM arrays and maps them into a sparse frequency-peak
/// representation: for each analysis frame, only the `max_peaks_per_frame` most
/// prominent spectral peaks (by magnitude, above a noise floor, with a minimum
/// semitone/bin spacing) are retained.
///
/// Each selected peak is refined via parabolic interpolation on the log-magnitude
/// spectrum, giving sub-bin accurate frequency, magnitude, and phase estimates.
/// This is an analysis-time-only cost (paid once per unique sample region via the
/// caller's spectral cache) — `spectral_generate_template` only ever reads the
/// resulting `Bin.frequency`/`magnitude`/`phase` as plain floats, so no downstream
/// code needs to change.
pub fn analyze_pcm_sample(
    pcm_channels: Arc<[Arc<[f32]>]>,
    original_sample_rate: u32,
    config: &SpectralConfig,
    fft_plans: &SpectralPlans,
) -> AnalyzedSample {
    let fft_size = config.fft_size;
    let fft_step = config.fft_step();
    let top_n = config.max_peaks_per_frame;
    let window_coeffs = fft_plans.window();

    let channels_count = pcm_channels.len();
    let bin_count = (fft_size / 2) + 1;
    let max_len = pcm_channels.iter().map(|c| c.len()).max().unwrap_or(0);

    let total_frames = if max_len <= fft_size {
        1
    } else {
        ((max_len - fft_size) / fft_step) + 1
    };

    let mut fft_buffer = vec![Complex::new(0.0f32, 0.0f32); fft_size];
    let scratch_len = fft_plans.forward_plan.get_inplace_scratch_len();
    let mut scratch_buffer = vec![Complex::new(0.0f32, 0.0f32); scratch_len];

    // Pre-size the output exactly: channels * frames * top_n entries, all zero
    // (zero magnitude == "empty slot"). We write into specific indices per frame
    // rather than push()ing, which avoids reallocation and guarantees the fixed
    // stride that get_frame_slice relies on.
    let mut flat_peaks_accumulator: Vec<Bin> =
        vec![Bin::default(); channels_count * total_frames * top_n];

    // Reusable scratch buffers for the per-frame peak-picking pass — avoids
    // allocating a Vec for every single frame of every sample.
    // magnitudes/log_magnitudes/phases are indexed by raw bin index (0..bin_count)
    // and are needed for parabolic interpolation, which reads neighbours of each
    // selected peak by bin index.
    let mut magnitudes: Vec<f32> = vec![0.0; bin_count];
    let mut log_magnitudes: Vec<f32> = vec![0.0; bin_count];
    let mut phases: Vec<f32> = vec![0.0; bin_count];
    let mut sorted_bins: Vec<usize> = Vec::with_capacity(bin_count);
    let mut selected_peaks: Vec<Bin> = Vec::with_capacity(top_n);

    let delta_f = original_sample_rate as f32 / fft_size as f32;
    let magnitude_threshold_db = -60.0;
    let semitone_ratio = 2f32.powf(1.0 / 12.0);

    // Floor on magnitude for log10 to avoid -inf for exact-zero bins.
    const MAG_FLOOR: f32 = 1e-12;

    for (channel_idx, pcm_channel) in pcm_channels.iter().enumerate() {
        let channel_stride = total_frames * top_n;

        for frame_idx in 0..total_frames {
            let start_sample = frame_idx * fft_step;
            fft_buffer.fill(Complex::new(0.0, 0.0));
            for i in 0..fft_size {
                let sample_pos = start_sample + i;
                if sample_pos < pcm_channel.len() {
                    fft_buffer[i].re = pcm_channel[sample_pos] * window_coeffs[i];
                }
            }
            fft_plans.forward_plan.process_with_scratch(&mut fft_buffer, &mut scratch_buffer);

            // Compute magnitude, log-magnitude, and phase for every bin.
            // log_magnitudes is needed for the parabolic fit; magnitudes/phases
            // are used for selection and as fallback values at spectrum edges.
            for bin_idx in 0..bin_count {
                let c = fft_buffer[bin_idx];
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                magnitudes[bin_idx] = mag;
                log_magnitudes[bin_idx] = mag.max(MAG_FLOOR).log10();
                phases[bin_idx] = c.im.atan2(c.re);
            }

            // Sort bin indices by magnitude descending so the strongest candidates
            // are considered first.
            sorted_bins.clear();
            sorted_bins.extend(0..bin_count);
            sorted_bins.sort_unstable_by(|&a, &b| {
                magnitudes[b].partial_cmp(&magnitudes[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Greedily select up to top_n peaks: above the noise floor, and at
            // least one semitone (or 1.5 bins, whichever is larger) away from
            // every already-selected peak's frequency.
            //
            // The bin-resolution floor (1.5 * delta_f) matters at low frequencies:
            // for a 4096-pt FFT at 44.1kHz, delta_f ~= 10.77 Hz, but a semitone at
            // 55 Hz (A1) is only ~3.27 Hz wide — without the floor, multiple bins
            // from the SAME spectral lobe would pass the spacing check and waste
            // top_n slots on redundant samples of one harmonic.
            selected_peaks.clear();
            for &idx in sorted_bins.iter() {
                if selected_peaks.len() >= top_n {
                    break;
                }

                let mag = magnitudes[idx];
                let mag_db = 20.0 * mag.max(MAG_FLOOR).log10();
                if mag_db < magnitude_threshold_db {
                    // Bins are sorted descending — once we drop below the floor,
                    // every remaining bin is also below it.
                    break;
                }

                let raw_freq = idx as f32 * delta_f;
                let min_dist = (raw_freq * (semitone_ratio - 1.0)).max(delta_f * 1.5);

                if selected_peaks.iter().any(|p| (p.frequency - raw_freq).abs() < min_dist) {
                    continue;
                }

                // --- Parabolic interpolation on log-magnitude ---
                //
                // Fits a parabola through (idx-1, idx, idx+1) in log-magnitude
                // space and finds its vertex, giving a sub-bin offset `p` in
                // [-0.5, 0.5]. This corrects the systematic bias where the true
                // sinusoid frequency falls between bin centers but the raw FFT
                // only reports energy at discrete bin frequencies.
                //
                // Edge bins (idx == 0 or idx == bin_count - 1) have no neighbour
                // on one side and are left uninterpolated — these correspond to
                // DC and Nyquist, which are not meaningful "peaks" for tonal
                // content anyway.
                let (refined_freq, refined_mag, refined_phase) =
                    if idx > 0 && idx < bin_count - 1 {
                        let alpha = log_magnitudes[idx - 1];
                        let beta = log_magnitudes[idx];
                        let gamma = log_magnitudes[idx + 1];

                        let denom = alpha - 2.0 * beta + gamma;

                        // denom == 0 means the three points are collinear (flat or
                        // linear region) — the parabola is degenerate, so fall back
                        // to the raw bin with no offset rather than dividing by zero.
                        let p = if denom.abs() > f32::EPSILON {
                            (0.5 * (alpha - gamma) / denom).clamp(-0.5, 0.5)
                        } else {
                            0.0
                        };

                        let interp_freq = (idx as f32 + p) * delta_f;

                        // Refined log-magnitude at the parabola vertex, converted
                        // back to linear magnitude.
                        let interp_log_mag = beta - 0.25 * (alpha - gamma) * p;
                        let interp_mag = 10f32.powf(interp_log_mag);

                        // Phase: linearly interpolate toward whichever neighbour
                        // the offset `p` points at, weighted by |p|. Phase varies
                        // far more slowly than magnitude near a peak, so this is
                        // a minor refinement compared to the frequency/magnitude
                        // correction, but keeps phase consistent with the
                        // corrected frequency.
                        let interp_phase = if p >= 0.0 {
                            phases[idx] * (1.0 - p) + phases[idx + 1] * p
                        } else {
                            phases[idx] * (1.0 + p) + phases[idx - 1] * (-p)
                        };

                        (interp_freq, interp_mag, interp_phase)
                    } else {
                        (raw_freq, mag, phases[idx])
                    };

                selected_peaks.push(Bin::new(refined_freq, refined_mag, refined_phase));
            }

            // Write into the fixed-stride output. Slots beyond selected_peaks.len()
            // remain Bin::default() (magnitude == 0.0), which spectral_generate_template
            // treats as "empty" and skips.
            let frame_start = (channel_idx * channel_stride) + (frame_idx * top_n);
            for (slot, peak) in selected_peaks.iter().enumerate() {
                flat_peaks_accumulator[frame_start + slot] = *peak;
            }
        }
    }

    AnalyzedSample {
        flat_peaks: Arc::from(flat_peaks_accumulator),
        total_frames,
        peaks_per_frame: top_n,
        channels_count,
        original_sample_rate,
        fft_size,
        fft_step,
    }
}