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
/// caller's spectral cache) — `spectral_process_voice` only ever reads the
/// resulting `Bin.frequency`/`magnitude`/`phase` as plain floats, so no downstream
/// code needs to change.
/*pub fn analyze_pcm_sample(
    pcm_channels: Arc<[Arc<[f32]>]>,
    original_sample_rate: u32,
    config: &SpectralConfig,
    fft_plans: &SpectralPlans,
) -> AnalyzedSample {
    let fft_size = config.fft_size;
    let fft_step = config.fft_step;
    let top_n = config.max_peaks_per_frame;
    let window_coeffs = fft_plans.window();

    let channels_count = pcm_channels.len();
    let bin_count = (fft_size / 2) + 1;
    let max_len = pcm_channels.iter().map(|c| c.len()).max().unwrap_or(0);

    //test stuff
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
    let call_id = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    eprintln!(
        "[analyze_pcm_sample] call #{} — max_len={} samples ({:.2}s @ {}Hz), channels={}",
        call_id,
        max_len,
        max_len as f32 / original_sample_rate as f32,
        original_sample_rate,
        pcm_channels.len()
    );
    //test stuff

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
    //let magnitude_threshold_db = -60.0;
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
                let raw_freq = idx as f32 * delta_f;
                let freq_khz = (raw_freq.max(1.0)) / 1000.0;
                let dynamic_threshold = -60.0 - 20.0 * freq_khz.log10();

                // relax below 1kHz by up to +12 dB
                let dynamic_threshold = if raw_freq < 1000.0 {
                    dynamic_threshold + 12.0 * (1.0 - (raw_freq / 1000.0))
                } else {
                    dynamic_threshold
                };

                // clamp extremes
                let dynamic_threshold = dynamic_threshold.clamp(-110.0, -30.0);
                if mag_db < dynamic_threshold {
                    break;
                }

                let semitone_dist = raw_freq * (semitone_ratio - 1.0);

                // make bin floor smaller at low freq, larger at high freq
                let bin_floor_multiplier = if raw_freq < 200.0 {
                    0.6 // allow closer bins at very low freq
                } else if raw_freq < 800.0 {
                    0.9
                } else {
                    1.5
                };
                let bin_floor = delta_f * bin_floor_multiplier;

                let min_dist = semitone_dist.max(bin_floor);

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
            // remain Bin::default() (magnitude == 0.0), which spectral_process_voice
            // treats as "empty" and skips.
            let frame_start = (channel_idx * channel_stride) + (frame_idx * top_n);
            for (slot, peak) in selected_peaks.iter().enumerate() {
                flat_peaks_accumulator[frame_start + slot] = *peak;
            }
        }
    }

    let result = AnalyzedSample {
        flat_peaks: Arc::from(flat_peaks_accumulator),
        total_frames,
        peaks_per_frame: top_n,
        channels_count,
        original_sample_rate,
        fft_size,
        fft_step,
    };
    //test stuff
    let bytes = result.flat_peaks.len() * std::mem::size_of::<Bin>();
    eprintln!(
        "[analyze_pcm_sample] call #{} done — total_frames={}, top_n={}, bytes={:.2}MB",
        call_id,
        total_frames,
        top_n,
        bytes as f32 / 1_000_000.0
    );
    //test stuff
    result
}*/


//last hope please
pub fn analyze_pcm_sample(
    pcm_channels: Arc<[Arc<[f32]>]>,
    original_sample_rate: u32,
    config: &SpectralConfig,
    fft_plans: &SpectralPlans,
) -> AnalyzedSample {
    let fft_size = config.fft_size;
    let fft_step = config.fft_step;
    let top_n = config.max_peaks_per_frame;
    let window_coeffs = fft_plans.window();

    let channels_count = pcm_channels.len();
    let bin_count = (fft_size / 2) + 1;
    let max_len = pcm_channels.iter().map(|c| c.len()).max().unwrap_or(0);

    //test stuff
    use std::sync::atomic::{AtomicUsize, Ordering};
    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
    let call_id = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    eprintln!(
        "[analyze_pcm_sample] call #{} — max_len={} samples ({:.2}s @ {}Hz), channels={}",
        call_id,
        max_len,
        max_len as f32 / original_sample_rate as f32,
        original_sample_rate,
        pcm_channels.len()
    );
    //test stuff

    let total_frames = if max_len <= fft_size {
        1
    } else {
        ((max_len - fft_size) / fft_step) + 1
    };

    let mut fft_buffer = vec![Complex::new(0.0f32, 0.0f32); fft_size];
    let scratch_len = fft_plans.forward_plan.get_inplace_scratch_len();
    let mut scratch_buffer = vec![Complex::new(0.0f32, 0.0f32); scratch_len];

    // Pre-size output
    let mut flat_peaks_accumulator: Vec<Bin> =
        vec![Bin::default(); channels_count * total_frames * top_n];

    // Reusable scratch buffers
    let mut magnitudes: Vec<f32> = vec![0.0; bin_count];
    let mut log_magnitudes: Vec<f32> = vec![0.0; bin_count];
    let mut phases: Vec<f32> = vec![0.0; bin_count];
    let mut sorted_bins: Vec<usize> = Vec::with_capacity(bin_count);
    let mut selected_peaks: Vec<Bin> = Vec::with_capacity(top_n);

    let delta_f = original_sample_rate as f32 / fft_size as f32;
    const MAG_FLOOR: f32 = 1e-12;

    // --- New: per-channel continuity tracker ---
    // For each channel we keep the last frame's selected frequencies (up to top_n).
    // This is a soft bias: if a candidate is near a previously tracked freq we
    // allow it even if it would otherwise be rejected by the threshold.
    let mut prev_selected_freqs: Vec<Vec<f32>> = vec![Vec::new(); channels_count];

    for (channel_idx, pcm_channel) in pcm_channels.iter().enumerate() {
        let channel_stride = total_frames * top_n;

        // Ensure prev_selected_freqs slot has capacity
        prev_selected_freqs[channel_idx].reserve(top_n);

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
            for bin_idx in 0..bin_count {
                let c = fft_buffer[bin_idx];
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                magnitudes[bin_idx] = mag;
                log_magnitudes[bin_idx] = mag.max(MAG_FLOOR).log10();
                phases[bin_idx] = c.im.atan2(c.re);
            }

            // Sort bin indices by magnitude descending
            sorted_bins.clear();
            sorted_bins.extend(0..bin_count);
            sorted_bins.sort_unstable_by(|&a, &b| {
                magnitudes[b].partial_cmp(&magnitudes[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Clear selected_peaks for this frame
            selected_peaks.clear();

            // --- Selection parameters (tweakable) ---
            // Base threshold constants (dB)
            let base_threshold_db: f32 = -60.0;
            // Low-frequency boost (makes threshold more permissive below low_freq_cut)
            let low_freq_cut: f32 = 200.0;
            let low_freq_boost_db: f32 = 15.0; // up to +12 dB at DC, tapering to 0 at low_freq_cut
            // Bin floor multipliers (vary with frequency)
            let low_bin_floor_mult: f32 = 0.5;
            let mid_bin_floor_mult: f32 = 0.9;
            let high_bin_floor_mult: f32 = 1.5;

            // Continuity tolerances
            let continuity_tolerance_hz: f32 = 5.0; // allow ~5 Hz tolerance to re-accept a previously tracked peak
            let continuity_strength_db: f32 = 6.0; // how much threshold is relaxed when continuity applies

            // Helper to compute dynamic threshold for a raw frequency
            let compute_threshold_db = |raw_freq: f32| -> f32 {
                // frequency-dependent base: more permissive at low freq
                let freq_khz = (raw_freq.max(1.0)) / 1000.0;
                let mut dynamic = base_threshold_db - 20.0 * freq_khz.log10();
                // apply low-frequency boost taper
                if raw_freq < low_freq_cut {
                    let t = (1.0 - (raw_freq / low_freq_cut)).clamp(0.0, 1.0);
                    dynamic += low_freq_boost_db * t;
                }
                dynamic.clamp(-110.0, -30.0)
            };

            // Helper to compute bin floor multiplier based on frequency
            let compute_bin_floor = |raw_freq: f32| -> f32 {
                if raw_freq < 200.0 {
                    low_bin_floor_mult
                } else if raw_freq < 800.0 {
                    mid_bin_floor_mult
                } else {
                    high_bin_floor_mult
                }
            };

            // Greedy selection with spacing and continuity bias
            for &idx in sorted_bins.iter() {
                if selected_peaks.len() >= top_n {
                    break;
                }

                let mag = magnitudes[idx];
                let mag_db = 20.0 * mag.max(MAG_FLOOR).log10();
                let raw_freq = idx as f32 * delta_f;

                // dynamic threshold for this frequency
                let mut dynamic_threshold = compute_threshold_db(raw_freq);

                // Continuity check: if this candidate is near any previously selected frequency,
                // relax the threshold so we keep the track.
                let mut continuity_hit = false;
                for &prev_f in prev_selected_freqs[channel_idx].iter() {
                    if (prev_f - raw_freq).abs() <= continuity_tolerance_hz {
                        continuity_hit = true;
                        break;
                    }
                }
                if continuity_hit {
                    dynamic_threshold -= continuity_strength_db; // more permissive
                }

                // If magnitude is below threshold and continuity didn't rescue it, stop scanning
                if mag_db < dynamic_threshold {
                    // Because sorted_bins is descending, once we hit below threshold we can stop.
                    // But if continuity_hit is true we already relaxed threshold and passed this check.
                    if !continuity_hit {
                        break;
                    }
                }

                // Spacing: compute semitone distance and bin floor distance, take max
                let semitone_ratio = 2f32.powf(1.0 / 12.0);
                let semitone_dist = raw_freq * (semitone_ratio - 1.0);
                let bin_floor = delta_f * compute_bin_floor(raw_freq);
                let min_dist = semitone_dist.max(bin_floor);

                // If too close to any already-selected peak, skip
                if selected_peaks.iter().any(|p| (p.frequency - raw_freq).abs() < min_dist) {
                    continue;
                }

                // Parabolic interpolation on log-magnitude (same as before)
                let (refined_freq, refined_mag, refined_phase) =
                    if idx > 0 && idx < bin_count - 1 {
                        let alpha = log_magnitudes[idx - 1];
                        let beta = log_magnitudes[idx];
                        let gamma = log_magnitudes[idx + 1];

                        let denom = alpha - 2.0 * beta + gamma;
                        let p = if denom.abs() > f32::EPSILON {
                            (0.5 * (alpha - gamma) / denom).clamp(-0.5, 0.5)
                        } else {
                            0.0
                        };

                        let interp_freq = (idx as f32 + p) * delta_f;
                        let interp_log_mag = beta - 0.25 * (alpha - gamma) * p;
                        let interp_mag = 10f32.powf(interp_log_mag);

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

            // If we selected fewer than top_n peaks, attempt a small continuity rescue pass:
            // re-scan the top few sorted bins (even if below threshold) and re-insert any
            // that are near prev_selected_freqs but were rejected due to spacing/threshold.
            if selected_peaks.len() < top_n && !prev_selected_freqs[channel_idx].is_empty() {
                for &prev_f in prev_selected_freqs[channel_idx].iter() {
                    if selected_peaks.len() >= top_n {
                        break;
                    }
                    // find best candidate near prev_f within a small window
                    let search_radius_bins = ((5.0 / delta_f).ceil() as usize).max(1); // ~5 Hz window
                    let center_bin = (prev_f / delta_f).round() as isize;
                    let lo = (center_bin - search_radius_bins as isize).max(0) as usize;
                    let hi = (center_bin + search_radius_bins as isize).min(bin_count as isize - 1) as usize;

                    // find the strongest bin in that small window
                    let mut best_bin: Option<usize> = None;
                    let mut best_mag = 0.0f32;
                    for b in lo..=hi {
                        if magnitudes[b] > best_mag {
                            best_mag = magnitudes[b];
                            best_bin = Some(b);
                        }
                    }
                    if let Some(bidx) = best_bin {
                        // compute refined values and insert if not too close to existing picks
                        let raw_freq = bidx as f32 * delta_f;
                        let semitone_ratio = 2f32.powf(1.0 / 12.0);
                        let semitone_dist = raw_freq * (semitone_ratio - 1.0);
                        let bin_floor = delta_f * compute_bin_floor(raw_freq);
                        let min_dist = semitone_dist.max(bin_floor);

                        if !selected_peaks.iter().any(|p| (p.frequency - raw_freq).abs() < min_dist) {
                            // parabolic refine
                            let (refined_freq, refined_mag, refined_phase) =
                                if bidx > 0 && bidx < bin_count - 1 {
                                    let alpha = log_magnitudes[bidx - 1];
                                    let beta = log_magnitudes[bidx];
                                    let gamma = log_magnitudes[bidx + 1];
                                    let denom = alpha - 2.0 * beta + gamma;
                                    let p = if denom.abs() > f32::EPSILON {
                                        (0.5 * (alpha - gamma) / denom).clamp(-0.5, 0.5)
                                    } else {
                                        0.0
                                    };
                                    let interp_freq = (bidx as f32 + p) * delta_f;
                                    let interp_log_mag = beta - 0.25 * (alpha - gamma) * p;
                                    let interp_mag = 10f32.powf(interp_log_mag);
                                    let interp_phase = if p >= 0.0 {
                                        phases[bidx] * (1.0 - p) + phases[bidx + 1] * p
                                    } else {
                                        phases[bidx] * (1.0 + p) + phases[bidx - 1] * (-p)
                                    };
                                    (interp_freq, interp_mag, interp_phase)
                                } else {
                                    (raw_freq, magnitudes[bidx], phases[bidx])
                                };
                            selected_peaks.push(Bin::new(refined_freq, refined_mag, refined_phase));
                        }
                    }
                }
            }

            // Write selected_peaks into the fixed-stride output
            let frame_start = (channel_idx * channel_stride) + (frame_idx * top_n);
            for (slot, peak) in selected_peaks.iter().enumerate() {
                flat_peaks_accumulator[frame_start + slot] = *peak;
            }
            // Remaining slots are already defaulted to zero magnitude

            // Update prev_selected_freqs for continuity in the next frame
            prev_selected_freqs[channel_idx].clear();
            for p in selected_peaks.iter() {
                prev_selected_freqs[channel_idx].push(p.frequency);
            }
            // Keep only up to top_n (should already be <= top_n)
            if prev_selected_freqs[channel_idx].len() > top_n {
                prev_selected_freqs[channel_idx].truncate(top_n);
            }
        }
    }

    let result = AnalyzedSample {
        flat_peaks: Arc::from(flat_peaks_accumulator),
        total_frames,
        peaks_per_frame: top_n,
        channels_count,
        original_sample_rate,
        fft_size,
        fft_step,
    };
    //test stuff
    let bytes = result.flat_peaks.len() * std::mem::size_of::<Bin>();
    eprintln!(
        "[analyze_pcm_sample] call #{} done — total_frames={}, top_n={}, bytes={:.2}MB",
        call_id,
        total_frames,
        top_n,
        bytes as f32 / 1_000_000.0
    );
    //test stuff
    result
}
