use std::sync::Arc;
use rustfft::num_complex::Complex;
use super::{HarmonicTrack, SpectralConfig, SpectralPlans};

/// Represents an analyzed frequency snapshot for an audio sample, containing fundemental harmonics and magnitude curve
#[derive(Clone, Debug)]
pub struct AnalyzedSample {
    pub harmonics: Arc<[HarmonicTrack]>,
    pub total_frames: usize,
    pub total_goertzel_samples: usize,
    pub harmonic_count: usize,
    pub channels_count: usize,
    pub original_sample_rate: u32,
    pub fft_size: usize,
    pub fft_step: usize,
}

impl AnalyzedSample {
    pub fn duration_seconds(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        let total_samples = (self.total_frames - 1) * self.fft_step + self.fft_size;
        total_samples as f32 / self.original_sample_rate as f32
    }
}


/// Consumes raw resampled PCM arrays and produces a fixed set of harmonic tracks.
pub fn analyze_pcm_sample(
    pcm_channels: Arc<[Arc<[f32]>]>,
    original_sample_rate: u32,
    config: &SpectralConfig,
    fft_plans: &SpectralPlans,
) -> AnalyzedSample {
    let fft_size = config.fft_size;
    let fft_step = config.fft_step;
    let harmonic_count = config.max_peaks_per_frame;
    let mag_res = config.magnitude_res;
    let window_coeffs = fft_plans.window();
    let channels_count = pcm_channels.len();
    let bin_count = (fft_size / 2) + 1;
    let max_len = pcm_channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let total_frames = if max_len <= fft_size {
        1
    } else {
        ((max_len - fft_size) / fft_step) + 1
    };
    let goertzel_window_size = mag_res * 4; //windowing over 2x the magnitude time-resolution since magnitude shouldn't wonk about in the sub-ms range, if it does then something's probably already wrong
    let total_goertzel_samples = if max_len < fft_size {
        0
    } else {
        ((max_len - goertzel_window_size) / mag_res) + 1
    };

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

    let mut fft_buffer = vec![Complex::new(0.0f32, 0.0f32); fft_size];
    let scratch_len = fft_plans.forward_plan.get_inplace_scratch_len();
    let mut scratch_buffer = vec![Complex::new(0.0f32, 0.0f32); scratch_len];

    let delta_f = original_sample_rate as f32 / fft_size as f32;
    let magnitude_threshold_db = -60.0;
    let semitone_ratio = 2f32.powf(1.0 / 12.0);
    const MAG_FLOOR: f32 = 1e-12;

    // --- Helper: run one frame's FFT on the mono-summed signal, writing into fft_buffer ---
    let compute_frame_fft = |frame_idx: usize, fft_buffer: &mut [Complex<f32>], scratch_buffer: &mut [Complex<f32>]| {
        let start_sample = frame_idx * fft_step;
        fft_buffer.fill(Complex::new(0.0, 0.0));

        if channels_count == 1 {
            let pcm = &pcm_channels[0];
            for i in 0..fft_size {
                let pos = start_sample + i;
                if pos < pcm.len() {
                    fft_buffer[i].re = pcm[pos] * window_coeffs[i];
                }
            }
        } else {
            let inv_channels = 1.0 / channels_count as f32;
            for i in 0..fft_size {
                let pos = start_sample + i;
                let mut sum = 0.0f32;
                for ch in pcm_channels.iter() {
                    if pos < ch.len() {
                        sum += ch[pos];
                    }
                }
                fft_buffer[i].re = sum * inv_channels * window_coeffs[i];
            }
        }

        fft_plans.forward_plan.process_with_scratch(fft_buffer, scratch_buffer);
    };

    // ===================== PASS 1: harmonic detection on a representative frame =====================
    //
    // Pick a frame in early-to-mid sustain — skips the attack transient (where the
    // spectrum is broadband/noisy and not representative of the note's harmonic
    // series) and lands where the signal is strong and harmonically clean.
    let representative_frame = (total_frames / 4).min(total_frames.saturating_sub(1));

    compute_frame_fft(representative_frame, &mut fft_buffer, &mut scratch_buffer);

    let mut magnitudes: Vec<f32> = vec![0.0; bin_count];
    let mut log_magnitudes: Vec<f32> = vec![0.0; bin_count];
    let mut phases: Vec<f32> = vec![0.0; bin_count];

    for bin_idx in 0..bin_count {
        let c = fft_buffer[bin_idx];
        let mag = (c.re * c.re + c.im * c.im).sqrt();
        magnitudes[bin_idx] = mag;
        log_magnitudes[bin_idx] = mag.max(MAG_FLOOR).log10();
        phases[bin_idx] = c.im.atan2(c.re);
    }

    let mut sorted_bins: Vec<usize> = (0..bin_count).collect();
    sorted_bins.sort_unstable_by(|&a, &b| {
        magnitudes[b].partial_cmp(&magnitudes[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Selected (frequency, phase) pairs — magnitude from this pass is discarded;
    // pass 2 derives the actual per-frame magnitude curve.
    let mut selected: Vec<(f32, f32)> = Vec::with_capacity(harmonic_count); // (frequency, phase)

    for &idx in sorted_bins.iter() {
        if selected.len() >= harmonic_count {
            break;
        }

        let mag = magnitudes[idx];
        let mag_db = 20.0 * mag.max(MAG_FLOOR).log10();
        if mag_db < magnitude_threshold_db {
            break; // sorted descending — everything after this is also below floor
        }

        let raw_freq = idx as f32 * delta_f;
        let min_dist = (raw_freq * (semitone_ratio - 1.0)).max(delta_f * 1.5);

        if selected.iter().any(|&(f, _)| (f - raw_freq).abs() < min_dist) {
            continue;
        }

        // Parabolic interpolation on log-magnitude for sub-bin frequency accuracy.
        let (refined_freq, refined_phase) = if idx > 0 && idx < bin_count - 1 {
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
            let interp_phase = if p >= 0.0 {
                phases[idx] * (1.0 - p) + phases[idx + 1] * p
            } else {
                phases[idx] * (1.0 + p) + phases[idx - 1] * (-p)
            };

            (interp_freq, interp_phase)
        } else {
            (raw_freq, phases[idx])
        };

        selected.push((refined_freq, refined_phase));
    }

    // Pad to exactly harmonic_count with inert entries (frequency=0, magnitude
    // curve all zero) — spectral_process_voice's `magnitude <= 0.0` check skips
    // these, same as the old zero-padded Bin scheme.
    while selected.len() < harmonic_count {
        selected.push((0.0, 0.0));
    }

    // Per-sample magnitude curve extraction
    // For each harmonic's fixed frequency, compute its complex DFT value directly with Goertzel on each sample in the audio sample. This avoids the full per-frame FFT.
    let mut magnitude_curves: Vec<Vec<f32>> = vec![vec![0.0; total_goertzel_samples]; harmonic_count];

    let inv_channels = 1.0 / channels_count as f32;

    // PRE-COMPUTE Goertzel coefficients for all harmonics ONCE
    let mut harmonic_coeffs = Vec::with_capacity(harmonic_count);
    for h in 0..harmonic_count {
        let (freq, _) = selected[h];
        if freq > 0.0 {
            let omega = 2.0 * std::f32::consts::PI * freq / original_sample_rate as f32;
            harmonic_coeffs.push(Some(2.0 * omega.cos()));
        } else {
            harmonic_coeffs.push(None);
        }
    }

    let mut goertzel_windows = Vec::with_capacity(goertzel_window_size);
    for n in 0..goertzel_window_size {
        goertzel_windows.push(0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / (goertzel_window_size - 1) as f32).cos()); //Hann window
    }

    for sample_idx in 0..total_goertzel_samples {
        let start_sample = sample_idx * mag_res;

        for h in 0..harmonic_count {
            let coeff = match harmonic_coeffs[h] {
                Some(c) => c,
                None => continue,
            };

            let mut q1 = 0.0f32;
            let mut q2 = 0.0f32;

            for n in 0..goertzel_window_size {
                let pos = start_sample + n;
                let mut sample = 0.0f32;

                if channels_count == 1 {
                    if let Some(&v) = pcm_channels[0].get(pos) {
                        sample = v;
                    }
                } else {
                    for ch in pcm_channels.iter() {
                        if let Some(&v) = ch.get(pos) {
                            sample += v;
                        }
                    }
                    sample *= inv_channels;
                }

                sample *= goertzel_windows[n];

                let q0 = coeff * q1 - q2 + sample;
                q2 = q1;
                q1 = q0;
            }

            let mag_squared = q1 * q1 + q2 * q2 - q1 * q2 * coeff;
        
            // Guard against tiny negative floating-point anomalies before sqrt, 2.0 since IFFT produces unscaled peak mag of Complex*2/fft_size (so we also eliminate the need for normalisation)
            let mag = if mag_squared > 0.0 { mag_squared.sqrt() * 2.0 / goertzel_window_size as f32 } else { 0.0 };

            magnitude_curves[h][sample_idx] = mag;
        }
    }

    let harmonics: Vec<HarmonicTrack> = (0..harmonic_count)
        .map(|h| {
            let (frequency, phase_at_origin) = selected[h];
            HarmonicTrack {
                frequency,
                phase_at_origin,
                magnitude_curve: Arc::from(std::mem::take(&mut magnitude_curves[h])),
            }
        })
        .collect();

    let result = AnalyzedSample {
        harmonics: Arc::from(harmonics),
        total_frames,
        total_goertzel_samples,
        harmonic_count,
        channels_count,
        original_sample_rate,
        fft_size,
        fft_step,
    };
    //test stuff
    let curve_bytes: usize = result.harmonics.iter()
    .map(|h| h.magnitude_curve.len() * std::mem::size_of::<f32>())
    .sum();
    let struct_bytes = result.harmonics.len() * std::mem::size_of::<HarmonicTrack>();
    let bytes = curve_bytes + struct_bytes;
    eprintln!(
        "[analyze_pcm_sample] call #{} done — total_frames={}, top_n={}, bytes={:.2}MB",
        call_id,
        total_frames,
        harmonic_count,
        bytes as f32 / 1_000_000.0
    );
    result
    //test stuff
}

//last hope please
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

    // Pre-size output
    let mut harmonics_accumulator: Vec<Bin> =
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
            let low_freq_boost_db: f32 = 12.0; // up to +12 dB at DC, tapering to 0 at low_freq_cut
            // Bin floor multipliers (vary with frequency)
            let low_bin_floor_mult: f32 = 0.6;
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
                harmonics_accumulator[frame_start + slot] = *peak;
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
        harmonics: Arc::from(harmonics_accumulator),
        total_frames,
        peaks_per_frame: top_n,
        channels_count,
        original_sample_rate,
        fft_size,
        fft_step,
    };
    //test stuff
    let bytes = result.harmonics.len() * std::mem::size_of::<Bin>();
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
