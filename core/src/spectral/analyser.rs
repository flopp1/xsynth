use std::sync::Arc;
use rustfft::num_complex::Complex;
use super::{HarmonicTrack, SpectralConfig, SpectralPlans};

/// Represents an analyzed frequency snapshot for an audio sample, containing fundemental harmonics and magnitude curve
#[derive(Clone, Debug)]
pub struct AnalyzedSample {
    pub harmonics: Arc<[HarmonicTrack]>,
    pub total_frames: usize,
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

fn cubic_maximize(y0: f32, y1: f32, y2: f32, y3: f32, max_val: &mut f32) -> f32 {
    let a = y0 / -6.0 + y1 / 2.0 - y2 / 2.0 + y3 / 6.0;
    let b = y0 - 5.0 * y1 / 2.0 + 2.0 * y2 - y3 / 2.0;
    let c = -11.0 * y0 / 6.0 + 3.0 * y1 - 3.0 * y2 / 2.0 + y3 / 3.0;
    let d = y0;

    let da = 3.0 * a;
    let db = 2.0 * b;
    let dc = c;

    let discriminant = db * db - 4.0 * da * dc;
    if discriminant < 0.0 {
        *max_val = -1.0;
        return -1.0;
    }
    let x1 = (-db + discriminant.sqrt()) / (2.0 * da);
    let x2 = (-db - discriminant.sqrt()) / (2.0 * da);

    let dda = 2.0 * da;
    let ddb = db;

    let chosen = if dda * x1 + ddb < 0.0 { x1 } else { x2 };
    *max_val = a * chosen * chosen * chosen + b * chosen * chosen + c * chosen + d;
    chosen
}

type FramePeaks = Vec<(f32, f32, f32)>;

/// Runs cubic-interpolation peak detection on a single frame's power spectrum.
fn detect_frame_peaks(
    power: &[f32],
    re: &[f32],
    im: &[f32],
    bin_count: usize,
    half: usize,
    delta_f: f32,
    magnitude_threshold_power: f32,
) -> FramePeaks {
    let mut peaks = Vec::new();
 
    if bin_count <= 4 || half < 4 {
        return peaks;
    }
 
    let mut up = power[1] > power[0];
    for bin in 2..(half - 1) {
        let now_up = power[bin] > power[bin - 1];
        if !now_up && up {
            let leftbin = bin - 2;
            let mut value_at_max = 0.0f32;
            let offset = cubic_maximize(
                power[leftbin],
                power[leftbin + 1],
                power[leftbin + 2],
                power[leftbin + 3],
                &mut value_at_max,
            );
 
            if offset.is_finite() && value_at_max.is_finite() && offset >= -0.5 && offset <= 2.5 {
                if value_at_max < magnitude_threshold_power {
                    up = now_up;
                    continue;
                }
 
                let refined_bin = leftbin as f32 + offset;
                let freq = refined_bin * delta_f;
 
                let bin_floor = refined_bin.floor() as isize;
                let phase = if bin_floor >= 0 && (bin_floor as usize) + 1 < bin_count {
                    let b0 = bin_floor as usize;
                    let frac = refined_bin - b0 as f32;
                
                    // 1. Compute the explicit phase angles for both adjacent bins
                    let phase0 = im[b0].atan2(re[b0]);
                    let phase1 = im[b0 + 1].atan2(re[b0 + 1]);
                
                    // 2. Find the shortest angular distance (delta) between the two phases
                    let mut diff = phase1 - phase0;
                    while diff > std::f32::consts::PI {
                        diff -= 2.0 * std::f32::consts::PI;
                    }
                    while diff < -std::f32::consts::PI {
                        diff += 2.0 * std::f32::consts::PI;
                    }
                
                    // 3. Linearly interpolate along the perimeter of the circle
                    let mut interp_phase = phase0 + frac * diff;
                
                    // 4. Ensure the final interpolated angle is bound between [-PI, PI]
                    while interp_phase > std::f32::consts::PI {
                        interp_phase -= 2.0 * std::f32::consts::PI;
                    }
                    while interp_phase < -std::f32::consts::PI {
                        interp_phase += 2.0 * std::f32::consts::PI;
                    }
                
                    interp_phase
                } else {
                    let idx = refined_bin.round().clamp(0.0, (bin_count - 1) as f32) as usize;
                    im[idx].atan2(re[idx])
                };
                // value_at_max is power-domain; convert to linear magnitude.
                peaks.push((freq, phase, value_at_max.max(0.0).sqrt()));
            }
        }
        up = now_up;
    }
 
    peaks
}

/// A partial being tracked across frames.
struct Track {
    start_frame: usize,
    /// (frequency, magnitude, phase) per frame, from start_frame onward.
    frames: Vec<(f32, f32, f32)>,
    /// Consecutive frames with no match — track dies if this exceeds max_gap.
    silent_run: usize,
    dead: bool,
}
 
impl Track {
    fn new(start_frame: usize, freq: f32, mag: f32, phase: f32) -> Self {
        Self {
            start_frame,
            frames: vec![(freq, mag, phase)],
            silent_run: 0,
            dead: false,
        }
    }
 
    fn last(&self) -> (f32, f32, f32) {
        *self.frames.last().unwrap()
    }
 
    fn extend(&mut self, freq: f32, mag: f32, phase: f32) {
        if self.silent_run > 0 {
            // The track was in a silent gap and fading out, but has now re-emerged!
            // To prevent a pop/click step discontinuity, we retroactively overwrite 
            // the artificial gap frames with a smooth linear transition.
            let gap_len = self.silent_run;
            let total_steps = gap_len + 1;
            
            // The last known genuine frame index sits right before the gap started
            let start_idx = self.frames.len() - gap_len - 1;
            let (start_freq, start_mag, start_phase) = self.frames[start_idx];
            
            for step in 1..=gap_len {
                let t = step as f32 / total_steps as f32;
                let idx = start_idx + step;
                
                // Linearly interpolate frequency and magnitude across the gap
                self.frames[idx].0 = start_freq * (1.0 - t) + freq * t;
                self.frames[idx].1 = start_mag * (1.0 - t) + mag * t;
                
                // Linearly interpolate the phase across the gap using wrap-aware logic
                let mut diff = phase - start_phase;
                while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
                while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
                
                let mut interp_p = start_phase + t * diff;
                while interp_p > std::f32::consts::PI { interp_p -= 2.0 * std::f32::consts::PI; }
                while interp_p < -std::f32::consts::PI { interp_p += 2.0 * std::f32::consts::PI; }
                
                self.frames[idx].2 = interp_p;
            }
        }
    
        // Append the new true frame data normally and reset the counter
        self.frames.push((freq, mag, phase));
        self.silent_run = 0;
    }
 
    fn extend_silent(&mut self, max_gap: usize) {
        let (last_freq, last_mag, last_phase) = self.last();
        // Fade out over the silent run rather than instant cutoff.
        let fade = (1.0 - (self.silent_run + 1) as f32 / (max_gap + 1) as f32).max(0.0);
        self.frames.push((last_freq, last_mag * fade, last_phase));
        self.silent_run += 1;
        if self.silent_run > max_gap {
            self.dead = true;
        }
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
    let window_coeffs = fft_plans.window();
    let channels_count = pcm_channels.len();
    let bin_count = (fft_size / 2) + 1;
    let max_len = pcm_channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let total_frames = if max_len <= fft_size {
        1
    } else {
        ((max_len - fft_size) / fft_step) + 1
    };
    let half = fft_size / 2;
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
    let magnitude_threshold_power = 10f32.powf(-60.0 / 10.0); // -60dB in power domain

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

    // ===================== Per-frame peak detection + tracking =====================
    let semitone_ratio = 2f32.powf(1.0 / 12.0);
 
    // Frequency-matching tolerance for linking a frame's peak to an existing
    // track: max of a fixed bin-based floor and a relative (semitone-scale)
    // tolerance — wider tolerance at high frequencies where the same
    // relative drift corresponds to a larger absolute Hz change.
    let match_tolerance = |track_freq: f32| -> f32 {
        (track_freq * (semitone_ratio - 1.0) * 0.5).max(delta_f * 1.5)
    };
 
    // Consecutive silent frames tolerated before a track dies. ~5 frames at
    // fft_step=2048/48000Hz =~ 213ms — survives a brief dip below threshold
    // without prematurely killing a real partial.
    let max_silent_gap = 5;
 
    let mut tracks: Vec<Track> = Vec::new();
    let mut power: Vec<f32> = vec![0.0; bin_count];
    let mut re: Vec<f32> = vec![0.0; bin_count];
    let mut im: Vec<f32> = vec![0.0; bin_count];
    let mut target_peak = 0.0f32;
 
    for frame_idx in 0..total_frames {
        compute_frame_fft(frame_idx, &mut fft_buffer, &mut scratch_buffer);
        for b in 0..bin_count {
            let c = fft_buffer[b];
            power[b] = c.re * c.re + c.im * c.im;
            re[b] = c.re;
            im[b] = c.im;
            let mag = power[b].sqrt();
            if mag > target_peak {
                target_peak = mag;
            }
        }
 
        let mut frame_peaks = detect_frame_peaks(
            &power, &re, &im, bin_count, half, delta_f, magnitude_threshold_power,
        );
 
        // Strong peaks get matched/birthed before weak ones compete for slots.
        frame_peaks.sort_unstable_by(|a, b| f32::total_cmp(&b.2, &a.2));
 
        let mut matched = vec![false; frame_peaks.len()];
 
        // --- Match active tracks to this frame's peaks ---
        for track in tracks.iter_mut() {
            if track.dead {
                continue;
            }
 
            let (last_freq, _, _) = track.last();
            let tol = match_tolerance(last_freq);
 
            let mut best_idx: Option<usize> = None;
            let mut best_dist = f32::INFINITY;
            for (i, &(freq, _, _)) in frame_peaks.iter().enumerate() {
                if matched[i] {
                    continue;
                }
                let dist = (freq - last_freq).abs();
                if dist < tol && dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(i);
                }
            }
 
            match best_idx {
                Some(i) => {
                    let (freq, phase, mag) = frame_peaks[i];
                    track.extend(freq, mag, phase);
                    matched[i] = true;
                }
                None => {
                    track.extend_silent(max_silent_gap);
                }
            }
        }
 
        // --- Birth new tracks for unmatched candidates, up to harmonic_count active ---
        let active_count = tracks.iter().filter(|t| !t.dead).count();
        let mut free_slots = harmonic_count.saturating_sub(active_count);
 
        if free_slots > 0 {
            for (i, &(freq, phase, mag)) in frame_peaks.iter().enumerate() {
                if matched[i] {
                    continue;
                }
                if free_slots == 0 {
                    break;
                }
                tracks.push(Track::new(frame_idx, freq, mag, phase));
                free_slots -= 1;
            }
        }
    }
 
    // Select top harmonic_count tracks by total energy
    tracks.sort_unstable_by(|a, b| {
        let ea: f32 = a.frames.iter().map(|&(_, m, _)| m).sum();
        let eb: f32 = b.frames.iter().map(|&(_, m, _)| m).sum();
        f32::total_cmp(&eb, &ea)
    });

    //tracks.sort_unstable_by(|a, b| {
    //    let ea = a.frames.iter().map(|&(_, m, _)| m).fold(0.0f32, f32::max);
    //    let eb = b.frames.iter().map(|&(_, m, _)| m).fold(0.0f32, f32::max);
    //    f32::total_cmp(&eb, &ea)
    //});

    tracks.truncate(harmonic_count);

    let per_frame_sums: Vec<f32> = (0..total_frames)
    .map(|f| tracks.iter().map(|t| {
        if f >= t.start_frame {
            let idx = f - t.start_frame;
            if idx < t.frames.len() { t.frames[idx].1 } else { 0.0 }
        } else {
            0.0
        }
    }).sum::<f32>())
    .collect();

    // Use a robust statistic (RMS) to avoid tiny single-frame denominators
    let _rms = if per_frame_sums.is_empty() {
        0.0
    } else {
        let s: f32 = per_frame_sums.iter().map(|v| v * v).sum();
        (s / per_frame_sums.len() as f32).sqrt()
    };

    // target_peak was computed earlier during analysis; if not, compute a conservative target
    let target_peak = target_peak.max(1e-9_f32); // ensure nonzero

    // Convert dB threshold or clamp scale to avoid runaway amplification
    let max_scale = 10.0_f32;      // tune: 4..16 typical
    let min_scale = 0.1_f32;       // avoid extreme attenuation
    let floor = 1e-6_f32;          // floor for tiny RMS
//
    //let denom = if rms > floor { rms } else { floor };
    //let scale = (target_peak / denom).clamp(min_scale, max_scale);

    // Optionally, if you prefer peak-based scaling, use:
    let reconstructed_peak = per_frame_sums.iter().cloned().fold(0.0_f32, f32::max);
    let scale = if reconstructed_peak > floor { (target_peak / reconstructed_peak).clamp(min_scale, max_scale) } else { 1.0 };

    // Apply scale to each track's stored magnitudes (in-place)
    for track in tracks.iter_mut() {
        for frame in track.frames.iter_mut() {
            frame.1 *= scale; // frame is (freq, mag, phase) — mag is amplitude
        }
    }

    // Now apply birth fade (after scaling) while `track` is still in scope.
    // Use a slightly longer/smoother fade to avoid audible artifacts.
    const BIRTH_FADE_FRAMES: usize = 16; // increase from 2 to 4 for smoother ramp
    for track in tracks.iter_mut() {
        let birth_len = BIRTH_FADE_FRAMES.min(track.frames.len());
        for i in 0..birth_len {
            // fade factor: smooth cosine ramp from 0 -> 1
            let t = (i + 1) as f32 / (BIRTH_FADE_FRAMES + 1) as f32;
            let fade = 0.5 - 0.5 * (std::f32::consts::PI * (1.0 - t)).cos(); // gentle curve
            track.frames[i].1 *= fade;
        }
    }

    // Finalize
    let mut harmonics: Vec<HarmonicTrack> = tracks.into_iter()
        .map(|track| {
            let mut freq_curve = vec![0.0f32; total_frames];
            let mut mag_curve = vec![0.0f32; total_frames];
            let mut phase_curve = vec![0.0f32; total_frames];
 
            for (i, &(f, m, p)) in track.frames.iter().enumerate() {
                let frame = track.start_frame + i;
                if frame < total_frames {
                    freq_curve[frame] = f;
                    mag_curve[frame] = m;
                    phase_curve[frame] = p;
                }
            }
            HarmonicTrack {
                frequency_curve: Arc::from(freq_curve),
                magnitude_curve: Arc::from(mag_curve),
                phase_curve: Arc::from(phase_curve),
            }
        })
        .collect();

    // Pad with inert tracks if fewer than harmonic_count were found.
    while harmonics.len() < harmonic_count {
        harmonics.push(HarmonicTrack {
            frequency_curve: Arc::from(vec![0.0f32; total_frames]),
            magnitude_curve: Arc::from(vec![0.0f32; total_frames]),
            phase_curve: Arc::from(vec![0.0f32; total_frames]),
        });
    }

    let result = AnalyzedSample {
        harmonics: Arc::from(harmonics),
        total_frames,
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
