use std::sync::Arc;
use rustfft::num_complex::Complex;
use super::{ComplexBin, SpectralConfig, SpectralPlans};

/// Represents an analyzed frequency snapshot for a voice asset.
/// Memory Optimized: Multi-channel input is downmixed to mono for the current
/// spectral pipeline, and all frames are packed into a single contiguous block to
/// eliminate vector tracking overhead and prevent heap fragmentation.
#[derive(Clone, Debug)]
pub struct AnalyzedSample {
    /// Contiguous block containing all data packed as: [Channel 0 Frames..., Channel 1 Frames...]
    pub flat_bins: Arc<[ComplexBin]>,
    pub total_frames: usize,
    pub bins_per_frame: usize,
    pub channels_count: usize,
    pub original_sample_rate: u32,
    pub fft_size: usize,
    pub fft_step: usize,
}

impl AnalyzedSample {
    /// Safe utility to pull out a single frame slice reference for a specific channel.
    /// Calculates the correct 1D memory offset mathematically instead of traversing nested pointers.

    #[inline(always)]
    pub fn get_frame_slice(&self, channel: usize, frame_idx: usize) -> Option<&[ComplexBin]> {
        if channel >= self.channels_count || frame_idx >= self.total_frames {
            return None;
        }
        
        let channel_stride = self.total_frames * self.bins_per_frame;
        let frame_stride = frame_idx * self.bins_per_frame;
        
        let start = (channel * channel_stride) + frame_stride;
        let end = start + self.bins_per_frame;
        
        Some(&self.flat_bins[start..end])
    }

    pub fn duration_seconds(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        let total_samples = (self.total_frames - 1) * self.fft_step + self.fft_size;
        total_samples as f32 / self.original_sample_rate as f32
    }
}

/// Consumes raw resampled PCM arrays and maps them into a flat frequency timeline.
pub fn analyze_pcm_sample(
    pcm_channels: Arc<[Arc<[f32]>]>,
    original_sample_rate: u32,
    config: &SpectralConfig,
    fft_plans: &SpectralPlans,
) -> AnalyzedSample {
    let fft_size = config.fft_size;
    let fft_step = config.fft_step;
    let window_coeffs = fft_plans.window();

    let mono_channel: Arc<[f32]> = if pcm_channels.len() <= 1 {
        pcm_channels
            .get(0)
            .cloned()
            .unwrap_or_else(|| Arc::from([]))
    } else {
        let channel_count = pcm_channels.len() as f32;
        let max_len = pcm_channels.iter().map(|channel| channel.len()).max().unwrap_or(0);
        let mut mixed = vec![0.0f32; max_len];

        for pcm_channel in pcm_channels.iter() {
            for (i, &sample) in pcm_channel.iter().enumerate() {
                mixed[i] += sample;
            }
        }

        let inv_channels = 1.0 / channel_count;
        for sample in mixed.iter_mut() {
            *sample *= inv_channels;
        }

        Arc::from(mixed)
    };

    let channels_count = 1;
    let bins_per_frame = (fft_size / 2) + 1;
    let mono_len = mono_channel.len();
    let total_frames = if mono_len <= fft_size {
        1
    } else {
        ((mono_len - fft_size) / fft_step) + 1
    };

    // 2. Allocate the exact total required memory upfront in a single step
    let total_required_bins = channels_count * total_frames * bins_per_frame;
    let mut flat_bins_accumulator = vec![ComplexBin::default(); total_required_bins];

    // 3. Pre-allocate temporary scratch vectors once
    let mut fft_buffer = vec![Complex::new(0.0f32, 0.0f32); fft_size];
    let scratch_len = fft_plans.forward_plan.get_inplace_scratch_len();
    let mut scratch_buffer = vec![Complex::new(0.0f32, 0.0f32); scratch_len];

    // 4. Populate the flat memory block directly
    for frame_idx in 0..total_frames {
        let start_sample = frame_idx * fft_step;

        fft_buffer.fill(Complex::new(0.0, 0.0));

        for i in 0..fft_size {
            let sample_pos = start_sample + i;
            if sample_pos < mono_len {
                fft_buffer[i].re = mono_channel[sample_pos] * window_coeffs[i];
            }
        }

        fft_plans.forward_plan.process_with_scratch(&mut fft_buffer, &mut scratch_buffer);

        let target_start = frame_idx * bins_per_frame;
        for bin_idx in 0..bins_per_frame {
            let c = fft_buffer[bin_idx];
            flat_bins_accumulator[target_start + bin_idx] = ComplexBin::new(c.re, c.im);
        }
    }

    AnalyzedSample {
        flat_bins: Arc::from(flat_bins_accumulator),
        total_frames,
        bins_per_frame,
        channels_count,
        original_sample_rate,
        fft_size,
        fft_step,
    }
}