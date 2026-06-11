use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

pub struct SpectralPlans {
    pub(crate) forward_plan: Arc<dyn Fft<f32>>,
    inverse_plan: Arc<dyn Fft<f32>>,
    fft_size: usize,
    fft_step: usize,
    window: Vec<f32>,
}

impl SpectralPlans {
    /// Creates and optimizes a new set of dynamic FFT/IFFT plans.
    pub fn new(fft_size: usize, fft_step: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let forward_plan = planner.plan_fft_forward(fft_size);
        let inverse_plan = planner.plan_fft_inverse(fft_size);

        // Pre-calculate Hanning window for the requested size
        let mut window = Vec::with_capacity(fft_size);
        for i in 0..fft_size {
            let val = 0.5
                * (1.0
                    - f32::cos((2.0 * std::f32::consts::PI * i as f32) / (fft_size as f32 - 1.0)));
            window.push(val);
        }

        Self {
            forward_plan,
            inverse_plan,
            fft_size,
            fft_step,
            window,
        }
    }

    #[inline(always)]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    #[inline(always)]
    pub fn fft_step(&self) -> usize {
        self.fft_step
    }

    /// Exposes the window coefficient array safely
    #[inline(always)]
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Executes high-performance forward FFT in-place on complex data
    #[inline(always)]
    pub fn execute_forward(&self, buffer: &mut [Complex<f32>]) {
        self.forward_plan.process(buffer);
    }

    /// Executes the inverse FFT in-place on complex data
    #[inline(always)]
    pub fn execute_inverse(&self, buffer: &mut [Complex<f32>]) {
        self.inverse_plan.process(buffer);
    }

    /// Applies the pre-calculated Hanning window to a time-domain grain slice.
    #[inline(always)]
    pub fn apply_synthesis_window(&self, grain: &mut [f32]) {
        debug_assert_eq!(grain.len(), self.fft_size);
        for (sample, &win_coeff) in grain.iter_mut().zip(self.window.iter()) {
            *sample *= win_coeff;
        }
    }
}
