use simdeez::{scalar::Scalar, Simd};
use std::{marker::PhantomData, ops::Mul, sync::Arc};

use crate::{
    effects::BiQuadFilter,
    voice::{
        BufferSampler, SIMDSample, SIMDSampleGrabber, SIMDSampleMono, SIMDSampleStereo,
        SIMDStereoVoiceCutoff, SIMDVoiceGenerator,
    },
    AudioStreamParams,
};
use crate::{
    voice::VoiceControlData,
    voice::{
        BufferSamplers, EnvelopeParameters, SIMDConstant, SIMDConstantStereo,
        SIMDLinearSampleGrabber, SIMDNearestSampleGrabber, SIMDStereoVoice, SIMDStereoVoiceSampler,
        SIMDVoiceControl, SIMDVoiceEnvelope, SampleReader, SampleReaderLoop,
        SampleReaderLoopSustain, SampleReaderNoLoop, Voice, VoiceBase, VoiceCombineSIMD,
    },
};

use crate::soundfont::{Interpolator, LoopParams, SampleVoiceSpawnerParams, VoiceSpawner};
use xsynth_soundfonts::LoopMode;

use crate::spectral::{AnalyzedSample, SpectralVoice};

pub struct StereoSampledVoiceSpawner<S: 'static + Simd + Send + Sync> {
    speed_mult: f32,
    filter: Option<BiQuadFilter>,
    loop_params: LoopParams,
    amp: f32,
    pan: f32,
    volume_envelope_params: Arc<EnvelopeParameters>,
    samples: Arc<[Arc<[f32]>]>,
    interpolator: Interpolator,
    exclusive_class: Option<u8>,
    vel: u8,
    stream_params: AudioStreamParams,
    root_key: u8,
    spectral_sample: Option<Arc<AnalyzedSample>>,
    _s: PhantomData<S>,
}

impl<S: Simd + Send + Sync> StereoSampledVoiceSpawner<S> {
    pub fn new(
        params: &SampleVoiceSpawnerParams,
        vel: u8,
        stream_params: AudioStreamParams,
    ) -> Self {
        let amp = params.volume;

        let filter = params.cutoff.map(|cutoff| {
            BiQuadFilter::new(
                params.filter_type,
                cutoff,
                stream_params.sample_rate as f32,
                Some(params.resonance),
            )
        });

        Self {
            speed_mult: params.speed_mult,
            filter,
            loop_params: params.loop_params.clone(),
            amp,
            pan: params.pan,
            volume_envelope_params: params.envelope.clone(),
            samples: params.sample.clone(),
            interpolator: params.interpolator,
            exclusive_class: params.exclusive_class,
            vel,
            stream_params,
            root_key: params.root_key,
            spectral_sample: params.spectral_sample.clone(),
            _s: PhantomData,
        }
    }

    fn begin_voice(&self, control: &VoiceControlData) -> Box<dyn Voice> {
        if self.samples.is_empty() {
            return self.spawn_spectral_voice(control);
        }

        #[allow(clippy::redundant_closure)]
        self.make_sample_reader(control, |s| BufferSamplers::new_f32(s))
    }

    fn make_sample_reader<BS: 'static + BufferSampler>(
        &self,
        control: &VoiceControlData,
        make_bs: impl Fn(Arc<[f32]>) -> BS,
    ) -> Box<dyn Voice> {
        match self.loop_params.mode {
            LoopMode::LoopContinuous => self.make_sample_grabber(control, move |s| {
                SampleReaderLoop::new(make_bs(s), self.loop_params.clone())
            }),
            LoopMode::LoopSustain => self.make_sample_grabber(control, move |s| {
                SampleReaderLoopSustain::new(make_bs(s), self.loop_params.clone())
            }),
            LoopMode::NoLoop | LoopMode::OneShot => self.make_sample_grabber(control, move |s| {
                SampleReaderNoLoop::new(make_bs(s), self.loop_params.clone())
            }),
        }
    }

    fn make_sample_grabber<SR: 'static + SampleReader>(
        &self,
        control: &VoiceControlData,
        make_bs: impl Fn(Arc<[f32]>) -> SR,
    ) -> Box<dyn Voice> {
        match self.interpolator {
            Interpolator::Nearest => {
                self.generate_sampler(control, |s| SIMDNearestSampleGrabber::new(make_bs(s)))
            }
            Interpolator::Linear => {
                self.generate_sampler(control, |s| SIMDLinearSampleGrabber::new(make_bs(s)))
            }
        }
    }

    fn generate_sampler<SG: 'static + SIMDSampleGrabber<S>>(
        &self,
        control: &VoiceControlData,
        make_sampler: impl Fn(Arc<[f32]>) -> SG,
    ) -> Box<dyn Voice> {
        let left = make_sampler(self.samples[0].clone());
        let right = make_sampler(self.samples[1].clone());

        let pitch_fac = self.create_pitch_fac(control);
        let sampler = SIMDStereoVoiceSampler::new(left, right, pitch_fac);
        self.apply_voice_params(sampler, control)
    }

    fn apply_velocity<Gen, Sample>(&self, gen: Gen) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let amp = SIMDConstant::<S>::new(self.amp);
        VoiceCombineSIMD::mult(amp, gen)
    }

    fn apply_pan<Gen, Sample>(&self, gen: Gen) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleStereo<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let pan = self.pan * std::f32::consts::PI / 2.0;
        let leftg = (pan.cos() * 1.42).min(1.0);
        let rightg = (pan.sin() * 1.42).min(1.0);

        let gains = SIMDConstantStereo::<S>::new(leftg, rightg);
        VoiceCombineSIMD::mult(gains, gen)
    }

    // FIXED: Changed &mut self to &self to resolve compiler diagnostic borrow errors
    fn create_pitch_fac(
        &self,
        control: &VoiceControlData,
    ) -> impl SIMDVoiceGenerator<S, SIMDSampleMono<S>> {
        let pitch_fac = SIMDConstant::<S>::new(self.speed_mult);
        let pitch_multiplier = SIMDVoiceControl::new(control, |vc| vc.voice_pitch_multiplier);
        VoiceCombineSIMD::mult(pitch_fac, pitch_multiplier)
    }

    fn apply_envelope<Gen, Sample>(
        &self,
        gen: Gen,
        control: &VoiceControlData,
    ) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let modified_params = SIMDVoiceEnvelope::<S>::get_modified_envelope(
            *self.volume_envelope_params.clone(),
            control.envelope,
            self.stream_params.sample_rate as f32,
        );

        let allow_release = self.loop_params.mode != LoopMode::OneShot;

        let volume_envelope = SIMDVoiceEnvelope::new(
            *self.volume_envelope_params.clone(),
            modified_params,
            allow_release,
            self.stream_params.sample_rate as f32,
        );

        VoiceCombineSIMD::mult(volume_envelope, gen)
    }

    fn convert_to_voice<Gen>(&self, gen: Gen) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let flattened = SIMDStereoVoice::new(gen);
        let base = VoiceBase::new(self.vel, self.exclusive_class(), flattened);
        Box::new(base)
    }

    fn apply_voice_params<Gen>(&self, gen: Gen, control: &VoiceControlData) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let gen = self.apply_velocity(gen);
        let gen = self.apply_pan(gen);
        let gen = self.apply_envelope(gen, control);

        self.apply_cutoff_effect(gen)
    }

    fn apply_cutoff_effect(
        &self,
        gen: impl 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    ) -> Box<dyn Voice> {
        if let Some(filter) = &self.filter {
            let active_filter = &filter.clone();

            // FIXED: Removed raw pointer/borrow reference `&active_filter`
            // and passed the cloned configuration instance safely by value.
            let gen = SIMDStereoVoiceCutoff::new(gen, active_filter);
            self.convert_to_voice(gen)
        } else {
            self.convert_to_voice(gen)
        }
    }
}

impl<S: 'static + Sync + Send + Simd> VoiceSpawner for StereoSampledVoiceSpawner<S> {
    fn spawn_voice(&self, control: &VoiceControlData) -> Box<dyn Voice> {
        self.begin_voice(control)
    }

    fn spawn_spectral_voice(&self, control: &VoiceControlData) -> Box<dyn Voice> {
        if let Some(ref spectral_sample) = self.spectral_sample {
            // FIXED: Integrated dynamic standard calculation logic for allow_release
            // to match structural requirements across execution loops.
            let allow_release = self.loop_params.mode != LoopMode::OneShot;

            let simd_envelope = SIMDVoiceEnvelope::<Scalar>::new(
                *self.volume_envelope_params,
                *self.volume_envelope_params,
                allow_release,
                self.stream_params.sample_rate as f32,
            );

            let derived_trigger_note =
                (self.root_key as f32 + 12.0 * self.speed_mult.log2()).round() as u8;

            // 1. Instantiate the inner spectral calculation voice core
            let spectral_generator = SpectralVoice::<Scalar>::new(
                spectral_sample.clone(),
                simd_envelope,
                self.root_key,
                derived_trigger_note,
                self.stream_params.sample_rate as f32, // Pass the actual numeric sample rate here!
            );

            // 2. Wrap it in VoiceBase to satisfy the `Voice` trait object boundary constraints
            let boxed_voice: Box<dyn Voice> = Box::new(VoiceBase::new(
                self.vel,
                self.exclusive_class,
                spectral_generator,
            ));

            // Now you can push `boxed_voice` straight into your pipeline voice list!
            boxed_voice
        } else {
            self.spawn_voice(control)
        }
    }

    fn exclusive_class(&self) -> Option<u8> {
        self.exclusive_class
    }
}
