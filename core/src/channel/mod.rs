use std::sync::{atomic::AtomicU64, Arc};

use crate::{
    effects::MultiChannelBiQuad,
    spectral::{SpectralConfig, SpectralPipeline, SpectralPlans},
    voice::{Voice, VoiceControlData},
    AudioStreamParams, ChannelCount,
};

use simdeez::scalar::Scalar;
use xsynth_soundfonts::FilterType;

use self::{control::ControlEventData, key::KeyData, params::VoiceChannelParams};

use super::AudioPipe;

mod channel_sf;
mod control;
mod event;
mod key;
mod params;
mod spectral_buffer;
mod voice_buffer;
mod voice_spawner;
pub use event::*;
pub use spectral_buffer::*;

pub(crate) use control::ValueLerp;
pub use params::VoiceChannelStatsReader;

struct Key {
    data: KeyData,
    event_cache: Vec<KeyNoteEvent>,
}

impl Key {
    pub fn new(key: u8, shared_voice_counter: Arc<AtomicU64>, options: ChannelInitOptions) -> Self {
        Key {
            data: KeyData::new(key, shared_voice_counter, options),
            event_cache: Vec::new(),
        }
    }
}

/// Options for initializing a new VoiceChannel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct ChannelInitOptions {
    /// If set to true, the voices killed due to the voice limit will fade out.
    /// If set to false, they will be killed immediately, usually causing clicking
    /// but improving performance.
    ///
    /// Default: `false`
    pub fade_out_killing: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for ChannelInitOptions {
    fn default() -> Self {
        Self {
            fade_out_killing: false,
        }
    }
}

/// Represents a single MIDI channel within XSynth supporting standard
/// time-domain streaming or centralized frequency-domain spectral accumulation.
pub struct VoiceChannel {
    key_voices: Vec<Key>,

    params: VoiceChannelParams,

    stream_params: AudioStreamParams,

    /// The helper struct for keeping track of MIDI control event data
    control_event_data: ControlEventData,

    /// Processed control data, ready to feed to voices
    voice_control_data: VoiceControlData,

    /// Effects
    cutoff: MultiChannelBiQuad,

    /// Storage container managing active spectral voice lists and tracking pedal states
    spectral_voices: Option<SpectralVoiceBuffer<Scalar>>,

    /// Centralized single-IFFT processing engine for spectral mode workloads
    spectral_pipeline: Option<SpectralPipeline<Scalar>>,
}

impl VoiceChannel {
    /// Initializes a new voice channel.
    pub fn new(
        options: ChannelInitOptions,
        stream_params: AudioStreamParams,
        spectral_config: Option<SpectralConfig>,
    ) -> VoiceChannel {
        fn fill_key_array<T, F: Fn(u8) -> T>(func: F) -> Vec<T> {
            let mut vec = Vec::with_capacity(128);
            for i in 0..128 {
                vec.push(func(i));
            }
            vec
        }

        let params = VoiceChannelParams::new(stream_params);
        let shared_voice_counter = params.stats.voice_counter.clone();

        // 1. Map configurations and build pipeline if requested
        let (spectral_pipeline, spectral_voices) = if let Some(config) = spectral_config {
            // Force buffer properties to follow the spectral config flags
            let buffer_options = ChannelInitOptions {
                fade_out_killing: config.enable_phase_fade_out,
            };

            (
                Some(SpectralPipeline::new(
                    config,
                    stream_params.sample_rate as f32,
                    Arc::new(SpectralPlans::new(config.fft_size, config.fft_step)),
                )),
                Some(SpectralVoiceBuffer::new(buffer_options)),
            )
        } else {
            (None, None)
        };

        VoiceChannel {
            params,
            key_voices: fill_key_array(|i| Key::new(i, shared_voice_counter.clone(), options)),
            stream_params,
            control_event_data: ControlEventData::new_defaults(stream_params.sample_rate),
            voice_control_data: VoiceControlData::new_defaults(),
            cutoff: MultiChannelBiQuad::new(
                stream_params.channels.count() as usize,
                FilterType::LowPass,
                stream_params.sample_rate as f32 / 2.0,
                stream_params.sample_rate as f32,
                None,
            ),
            spectral_voices,
            spectral_pipeline,
        }
    }

    fn apply_channel_effects(&mut self, out: &mut [f32]) {
        let control = &mut self.control_event_data;

        match self.stream_params.channels {
            ChannelCount::Mono => {
                for sample in out.iter_mut() {
                    let vol = control.volume.get_next() * control.expression.get_next();
                    let vol = vol.powi(2);
                    *sample *= vol;
                }
            }
            ChannelCount::Stereo => {
                for sample in out.chunks_mut(2) {
                    let vol = control.volume.get_next() * control.expression.get_next();
                    let vol = vol.powi(2);
                    sample[0] *= vol;
                    sample[1] *= vol;
                }

                for sample in out.chunks_mut(2) {
                    let pan = control.pan.get_next();
                    sample[0] *= ((pan * std::f32::consts::PI / 2.0).cos()).min(1.0);
                    sample[1] *= ((pan * std::f32::consts::PI / 2.0).sin()).min(1.0);
                }
            }
        }

        if let Some(cutoff) = control.cutoff {
            self.cutoff
                .set_filter_type(FilterType::LowPass, cutoff, control.resonance);
            self.cutoff.process(out);
        }
    }

    /// Pulls events from cached key registers and drives voice rendering loops.
    fn push_key_events_and_render(&mut self, out: &mut [f32]) {
        self.params.load_program();

        // ROUTE A: Centralized Frequency-Domain Pipeline Mode
        if let (Some(ref mut buffer), Some(ref mut pipeline)) =
            (&mut self.spectral_voices, &mut self.spectral_pipeline)
        {
            let params = &self.params;
            let control_data = &self.voice_control_data;
            let max_voices = pipeline.config().max_voices;

            // 1. Drain event logs from all keys and push spawned notes directly into the unified spectral buffer
            for key in self.key_voices.iter_mut() {
                for e in key.event_cache.drain(..) {
                    match e {
                        KeyNoteEvent::On(vel) => {
                            let voices = params.channel_sf.spawn_voices_attack_spectral(
                                control_data,
                                key.data.key,
                                vel,
                            );
                            buffer.push_voices(voices, max_voices);
                        }
                        KeyNoteEvent::Off => {
                            if let Some(vel) = buffer.release_next_voice() {
                                let voices = params.channel_sf.spawn_voices_release_spectral(
                                    control_data,
                                    key.data.key,
                                    vel,
                                );
                                buffer.push_voices(voices, max_voices);
                            }
                        }
                        KeyNoteEvent::AllOff => while buffer.release_next_voice().is_some() {},
                        KeyNoteEvent::AllKilled => {
                            buffer.kill_all_voices();
                        }
                    }
                }
            }

            // 2. Sync structural parameters down into the global buffer before processing
            buffer.set_damper(self.control_event_data.damper);

            // 3. Drop voices that completed their tracking curves
            buffer.remove_ended_voices();

            // 4. Create a temporary reference vector on the stack
            let mut active_references: Vec<&mut dyn Voice> =
                Vec::with_capacity(buffer.voice_count());
            for voice_box in buffer.iter_voices_mut() {
                active_references.push(&mut **voice_box);
            }

            // 5. Pass our short-lived references into the pipeline execution pass
            pipeline.drain_into(out, &mut active_references);

            // 6. Mirror active counts to the global AtomicU64 statistics tracker
            let active_count = buffer.voice_count();
            self.params
                .stats
                .voice_counter
                .store(active_count as u64, std::sync::atomic::Ordering::SeqCst);
        } else {
            // ROUTE B: Legacy Time-Domain Mode
            // Process voices sequentially within each channel thread.
            // Per-channel threading already provides parallelism; per-key Rayon parallelism
            // causes excessive lock contention and thread blocking overhead.
            out.fill(0.0);
            for key in self.key_voices.iter_mut() {
                for e in key.event_cache.drain(..) {
                    key.data.send_event(
                        e,
                        &self.voice_control_data,
                        &self.params.channel_sf,
                        self.params.layers,
                    );
                }

                key.data.render_to(out);
            }
        }

        self.apply_channel_effects(out);
    }

    fn propagate_voice_controls(&mut self) {
        if let Some(ref mut buffer) = self.spectral_voices {
            for voice in buffer.iter_voices_mut() {
                voice.process_controls(&self.voice_control_data);
            }
        } else {
            for key in self.key_voices.iter_mut() {
                key.data.process_controls(&self.voice_control_data);
            }
        }
    }

    fn kill_voices_in_exclusive_class(&mut self, class: u8) {
        if let Some(ref mut buffer) = self.spectral_voices {
            buffer.kill_by_exclusive_class(class);
        } else {
            for key in self.key_voices.iter_mut() {
                key.data.kill_by_exclusive_class(class);
            }
        }
    }

    pub fn process_event(&mut self, event: ChannelEvent) {
        self.push_events_iter(std::iter::once(event));
    }

    pub fn push_events_iter<T: Iterator<Item = ChannelEvent>>(&mut self, iter: T) {
        for e in iter {
            match e {
                ChannelEvent::Audio(audio) => match audio {
                    ChannelAudioEvent::NoteOn { key, vel } => {
                        let classes: Vec<_> = self
                            .params
                            .channel_sf
                            .exclusive_classes_attack(key, vel)
                            .collect();
                        for class in classes {
                            self.kill_voices_in_exclusive_class(class);
                        }
                        if let Some(key) = self.key_voices.get_mut(key as usize) {
                            let ev = KeyNoteEvent::On(vel);
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::NoteOff { key } => {
                        if let Some(key) = self.key_voices.get_mut(key as usize) {
                            let ev = KeyNoteEvent::Off;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::AllNotesOff => {
                        for key in self.key_voices.iter_mut() {
                            let ev = KeyNoteEvent::AllOff;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::AllNotesKilled => {
                        for key in self.key_voices.iter_mut() {
                            let ev = KeyNoteEvent::AllKilled;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::ResetControl => {
                        self.reset_control();
                    }
                    ChannelAudioEvent::Control(control) => {
                        self.process_control_event(control);
                    }
                    ChannelAudioEvent::ProgramChange(preset) => {
                        self.params.set_preset(preset);
                    }
                    ChannelAudioEvent::SystemReset => {
                        for key in self.key_voices.iter_mut() {
                            key.event_cache.clear();
                            key.event_cache.push(KeyNoteEvent::AllKilled);
                        }
                        self.reset_control();
                        self.reset_program();
                    }
                },
                ChannelEvent::Config(config) => self.params.process_config_event(config),
            }
        }
    }

    pub fn get_channel_stats(&self) -> VoiceChannelStatsReader {
        let stats = self.params.stats.clone();
        VoiceChannelStatsReader::new(stats)
    }
}

impl AudioPipe for VoiceChannel {
    fn stream_params(&self) -> &AudioStreamParams {
        &self.params.constant.stream_params
    }

    fn read_samples_unchecked(&mut self, out: &mut [f32]) {
        self.push_key_events_and_render(out);
    }
}
