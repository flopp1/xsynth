#![allow(non_camel_case_types)]
use std::{
    collections::{HashMap, HashSet},
    io,
    path::PathBuf,
    sync::Arc,
};

use biquad::Q_BUTTERWORTH_F32;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;
use xsynth_soundfonts::{convert_sample_index, FilterType, LoopMode};

use self::audio::load_audio_file;
pub use self::audio::AudioLoadError;

use super::{
    voice::VoiceControlData,
    voice::{EnvelopeParameters, Voice},
};
use crate::{helpers::db_to_amp, voice::EnvelopeDescriptor, AudioStreamParams, ChannelCount};
use crate::spectral::{AnalyzedSample, analyze_pcm_sample, SpectralPlans};

pub use xsynth_soundfonts::{sf2::Sf2ParseError, sfz::SfzParseError};

mod audio;
mod config;
mod utils;
mod voice_spawners;
use utils::*;
use voice_spawners::*;

pub use config::*;

pub trait VoiceSpawner: Sync + Send {
    fn spawn_voice(&self, control: &VoiceControlData) -> Box<dyn Voice>;
    
    /// NEW: Extends the top-level definition to declare the spectral initialization hook.
    /// Automatically drops back to standard time-domain generation by default for system stability.
    fn spawn_spectral_voice(&self, control: &VoiceControlData) -> Box<dyn Voice> {
        self.spawn_voice(control)
    }
    
    fn exclusive_class(&self) -> Option<u8> {
        None
    }
}

pub trait SoundfontBase: Sync + Send + std::fmt::Debug {
    fn stream_params(&self) -> &'_ AudioStreamParams;

    fn get_attack_voice_spawners_at(
        &self,
        bank: u8,
        preset: u8,
        key: u8,
        vel: u8,
    ) -> Vec<Box<dyn VoiceSpawner>>;
    fn get_release_voice_spawners_at(
        &self,
        bank: u8,
        preset: u8,
        key: u8,
        vel: u8,
    ) -> Vec<Box<dyn VoiceSpawner>>;
}

#[derive(Clone)]
pub(super) struct LoopParams {
    pub mode: LoopMode,
    pub offset: u32,
    pub start: u32,
    pub end: u32,
    pub stop: Option<u32>,
}

pub(super) struct SampleVoiceSpawnerParams {
    pub volume: f32,
    pub pan: f32,
    pub speed_mult: f32,
    pub cutoff: Option<f32>,
    pub resonance: f32,
    pub filter_type: FilterType,
    pub loop_params: LoopParams,
    pub envelope: Arc<EnvelopeParameters>,
    pub sample: Arc<[Arc<[f32]>]>,
    pub interpolator: Interpolator,
    pub exclusive_class: Option<u8>,
    pub root_key: u8,
    
    /// NEW: The pre-analyzed frequency matrices used by your SpectralVoice<T> structure.
    pub spectral_sample: Option<Arc<AnalyzedSample>>,
}

pub(super) struct SoundfontInstrument {
    bank: u8,
    preset: u8,
    spawner_params_list: Vec<Vec<Arc<SampleVoiceSpawnerParams>>>,
}

pub struct SampleSoundfont {
    instruments: Vec<SoundfontInstrument>,
    stream_params: AudioStreamParams,
}

#[derive(Debug, Error)]
pub enum LoadSfzError {
    #[error("IO Error")]
    IOError(#[from] io::Error),

    #[error("Error loading samples")]
    AudioLoadError(#[from] AudioLoadError),

    #[error("Error parsing the SFZ: {0}")]
    SfzParseError(#[from] SfzParseError),
}

#[derive(Debug, Error)]
pub enum LoadSfError {
    #[error("Error loading the SFZ: {0}")]
    LoadSfzError(#[from] LoadSfzError),

    #[error("Error loading the SF2: {0}")]
    LoadSf2Error(#[from] Sf2ParseError),

    #[error("Unsupported format")]
    Unsupported,
}

impl SampleSoundfont {
    pub fn new(
        path: impl Into<PathBuf>,
        stream_params: AudioStreamParams,
        options: SoundfontInitOptions,
    ) -> Result<Self, LoadSfError> {
        let path: PathBuf = path.into();
        if let Some(ext) = path.extension() {
            match ext.to_str().unwrap_or("").to_lowercase().as_str() {
                "sfz" => {
                    Self::new_sfz(path, stream_params, options).map_err(LoadSfError::LoadSfzError)
                }
                "sf2" => {
                    Self::new_sf2(path, stream_params, options).map_err(LoadSfError::LoadSf2Error)
                }
                _ => Err(LoadSfError::Unsupported),
            }
        } else {
            Err(LoadSfError::Unsupported)
        }
    }

    pub fn new_sfz(
        sfz_path: impl Into<PathBuf>,
        stream_params: AudioStreamParams,
        options: SoundfontInitOptions,
    ) -> Result<Self, LoadSfzError> {
        let regions = xsynth_soundfonts::sfz::parse_soundfont(sfz_path.into())?;

        let unique_sample_params: HashSet<_> = regions
            .iter()
            .map(sample_cache_from_region_params)
            .collect();

        let samples: Result<HashMap<_, _>, _> = unique_sample_params
            .into_par_iter()
            .map(|params| -> Result<(_, _), LoadSfzError> {
                let sample = load_audio_file(&params.path, stream_params)?;
                Ok((params, sample))
            })
            .collect();
        let samples = samples?;

        let spectral_config = options.spectral_config.unwrap_or_default();

        let spectral_plans = SpectralPlans::new(spectral_config.fft_size, spectral_config.fft_step());

        let mut spawner_params_list = Vec::<Vec<Arc<SampleVoiceSpawnerParams>>>::new();
        for _ in 0..(128 * 128) {
            spawner_params_list.push(Vec::new());
        }

        let mut spectral_cache: HashMap<SampleCache, Arc<AnalyzedSample>> = HashMap::new();

        for region in regions {
            let params = sample_cache_from_region_params(&region);

            if region.keyrange.contains(&-1) {
                continue;
            }

            let sample_rate = samples[&params].1;

            let mut region_samples = samples[&params].0.clone();

            // SF2 / SFZ: compute ONCE per region, before the key/vel loops
            let spectral_sample = if options.spectral_config.is_some() {
                Some(
                    spectral_cache.entry(params.clone())
                        .or_insert_with(|| {
                            Arc::new(analyze_pcm_sample(region_samples.clone(), sample_rate, &spectral_config, &spectral_plans))
                        })
                        .clone(),
                )
            } else {
                None
            };

            let sample = if spectral_sample.is_some() {
                // spectral-only mode: drop the raw PCM reference after analysis to reduce memory
                Arc::new([])
            } else {
                if stream_params.channels == ChannelCount::Stereo && region_samples.len() == 1 {
                    region_samples = Arc::new([region_samples[0].clone(), region_samples[0].clone()]);
                }
                region_samples.clone()
            };

            for key in region.keyrange.clone() {
                for vel in region.velrange.clone() {
                    let index = key_vel_to_index(key as u8, vel);
                    let speed_mult =
                        get_speed_mult_from_keys(key as u8, region.pitch_keycenter as u8)
                            * cents_factor(region.tune as f32);

                    let mut envelope = region.ampeg_envelope.clone();
                    envelope.ampeg_release +=
                        (vel as f32 / 127.0) * region.ampeg_envelope.ampeg_vel2release;
                    let envelope_params = Arc::new(
                        envelope_descriptor_from_region_params(&envelope).to_envelope_params(
                            stream_params.sample_rate,
                            options.vol_envelope_options,
                        ),
                    );

                    let mut cutoff = None;
                    if options.use_effects {
                        if let Some(mut cutoff_t) = region.cutoff {
                            if cutoff_t >= 1.0 {
                                let cents = vel as f32 / 127.0 * region.fil_veltrack as f32
                                    + (key as f32 - region.fil_keycenter as f32)
                                        * region.fil_keytrack as f32;
                                cutoff_t *= cents_factor(cents);
                                cutoff = Some(
                                    cutoff_t
                                        .clamp(1.0, stream_params.sample_rate as f32 / 2.0 - 100.0),
                                );
                            }
                        }
                    }

                    let pan_mult = vel as f32 / 127.0 * region.pan_veltrack
                        + (key as f32 - region.pan_keycenter as f32) * region.pan_keytrack;
                    let pan = (region.pan as f32 + pan_mult).clamp(-100.0, 100.0) / 100.0;
                    let pan = (pan + 1.0) / 2.0;

                    let vol_vel = {
                        let a = region.amp_veltrack / 100.0;
                        let aabs = a.abs();
                        let vel = vel as f32;

                        127.0 * (1.0 - aabs)
                            + vel * (a + aabs) / 2.0
                            + (127.0 - vel) * (aabs - a) / 2.0
                    };
                    let vol_mult = (vol_vel / 127.0).powi(2);
                    let vol_db_add =
                        (key as f32 - region.amp_keycenter as f32) * region.amp_keytrack;
                    let vol_db = (region.volume as f32 + vol_db_add).clamp(-96.0, 12.0);
                    let volume = vol_mult * db_to_amp(vol_db);

                    

                    let loop_params = LoopParams {
                        mode: if region.loop_start == region.loop_end {
                            LoopMode::NoLoop
                        } else {
                            region.loop_mode
                        },
                        offset: convert_sample_index(
                            region.offset,
                            sample_rate,
                            stream_params.sample_rate,
                        ),
                        start: convert_sample_index(
                            region.loop_start,
                            sample_rate,
                            stream_params.sample_rate,
                        ),
                        end: convert_sample_index(
                            region.loop_end,
                            sample_rate,
                            stream_params.sample_rate,
                        ),
                        stop: None,
                    };

                    if stream_params.channels == ChannelCount::Stereo && region_samples.len() == 1 {
                        region_samples =
                            Arc::new([region_samples[0].clone(), region_samples[0].clone()]);
                    }

                    let spawner_params = Arc::new(SampleVoiceSpawnerParams {
                        pan,
                        volume,
                        envelope: envelope_params,
                        speed_mult,
                        cutoff,
                        resonance: db_to_amp(region.resonance) * Q_BUTTERWORTH_F32,
                        filter_type: region.filter_type,
                        interpolator: options.interpolator,
                        loop_params,
                        sample: sample.clone(),
                        exclusive_class: None,
                        root_key: region.pitch_keycenter as u8,
                        spectral_sample: spectral_sample.clone(),
                    });

                    spawner_params_list[index].push(spawner_params.clone());
                }
            }
        }

        Ok(SampleSoundfont {
            instruments: vec![SoundfontInstrument {
                bank: options.bank.unwrap_or(0),
                preset: options.preset.unwrap_or(0),
                spawner_params_list,
            }],
            stream_params,
        })
    }

    pub fn new_sf2(
        sf2_path: impl Into<PathBuf>,
        stream_params: AudioStreamParams,
        options: SoundfontInitOptions,
    ) -> Result<Self, Sf2ParseError> {
        let presets =
            xsynth_soundfonts::sf2::load_soundfont(sf2_path.into(), stream_params.sample_rate)?;

        let mut instruments = Vec::new();

        let spectral_config = options.spectral_config.unwrap_or_default();

        let spectral_plans = SpectralPlans::new(spectral_config.fft_size, spectral_config.fft_step());

        for preset in presets {
            if let Some(bank) = options.bank {
                if bank != preset.bank as u8 {
                    continue;
                }
            }
            if let Some(presetn) = options.preset {
                if presetn != preset.preset as u8 {
                    continue;
                }
            }

            let mut spawner_params_list = Vec::<Vec<Arc<SampleVoiceSpawnerParams>>>::new();
            for _ in 0..(128 * 128) {
                spawner_params_list.push(Vec::new());
            }

            let mut spectral_cache: HashMap<*const [Arc<[f32]>], Arc<AnalyzedSample>> = HashMap::new();

            let mut unique_envelope_params =
                Vec::<(EnvelopeDescriptor, Arc<EnvelopeParameters>)>::new();

            for region in preset.regions {
                let sample_rate = region.sample_rate.clone();

                let mut region_samples = region.sample.clone();

                let key = Arc::as_ptr(&region_samples);

                let spectral_sample = if options.spectral_config.is_some() {
                    Some(spectral_cache.entry(key)
                        .or_insert_with(|| {
                            Arc::new(analyze_pcm_sample(region_samples.clone(), sample_rate, &spectral_config, &spectral_plans))
                        })
                        .clone())
                } else {
                    None
                };

                let sample = if spectral_sample.is_some() {
                    Arc::new([])
                } else {
                    if stream_params.channels == ChannelCount::Stereo && region_samples.len() == 1 {
                        region_samples = Arc::new([region_samples[0].clone(), region_samples[0].clone()]);
                    }
                    region_samples.clone()
                };

                for key in region.keyrange.clone() {
                    for vel in region.velrange.clone() {
                        let index = key_vel_to_index(key, vel);
                        let note_params = region.note_params(key, vel);
                        let envelope =
                            envelope_descriptor_from_region_params(&note_params.ampeg_envelope);
                        let envelope_params = if let Some((_, params)) = unique_envelope_params
                            .iter()
                            .find(|(descriptor, _)| *descriptor == envelope)
                        {
                            params.clone()
                        } else {
                            let params = Arc::new(envelope.to_envelope_params(
                                stream_params.sample_rate,
                                options.vol_envelope_options,
                            ));
                            unique_envelope_params.push((envelope, params.clone()));
                            params
                        };
                        let tuned_key_cents =
                            (key as f32 - region.root_key as f32) * region.scale_tuning as f32;
                        let speed_mult = cents_factor(
                            tuned_key_cents
                                + region.fine_tune as f32
                                + region.coarse_tune as f32 * 100.0
                                + note_params.tune_cents,
                        );

                        let mut cutoff = None;
                        if options.use_effects {
                            if let Some(cutoff_t) = note_params.cutoff {
                                if cutoff_t >= 1.0 {
                                    cutoff = Some(cutoff_t.clamp(
                                        1.0,
                                        stream_params.sample_rate as f32 / 2.0 - 100.0,
                                    ));
                                }
                            }
                        }

                        let pan = ((note_params.pan as f32 / 500.0) + 1.0) / 2.0;

                        let loop_params = LoopParams {
                            mode: if region.loop_start == region.loop_end {
                                LoopMode::NoLoop
                            } else {
                                region.loop_mode
                            },
                            offset: region.offset,
                            start: region.loop_start,
                            end: region.loop_end,
                            stop: Some(region.sample_end),
                        };
                        
                        if stream_params.channels == ChannelCount::Stereo && region_samples.len() == 1
                        {
                            region_samples =
                                Arc::new([region_samples[0].clone(), region_samples[0].clone()]);
                        }

                        let spawner_params = Arc::new(SampleVoiceSpawnerParams {
                            pan,
                            volume: note_params.volume,
                            envelope: envelope_params,
                            speed_mult,
                            cutoff,
                            resonance: db_to_amp(note_params.resonance) * Q_BUTTERWORTH_F32,
                            filter_type: FilterType::LowPass,
                            interpolator: options.interpolator,
                            loop_params,
                            sample: sample.clone(),
                            exclusive_class: region.exclusive_class,
                            root_key: region.root_key as u8,
                            spectral_sample: spectral_sample.clone(),
                        });

                        spawner_params_list[index].push(spawner_params.clone());
                    }
                }
            }

            let new = SoundfontInstrument {
                bank: preset.bank as u8,
                preset: preset.preset as u8,
                spawner_params_list,
            };
            instruments.push(new);
        }

        Ok(SampleSoundfont {
            instruments,
            stream_params,
        })
    }
}

impl std::fmt::Debug for SampleSoundfont {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "SampleSoundfont")
    }
}

impl SoundfontBase for SampleSoundfont {
    fn stream_params(&self) -> &'_ AudioStreamParams {
        &self.stream_params
    }

    fn get_attack_voice_spawners_at(
        &self,
        bank: u8,
        preset: u8,
        key: u8,
        vel: u8,
    ) -> Vec<Box<dyn VoiceSpawner>> {
        use simdeez::*;
        use simdeez::prelude::*;

        simd_runtime_generate!(
            fn get(
                key: u8,
                vel: u8,
                sf: &SoundfontInstrument,
                stream_params: &AudioStreamParams,
            ) -> Vec<Box<dyn VoiceSpawner>> {
                if sf.spawner_params_list.is_empty() {
                    return Vec::new();
                }

                let index = key_vel_to_index(key, vel);
                let mut vec = Vec::<Box<dyn VoiceSpawner>>::new();
                for spawner in &sf.spawner_params_list[index] {
                    match stream_params.channels {
                        ChannelCount::Stereo => vec.push(Box::new(
                            StereoSampledVoiceSpawner::<S>::new(spawner, vel, *stream_params),
                        )),
                        ChannelCount::Mono => vec.push(Box::new(
                            MonoSampledVoiceSpawner::<S>::new(spawner, vel, *stream_params),
                        )),
                    }
                }
                vec
            }
        );

        let empty = SoundfontInstrument {
            bank: 0,
            preset: 0,
            spawner_params_list: Vec::new(),
        };

        let instrument = self
            .instruments
            .iter()
            .find(|i| i.bank == bank && i.preset == preset)
            .unwrap_or(&empty);

        get(key, vel, instrument, self.stream_params())
    }

    fn get_release_voice_spawners_at(
        &self,
        _bank: u8,
        _preset: u8,
        _key: u8,
        _vel: u8,
    ) -> Vec<Box<dyn VoiceSpawner>> {
        vec![]
    }
}