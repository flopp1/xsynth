use std::sync::Arc;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use xsynth_core::{
    channel::{ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, ChannelInitOptions, VoiceChannel},
    soundfont::{EnvelopeCurveType, EnvelopeOptions, Interpolator, SampleSoundfont, SoundfontBase},
    AudioPipe, AudioStreamParams, ChannelCount,
};

fn stress_channel(channel: &mut VoiceChannel) {
    let mut buffer = vec![0.0; 0];
    for _ in 0..400 {
        for i in 0..127 {
            channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                key: i as u8,
                vel: 127,
            }));
        }
        for i in 0..127 {
            channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOff {
                key: i as u8,
            }));
        }

        // Key events get processed when we read samples
        channel.read_samples(&mut buffer);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let Some(sfz) = std::env::var("XSYNTH_EXAMPLE_SFZ").ok() else {
        println!(
            "Usage: {} [sfz]",
            std::env::current_exe()
                .unwrap_or("example".into())
                .display()
        );
        return;
    };

    let stream_params = AudioStreamParams::new(48000, ChannelCount::Stereo);

    println!("Loading soundfont...");

    let layer_count = 512 * 4;

    let spectral_config = xsynth_core::spectral::SpectralConfig {
        fft_size: 1024,          // Standard FFT window sizing
        fft_step: 256,             // 75% overlap for smooth phase tracking
        max_voices: Some(layer_count),
        enable_phase_fade_out: true,
    };

    let soundfonts: Vec<Arc<dyn SoundfontBase>> = vec![Arc::new(
        SampleSoundfont::new(sfz, 
                            stream_params, 
                            xsynth_core::soundfont::SoundfontInitOptions {
                                bank: None,
                                preset: None,
                                vol_envelope_options: EnvelopeOptions {
                                    attack_curve: EnvelopeCurveType::Exponential,
                                    decay_curve: EnvelopeCurveType::Exponential,
                                    release_curve: EnvelopeCurveType::Exponential,
                                },
                                interpolator: Interpolator::Nearest,
                                use_effects: false,
                                spectral_config: Some(spectral_config),
                            },
        ).unwrap(),
    )];

    c.bench_function("send events (4 layers, kill notes)", |f| {
        f.iter(|| {
            let init = ChannelInitOptions {
                fade_out_killing: false,
            };
            let mut channel = VoiceChannel::new(init, stream_params, /*None,*/ Some(spectral_config));
            channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(
                soundfonts.clone(),
            )));
            channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetLayerCount(
                Some(4),
            )));

            stress_channel(&mut channel)
        })
    });

    c.bench_function("send events (4 layers, release notes)", |f| {
        f.iter(|| {
            let init = ChannelInitOptions {
                fade_out_killing: true,
            };
            let mut channel = VoiceChannel::new(init, stream_params, /*None,*/ Some(spectral_config));
            channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(
                soundfonts.clone(),
            )));
            channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetLayerCount(
                Some(4),
            )));

            stress_channel(&mut channel)
        })
    });

    c.bench_function("send events (unlimited layers, kill notes)", |f| {
        f.iter(|| {
            let init = ChannelInitOptions {
                fade_out_killing: false,
            };
            let mut channel = VoiceChannel::new(init, stream_params, /*None,*/ Some(spectral_config));
            channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(
                soundfonts.clone(),
            )));
            channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetLayerCount(
                None,
            )));

            stress_channel(&mut channel)
        })
    });

    c.bench_function("send events (unlimited layers, release notes)", |f| {
        f.iter(|| {
            let init = ChannelInitOptions {
                fade_out_killing: true,
            };
            let mut channel = VoiceChannel::new(init, stream_params, /*None,*/ Some(spectral_config));
            channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(
                soundfonts.clone(),
            )));
            channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetLayerCount(
                None,
            )));

            stress_channel(&mut channel)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
