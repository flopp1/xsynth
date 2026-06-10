use std::{sync::Arc, time::Duration};

use rand::Rng;
use xsynth_core::{
    channel::{ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, VoiceChannel},
    soundfont::{EnvelopeCurveType, EnvelopeOptions, Interpolator, SampleSoundfont, SoundfontBase},
    AudioPipe, AudioStreamParams, ChannelCount,
};

pub fn run_bench(
    name: &str,
    count: u32,
    mut make_new_channel: impl FnMut() -> VoiceChannel,
    mut bench: impl FnMut(VoiceChannel),
) -> Duration {
    let mut total_duration = Duration::new(0, 0);
    for _ in 0..count {
        let channel = make_new_channel();
        let start = std::time::Instant::now();
        bench(channel);
        let duration = start.elapsed();
        total_duration += duration;
    }
    println!("{}: {}ms", name, total_duration.as_millis() / count as u128);
    total_duration
}

pub fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let Some(sfz) = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("XSYNTH_EXAMPLE_SFZ").ok())
    else {
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
        overlap_percent: 0.75,   // 75% overlap for smooth phase tracking
        max_voices: Some(layer_count),
        enable_phase_fade_out: true,
        max_peaks_per_frame: 32, //32 seems like a decent default for now, i hope
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

    println!("Running benches");

    let make_new_channel = || {
        let mut channel = VoiceChannel::new(Default::default(), 
                                                            stream_params, 
                                                            Some(spectral_config));
        channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(
            soundfonts.clone(),
        )));
        channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetLayerCount(
            None,
        )));
        channel
    };

    bench_events(make_new_channel);
    bench_rendering(make_new_channel);
    bench_random_rendering(make_new_channel);
}

fn bench_events(make_new_channel: impl FnMut() -> VoiceChannel) {
    run_bench("Push Events", 100, make_new_channel, |mut channel| {
        for _ in 0..1000 {
            for i in 0..127 {
                channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                    key: i as u8,
                    vel: 127,
                }));
            }
        }
    });
}

fn bench_rendering(make_new_channel: impl FnMut() -> VoiceChannel) {
    let mut buffer = vec![0.0; 480];
    run_bench(
        "Render events",
        100,
        make_new_channel,
        move |mut channel| {
            for _ in 0..100 {
                for i in 0..127 {
                    channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                        key: i as u8,
                        vel: 127,
                    }));
                }

                channel.read_samples(&mut buffer);
                for i in 0..127 {
                    channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOff {
                        key: i as u8,
                    }));
                }
            }
        },
    );
}

fn bench_random_rendering(make_new_channel: impl FnMut() -> VoiceChannel) {
    let mut buffer = vec![0.0; 480];

    // It asks for a 32 byte long array for the seed
    let mut random: rand::rngs::StdRng = rand::SeedableRng::from_seed([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
        1, 2,
    ]);

    run_bench(
        "Render random many",
        100,
        make_new_channel,
        move |mut channel| {
            let mut off_events = Vec::new();
            let halflife = 1000.0;
            for _ in 0..100 {
                for _ in 0..1000 {
                    let note_on_chance = 1.0 / (off_events.len() as f64 / halflife + 1.0);
                    if random.gen_bool(note_on_chance) {
                        let key = random.gen_range(0u8..=127u8);
                        let vel = random.gen_range(1u8..=127u8);

                        channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                            key,
                            vel,
                        }));

                        off_events.push(key);
                    } else {
                        let key = off_events.swap_remove(random.gen_range(0..off_events.len()));
                        channel
                            .process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOff { key }));
                    }
                }

                let buffer_len = random.gen_range(1..=(buffer.len() / 2)) * 2;
                channel.read_samples(&mut buffer[..buffer_len]);
            }
        },
    );
}
