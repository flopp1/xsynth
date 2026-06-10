use crate::utils::*;
use clap::{command, Arg, ArgAction, ArgMatches, Command};
use std::path::PathBuf;
use xsynth_core::{
    channel::ChannelInitOptions,
    channel_group::{ChannelGroupConfig, ParallelismOptions, SynthFormat, ThreadCount},
    soundfont::{EnvelopeCurveType, EnvelopeOptions, Interpolator, SoundfontInitOptions},
    AudioStreamParams, ChannelCount,
    spectral::SpectralConfig,
};

#[derive(Clone, Debug, PartialEq)]
pub struct XSynthRenderConfig {
    pub group_options: ChannelGroupConfig,

    pub sf_options: SoundfontInitOptions,

    pub use_limiter: bool,

    pub spectral_config: Option<SpectralConfig>,
}

#[derive(Clone, Debug)]
pub struct State {
    pub config: XSynthRenderConfig,
    pub layers: Option<usize>,
    pub midi: PathBuf,
    pub soundfonts: Vec<PathBuf>,
    pub output: PathBuf,
}

impl State {
    const THREADING_HELP: &'static str =
        "Use \"none\" for no multithreading, \"auto\" for multithreading with\n\
        an automatically determined thread count or any number to specify the\n\
        amount of threads that should be used.\n\
        Default: \"auto\"";

    pub fn command() -> Command {
        command!().args([
            Arg::new("midi")
                .required(true)
                .help("The path of the MIDI file to be converted."),
                
            // FIXED: Replaced ArgAction::Append with num_args(1..) 
            // This tells clap to safely ingest all remaining trailing positions as a slice.
            Arg::new("soundfonts")
                .required(true)
                .num_args(1..) 
                .help(
                    "Paths of the soundfonts to be used.\n\
                    Will be loaded in the order they are typed.",
                ),
                
            Arg::new("output")
                .short('o')
                .long("output")
                .help("The path of the output audio file.\nDefault: \"out.wav\""),

            Arg::new("sample rate")
                .short('s')
                .long("sample-rate")
                .help(
                    "The sample rate of the output audio in Hz.\n\
                    Default: 48000 (48kHz)",
                )
                .value_parser(int_parser),
            Arg::new("audio channels")
                .short('c')
                .long("audio-channels")
                .help(
                    "The audio channel count of the output audio.\n\
                    Supported: \"mono\" and \"stereo\"\n\
                    Default: stereo",
                )
                .value_parser(audio_channels_parser),
            Arg::new("layer limit")
                .short('l')
                .long("layers")
                .help(
                    "The layer limit for each channel. Use \"0\" for unlimited layers.\n\
                    One layer is one voice per key per channel.\n\
                    Default: 32",
                )
                .value_parser(layers_parser),
            Arg::new("channel threading")
                .long("channel-threading")
                .help("Per-channel multithreading options.\n".to_owned() + Self::THREADING_HELP)
                .value_parser(threading_parser),
            Arg::new("key threading")
                .long("key-threading")
                .help("Per-key multithreading options.\n".to_owned() + Self::THREADING_HELP)
                .value_parser(threading_parser),
            Arg::new("limiter")
                .short('L')
                .long("apply-limiter")
                .help("Apply an audio limiter to the output audio to prevent clipping.")
                .action(ArgAction::SetTrue),
            Arg::new("disable fade out voice killing")
                .long("disable-fade-out")
                .help("Disables fade out when killing a voice. This may cause popping.")
                .action(ArgAction::SetFalse),
            Arg::new("linear envelope")
                .long("linear-envelope")
                .help("Use a linear decay and release phase in the volume envelope, in amplitude units.")
                .action(ArgAction::SetTrue),
            Arg::new("interpolation")
                .short('I')
                .long("interpolation")
                .help(
                    "The interpolation algorithm to use. Available options are\n\
                    \"none\" (no interpolation) and \"linear\" (linear interpolation).\n\
                    Default: \"linear\"",
                )
                .value_parser(interpolation_parser),
            Arg::new("spectral")
                .long("spectral")
                .help("Enable frequency-domain spectral rendering instead of standard time-domain.")
                .action(ArgAction::SetTrue),
            Arg::new("fft size")
                .long("fft-size")
                .help("FFT window size for spectral processing. Must be a power of 2.\nDefault: 1024")
                .value_parser(int_parser),
            Arg::new("fft step")
                .long("fft-step")
                .help("Frame advance distance for overlap-add processing.\nDefault: 256")
                .value_parser(int_parser),
        ])
    }

    pub fn from_args() -> Self {
        let matches = Self::command().get_matches();
        Self::from_matches(&matches)
    }

    fn from_matches(matches: &ArgMatches) -> Self {
    let midi = matches
        .get_one::<String>("midi")
        .cloned()
        .unwrap_or_default();

    let output = matches
        .get_one::<String>("output")
        .cloned()
        .unwrap_or_else(|| "out.wav".to_owned());

    let soundfonts = matches
        .get_many::<String>("soundfonts")
        .unwrap_or_default()
        .map(PathBuf::from)
        .collect::<Vec<_>>();

    // FIXED: Added explicit type annotation matching whatever your `layers_parser` outputs.
    // Assuming layers_parser returns a usize or Option<usize>
    let layer_limit = matches
        .get_one::<Option<usize>>("layer limit")
        .copied()
        .unwrap_or(Some(32));

    // FIXED: Action flags (SetTrue/SetFalse) must use .get_flag()
    let disable_fade_out = matches.get_flag("disable fade out voice killing"); 
    let linear_envelope = matches.get_flag("linear envelope");
    let use_limiter = matches.get_flag("limiter");
    let is_spectral = matches.get_flag("spectral");

    let spectral_config = if is_spectral {
        // Pull them out using <u32> to match your int_parser, then cast to usize
        let fft_size = matches
            .get_one::<u32>("fft size")
            .copied()
            .unwrap_or(1024) as usize;

        let fft_step = matches
            .get_one::<u32>("fft step")
            .copied()
            .unwrap_or(256) as usize;
        
        Some(SpectralConfig {
            fft_size,
            fft_step,
            max_voices: layer_limit,
            enable_phase_fade_out: disable_fade_out,
        })
    } else {
        None
    };

    let config = XSynthRenderConfig {
        group_options: ChannelGroupConfig {
            channel_init_options: ChannelInitOptions {
                fade_out_killing: disable_fade_out,
            },
            format: SynthFormat::Midi,
            audio_params: AudioStreamParams::new(
                matches.get_one::<u32>("sample rate").copied().unwrap_or(48000),
                matches
                    .get_one::<ChannelCount>("audio channels")
                    .copied()
                    .unwrap_or(ChannelCount::Stereo),
            ),
            parallelism: ParallelismOptions {
                channel: matches
                    .get_one::<ThreadCount>("channel threading")
                    .copied()
                    .unwrap_or(ThreadCount::Auto),
                key: matches
                    .get_one::<ThreadCount>("key threading")
                    .copied()
                    .unwrap_or(ThreadCount::Auto),
            },
            spectral_config: spectral_config.clone(),
        },
        sf_options: SoundfontInitOptions {
            bank: None,
            preset: None,
            vol_envelope_options: if linear_envelope {
                EnvelopeOptions {
                    attack_curve: EnvelopeCurveType::Exponential,
                    decay_curve: EnvelopeCurveType::Exponential,
                    release_curve: EnvelopeCurveType::Exponential,
                }
            } else {
                EnvelopeOptions {
                    attack_curve: EnvelopeCurveType::Exponential,
                    decay_curve: EnvelopeCurveType::Linear,
                    release_curve: EnvelopeCurveType::Linear,
                }
            },
            use_effects: true,
            interpolator: matches
                .get_one::<Interpolator>("interpolation")
                .copied()
                .unwrap_or(Interpolator::Linear),
            spectral_config: spectral_config.clone(),
        },
        use_limiter,
        spectral_config,
    };

    Self {
        config,
        layers: layer_limit,
        midi: PathBuf::from(midi),
        output: PathBuf::from(output),
        soundfonts,
    }
}
}

#[cfg(test)]
mod tests {
    use super::State;
    use xsynth_core::soundfont::EnvelopeCurveType;

    #[test]
    fn linear_envelope_flag_uses_db_curves_that_render_linear_in_amplitude() {
        let matches = State::command().get_matches_from([
            "xsynth-render",
            "song.mid",
            "piano.sf2",
            "--linear-envelope",
        ]);
        let state = State::from_matches(&matches);

        assert_eq!(
            state.config.sf_options.vol_envelope_options.decay_curve,
            EnvelopeCurveType::Exponential
        );
        assert_eq!(
            state.config.sf_options.vol_envelope_options.release_curve,
            EnvelopeCurveType::Exponential
        );
    }

    #[test]
    fn default_envelope_curves_remain_soundfont_style() {
        let matches = State::command().get_matches_from(["xsynth-render", "song.mid", "piano.sf2"]);
        let state = State::from_matches(&matches);

        assert_eq!(
            state.config.sf_options.vol_envelope_options.decay_curve,
            EnvelopeCurveType::Linear
        );
        assert_eq!(
            state.config.sf_options.vol_envelope_options.release_curve,
            EnvelopeCurveType::Linear
        );
    }
}
