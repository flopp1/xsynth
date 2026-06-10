use std::{
    collections::VecDeque,
    io,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread::{self},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, DefaultStreamConfigError, Device, PauseStreamError, PlayStreamError,
    SizedSample, Stream, SupportedStreamConfig,
};
use crossbeam_channel::{bounded, unbounded};
use thiserror::Error;

use xsynth_core::{
    buffered_renderer::{BufferedRenderer, BufferedRendererStatsReader},
    channel::{ChannelConfigEvent, ChannelEvent, VoiceChannel},
    channel_group::SynthFormat,
    effects::VolumeLimiter,
    helpers::{prepapre_cache_vec, sum_simd},
    AudioPipe, AudioStreamParams, FunctionAudioPipe,
};

use crate::{
    util::ReadWriteAtomicU64, RealtimeEventSender, SynthEvent, XSynthRealtimeConfig,
};

#[derive(Debug, Error)]
pub enum RealtimeSynthError {
    #[error("failed to find output device")]
    NoOutputDevice,

    #[error("failed to get default output config: {0}")]
    DefaultOutputConfig(#[from] DefaultStreamConfigError),

    #[error("failed to build thread pool: {0}")]
    ThreadPoolBuild(#[from] rayon::ThreadPoolBuildError),

    #[error("failed to spawn realtime channel thread: {0}")]
    ChannelThreadSpawn(#[source] io::Error),

    #[error("failed to spawn realtime stream thread: {0}")]
    StreamThreadSpawn(#[source] io::Error),

    #[error("realtime stream thread terminated during startup")]
    StreamThreadInit,

    #[error("failed to create realtime event sender: {0}")]
    EventSenderInit(#[source] io::Error),

    #[error("failed to spawn buffered renderer thread: {0}")]
    BufferedRendererThreadSpawn(#[source] io::Error),

    #[error("failed to create audio stream: {0}")]
    BuildStream(#[from] BuildStreamError),

    #[error("failed to start audio stream: {0}")]
    PlayStream(#[from] PlayStreamError),

    #[error("unsupported sample format: {0:?}")]
    UnsupportedSampleFormat(cpal::SampleFormat),
}

/// Holds the statistics for an instance of RealtimeSynth.
#[derive(Debug, Clone)]
struct RealtimeSynthStats {
    voice_count: Arc<AtomicU64>,
}

impl RealtimeSynthStats {
    pub fn new() -> RealtimeSynthStats {
        RealtimeSynthStats {
            voice_count: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Reads the statistics of an instance of RealtimeSynth in a usable way.
pub struct RealtimeSynthStatsReader {
    buffered_stats: BufferedRendererStatsReader,
    stats: RealtimeSynthStats,
}

impl RealtimeSynthStatsReader {
    pub(self) fn new(
        stats: RealtimeSynthStats,
        buffered_stats: BufferedRendererStatsReader,
    ) -> RealtimeSynthStatsReader {
        RealtimeSynthStatsReader {
            stats,
            buffered_stats,
        }
    }

    /// Returns the active voice count of all the MIDI channels.
    pub fn voice_count(&self) -> u64 {
        self.stats.voice_count.load(Ordering::Relaxed)
    }

    /// Returns the statistics of the buffered renderer used.
    ///
    /// See the BufferedRendererStatsReader documentation for more information.
    pub fn buffer(&self) -> &BufferedRendererStatsReader {
        &self.buffered_stats
    }
}

struct RealtimeSynthThreadSharedData {
    buffered_renderer: Arc<Mutex<BufferedRenderer>>,

    stream_control: crossbeam_channel::Sender<StreamCommand>,

    event_senders: RealtimeEventSender,
}

struct PreparedRealtimeChannels {
    channel_stats: Vec<xsynth_core::channel::VoiceChannelStatsReader>,
    senders: Vec<crossbeam_channel::Sender<ChannelEvent>>,
    command_senders: Vec<crossbeam_channel::Sender<Vec<f32>>>,
    join_handles: Vec<thread::JoinHandle<()>>,
    output_receiver: crossbeam_channel::Receiver<Vec<f32>>,
}

/// A realtime MIDI synthesizer using an audio device for output.
pub struct RealtimeSynth {
    data: Option<RealtimeSynthThreadSharedData>,
    stream_owner: Option<thread::JoinHandle<()>>,
    join_handles: Vec<thread::JoinHandle<()>>,

    stats: RealtimeSynthStats,

    stream_params: AudioStreamParams,
}

enum StreamCommand {
    Pause(crossbeam_channel::Sender<Result<(), PauseStreamError>>),
    Resume(crossbeam_channel::Sender<Result<(), PlayStreamError>>),
    Shutdown,
}

impl RealtimeSynth {
    /// Initializes a new realtime synthesizer using the default config and
    /// the default audio output.
    pub fn open_with_all_defaults() -> Result<Self, RealtimeSynthError> {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .ok_or(RealtimeSynthError::NoOutputDevice)?;
        if let Ok(name) = device.name() {
            println!("Output device: {name}");
        }

        let stream_config = device.default_output_config()?;

        RealtimeSynth::open(Default::default(), &device, stream_config)
    }

    /// Initializes as new realtime synthesizer using a given config and
    /// the default audio output.
    ///
    /// See the `XSynthRealtimeConfig` documentation for the available options.
    pub fn open_with_default_output(
        config: XSynthRealtimeConfig,
    ) -> Result<Self, RealtimeSynthError> {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .ok_or(RealtimeSynthError::NoOutputDevice)?;
        if let Ok(name) = device.name() {
            println!("Output device: {name}");
        }

        let stream_config = device.default_output_config()?;

        RealtimeSynth::open(config, &device, stream_config)
    }

    /// Initializes a new realtime synthesizer using a given config and a
    /// specified audio output device.
    ///
    /// See the `XSynthRealtimeConfig` documentation for the available options.
    /// See the `cpal` crate documentation for the `device` and `stream_config` parameters.
    pub fn open(
        config: XSynthRealtimeConfig,
        device: &Device,
        stream_config: SupportedStreamConfig,
    ) -> Result<Self, RealtimeSynthError> {
        let sample_rate = stream_config.sample_rate().0;
        let stream_params = AudioStreamParams::new(sample_rate, stream_config.channels().into());
        let channel_count = channel_count(config.format);

        let PreparedRealtimeChannels {
            channel_stats,
            senders,
            command_senders,
            join_handles,
            output_receiver,
        } = prepare_channels(
            channel_count,
            config.channel_init_options,
            stream_params,
            config.format,
            config.spectral_config,
        )?;

        let stats = RealtimeSynthStats::new();
        let render = build_render_pipe(
            stream_params,
            channel_count,
            command_senders,
            output_receiver,
            channel_stats,
            &stats,
        );
        let buffered = Arc::new(Mutex::new(
            BufferedRenderer::new(
                render,
                stream_params,
                calculate_render_size(sample_rate, config.render_window_ms),
            )
            .map_err(RealtimeSynthError::BufferedRendererThreadSpawn)?,
        ));
        let (stream_control, stream_owner) =
            spawn_stream_thread(device.clone(), stream_config, buffered.clone())?;

        let max_nps = Arc::new(ReadWriteAtomicU64::new(10000));

        Ok(Self {
            data: Some(RealtimeSynthThreadSharedData {
                buffered_renderer: buffered,

                event_senders: RealtimeEventSender::new(senders, max_nps, config.ignore_range)
                    .map_err(RealtimeSynthError::EventSenderInit)?,
                stream_control,
            }),
            stream_owner: Some(stream_owner),
            join_handles,

            stats,
            stream_params,
        })
    }

    /// Sends a SynthEvent to the realtime synthesizer.
    ///
    /// See the `SynthEvent` documentation for more information.
    pub fn send_event(&mut self, event: SynthEvent) {
        let data = self.data.as_mut().unwrap();
        data.event_senders.send_event(event);
    }

    /// Sends a u32 event to the realtime synthesizer.
    pub fn send_event_u32(&mut self, event: u32) {
        let data = self.data.as_mut().unwrap();
        data.event_senders.send_event_u32(event);
    }

    /// Returns a reference to the event sender of the realtime synthesizer.
    /// This can be used to clone the sender so it can be passed in threads.
    ///
    /// See the `RealtimeEventSender` documentation for more information
    /// on how to use.
    pub fn get_sender_ref(&self) -> &RealtimeEventSender {
        let data = self.data.as_ref().unwrap();
        &data.event_senders
    }

    /// Returns a mutable reference the event sender of the realtime synthesizer.
    /// This can be used to modify its parameters (eg. ignore range).
    /// Please note that each clone will store its own distinct parameters.
    ///
    /// See the `RealtimeEventSender` documentation for more information
    /// on how to use.
    pub fn get_sender_mut(&mut self) -> &mut RealtimeEventSender {
        let data = self.data.as_mut().unwrap();
        &mut data.event_senders
    }

    /// Returns the statistics reader of the realtime synthesizer.
    ///
    /// See the `RealtimeSynthStatsReader` documentation for more information
    /// on how to use.
    pub fn get_stats(&self) -> RealtimeSynthStatsReader {
        let data = self.data.as_ref().unwrap();
        let buffered_stats = data.buffered_renderer.lock().unwrap().get_buffer_stats();

        RealtimeSynthStatsReader::new(self.stats.clone(), buffered_stats)
    }

    /// Returns the stream parameters of the audio output device.
    pub fn stream_params(&self) -> AudioStreamParams {
        self.stream_params
    }

    /// Pauses the playback of the audio output device.
    pub fn pause(&mut self) -> Result<(), PauseStreamError> {
        let data = self.data.as_ref().unwrap();
        let (sender, receiver) = bounded(1);
        if data
            .stream_control
            .send(StreamCommand::Pause(sender))
            .is_err()
        {
            return Err(PauseStreamError::DeviceNotAvailable);
        }
        receiver
            .recv()
            .unwrap_or(Err(PauseStreamError::DeviceNotAvailable))
    }

    /// Resumes the playback of the audio output device.
    pub fn resume(&mut self) -> Result<(), PlayStreamError> {
        let data = self.data.as_ref().unwrap();
        let (sender, receiver) = bounded(1);
        if data
            .stream_control
            .send(StreamCommand::Resume(sender))
            .is_err()
        {
            return Err(PlayStreamError::DeviceNotAvailable);
        }
        receiver
            .recv()
            .unwrap_or(Err(PlayStreamError::DeviceNotAvailable))
    }

    /// Changes the length of the buffer reader.
    pub fn set_buffer(&self, render_window_ms: f64) {
        let data = self.data.as_ref().unwrap();
        let sample_rate = self.stream_params.sample_rate;
        let size = calculate_render_size(sample_rate, render_window_ms);
        data.buffered_renderer.lock().unwrap().set_render_size(size);
    }
}

fn channel_count(format: SynthFormat) -> u32 {
    match format {
        SynthFormat::Midi => 16,
        SynthFormat::Custom { channels } => channels,
    }
}

fn prepare_channels(
    channel_count: u32,
    init_options: xsynth_core::channel::ChannelInitOptions,
    stream_params: AudioStreamParams,
    format: SynthFormat,
    spectral_config: Option<xsynth_core::spectral::SpectralConfig>,
) -> Result<PreparedRealtimeChannels, RealtimeSynthError> {
    let (output_sender, output_receiver) = bounded::<Vec<f32>>(channel_count as usize);

    let mut channel_stats = Vec::new();
    let mut senders = Vec::new();
    let mut command_senders = Vec::new();
    let mut join_handles = Vec::new();

    for _ in 0..channel_count {
        let channel = VoiceChannel::new(init_options, stream_params, spectral_config.clone());
        channel_stats.push(channel.get_channel_stats());

        let (event_sender, event_receiver) = unbounded();
        senders.push(event_sender);

        let (command_sender, command_receiver) = bounded::<Vec<f32>>(1);
        command_senders.push(command_sender);

        let output_sender = output_sender.clone();
        let join_handle =
            spawn_channel_thread(channel, event_receiver, command_receiver, output_sender)?;
        join_handles.push(join_handle);
    }

    if format == SynthFormat::Midi {
        let _ = senders[9].send(ChannelEvent::Config(ChannelConfigEvent::SetPercussionMode(
            true,
        )));
    }

    Ok(PreparedRealtimeChannels {
        channel_stats,
        senders,
        command_senders,
        join_handles,
        output_receiver,
    })
}

fn spawn_channel_thread(
    mut channel: VoiceChannel,
    event_receiver: crossbeam_channel::Receiver<ChannelEvent>,
    command_receiver: crossbeam_channel::Receiver<Vec<f32>>,
    output_sender: crossbeam_channel::Sender<Vec<f32>>,
) -> Result<thread::JoinHandle<()>, RealtimeSynthError> {
    thread::Builder::new()
        .name("xsynth_channel_handler".to_string())
        .spawn(move || loop {
            channel.push_events_iter(event_receiver.try_iter());
            let mut vec = match command_receiver.recv() {
                Ok(vec) => vec,
                Err(_) => break,
            };
            channel.push_events_iter(event_receiver.try_iter());
            channel.read_samples(&mut vec);
            if output_sender.send(vec).is_err() {
                break;
            }
        })
        .map_err(RealtimeSynthError::ChannelThreadSpawn)
}

fn build_render_pipe(
    stream_params: AudioStreamParams,
    channel_count: u32,
    command_senders: Vec<crossbeam_channel::Sender<Vec<f32>>>,
    output_receiver: crossbeam_channel::Receiver<Vec<f32>>,
    channel_stats: Vec<xsynth_core::channel::VoiceChannelStatsReader>,
    stats: &RealtimeSynthStats,
) -> FunctionAudioPipe<impl FnMut(&mut [f32]) + Send> {
    let mut vec_cache: VecDeque<Vec<f32>> = VecDeque::new();
    for _ in 0..channel_count {
        vec_cache.push_front(Vec::new());
    }

    let total_voice_count = stats.voice_count.clone();

    FunctionAudioPipe::new(stream_params, move |out| {
        for sender in &command_senders {
            let mut buf = vec_cache.pop_front().unwrap();
            prepapre_cache_vec(&mut buf, out.len(), 0.0);
            sender.send(buf).unwrap();
        }

        for _ in 0..channel_count {
            let buf = output_receiver.recv().unwrap();
            sum_simd(&buf, out);
            vec_cache.push_front(buf);
        }

        let total_voices = channel_stats.iter().map(|c| c.voice_count()).sum();
        total_voice_count.store(total_voices, Ordering::SeqCst);
    })
}

fn build_output_stream(
    device: &Device,
    stream_config: SupportedStreamConfig,
    buffered: Arc<Mutex<BufferedRenderer>>,
) -> Result<Stream, RealtimeSynthError> {
    match stream_config.sample_format() {
        cpal::SampleFormat::F32 => build_output_stream_for::<f32>(device, stream_config, buffered),
        cpal::SampleFormat::I16 => build_output_stream_for::<i16>(device, stream_config, buffered),
        cpal::SampleFormat::U16 => build_output_stream_for::<u16>(device, stream_config, buffered),
        _ => Err(RealtimeSynthError::UnsupportedSampleFormat(
            stream_config.sample_format(),
        )),
    }
}

fn build_output_stream_for<T: SizedSample + ConvertSample>(
    device: &Device,
    stream_config: SupportedStreamConfig,
    buffered: Arc<Mutex<BufferedRenderer>>,
) -> Result<Stream, RealtimeSynthError> {
    let err_fn = |err| eprintln!("an error occurred on stream: {err}");
    let mut output_vec = Vec::new();
    let mut limiter = VolumeLimiter::new(stream_config.channels());

    Ok(device.build_output_stream(
        &stream_config.into(),
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            output_vec.resize(data.len(), 0.0);
            buffered.lock().unwrap().read(&mut output_vec);
            for (i, s) in limiter.limit_iter(output_vec.drain(0..)).enumerate() {
                data[i] = ConvertSample::from_f32(s);
            }
        },
        err_fn,
        None,
    )?)
}

fn spawn_stream_thread(
    device: Device,
    stream_config: SupportedStreamConfig,
    buffered: Arc<Mutex<BufferedRenderer>>,
) -> Result<
    (
        crossbeam_channel::Sender<StreamCommand>,
        thread::JoinHandle<()>,
    ),
    RealtimeSynthError,
> {
    let (command_sender, command_receiver) = unbounded();
    let (ready_sender, ready_receiver) = bounded(1);
    let join_handle = thread::Builder::new()
        .name("xsynth_stream_owner".to_string())
        .spawn(move || {
            let stream = match build_output_stream(&device, stream_config, buffered) {
                Ok(stream) => stream,
                Err(err) => {
                    let _ = ready_sender.send(Err(err));
                    return;
                }
            };
            if let Err(err) = stream.play() {
                let _ = ready_sender.send(Err(err.into()));
                return;
            }
            if ready_sender.send(Ok(())).is_err() {
                return;
            }

            while let Ok(command) = command_receiver.recv() {
                match command {
                    StreamCommand::Pause(reply) => {
                        let _ = reply.send(stream.pause());
                    }
                    StreamCommand::Resume(reply) => {
                        let _ = reply.send(stream.play());
                    }
                    StreamCommand::Shutdown => break,
                }
            }
        })
        .map_err(RealtimeSynthError::StreamThreadSpawn)?;

    match ready_receiver.recv() {
        Ok(Ok(())) => Ok((command_sender, join_handle)),
        Ok(Err(err)) => {
            let _ = join_handle.join();
            Err(err)
        }
        Err(_) => {
            let _ = join_handle.join();
            Err(RealtimeSynthError::StreamThreadInit)
        }
    }
}

impl Drop for RealtimeSynth {
    fn drop(&mut self) {
        let data = self.data.take().unwrap();
        let _ = data.stream_control.send(StreamCommand::Shutdown);
        drop(data);
        if let Some(handle) = self.stream_owner.take() {
            if handle.join().is_err() {
                eprintln!("xsynth-realtime: stream owner thread panicked during shutdown");
            }
        }
        for handle in self.join_handles.drain(..) {
            if handle.join().is_err() {
                eprintln!("xsynth-realtime: channel handler thread panicked during shutdown");
            }
        }
    }
}

trait ConvertSample: SizedSample {
    fn from_f32(s: f32) -> Self;
}

impl ConvertSample for f32 {
    fn from_f32(s: f32) -> Self {
        s
    }
}

impl ConvertSample for i16 {
    fn from_f32(s: f32) -> Self {
        (s * i16::MAX as f32) as i16
    }
}

impl ConvertSample for u16 {
    fn from_f32(s: f32) -> Self {
        ((s * u16::MAX as f32) as i32 + i16::MIN as i32) as u16
    }
}

fn calculate_render_size(sample_rate: u32, buffer_ms: f64) -> usize {
    (sample_rate as f64 * buffer_ms / 1000.0) as usize
}

#[cfg(test)]
mod tests {
    use super::RealtimeSynth;

    #[test]
    fn realtime_synth_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RealtimeSynth>();
    }
}
