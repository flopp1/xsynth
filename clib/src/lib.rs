#![allow(clippy::missing_safety_doc)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::too_long_first_doc_paragraph)]
#![allow(clippy::doc_overindented_list_items)]

pub mod consts;
pub mod group;
pub mod handles;
pub mod realtime;
pub mod soundfont;
mod utils;

use consts::*;
use pkg_version::*;

const XSYNTH_VERSION: u32 =
    pkg_version_patch!() | (pkg_version_minor!() << 8) | (pkg_version_major!() << 16);

/// Returns the version of XSynth
///
/// --Returns--
/// The XSynth version. For example, 0x010102 (hex), would be version 1.1.2
#[no_mangle]
pub extern "C" fn XSynth_GetVersion() -> u32 {
    XSYNTH_VERSION
}

/// Parameters of the output audio
/// - sample_rate: Audio sample rate
/// - audio_channels: Number of audio channels
///   Supported: XSYNTH_AUDIO_CHANNELS_MONO (mono),
///   XSYNTH_AUDIO_CHANNELS_STEREO (stereo)
#[repr(C)]
pub struct XSynth_StreamParams {
    pub sample_rate: u32,
    pub audio_channels: u16,
}

#[repr(C)]
pub struct XSynth_SpectralConfig {
    pub fft_size: usize,
    pub fft_step: usize,
    /// Maximum number of simultaneous spectral voices.
    /// If set to 0, spectral voice allocation is unlimited.
    pub max_voices: usize,
    pub enable_phase_fade_out: bool,
}

/// Generates the default values for the XSynth_StreamParams struct
/// Default values are:
/// - sample_rate = 44.1kHz
/// - audio_channels = XSYNTH_AUDIO_CHANNELS_STEREO
#[no_mangle]
pub extern "C" fn XSynth_GenDefault_StreamParams() -> XSynth_StreamParams {
    XSynth_StreamParams {
        sample_rate: 44100,
        audio_channels: XSYNTH_AUDIO_CHANNELS_STEREO,
    }
}

/// A helper struct to specify a range of bytes.
/// - start: The start of the range
/// - end: The end of the range
#[repr(C)]
pub struct XSynth_ByteRange {
    pub start: u8,
    pub end: u8,
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use crate::{
        group::{
            XSynth_ChannelGroup_Drop, XSynth_ChannelGroup_GetStreamParams,
            XSynth_ChannelGroup_ReadSamples, XSynth_ChannelGroup_SetSoundfonts,
            XSynth_ChannelGroup_VoiceCount,
        },
        handles::{XSynth_ChannelGroup, XSynth_RealtimeSynth, XSynth_Soundfont},
        realtime::{
            XSynth_Realtime_Drop, XSynth_Realtime_GetStats, XSynth_Realtime_GetStreamParams,
            XSynth_Realtime_Reset, XSynth_Realtime_SetSoundfonts,
        },
        soundfont::{
            XSynth_GenDefault_SoundfontOptions, XSynth_SoundfontOptions, XSynth_Soundfont_LoadNew,
            XSynth_Soundfont_Remove,
        },
        XSynth_GenDefault_StreamParams,
    };

    #[test]
    fn null_channel_group_handle_operations_are_noops() {
        let handle = XSynth_ChannelGroup {
            group: ptr::null_mut(),
        };

        unsafe {
            XSynth_ChannelGroup_SetSoundfonts(handle, ptr::null(), 1);
            XSynth_ChannelGroup_ReadSamples(handle, ptr::null_mut(), 0);
        }

        let params = XSynth_ChannelGroup_GetStreamParams(handle);
        let default_params = XSynth_GenDefault_StreamParams();
        assert_eq!(params.sample_rate, default_params.sample_rate);
        assert_eq!(params.audio_channels, default_params.audio_channels);
        assert_eq!(XSynth_ChannelGroup_VoiceCount(handle), 0);

        XSynth_ChannelGroup_Drop(handle);
    }

    #[test]
    fn null_realtime_handle_operations_are_noops() {
        let handle = XSynth_RealtimeSynth {
            synth: ptr::null_mut(),
        };

        unsafe {
            XSynth_Realtime_SetSoundfonts(handle, ptr::null(), 1);
        }
        XSynth_Realtime_Reset(handle);

        let params = XSynth_Realtime_GetStreamParams(handle);
        assert_eq!(params.sample_rate, 0);
        assert_eq!(params.audio_channels, 0);

        let stats = XSynth_Realtime_GetStats(handle);
        assert_eq!(stats.voice_count, 0);
        assert_eq!(stats.buffer, 0);
        assert_eq!(stats.render_time, 0.0);

        XSynth_Realtime_Drop(handle);
    }

    #[test]
    fn soundfont_load_rejects_null_path() {
        let handle =
            unsafe { XSynth_Soundfont_LoadNew(ptr::null(), XSynth_GenDefault_SoundfontOptions()) };
        assert!(handle.soundfont.is_null());

        XSynth_Soundfont_Remove(handle);
        XSynth_Soundfont_Remove(XSynth_Soundfont {
            soundfont: ptr::null_mut(),
        });
    }

    #[test]
    fn soundfont_load_rejects_invalid_envelope_values() {
        let mut options: XSynth_SoundfontOptions = XSynth_GenDefault_SoundfontOptions();
        options.vol_envelope_options.attack_curve = 255;
        let path = std::ffi::CString::new("missing.sf2").unwrap();

        let handle = unsafe { XSynth_Soundfont_LoadNew(path.as_ptr(), options) };
        assert!(handle.soundfont.is_null());
    }
}
