use super::ChannelInitOptions;
use crate::voice::{ReleaseType, Voice};
use simdeez::Simd;
use std::{
    collections::VecDeque,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

struct GroupSpectralVoice {
    pub id: usize,
    pub voice: Box<dyn Voice>,
}

impl Deref for GroupSpectralVoice {
    type Target = Box<dyn Voice>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.voice
    }
}

impl DerefMut for GroupSpectralVoice {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Box<dyn Voice> {
        &mut self.voice
    }
}

impl Debug for GroupSpectralVoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SpectralVoiceGroup")
            .field(&self.id)
            .field(&self.voice.velocity())
            .field(&self.voice.is_killed())
            .finish()
    }
}

pub struct SpectralVoiceBuffer<T: Simd> {
    options: ChannelInitOptions,
    id_counter: usize,
    buffer: VecDeque<GroupSpectralVoice>,
    damper_held: bool,
    held_by_damper: Vec<usize>,
    _marker: std::marker::PhantomData<T>, // Tells the compiler T is evaluated inside inner voice references
}

impl<T: Simd> SpectralVoiceBuffer<T> {
    pub fn new(options: ChannelInitOptions) -> Self {
        SpectralVoiceBuffer {
            options,
            id_counter: 0,
            buffer: VecDeque::new(),
            damper_held: false,
            held_by_damper: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    fn get_id(&mut self) -> usize {
        self.id_counter = self.id_counter.wrapping_add(1);
        self.id_counter
    }

    /// Pops the quietest spectral voice group. Multiple voices can belong to the same
    /// polyphonic group if spawned by a singular event (e.g., overlapping multi-samples).
    fn pop_quietest_voice_group(&mut self, ignored_id: usize) {
        if self.buffer.is_empty() {
            return;
        }

        let mut quietest = u8::MAX;
        let mut quietest_index = 0;
        let mut quietest_id = 0;
        let mut count = 0;

        for i in 0..self.buffer.len() {
            let voice = &self.buffer[i];
            if voice.id == ignored_id || voice.is_killed() {
                continue;
            }
            let vel = voice.velocity();
            if quietest_id == voice.id {
                count += 1;
            } else if vel < quietest || i == 0 {
                quietest = vel;
                quietest_index = i;
                quietest_id = voice.id;
                count = 1;
            }
        }

        if count > 0 {
            if self.options.fade_out_killing {
                // For spectral audio, fading out prevents sudden windowed phase discontinuities
                for i in quietest_index..(quietest_index + count) {
                    self.kill_voice_fade_out(i);
                }
            } else {
                self.buffer.drain(quietest_index..(quietest_index + count));
            }

            if let Some(index) = self.held_by_damper.iter().position(|&x| x == quietest_id) {
                self.held_by_damper.remove(index);
            }
        }
    }

    fn kill_voice_fade_out(&mut self, index: usize) {
        self.buffer[index]
            .deref_mut()
            .signal_release(ReleaseType::Kill);
    }

    pub fn kill_all_voices(&mut self) {
        if self.options.fade_out_killing {
            for i in 0..self.buffer.len() {
                self.kill_voice_fade_out(i);
            }
            self.id_counter = 0;
        } else {
            self.buffer.clear();
        }
    }

    pub fn kill_by_exclusive_class(&mut self, class: u8) {
        for voice in &mut self.buffer {
            if voice.exclusive_class() == Some(class) {
                voice.signal_release(ReleaseType::Kill);
            }
        }
    }

    fn get_active_count(&self) -> usize {
        let mut active = 0;
        for i in 0..self.buffer.len() {
            if !self.buffer[i].deref().is_killed() {
                active += 1;
            }
        }
        active
    }

    /// Pushes a new collection of running frequency domain generators into the track list.
    pub fn push_voices(
        &mut self,
        voices: impl Iterator<Item = Box<dyn Voice>>,
        max_voices: Option<usize>,
    ) {
        let mut len = 0;
        let id = self.get_id();

        for voice in voices {
            self.buffer.push_back(GroupSpectralVoice { id, voice });
            len += 1;
        }

        if let Some(max_voices) = max_voices {
            if len > max_voices {
                self.pop_quietest_voice_group(id);
            } else if self.options.fade_out_killing {
                while self.get_active_count() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            } else {
                while self.buffer.len() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            }
        }
    }

    /// Triggers standard release decays over active target spectral bands.
    pub fn release_next_voice(&mut self) -> Option<u8> {
        if !self.damper_held {
            let mut id: Option<usize> = None;
            let mut vel = None;

            for voice in self.buffer.iter_mut() {
                if voice.is_releasing() {
                    continue;
                }

                if id.is_none() {
                    id = Some(voice.id);
                    vel = Some(voice.velocity())
                }

                if id != Some(voice.id) {
                    break;
                }

                voice.signal_release(ReleaseType::Standard);
            }

            vel
        } else {
            // Damper pedal behavior: hold the spectral slots back from starting their release envelopes
            for voice in self.buffer.iter_mut() {
                if voice.is_releasing() {
                    continue;
                }

                if self.held_by_damper.contains(&voice.id) {
                    continue;
                }

                self.held_by_damper.push(voice.id);
                break;
            }

            None
        }
    }

    pub fn remove_ended_voices(&mut self) {
        self.buffer.retain(|voice_group| !voice_group.ended());
    }
    //  pub fn iter_voices<'a>(&'a self) -> impl Iterator<Item = &Box<dyn Voice>> + 'a {
    //     self.buffer.iter().map(|group| &group.voice)
    //  }

    pub fn iter_voices_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Voice>> {
        self.buffer.iter_mut().map(|group| &mut group.voice)
    }

    pub fn has_voices(&self) -> bool {
        !self.buffer.is_empty()
    }

    pub fn voice_count(&self) -> usize {
        self.buffer.len()
    }

    pub fn set_damper(&mut self, damper: bool) {
        if self.damper_held && !damper {
            for voice in self.buffer.iter_mut() {
                if self.held_by_damper.contains(&voice.id) {
                    voice.signal_release(ReleaseType::Standard);
                }
            }
            self.held_by_damper.clear();
        }
        self.damper_held = damper;
    }
}
