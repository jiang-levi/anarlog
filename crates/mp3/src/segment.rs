use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anlg_audio_utils::Source;

use crate::encoder::f32_to_i16;
use crate::{Error, MonoStreamEncoder, StereoStreamEncoder};

const ENCODE_FRAMES: usize = 4096;

/// Decodes any supported audio file, downmixes it to mono, and writes it back out
/// as MP3 segments of at most `segment_duration`. Providers that cap upload size
/// or time out on long audio get one request per segment.
pub fn encode_mono_segments(
    source_path: &Path,
    output_dir: &Path,
    segment_duration: Duration,
) -> Result<Vec<PathBuf>, Error> {
    encode_segments(source_path, output_dir, segment_duration, 1)
}

/// Encode mono or preserve two interleaved input channels in stereo segments.
/// Stereo requests reject mismatched sources instead of silently downmixing.
pub fn encode_segments(
    source_path: &Path,
    output_dir: &Path,
    segment_duration: Duration,
    output_channels: u16,
) -> Result<Vec<PathBuf>, Error> {
    let source = anlg_audio_utils::source_from_path(source_path)?;
    let sample_rate: u32 = source.sample_rate().into();
    let channels = usize::from(u16::from(source.channels())).max(1);
    if !(1..=2).contains(&output_channels) || (output_channels == 2 && channels != 2) {
        return Err(Error::UnsupportedChannelCount(channels as u16));
    }
    let mut frames: Box<dyn Iterator<Item = (f32, f32)>> = if output_channels == 1 {
        Box::new(anlg_audio_utils::mono_frames(source, channels).map(|sample| (sample, 0.0)))
    } else {
        let mut source = source;
        Box::new(std::iter::from_fn(move || {
            Some((source.next()?, source.next()?))
        }))
    };
    let frames_per_segment = frames_per_segment(sample_rate, segment_duration)?;

    let mut paths: Vec<PathBuf> = Vec::new();
    let mut current: Option<SegmentEncoder> = None;

    for (left, right) in &mut frames {
        if current
            .as_ref()
            .is_some_and(|segment| segment.frames >= frames_per_segment)
        {
            current
                .take()
                .expect("segment was just observed")
                .finish()?;
        }

        if current.is_none() {
            let path = output_dir.join(format!("segment-{:04}.mp3", paths.len()));
            current = Some(SegmentEncoder::create(&path, sample_rate, output_channels)?);
            paths.push(path);
        }

        current
            .as_mut()
            .expect("segment was just created")
            .push(left, right)?;
    }

    if let Some(segment) = current {
        segment.finish()?;
    }

    Ok(paths)
}

fn frames_per_segment(sample_rate: u32, segment_duration: Duration) -> Result<usize, Error> {
    let frames = (u128::from(sample_rate) * segment_duration.as_millis()) / 1000;
    usize::try_from(frames)
        .ok()
        .filter(|frames| *frames > 0)
        .ok_or(Error::InvalidSegmentDuration(segment_duration))
}

enum Encoder {
    Mono(MonoStreamEncoder),
    Stereo(StereoStreamEncoder),
}

struct SegmentEncoder {
    encoder: Encoder,
    writer: BufWriter<File>,
    pcm: Vec<i16>,
    right: Vec<i16>,
    encoded: Vec<u8>,
    frames: usize,
}

impl SegmentEncoder {
    fn create(path: &Path, sample_rate: u32, channels: u16) -> Result<Self, Error> {
        Ok(Self {
            encoder: if channels == 2 {
                Encoder::Stereo(StereoStreamEncoder::new(sample_rate)?)
            } else {
                Encoder::Mono(MonoStreamEncoder::new(sample_rate)?)
            },
            writer: BufWriter::new(File::create(path)?),
            pcm: Vec::with_capacity(ENCODE_FRAMES),
            right: Vec::with_capacity(ENCODE_FRAMES),
            encoded: Vec::new(),
            frames: 0,
        })
    }

    fn push(&mut self, sample: f32, right: f32) -> Result<(), Error> {
        self.pcm.push(f32_to_i16(sample));
        if matches!(self.encoder, Encoder::Stereo(_)) {
            self.right.push(f32_to_i16(right));
        }
        self.frames += 1;

        if self.pcm.len() >= ENCODE_FRAMES {
            self.drain()?;
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), Error> {
        self.encoded.clear();
        match &mut self.encoder {
            Encoder::Mono(encoder) => encoder.encode_i16(&self.pcm, &mut self.encoded)?,
            Encoder::Stereo(encoder) => {
                encoder.encode_i16(&self.pcm, &self.right, &mut self.encoded)?
            }
        }
        self.writer.write_all(&self.encoded)?;
        self.pcm.clear();
        self.right.clear();
        Ok(())
    }

    fn finish(mut self) -> Result<(), Error> {
        self.drain()?;
        self.encoded.clear();
        match &mut self.encoder {
            Encoder::Mono(encoder) => encoder.flush(&mut self.encoded)?,
            Encoder::Stereo(encoder) => encoder.flush(&mut self.encoded)?,
        }
        self.writer.write_all(&self.encoded)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_per_segment_scales_with_sample_rate() {
        assert_eq!(
            frames_per_segment(16_000, Duration::from_secs(10)).unwrap(),
            160_000
        );
        assert!(frames_per_segment(16_000, Duration::ZERO).is_err());
    }
}

#[cfg(test)]
mod stereo_tests {
    use super::*;
    #[test]
    fn stereo_segments_preserve_distinct_signals_and_duration() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("stereo.wav");
        let mut writer = hound::WavWriter::create(
            &source,
            hound::WavSpec {
                channels: 2,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for i in 0..32000 {
            writer
                .write_sample(
                    (8000.0 * (i as f32 * std::f32::consts::TAU * 440.0 / 16000.0).sin()) as i16,
                )
                .unwrap();
            writer
                .write_sample(
                    (8000.0 * (i as f32 * std::f32::consts::TAU * 1100.0 / 16000.0).sin()) as i16,
                )
                .unwrap();
        }
        writer.finalize().unwrap();
        let segments = encode_segments(&source, dir.path(), Duration::from_secs(1), 2).unwrap();
        assert_eq!(segments.len(), 2);
        for path in segments {
            let decoded = anlg_audio_utils::source_from_path(&path).unwrap();
            assert_eq!(u16::from(decoded.channels()), 2);
            let samples: Vec<f32> = decoded.collect();
            assert!(samples.len() >= 32000);
            let (mut difference, mut energy) = (0.0, 0.0);
            for frame in samples.chunks_exact(2) {
                difference += (frame[0] - frame[1]).abs();
                energy += frame[0].abs() + frame[1].abs();
            }
            assert!(difference > energy * 0.3, "stereo channels were collapsed");
        }
    }
}
