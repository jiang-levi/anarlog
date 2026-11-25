use std::path::Path;

use base64::{Engine, engine::general_purpose::STANDARD};
use owhisper_interface::{
    ListenParams,
    batch::{Alternatives, Channel, Response, Results, Word},
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ZoomScribeAdapter, config, endpoint, speaker};
use crate::{
    Error,
    adapter::{BatchFuture, BatchSttAdapter, ClientWithMiddleware, http::ensure_success},
};

// Application cap for the complete JSON body, not a server acceptance guarantee.
// Match OpenWhispr; historical probes observed a 65–72 MB gateway boundary.
const MAX_REQUEST_BYTES: usize = 50 * 1024 * 1024;

impl BatchSttAdapter for ZoomScribeAdapter {
    fn accepted_audio_extensions(&self) -> Option<&'static [&'static str]> {
        Some(&["wav", "mp3", "m4a", "mp4", "webm"])
    }
    fn compression_retry_duration(&self) -> Option<std::time::Duration> {
        Some(std::time::Duration::from_secs(10 * 60))
    }
    fn preserve_upload_channels(&self) -> bool {
        true
    }

    fn normalize_batch_response(&self, response: &mut Response) {
        normalize_text(response);
    }

    fn provider_name(&self) -> &'static str {
        "zoom_scribe"
    }

    fn is_supported_languages(
        &self,
        languages: &[anlg_language::Language],
        model: Option<&str>,
    ) -> bool {
        Self::language_support_batch(languages, model).is_supported()
    }

    fn transcribe_file<'a, P: AsRef<Path> + Send + 'a>(
        &'a self,
        client: &'a ClientWithMiddleware,
        base: &'a str,
        key: &'a str,
        params: &'a ListenParams,
        file_path: P,
    ) -> BatchFuture<'a> {
        Box::pin(async move {
            let config = config(params, false)?;
            if !(1..=2).contains(&params.channels) {
                return Err(Error::provider_configuration(
                    "zoom_scribe",
                    "Fast supports mono or stereo audio",
                ));
            }
            let path = file_path.as_ref();
            let file = tokio::fs::File::open(path)
                .await
                .map_err(|e| Error::AudioProcessing(e.to_string()))?;
            let length = file
                .metadata()
                .await
                .map_err(|e| Error::AudioProcessing(e.to_string()))?
                .len();
            let max_audio_bytes = MAX_REQUEST_BYTES / 4 * 3;
            if length > max_audio_bytes as u64 {
                return Err(request_too_large());
            }
            use tokio::io::AsyncReadExt;
            let mut audio = Vec::new();
            file.take(max_audio_bytes as u64 + 1)
                .read_to_end(&mut audio)
                .await
                .map_err(|e| Error::AudioProcessing(e.to_string()))?;
            if audio.len() > max_audio_bytes {
                return Err(request_too_large());
            }
            let mime = audio_mime(&audio).ok_or_else(|| {
                Error::AudioProcessing(
                    "Zoom Scribe requires WAV, MP3, M4A, MP4, or WebM audio".into(),
                )
            })?;
            let body = encoded_body(&audio, mime, config, MAX_REQUEST_BYTES)?;
            let response = client
                .post(endpoint(base, "transcribe")?)
                .bearer_auth(key)
                .header("Content-Type", "application/json")
                .header("x-zm-trackingid", super::tracking_id())
                .body(body)
                .send()
                .await?;
            if response.status() == reqwest::StatusCode::PAYLOAD_TOO_LARGE {
                return Err(Error::UnexpectedStatus {
                    status: response.status(),
                    body: "The audio is too large for a single Zoom Scribe request; compress it and retry."
                        .into(),
                });
            }
            let response: FastResponse = ensure_success(response).await?.json().await?;
            convert(response, params)
        })
    }
}

pub(super) fn encoded_body(
    audio: &[u8],
    mime: &str,
    config: Value,
    limit: usize,
) -> Result<String, Error> {
    let body = json!({
        "file": format!("data:{mime};base64,{}", STANDARD.encode(audio)),
        "config": config,
        "outputs": {"text_polishing": {"enabled": false}},
    })
    .to_string();
    if body.len() > limit {
        return Err(Error::RequestTooLarge { limit_bytes: limit });
    }
    Ok(body)
}

fn request_too_large() -> Error {
    Error::RequestTooLarge {
        limit_bytes: MAX_REQUEST_BYTES,
    }
}

fn audio_mime(audio: &[u8]) -> Option<&'static str> {
    if audio.starts_with(b"RIFF") && audio.get(8..12) == Some(b"WAVE") {
        Some("audio/wav")
    } else if audio.get(4..8) == Some(b"ftyp") {
        Some("audio/mp4")
    } else if audio.starts_with(b"\x1a\x45\xdf\xa3") {
        Some("audio/webm")
    } else if audio.starts_with(b"ID3")
        || (audio.len() >= 2 && audio[0] == 0xff && audio[1] & 0xe0 == 0xe0)
    {
        Some("audio/mpeg")
    } else {
        None
    }
}

#[derive(Deserialize)]
struct FastResponse {
    result: FastResult,
    request_id: Option<String>,
    duration_sec: Option<f64>,
    model: Option<String>,
}

#[derive(Deserialize)]
struct FastResult {
    #[serde(default)]
    text_display: String,
    #[serde(default)]
    segments: Vec<FastSegment>,
}

#[derive(Deserialize)]
struct FastSegment {
    start: f64,
    end: f64,
    text: String,
    #[serde(default)]
    channel: usize,
    #[serde(default)]
    speaker: Value,
    #[serde(default)]
    words: Vec<FastWord>,
}

#[derive(Deserialize)]
struct FastWord {
    word: String,
    start: f64,
    end: f64,
}

fn convert(response: FastResponse, params: &ListenParams) -> Result<Response, Error> {
    let channels = params.channels;
    let selected_model = super::resolve_model(params.model.as_deref());
    let mut alternatives: Vec<_> = (0..channels)
        .map(|_| Alternatives {
            transcript: String::new(),
            confidence: 1.0,
            words: Vec::new(),
        })
        .collect();
    for segment in response.result.segments {
        let Some(alternative) = alternatives.get_mut(segment.channel) else {
            return Err(Error::provider_failure(
                "Zoom Scribe returned an invalid audio channel",
                false,
            ));
        };
        if !alternative.transcript.is_empty() {
            alternative.transcript.push(' ');
        }
        alternative.transcript.push_str(&segment.text);
        let speaker = speaker(&segment.speaker).and_then(|id| usize::try_from(id).ok());
        let words = if segment.words.is_empty() && !segment.text.is_empty() {
            vec![FastWord {
                word: segment.text,
                start: segment.start,
                end: segment.end,
            }]
        } else {
            segment.words
        };
        alternative.words.extend(words.into_iter().map(|word| Word {
            punctuated_word: Some(word.word.clone()),
            word: word.word,
            start: word.start,
            end: word.end,
            confidence: 1.0,
            channel: if channels == 1 && params.num_speakers != Some(1) {
                crate::adapter::MIXED_CAPTURE_CHANNEL
            } else {
                segment.channel as i32
            },
            speaker,
        }));
    }
    if channels == 1 && !response.result.text_display.is_empty() {
        alternatives[0].transcript = response.result.text_display;
    }
    let has_timing = alternatives
        .iter()
        .any(|alternative| !alternative.words.is_empty());
    let mut result = Response {
        metadata: json!({
            "provider": "zoom_scribe", "request_id": response.request_id,
            "duration": response.duration_sec, "model": selected_model, "provider_model": response.model,
            "timing_source": if has_timing { "provider" } else { "synthetic_text" },
        }),
        results: Results {
            channels: alternatives
                .into_iter()
                .map(|alternative| Channel {
                    alternatives: vec![alternative],
                })
                .collect(),
        },
    };
    normalize_text(&mut result);
    Ok(result)
}

fn normalize_text(response: &mut Response) {
    let model = response.metadata["model"].as_str().unwrap_or("");
    for channel in &mut response.results.channels {
        for alternative in &mut channel.alternatives {
            alternative.transcript = super::clean_text(&alternative.transcript, model);
            for word in &mut alternative.words {
                word.word = super::clean_text(&word.word, model);
                if let Some(text) = &mut word.punctuated_word {
                    *text = super::clean_text(text, model);
                }
            }
        }
    }
}
