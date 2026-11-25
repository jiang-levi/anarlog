use anlg_ws_client::client::Message;
use owhisper_interface::{ListenParams, stream::StreamResponse};
use serde_json::{Value, json};

use super::{Session, ZoomScribeAdapter, config, endpoint, speaker};
use crate::adapter::{RealtimeSttAdapter, segment};

impl RealtimeSttAdapter for ZoomScribeAdapter {
    fn validate_configuration(
        &self,
        base: &str,
        params: &ListenParams,
        channels: u8,
    ) -> Result<(), crate::Error> {
        endpoint(base, "live")?;
        config(params, true)?;
        if channels != 1 {
            return Err(crate::Error::provider_configuration(
                "zoom_scribe",
                "Live requires one mono stream per connection",
            ));
        }
        Ok(())
    }

    fn fork_session(&self) -> Self {
        Self::default()
    }

    fn provider_name(&self) -> &'static str {
        "zoom_scribe"
    }

    fn initial_response_type(&self) -> Option<&'static str> {
        Some("session.updated")
    }

    fn initial_response_ignored_types(&self) -> &'static [&'static str] {
        &["session.created"]
    }

    fn websocket_subprotocol(&self) -> Option<&'static str> {
        Some("live-asr")
    }

    fn required_sample_rate(&self) -> Option<u32> {
        Some(16_000)
    }

    fn is_supported_languages(
        &self,
        languages: &[anlg_language::Language],
        model: Option<&str>,
    ) -> bool {
        Self::language_support_live(languages, model).is_supported()
    }

    fn supports_native_multichannel(&self) -> bool {
        false
    }

    fn build_ws_url(&self, base: &str, _: &ListenParams, _: u8) -> url::Url {
        let mut url = endpoint(base, "live").expect("invalid Zoom Scribe base URL");
        let scheme = if matches!(url.scheme(), "http" | "ws") {
            "ws"
        } else {
            "wss"
        };
        url.set_scheme(scheme)
            .expect("invalid Zoom Scribe URL scheme");
        url
    }

    fn build_auth_header(&self, key: Option<&str>) -> Option<(&'static str, String)> {
        key.map(|key| ("Authorization", format!("Bearer {key}")))
    }

    fn extra_headers(&self) -> Vec<(&'static str, String)> {
        vec![("x-zm-trackingid", super::tracking_id())]
    }

    fn keep_alive_message(&self) -> Option<Message> {
        None
    }

    fn finalize_message(&self) -> Message {
        self.session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .closing = true;
        segment::message(json!({"type": "session.close"}))
    }

    fn initial_message(&self, _: Option<&str>, params: &ListenParams, _: u8) -> Option<Message> {
        *self.session.lock().unwrap_or_else(|e| e.into_inner()) = Session {
            model: super::resolve_model(params.model.as_deref()).to_string(),
            ..Session::default()
        };
        Some(segment::message(json!({
            "type": "session.update",
            "config": config(params, true).ok()?,
            "audio": {"format": "pcm16"},
            "outputs": {"text_polishing": {"enabled": false}},
        })))
    }

    fn audio_to_message(&self, audio: bytes::Bytes) -> Message {
        self.session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .samples += audio.len() as u64 / 2;
        Message::Binary(audio)
    }

    fn parse_response(&self, raw: &str) -> Vec<StreamResponse> {
        let Ok(value) = serde_json::from_str::<Value>(raw) else {
            return vec![];
        };
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        match value["type"].as_str().unwrap_or_default() {
            "input_audio_buffer.speech_started" => {
                if let Some(start) = value["audio_start_ms"].as_f64() {
                    session.start = start / 1000.0;
                }
                vec![]
            }
            "transcription.delta" | "transcription.completed" => {
                let is_final = value["type"] == "transcription.completed";
                let field = if is_final { "transcript" } else { "delta" };
                let Some(text) = value[field].as_str() else {
                    return vec![];
                };
                let start = value["audio_start_ms"]
                    .as_f64()
                    .map(|ms| ms / 1000.0)
                    .unwrap_or(session.start);
                let end = value["audio_end_ms"]
                    .as_f64()
                    .map(|ms| ms / 1000.0)
                    .unwrap_or(session.samples as f64 / 16_000.0)
                    .max(start);
                if is_final {
                    session.start = end;
                    session.end = end;
                }
                let mut response = segment::response(
                    &super::clean_text(text, &session.model),
                    start,
                    end,
                    is_final,
                    speaker(&value["speaker"]),
                );
                if let StreamResponse::TranscriptResponse { channel, .. } = &mut response
                    && let Some(language) = value["language"].as_str()
                {
                    channel.alternatives[0].languages = vec![language.into()];
                }
                vec![response]
            }
            "session.closed" if session.closing => vec![segment::finished(session.end)],
            // A provider-side session limit/idle close is not a successful user
            // finalize. Surface it so the listener supervisor can reconnect/replay.
            "session.closed" => vec![segment::error(
                "zoom_scribe",
                &json!({
                    "message": "Zoom Scribe session ended unexpectedly; reconnect transcription"
                }),
            )],
            "error" if value["error"]["fatal"] != false => {
                let mut error = value["error"].clone();
                if let Some(code) = error["code"].as_str() {
                    error["message"] = json!(format!(
                        "{}: {}",
                        code,
                        error["message"].as_str().unwrap_or("Zoom Scribe error")
                    ));
                }
                vec![segment::error("zoom_scribe", &error)]
            }
            "error" => {
                // Codes only: server messages can include request content.
                let code = value["error"]["code"].as_str().unwrap_or("");
                let safe_code: String = code
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
                    .take(80)
                    .collect();
                tracing::debug!(code = safe_code, "zoom_scribe_nonfatal_error");
                vec![]
            }
            _ => vec![],
        }
    }
}
