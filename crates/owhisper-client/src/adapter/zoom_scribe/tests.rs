use std::time::Duration;

use anlg_ws_client::client::Message;
use futures_util::{SinkExt, StreamExt};
use owhisper_interface::{ListenParams, stream::StreamResponse};
use serde_json::{Value, json};

use super::ZoomScribeAdapter;
use crate::{
    BatchClient, FinalizeHandle, ListenClient,
    adapter::{BatchSttAdapter, RealtimeSttAdapter},
};

fn transcript(response: &StreamResponse) -> (&str, bool, f64, f64) {
    let StreamResponse::TranscriptResponse {
        channel,
        is_final,
        start,
        duration,
        ..
    } = response
    else {
        panic!("expected transcript");
    };
    (
        &channel.alternatives[0].transcript,
        *is_final,
        *start,
        *duration,
    )
}

#[test]
fn zoom_scribe_replaces_partials_and_preserves_final_timing_and_speaker() {
    let adapter = ZoomScribeAdapter::default();
    adapter.audio_to_message(vec![0; 96_000].into());
    adapter.parse_response(r#"{"type":"input_audio_buffer.speech_started","audio_start_ms":1200}"#);
    let first = adapter.parse_response(r#"{"type":"transcription.delta","delta":"Hello"}"#);
    let revised = adapter.parse_response(r#"{"type":"transcription.delta","delta":"Hello world"}"#);
    assert_eq!(transcript(&first[0]).0, "Hello");
    assert_eq!(transcript(&revised[0]).0, "Hello world");
    let final_result = adapter.parse_response(r#"{"type":"transcription.completed","transcript":"Hello world.","audio_start_ms":1200,"audio_end_ms":2800,"speaker":"Speaker 2","language":"en"}"#);
    let (text, is_final, start, duration) = transcript(&final_result[0]);
    assert_eq!((text, is_final, start), ("Hello world.", true, 1.2));
    assert!((duration - 1.6).abs() < 1e-9);
    let StreamResponse::TranscriptResponse {
        channel,
        from_finalize,
        ..
    } = &final_result[0]
    else {
        unreachable!()
    };
    assert!(!from_finalize);
    assert_eq!(channel.alternatives[0].words[0].speaker, Some(2));
    assert_eq!(channel.alternatives[0].words[0].end, 2.8);
    assert_eq!(channel.alternatives[0].languages, ["en"]);
    adapter.finalize_message();
    let closed = adapter.parse_response(r#"{"type":"session.closed"}"#);
    assert!(matches!(
        closed[0],
        StreamResponse::TranscriptResponse {
            from_finalize: true,
            ..
        }
    ));
}

#[test]
fn zoom_scribe_keeps_microphone_and_speaker_session_clocks_independent() {
    let speaker = ZoomScribeAdapter::default();
    let microphone = speaker.fork_session();
    speaker.audio_to_message(vec![0; 64_000].into());
    microphone.audio_to_message(vec![0; 16_000].into());
    let event = r#"{"type":"transcription.completed","transcript":"test"}"#;
    assert_eq!(transcript(&speaker.parse_response(event)[0]).3, 2.0);
    assert_eq!(transcript(&microphone.parse_response(event)[0]).3, 0.5);
}

#[test]
fn zoom_scribe_rejects_incompatible_languages_and_builds_supported_locales() {
    let adapter = ZoomScribeAdapter::default();
    let initial = |languages: &[&str], model: &str| {
        let params = ListenParams {
            languages: languages.iter().map(|code| code.parse().unwrap()).collect(),
            model: Some(model.into()),
            ..Default::default()
        };
        adapter
            .initial_message(None, &params, 1)
            .map(|message| serde_json::from_str::<Value>(&message.into_text().unwrap()).unwrap())
    };
    let en = initial(&["en"], "zoom-scribe-en-pro").unwrap();
    assert_eq!(en["config"]["language"], "en-US");
    assert_eq!(
        initial(&["en-GB"], "zoom-scribe-en").unwrap()["config"]["language"],
        "en-US"
    );
    assert!(en["config"].get("language_detection").is_none());
    let multi = initial(&["de-DE", "ko"], "zoom-scribe-multilingual").unwrap();
    assert_eq!(
        multi["config"]["language_detection"],
        json!({"mode": "continuous"})
    );
    assert!(initial(&["zh"], "zoom-scribe-en").is_none());
    assert!(initial(&["ru"], "zoom-scribe-multilingual").is_some());
    assert!(initial(&["ko"], "zoom-scribe-ko").is_none());
}

#[test]
fn zoom_scribe_reports_fatal_errors_without_treating_warnings_as_final_results() {
    let adapter = ZoomScribeAdapter::default();
    assert!(
        adapter
            .parse_response(r#"{"type":"error","error":{"fatal":false,"message":"warning"}}"#)
            .is_empty()
    );
    assert!(matches!(
        &adapter.parse_response(r#"{"type":"error","error":{"code":"UNSUPPORTED_MODEL","fatal":true,"message":"Model unavailable"}}"#)[0],
        StreamResponse::ErrorResponse { provider, error_message, .. }
            if provider == "zoom_scribe" && error_message == "UNSUPPORTED_MODEL: Model unavailable"
    ));
}

#[tokio::test]
// Tungstenite fixes the handshake callback's error type to an HTTP response.
#[allow(clippy::result_large_err)]
async fn zoom_scribe_waits_for_acknowledgement_and_drains_results_after_stop() {
    use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
    tokio::time::timeout(Duration::from_secs(5), async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (audio_tx, audio_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_hdr_async(socket, |request: &Request, mut response: Response| {
                assert_eq!(request.headers()["authorization"], "Bearer test-key");
                assert_eq!(request.headers()["sec-websocket-protocol"], "live-asr");
                assert_eq!(request.uri().path(), "/v2/aiservices/scribe/live");
                response.headers_mut().insert("sec-websocket-protocol", "live-asr".parse().unwrap());
                Ok(response)
            }).await.unwrap();
            ws.send(Message::Text(json!({"type":"session.created"}).to_string().into())).await.unwrap();
            let update: Value = serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap()).unwrap();
            assert_eq!(update["type"], "session.update");
            assert_eq!(update["outputs"]["text_polishing"]["enabled"], false);
            ws.send(Message::Ping(vec![1].into())).await.unwrap();
            assert!(matches!(ws.next().await.unwrap().unwrap(), Message::Pong(_)));
            ws.send(Message::Text(json!({"type":"session.updated"}).to_string().into())).await.unwrap();
            assert!(matches!(ws.next().await.unwrap().unwrap(), Message::Binary(bytes) if bytes.len() == 3200));
            audio_tx.send(()).unwrap();
            let close: Value = serde_json::from_str(&ws.next().await.unwrap().unwrap().into_text().unwrap()).unwrap();
            assert_eq!(close["type"], "session.close");
            for event in [
                json!({"type":"transcription.completed","transcript":"Last words.","audio_start_ms":0,"audio_end_ms":100}),
                json!({"type":"session.closed"}),
            ] {
                ws.send(Message::Text(event.to_string().into())).await.unwrap();
            }
            ws.close(None).await.unwrap();
        });
        let client = ListenClient::builder().adapter::<ZoomScribeAdapter>()
            .api_base(format!("http://{address}/v2/aiservices/scribe"))
            .api_key("test-key").params(ListenParams::default())
            .build_single().await.unwrap();
        let (stream, handle) = client.from_realtime_audio(futures_util::stream::iter([
            owhisper_interface::MixedMessage::Audio(vec![0; 3200].into()),
        ])).await.unwrap();
        audio_rx.await.unwrap();
        handle.finalize().await;
        futures_util::pin_mut!(stream);
        let last = stream.next().await.unwrap().unwrap();
        assert_eq!(transcript(&last), ("Last words.", true, 0.0, 0.1));
        assert!(matches!(stream.next().await.unwrap().unwrap(), StreamResponse::TranscriptResponse { from_finalize: true, .. }));
        server.await.unwrap();
    }).await.unwrap();
}

#[tokio::test]
async fn zoom_scribe_rejects_invalid_live_configuration_before_connecting() {
    for (base, params) in [
        ("invalid", ListenParams::default()),
        (
            "https://api.zoom.us/v2/aiservices/scribe",
            ListenParams {
                sample_rate: 24_000,
                ..Default::default()
            },
        ),
        (
            "https://api.zoom.us/v2/aiservices/scribe",
            ListenParams {
                model: Some("zoom-scribe-en".into()),
                languages: vec!["zh".parse().unwrap()],
                ..Default::default()
            },
        ),
    ] {
        assert!(
            ListenClient::builder()
                .adapter::<ZoomScribeAdapter>()
                .api_base(base)
                .params(params)
                .build_single()
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn zoom_scribe_fast_preserves_stereo_channels_timestamps_and_speakers() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_partial_json, header, method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/scribe/transcribe"))
        .and(header("Authorization", "Bearer test-key"))
        .and(body_partial_json(json!({"config":{"word_time_offsets":true,"channel_separation":true}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {"text_display":"left right", "segments":[
                {"text":"left", "start":0.1,"end":0.5,"channel":0,"speaker":"Speaker 1", "words":[{"word":"left","start":0.2,"end":0.4}]},
                {"text":"right", "start":0.6,"end":0.9,"channel":1,"speaker":"Speaker 2"}
            ]}
        })))
        .expect(1).mount(&server).await;
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), b"RIFF\x00\x00\x00\x00WAVE").unwrap();
    let client = reqwest_middleware::ClientBuilder::new(reqwest::Client::new()).build();
    let response = BatchSttAdapter::transcribe_file(
        &ZoomScribeAdapter::default(),
        &client,
        &format!("{}/scribe", server.uri()),
        "test-key",
        &ListenParams {
            channels: 2,
            ..Default::default()
        },
        file.path(),
    )
    .await
    .unwrap();
    let left = &response.results.channels[0].alternatives[0];
    let right = &response.results.channels[1].alternatives[0];
    assert_eq!(left.transcript, "left");
    assert_eq!((left.words[0].start, left.words[0].end), (0.2, 0.4));
    assert_eq!(right.transcript, "right");
    assert_eq!(
        (right.words[0].channel, right.words[0].speaker),
        (1, Some(2))
    );
    assert_eq!((right.words[0].start, right.words[0].end), (0.6, 0.9));
}

#[tokio::test]
async fn zoom_scribe_fast_uploads_beyond_documented_limit_and_preserves_gateway_413() {
    use base64::{Engine, engine::general_purpose::STANDARD};
    use std::io::Write;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(b"RIFF\x00\x00\x00\x00WAVE").unwrap();
    file.as_file().set_len(2_100_000).unwrap();

    for status in [200, 413] {
        let server = MockServer::start().await;
        let response = if status == 200 {
            ResponseTemplate::new(status).set_body_json(json!({
                "result": {"text_display": "Large recording.", "segments": []}
            }))
        } else {
            ResponseTemplate::new(status).set_body_raw(
                "<html><body>413 Request Entity Too Large</body></html>",
                "text/html",
            )
        };
        Mock::given(method("POST"))
            .and(path("/transcribe"))
            .respond_with(response)
            .expect(1)
            .mount(&server)
            .await;
        let result = BatchClient::<ZoomScribeAdapter>::builder()
            .api_base(server.uri())
            .api_key("test-key")
            .build()
            .transcribe_file(file.path())
            .await;
        if status == 200 {
            assert_eq!(
                result.unwrap().results.channels[0].alternatives[0].transcript,
                "Large recording."
            );
        } else {
            let crate::Error::UnexpectedStatus { status, body } = result.unwrap_err() else {
                panic!("expected the gateway HTTP status to reach the caller");
            };
            assert_eq!(status, reqwest::StatusCode::PAYLOAD_TOO_LARGE);
            assert!(!body.contains("<html>"));
            assert!(body.contains("compress"));
        }
        let requests = server.received_requests().await.unwrap();
        assert!(requests[0].body.len() > 2_000_000);
        let payload: Value = serde_json::from_slice(&requests[0].body).unwrap();
        let audio = payload["file"]
            .as_str()
            .unwrap()
            .strip_prefix("data:audio/wav;base64,")
            .unwrap();
        let decoded = STANDARD.decode(audio).unwrap();
        assert_eq!(decoded.len(), 2_100_000);
        assert_eq!(&decoded[..12], b"RIFF\x00\x00\x00\x00WAVE");
    }
}

#[tokio::test]
async fn zoom_scribe_fast_rejects_encoded_requests_over_50_mib_before_upload() {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(b"RIFF\x00\x00\x00\x00WAVE").unwrap();
    let client = BatchClient::<ZoomScribeAdapter>::builder()
        .api_base("http://127.0.0.1:1")
        .api_key("test-key")
        .build();
    // The first file fits after Base64 alone; its JSON envelope pushes it over.
    // Keep one real production-limit check; the smaller exact-boundary test covers
    // envelope behavior without repeatedly allocating large buffers.
    for raw_bytes in [39_321_600] {
        file.as_file().set_len(raw_bytes).unwrap();
        let error = client.transcribe_file(file.path()).await.unwrap_err();
        assert!(matches!(
            error,
            crate::Error::RequestTooLarge {
                limit_bytes: 52_428_800
            }
        ));
    }
}

#[tokio::test]
async fn zoom_scribe_fast_marks_mono_diarization_as_mixed_capture() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {"text_display":"Hello.","segments":[
                {"text":"Hello.","start":0.1,"end":0.5,"speaker":"Speaker 1"}
            ]}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), b"RIFF\x00\x00\x00\x00WAVE").unwrap();
    let response = BatchClient::<ZoomScribeAdapter>::builder()
        .api_base(server.uri())
        .api_key("test-key")
        .build()
        .transcribe_file(file.path())
        .await
        .unwrap();
    let alternative = &response.results.channels[0].alternatives[0];
    assert_eq!(alternative.transcript, "Hello.");
    assert_eq!(
        alternative.words[0].channel,
        crate::adapter::MIXED_CAPTURE_CHANNEL
    );
    assert_eq!(alternative.words[0].speaker, Some(1));
}

#[test]
fn zoom_scribe_routes_official_and_custom_endpoints_without_leaking_routing_params() {
    use crate::AdapterKind;
    for base in [
        "https://api.zoom.us/v2/aiservices/scribe",
        "http://127.0.0.1:8000/v2/aiservices/scribe?provider=zoom_scribe",
    ] {
        let kind = AdapterKind::from_url_and_languages(base, &[], None);
        assert_eq!(kind, AdapterKind::ZoomScribe);
        assert!(kind.has_live_mode());
        assert!(kind.is_supported_languages_live(
            &["zh".parse().unwrap()],
            Some("zoom-scribe-multilingual")
        ));
        assert!(
            !kind.is_supported_languages_batch(&["zh".parse().unwrap()], Some("zoom-scribe-en"))
        );
        let url = ZoomScribeAdapter::default().build_ws_url(base, &ListenParams::default(), 1);
        assert_eq!(url.path(), "/v2/aiservices/scribe/live");
        assert!(!url.query_pairs().any(|(key, _)| key == "provider"));
    }
}

#[test]
fn zoom_scribe_auto_detection_and_dictation_interim_are_independent_of_meeting_speakers() {
    let adapter = ZoomScribeAdapter::default();
    let mut params = ListenParams {
        model: Some("zoom-scribe-multilingual".into()),
        num_speakers: Some(1),
        languages: vec!["en".parse().unwrap(), "ru".parse().unwrap()],
        ..Default::default()
    };
    let message = |params: &ListenParams| {
        serde_json::from_str::<Value>(
            &adapter
                .initial_message(None, params, 1)
                .unwrap()
                .into_text()
                .unwrap(),
        )
        .unwrap()
    };
    assert_eq!(message(&params)["config"]["interim_results"], true);
    assert_eq!(
        message(&params)["config"]["language_detection"],
        json!({"mode":"continuous"})
    );
    for model in [
        None,
        Some(""),
        Some("cloud"),
        Some("zoom-scribe-multilingual"),
    ] {
        params.model = model.map(str::to_string);
        params.custom_query = None;
        ZoomScribeAdapter::configure_dictation(&mut params);
        let update = message(&params);
        assert_eq!(update["config"]["model"], "zoom-scribe-multilingual");
        assert_eq!(update["config"]["interim_results"], false);
    }
    params.model = Some("zoom-scribe-en".into());
    params.languages.clear();
    params.custom_query = None;
    ZoomScribeAdapter::configure_dictation(&mut params);
    assert_eq!(message(&params)["config"]["interim_results"], true);
    let first = adapter.extra_headers();
    let second = adapter.extra_headers();
    assert_eq!(first[0].0, "x-zm-trackingid");
    assert!(first[0].1.starts_with("zoom_ai_community_anarlog_"));
    assert_ne!(first[0].1, second[0].1);
}

#[test]
fn zoom_scribe_chinese_cleaning_preserves_other_models_and_newlines() {
    let text = "你 好 ， 世 界 hello world\n新 的 一 行 [Speaker 1]";
    assert_eq!(
        super::clean_text(text, "zoom-scribe-zh"),
        "你好，世界 hello world\n新的一行 [Speaker 1]"
    );
    assert_eq!(super::clean_text(text, "zoom-scribe-multilingual"), text);
    assert_eq!(
        super::clean_text("  hello world\n", "zoom-scribe-en"),
        "  hello world\n"
    );
    let adapter = ZoomScribeAdapter::default();
    let params = ListenParams {
        model: Some("zoom-scribe-zh".into()),
        ..Default::default()
    };
    adapter.initial_message(None, &params, 1);
    let messages = adapter.parse_response(r#"{"type":"transcription.delta","delta":"你 好"}"#);
    let StreamResponse::TranscriptResponse { channel, .. } = &messages[0] else {
        panic!("transcript expected")
    };
    assert_eq!(channel.alternatives[0].transcript, "你好");
    assert_eq!(channel.alternatives[0].words[0].word, "你好");
}

#[test]
fn zoom_scribe_encoded_body_caps_the_actual_json_envelope() {
    let audio = [0u8; 12];
    let body =
        super::batch::encoded_body(&audio, "audio/wav", json!({"model":"zoom-scribe-zh"}), 1024)
            .unwrap();
    assert!(
        super::batch::encoded_body(
            &audio,
            "audio/wav",
            json!({"model":"zoom-scribe-zh"}),
            body.len()
        )
        .is_ok()
    );
    assert!(matches!(
        super::batch::encoded_body(
            &audio,
            "audio/wav",
            json!({"model":"zoom-scribe-zh"}),
            body.len() - 1
        ),
        Err(crate::Error::RequestTooLarge { .. })
    ));
}

#[test]
fn zoom_scribe_does_not_infer_unrelated_zoom_endpoints() {
    assert_ne!(
        crate::AdapterKind::from_url_and_languages("https://api.zoom.us/v2/meetings", &[], None),
        crate::AdapterKind::ZoomScribe
    );
}

#[tokio::test]
async fn zoom_scribe_fast_cleans_chinese_transcript_and_words_without_changing_timing() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(200).set_body_json(json!({
        "result": { "text_display": "你 好 ， hello world\n新 的 一 行", "segments": [
            {"text":"你 好", "start":0.1,"end":0.8,"channel":0,"words":[{"word":"你 好","start":0.2,"end":0.7}]}
        ]}
    }))).mount(&server).await;
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), b"RIFF\0\0\0\0WAVE").unwrap();
    let response = BatchClient::<ZoomScribeAdapter>::builder()
        .api_base(server.uri())
        .api_key("test-key")
        .params(ListenParams {
            model: Some("zoom-scribe-zh".into()),
            ..Default::default()
        })
        .build()
        .transcribe_file(file.path())
        .await
        .unwrap();
    let alternative = &response.results.channels[0].alternatives[0];
    assert_eq!(alternative.transcript, "你好， hello world\n新的一行");
    assert_eq!(alternative.words[0].word, "你好");
    assert_eq!(
        alternative.words[0].punctuated_word.as_deref(),
        Some("你好")
    );
    assert_eq!(
        (alternative.words[0].start, alternative.words[0].end),
        (0.2, 0.7)
    );
    let requests = server.received_requests().await.unwrap();
    assert!(
        requests[0].headers["x-zm-trackingid"]
            .to_str()
            .unwrap()
            .starts_with("zoom_ai_community_anarlog_")
    );
}

#[test]
fn zoom_scribe_unexpected_session_close_requests_recovery_and_fork_resets_closing_state() {
    let adapter = ZoomScribeAdapter::default();
    assert!(matches!(
        adapter.parse_response(r#"{"type":"session.closed"}"#)[0],
        StreamResponse::ErrorResponse { .. }
    ));
    adapter.finalize_message();
    assert!(matches!(
        adapter.parse_response(r#"{"type":"session.closed"}"#)[0],
        StreamResponse::TranscriptResponse {
            from_finalize: true,
            ..
        }
    ));
    let next = adapter.fork_session();
    assert!(matches!(
        next.parse_response(r#"{"type":"session.closed"}"#)[0],
        StreamResponse::ErrorResponse { .. }
    ));
}
