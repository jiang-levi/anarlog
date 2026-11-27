mod batch;
mod live;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};

use anlg_language::Language;
use owhisper_interface::ListenParams;
use serde_json::{Value, json};

use super::{LanguageQuality, LanguageSupport, parsing::parse_speaker_id};
use crate::Error;

const DEFAULT_BASE: &str = "https://api.zoom.us/v2/aiservices/scribe";
const DEFAULT_MODEL: &str = "zoom-scribe-multilingual";
// Keep in sync with apps/desktop/src/stt/zoom-scribe.ts. Korean is
// available through Multilingual; a standalone ko model is not verified.
const LOCALES: &[(&str, &str)] = &[
    ("en", "en-US"),
    ("ar", "ar-SA"),
    ("zh", "zh-CN"),
    ("fr", "fr-FR"),
    ("de", "de-DE"),
    ("it", "it-IT"),
    ("ja", "ja-JP"),
    ("ko", "ko-KR"),
    ("pt", "pt-BR"),
    ("es", "es-ES"),
];

#[derive(Default)]
struct Session {
    model: String,
    closing: bool,
    samples: u64,
    start: f64,
    end: f64,
}

#[derive(Clone, Default)]
pub struct ZoomScribeAdapter {
    session: Arc<Mutex<Session>>,
}

impl ZoomScribeAdapter {
    /// Apply dictation preview policy to the same resolved model sent to Scribe.
    /// Meeting callers do not use this hook and retain interim captions.
    pub fn configure_dictation(params: &mut ListenParams) {
        if resolve_model(params.model.as_deref()) == DEFAULT_MODEL {
            params
                .custom_query
                .get_or_insert_with(Default::default)
                .insert("zoom_interim_results".into(), "false".into());
        }
    }

    /// The dictation UI also joins completed turns; apply the same model-specific rules there.
    pub fn normalize_transcript(text: &str, model: &str) -> String {
        clean_text(text, model)
    }

    pub fn language_support_live(languages: &[Language], model: Option<&str>) -> LanguageSupport {
        let model = resolve_model(model);
        let supported = supported_model(model)
            && (model == DEFAULT_MODEL
                || (languages.len() <= 1
                    && languages.iter().all(|lang| {
                        locale(lang).is_some()
                            && (model.strip_prefix("zoom-scribe-") == Some(lang.iso639_code())
                                || (model == "zoom-scribe-en-pro" && lang.iso639_code() == "en"))
                    })));
        if supported {
            LanguageSupport::Supported {
                quality: LanguageQuality::NoData,
            }
        } else {
            LanguageSupport::NotSupported
        }
    }

    pub fn language_support_batch(languages: &[Language], model: Option<&str>) -> LanguageSupport {
        Self::language_support_live(languages, model)
    }
}

fn resolve_model(model: Option<&str>) -> &str {
    model
        .filter(|model| !model.is_empty() && !crate::is_meta_model(model))
        .unwrap_or(DEFAULT_MODEL)
}

fn supported_model(model: &str) -> bool {
    model == DEFAULT_MODEL
        || model == "zoom-scribe-en-pro"
        || LOCALES.iter().any(|(language, _)| {
            *language != "ko" && model.strip_prefix("zoom-scribe-") == Some(*language)
        })
}

fn locale(language: &Language) -> Option<String> {
    let (_, default) = LOCALES
        .iter()
        .find(|(code, _)| *code == language.iso639_code())?;
    if language.region().is_none() {
        return Some((*default).into());
    }
    let bcp47 = language.bcp47_code();
    LOCALES
        .iter()
        .map(|(_, locale)| *locale)
        .chain(["ar-AE", "pt-PT"])
        .find(|locale| locale.eq_ignore_ascii_case(&bcp47))
        .or(Some(*default))
        .map(str::to_owned)
}

fn config(params: &ListenParams, live: bool) -> Result<Value, Error> {
    let model = resolve_model(params.model.as_deref());
    if !ZoomScribeAdapter::language_support_live(&params.languages, Some(model)).is_supported() {
        return Err(Error::provider_configuration(
            "zoom_scribe",
            "the selected model does not support these languages",
        ));
    }
    let mut config = json!({
        "model": model,
        "diarization": params.num_speakers != Some(1),
    });
    if model == DEFAULT_MODEL {
        // Ignore app language/candidate settings for continuous automatic detection.
        config["language_detection"] = json!({"mode": "continuous"});
    } else {
        config["language"] = json!(params.languages.first().and_then(locale).unwrap_or_else(
            || {
                LOCALES
                    .iter()
                    .find(|(code, _)| model.strip_prefix("zoom-scribe-") == Some(*code))
                    .map(|(_, locale)| *locale)
                    .unwrap_or("en-US")
                    .into()
            }
        ));
    }
    let phrases: Vec<_> = params
        .keywords
        .iter()
        .map(|word| word.trim())
        .filter(|word| !word.is_empty() && word.chars().count() <= 50)
        .take(100)
        .collect();
    if !phrases.is_empty() {
        config["vocabulary"] = json!({"phrases": phrases});
    }
    if live {
        config["interim_results"] = json!(
            params
                .custom_query
                .as_ref()
                .and_then(|query| query.get("zoom_interim_results"))
                .is_none_or(|value| value != "false")
        );
    } else {
        config["word_time_offsets"] = json!(true);
        config["channel_separation"] = json!(params.channels == 2);
    }
    Ok(config)
}

fn endpoint(base: &str, operation: &str) -> Result<url::Url, Error> {
    let mut url = url::Url::parse(if base.is_empty() { DEFAULT_BASE } else { base })
        .map_err(|_| Error::provider_configuration("zoom_scribe", "invalid base URL"))?;
    if !matches!(url.scheme(), "http" | "https" | "ws" | "wss")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(Error::provider_configuration(
            "zoom_scribe",
            "invalid base URL",
        ));
    }
    let path = url.path().trim_end_matches('/');
    let path = path
        .strip_suffix("/live")
        .or_else(|| path.strip_suffix("/transcribe"))
        .unwrap_or(path);
    url.set_path(&format!("{path}/{operation}"));
    let query: Vec<_> = url
        .query_pairs()
        .filter(|(key, _)| key != "provider")
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    url.set_query(None);
    if !query.is_empty() {
        url.query_pairs_mut().extend_pairs(query);
    }
    url.set_fragment(None);
    Ok(url)
}

fn speaker(value: &Value) -> Option<i32> {
    value
        .as_str()
        .and_then(parse_speaker_id)
        .and_then(|id| i32::try_from(id).ok())
}

fn tracking_id() -> String {
    format!("zoom_ai_community_anarlog_{}", uuid::Uuid::new_v4())
}

// Clean only the dedicated Chinese model; preserve Latin spaces and newlines.
fn clean_text(text: &str, model: &str) -> String {
    if model != "zoom-scribe-zh" {
        return text.to_string();
    }
    fn chinese(c: char) -> bool {
        matches!(c, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}' | '\u{20000}'..='\u{323af}')
            || "，。！？；：、（）《》“”‘’".contains(c)
    }
    let mut output = String::new();
    let mut pending = String::new();
    let mut previous = None;
    for c in text.trim().chars() {
        if matches!(c, ' ' | '\t') {
            pending.push(c);
            continue;
        }
        if !(previous.is_some_and(chinese) && chinese(c)) {
            output.push_str(&pending);
        }
        pending.clear();
        output.push(c);
        previous = Some(c);
    }
    output
}
