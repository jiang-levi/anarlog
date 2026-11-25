// Keep the standalone choices aligned with crates/owhisper-client/src/adapter/zoom_scribe/mod.rs.
// Korean is available through Multilingual, not a verified standalone model.
export const ZOOM_SCRIBE_LANGUAGES = [
  "en",
  "ar",
  "zh",
  "fr",
  "de",
  "it",
  "ja",
  "pt",
  "es",
] as const;

export function getZoomScribeLanguage(model: string | null | undefined) {
  return ZOOM_SCRIBE_LANGUAGES.find(
    (language) => model === `zoom-scribe-${language}`,
  );
}

export function getZoomScribeTranscriptionLanguages(
  provider: string | null | undefined,
  model: string | null | undefined,
  languages: readonly string[],
): string[] {
  if (
    provider === "zoom_scribe" &&
    (!model || model === "zoom-scribe-multilingual")
  )
    return [];
  const language =
    provider === "zoom_scribe"
      ? (getZoomScribeLanguage(model) ??
        (model === "zoom-scribe-en-pro" ? "en" : undefined))
      : undefined;
  return language ? [language] : [...languages];
}
