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
  const language =
    provider === "zoom_scribe"
      ? (getZoomScribeLanguage(model) ??
        (model === "zoom-scribe-en-pro" ? "en" : undefined))
      : undefined;
  return language ? [language] : [...languages];
}
