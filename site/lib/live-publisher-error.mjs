export function livePublisherErrorMessage(error, fallback) {
  if (isAbortError(error)) return fallback;
  return error instanceof Error && error.message ? error.message : fallback;
}

function isAbortError(error) {
  return (error instanceof DOMException && error.name === "AbortError")
    || (error instanceof Error && /signal is aborted/i.test(error.message));
}
