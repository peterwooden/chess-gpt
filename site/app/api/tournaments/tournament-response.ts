import { HistoryError } from "../../../lib/history";

/** Turn a thrown error into the same JSON shape every tournament route returns. */
export function failure(error: unknown): Response {
  const status = error instanceof HistoryError ? error.status : 500;
  const message = error instanceof HistoryError
    ? error.message
    : "The tournament request could not be completed.";
  return Response.json({ error: message }, { status });
}
