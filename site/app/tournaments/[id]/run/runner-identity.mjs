const ID_KEY = "chess-gpt-runner-id";
const LABEL_KEY = "chess-gpt-runner-label";

/**
 * A browser exposes no stable machine identity, so one is constructed and kept
 * in localStorage. It is stable for this browser profile on this machine, which
 * is what "the same hardware" has to mean in practice.
 */
export function loadRunnerIdentity() {
  let id = safeGet(ID_KEY);
  if (!id) {
    id = crypto.randomUUID();
    safeSet(ID_KEY, id);
  }
  return {
    id,
    label: safeGet(LABEL_KEY) ?? "",
    saveLabel(label) {
      safeSet(LABEL_KEY, label);
    },
  };
}

/** Enough detail to tell two machines apart after the fact. */
export function describeMachine() {
  const navigatorRef = globalThis.navigator ?? {};
  return {
    userAgent: navigatorRef.userAgent ?? null,
    platform: navigatorRef.platform ?? null,
    hardwareConcurrency: navigatorRef.hardwareConcurrency ?? null,
    deviceMemory: navigatorRef.deviceMemory ?? null,
    language: navigatorRef.language ?? null,
    capturedAt: new Date().toISOString(),
  };
}

function safeGet(key) {
  try {
    return globalThis.localStorage?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

function safeSet(key, value) {
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    // A runner without persistent storage simply gets a new identity each visit,
    // which the pinning check will surface rather than silently accept.
  }
}
