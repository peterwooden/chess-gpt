/**
 * Hugging Face Git commits currently use full 40-character hexadecimal IDs.
 * Branches, short hashes, and tags remain movable and are deliberately excluded.
 *
 * @param {string} revision
 */
export function isImmutableRevision(revision) {
  return /^[0-9a-f]{40}$/i.test(revision);
}

const CACHE_NAME = "chess-gpt-immutable-model-files-v1";
const BYTE_LENGTH_HEADER = "x-chess-gpt-byte-length";

/**
 * Build a best-effort, verified read-through cache for immutable model files.
 * Cache failures never prevent a network load.
 *
 * @param {CacheStorage | undefined} cacheStorage
 */
export function createImmutableDownloadCache(cacheStorage = globalThis.caches) {
  async function openCache() {
    if (!cacheStorage || typeof cacheStorage.open !== "function") return null;
    try {
      return await cacheStorage.open(CACHE_NAME);
    } catch {
      return null;
    }
  }

  async function remove(url) {
    const cache = await openCache();
    if (!cache) return;
    try {
      await cache.delete(cacheRequest(url));
    } catch {
      // Cache Storage is optional; the network path remains authoritative.
    }
  }

  async function read(url, maximumBytes) {
    const cache = await openCache();
    if (!cache) return null;
    try {
      const response = await cache.match(cacheRequest(url));
      if (!response) return null;
      const declaredBytes = Number(response.headers.get(BYTE_LENGTH_HEADER));
      if (Number.isFinite(declaredBytes) && declaredBytes > maximumBytes) {
        await remove(url);
        return null;
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength > maximumBytes) {
        await remove(url);
        return null;
      }
      return bytes;
    } catch {
      return null;
    }
  }

  async function write(url, bytes) {
    const cache = await openCache();
    if (!cache) return;
    try {
      const body = bytes.slice().buffer;
      await cache.put(
        cacheRequest(url),
        new Response(body, {
          headers: {
            "content-type": "application/octet-stream",
            [BYTE_LENGTH_HEADER]: String(bytes.byteLength),
          },
        }),
      );
    } catch {
      // Quota pressure, private mode, or eviction must degrade to a re-download.
    }
  }

  return {
    /**
     * @param {{
     *   url: string,
     *   immutable: boolean,
     *   maximumBytes: number,
     *   download(): Promise<Uint8Array>,
     *   validate(bytes: Uint8Array): Promise<void>
     * }} options
     */
    async load({ url, immutable, maximumBytes, download, validate }) {
      if (immutable) {
        const cached = await read(url, maximumBytes);
        if (cached) {
          try {
            await validate(cached);
            return cached;
          } catch {
            await remove(url);
          }
        }
      }

      const bytes = await download();
      if (bytes.byteLength > maximumBytes) {
        throw new Error("The downloaded file exceeded its allowed byte length.");
      }
      await validate(bytes);
      if (immutable) await write(url, bytes);
      return bytes;
    },
  };
}

function cacheRequest(url) {
  return new Request(url, { credentials: "omit" });
}
