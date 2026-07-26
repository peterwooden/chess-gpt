import assert from "node:assert/strict";
import test from "node:test";

import {
  createImmutableDownloadCache,
  isImmutableRevision,
} from "../app/arena/immutable-download-cache.mjs";

test("only a full Hugging Face commit SHA is treated as immutable", () => {
  assert.equal(isImmutableRevision("bea221167728c33f0a5df54051cd27717cae6586"), true);
  assert.equal(isImmutableRevision("main"), false);
  assert.equal(isImmutableRevision("v1.0.0"), false);
  assert.equal(isImmutableRevision("bea2211"), false);
});

test("an immutable verified download is reused from Cache Storage", async () => {
  const { storage } = memoryCacheStorage();
  const cache = createImmutableDownloadCache(storage);
  let downloads = 0;
  const options = {
    url: "https://huggingface.co/alice/model/resolve/bea221167728c33f0a5df54051cd27717cae6586/browser/model.bin",
    immutable: true,
    maximumBytes: 3,
    async download() {
      downloads += 1;
      return new Uint8Array([1, 2, 3]);
    },
    async validate(bytes) {
      assert.deepEqual([...bytes], [1, 2, 3]);
    },
  };

  assert.deepEqual([...(await cache.load(options))], [1, 2, 3]);
  assert.deepEqual([...(await cache.load(options))], [1, 2, 3]);
  assert.equal(downloads, 1);
});

test("a movable branch or tag is always downloaded and never stored", async () => {
  const { entries, storage } = memoryCacheStorage();
  const cache = createImmutableDownloadCache(storage);
  let downloads = 0;
  const options = {
    url: "https://huggingface.co/alice/model/resolve/v1/browser/model.bin",
    immutable: false,
    maximumBytes: 1,
    async download() {
      downloads += 1;
      return new Uint8Array([downloads]);
    },
    async validate() {},
  };

  assert.deepEqual([...(await cache.load(options))], [1]);
  assert.deepEqual([...(await cache.load(options))], [2]);
  assert.equal(downloads, 2);
  assert.equal(entries.size, 0);
});

test("a corrupted immutable cache entry is discarded and downloaded again", async () => {
  const url = "https://huggingface.co/alice/model/resolve/bea221167728c33f0a5df54051cd27717cae6586/browser/model.bin";
  const initial = new Map([[url, new Response(new Uint8Array([9]))]]);
  const { storage } = memoryCacheStorage(initial);
  const cache = createImmutableDownloadCache(storage);
  let downloads = 0;
  const options = {
    url,
    immutable: true,
    maximumBytes: 1,
    async download() {
      downloads += 1;
      return new Uint8Array([1]);
    },
    async validate(bytes) {
      if (bytes[0] !== 1) throw new Error("digest mismatch");
    },
  };

  assert.deepEqual([...(await cache.load(options))], [1]);
  assert.deepEqual([...(await cache.load(options))], [1]);
  assert.equal(downloads, 1);
});

function memoryCacheStorage(initialEntries = new Map()) {
  const entries = new Map(initialEntries);
  const browserCache = {
    async match(request) {
      return entries.get(request.url)?.clone();
    },
    async put(request, response) {
      entries.set(request.url, response.clone());
    },
    async delete(request) {
      return entries.delete(request.url);
    },
  };
  return {
    entries,
    storage: { async open() { return browserCache; } },
  };
}
