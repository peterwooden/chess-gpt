import type { Plugin } from "vite";

export declare const CROSS_ORIGIN_ISOLATION_HEADERS: Readonly<Record<string, string>>;

export declare function crossOriginIsolation(): Plugin;
