import type { NextConfig } from "next";
import { CROSS_ORIGIN_ISOLATION_HEADERS } from "./build/cross-origin-isolation.mjs";

const isolationHeaders = Object.entries(CROSS_ORIGIN_ISOLATION_HEADERS).map(([key, value]) => ({
  key,
  value,
}));

const nextConfig: NextConfig = {
  async headers() {
    // Every route, so a navigation never lands on a non-isolated window.
    // `/:path*` does not match the empty path, so "/" needs its own rule.
    // See build/cross-origin-isolation.mjs for why the Vite dev server and
    // public/_headers repeat these.
    return [
      { source: "/", headers: isolationHeaders },
      { source: "/:path*", headers: isolationHeaders },
    ];
  },
};

export default nextConfig;
