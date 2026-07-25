import type { Metadata } from "next";

import LessonClient from "./lesson-client";

export const metadata: Metadata = {
  title: "Split games, not positions · Chess GPT Learning Lab",
  description: "Chapter 1 Mission 1: predict leakage, compare chess data splits, and earn a completion code.",
};

export default function DataSplitsLesson() {
  return <LessonClient />;
}
