import type { Metadata } from "next";
import "./globals.css";

const title = "Chess GPT Learning Lab";
const description = "A first-principles machine-learning course built around training a competitive small chess language model.";

export const metadata: Metadata = {
  title,
  description,
  openGraph: { title, description },
  twitter: { card: "summary", title, description },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
