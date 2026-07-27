import Link from "next/link";

export function HistoryNav() {
  return (
    <nav className="history-nav" aria-label="Site navigation">
      <Link className="arena-title" href="/arena">ChessGPT arena</Link>
      <div><Link href="/">Learn</Link><Link href="/history">History</Link></div>
    </nav>
  );
}

export function formatDate(timestamp: number) {
  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(timestamp);
}
