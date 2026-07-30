import Link from "next/link";

export function HistoryNav({ active }: { active?: "models" | "players" }) {
  return (
    <nav className="history-nav" aria-label="Site navigation">
      <Link className="arena-title" href="/arena">ChessGPT arena</Link>
      <div>
        <Link href="/">Learn</Link>
        <Link className={active === "models" ? "active" : ""} href="/models">Models</Link>
        <Link className={active === "players" ? "active" : ""} href="/history">Players</Link>
      </div>
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
