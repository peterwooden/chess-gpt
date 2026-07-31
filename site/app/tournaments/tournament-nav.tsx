import Link from "next/link";

export function TournamentNav() {
  return (
    <nav className="history-nav" aria-label="Site navigation">
      <Link className="arena-title" href="/arena">ChessGPT arena</Link>
      <div>
        <Link href="/">Learn</Link>
        <Link href="/models">Models</Link>
        <Link href="/history">Players</Link>
        <Link className="active" href="/tournaments">Tournaments</Link>
      </div>
    </nav>
  );
}

export function formatDateTime(timestamp: number | null) {
  if (timestamp === null) return "—";
  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(timestamp);
}

export function formatDuration(ms: number) {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const totalMinutes = Math.round(ms / 60_000);
  if (totalMinutes < 60) return `${totalMinutes} min`;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes === 0 ? `${hours} h` : `${hours} h ${minutes} min`;
}

export function formatScore(points: number) {
  return Number.isInteger(points) ? String(points) : points.toFixed(1);
}
