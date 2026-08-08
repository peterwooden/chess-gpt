import { getLiveGameEventBatches, getLiveGameResponse } from "../../../../../lib/live-games";

const POLL_MS = 500;
const HEARTBEAT_MS = 15_000;

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const initial = await getLiveGameResponse(id);
  if (!initial.live) return Response.json({ error: "Live game not found." }, { status: 404 });
  const url = new URL(request.url);
  let cursor = parseCursor(
    request.headers.get("last-event-id") ?? url.searchParams.get("after"),
  );
  const encoder = new TextEncoder();
  let stop = () => {};
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      let polling = false;
      let closed = false;
      const enqueue = (value: string) => controller.enqueue(encoder.encode(value));
      const poll = async () => {
        if (polling || closed) return;
        polling = true;
        try {
          const batches = await getLiveGameEventBatches(id, cursor);
          for (const batch of batches) {
            cursor = Math.max(cursor, batch.lastSeq);
            enqueue(`id: ${batch.lastSeq}\nevent: live-game-batch\ndata: ${JSON.stringify(batch)}\n\n`);
          }
        } catch (error) {
          stop();
          controller.error(error);
        } finally {
          polling = false;
        }
      };
      enqueue("retry: 1000\n\n");
      const pollTimer = setInterval(() => void poll(), POLL_MS);
      const heartbeatTimer = setInterval(() => enqueue(": keepalive\n\n"), HEARTBEAT_MS);
      stop = () => {
        closed = true;
        clearInterval(pollTimer);
        clearInterval(heartbeatTimer);
      };
      void poll();
    },
    cancel() {
      stop();
    },
  });
  return new Response(stream, {
    headers: {
      "Cache-Control": "no-cache, no-transform",
      "Content-Type": "text/event-stream; charset=utf-8",
      "X-Accel-Buffering": "no",
    },
  });
}

function parseCursor(value: string | null): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : 0;
}
