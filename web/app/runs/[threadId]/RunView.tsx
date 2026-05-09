"use client";

import { useEffect, useState } from "react";

import { NodePanel } from "@/components/NodePanel";
import { ReportMarkdown } from "@/components/ReportMarkdown";
import { streamUrl } from "@/lib/api";
import type { NodeEvent, PauseInfo } from "@/lib/types";

type Status = "streaming" | "paused" | "done" | "error";

export function RunView({ threadId }: { threadId: string }) {
  const [events, setEvents] = useState<NodeEvent[]>([]);
  const [pause, setPause] = useState<PauseInfo | null>(null);
  const [status, setStatus] = useState<Status>("streaming");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    const es = new EventSource(streamUrl(threadId));

    es.addEventListener("node", (e) => {
      try {
        const parsed = JSON.parse((e as MessageEvent).data) as NodeEvent;
        setEvents((prev) => [...prev, parsed]);
      } catch {
        /* swallow malformed event */
      }
    });

    es.addEventListener("paused", (e) => {
      try {
        const parsed = JSON.parse((e as MessageEvent).data) as PauseInfo;
        setPause(parsed);
      } catch {
        /* ignore */
      }
      setStatus("paused");
      es.close();
    });

    es.addEventListener("end", () => {
      setStatus("done");
      es.close();
    });

    es.addEventListener("error", (e) => {
      const data = (e as MessageEvent).data;
      if (typeof data === "string") {
        try {
          const parsed = JSON.parse(data) as { message?: string };
          setErrorMsg(parsed.message ?? "stream error");
        } catch {
          setErrorMsg("stream error");
        }
      }
      setStatus("error");
      es.close();
    });

    return () => es.close();
  }, [threadId]);

  const finalReport = [...events]
    .reverse()
    .find((e) => typeof e.update.final_report === "string" && e.update.final_report.length > 0)
    ?.update.final_report as string | undefined;

  return (
    <div className="space-y-4">
      <StatusBar status={status} count={events.length} errorMsg={errorMsg} />

      <ol className="space-y-2">
        {events.map((evt, i) => (
          <NodePanel key={i} event={evt} />
        ))}
      </ol>

      {pause && (
        <section className="rounded border border-amber-700 bg-amber-950/40 p-4">
          <h2 className="font-semibold text-amber-300">Paused for review</h2>
          <p className="text-sm text-zinc-300">
            Next: <code>{pause.next.join(", ")}</code>. (Resume + edit UI lands in PR-3 / PR-5.)
          </p>
        </section>
      )}

      {finalReport && (
        <section className="rounded border border-zinc-800 bg-zinc-900 p-6">
          <h2 className="text-lg font-semibold mb-2">Final Report</h2>
          <ReportMarkdown markdown={finalReport} />
        </section>
      )}
    </div>
  );
}

function StatusBar({
  status,
  count,
  errorMsg,
}: {
  status: Status;
  count: number;
  errorMsg: string | null;
}) {
  const tone =
    status === "done"
      ? "text-emerald-400"
      : status === "error"
        ? "text-red-400"
        : status === "paused"
          ? "text-amber-300"
          : "text-zinc-300";
  const label =
    status === "done"
      ? "complete"
      : status === "error"
        ? `error: ${errorMsg ?? "stream failed"}`
        : status === "paused"
          ? "paused"
          : "streaming";
  return (
    <div className={`flex items-center gap-3 text-sm ${tone}`}>
      <span>● {label}</span>
      <span className="text-zinc-500">({count} events)</span>
    </div>
  );
}
