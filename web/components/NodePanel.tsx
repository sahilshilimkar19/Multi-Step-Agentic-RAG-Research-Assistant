import type { NodeEvent } from "@/lib/types";

const NODE_COLORS: Record<string, string> = {
  planner: "border-sky-700 bg-sky-950/40",
  searcher: "border-violet-700 bg-violet-950/40",
  grader: "border-emerald-700 bg-emerald-950/40",
  synthesizer: "border-amber-700 bg-amber-950/40",
};

export function NodePanel({ event }: { event: NodeEvent }) {
  const { node, update } = event;
  const tone = NODE_COLORS[node] ?? "border-zinc-700 bg-zinc-900";

  const lines: { label: string; value: React.ReactNode }[] = [];
  if (typeof update.iteration_count === "number") {
    lines.push({ label: "iteration", value: update.iteration_count });
  }
  if (update.research_plan && update.research_plan.length > 0) {
    lines.push({
      label: "plan",
      value: (
        <ul className="list-disc pl-5 text-zinc-200">
          {update.research_plan.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      ),
    });
  }
  if (update.search_queries && update.search_queries.length > 0) {
    lines.push({
      label: "queries",
      value: (
        <ul className="list-disc pl-5 text-zinc-200">
          {update.search_queries.map((q, i) => (
            <li key={i}>{q}</li>
          ))}
        </ul>
      ),
    });
  }
  if (update.graded_documents) {
    const total = update.graded_documents.length;
    const relevant = update.graded_documents.filter(
      (d) => d.relevance_score >= 0.6 && d.is_grounded,
    ).length;
    lines.push({
      label: "graded",
      value: `${total} (${relevant} relevant)`,
    });
  }
  if (update.next_action) {
    lines.push({ label: "next", value: update.next_action });
  }

  return (
    <li className={`rounded border ${tone} p-3`}>
      <div className="flex items-baseline justify-between">
        <span className="font-mono font-semibold">{node}</span>
      </div>
      {lines.length > 0 && (
        <dl className="mt-1 space-y-1 text-sm">
          {lines.map((l, i) => (
            <div key={i} className="flex gap-2">
              <dt className="w-20 shrink-0 text-zinc-500">{l.label}</dt>
              <dd className="text-zinc-200">{l.value}</dd>
            </div>
          ))}
        </dl>
      )}
    </li>
  );
}
