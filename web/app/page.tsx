"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { startRun } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [maxIter, setMaxIter] = useState(4);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!query.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const { thread_id } = await startRun(query.trim(), maxIter);
      router.push(`/runs/${thread_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <main className="space-y-6">
      <header>
        <h1 className="text-3xl font-semibold">Agentic RAG</h1>
        <p className="text-zinc-400">
          Multi-step research with plan / search / grade / synthesize loops.
        </p>
      </header>

      <form onSubmit={onSubmit} className="space-y-4 rounded-lg border border-zinc-800 p-6">
        <label className="block">
          <span className="block text-sm text-zinc-300 mb-1">Research question</span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="How does Mamba compare to Transformers on long-context tasks?"
            className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2 outline-none focus:border-zinc-500"
            autoFocus
          />
        </label>
        <label className="block w-40">
          <span className="block text-sm text-zinc-300 mb-1">Max iterations</span>
          <input
            type="number"
            value={maxIter}
            min={1}
            max={20}
            onChange={(e) => setMaxIter(Number(e.target.value))}
            className="w-full rounded border border-zinc-700 bg-zinc-900 px-3 py-2 outline-none focus:border-zinc-500"
          />
        </label>
        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={submitting || !query.trim()}
            className="rounded bg-emerald-600 px-4 py-2 font-medium disabled:opacity-50 hover:bg-emerald-500"
          >
            {submitting ? "Starting..." : "Start"}
          </button>
          {error && <span className="text-sm text-red-400">{error}</span>}
        </div>
      </form>
    </main>
  );
}
