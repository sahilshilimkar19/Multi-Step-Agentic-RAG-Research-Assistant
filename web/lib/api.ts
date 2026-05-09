import type { StartRunResponse } from "./types";

// Empty string -> use Next.js rewrites to proxy /api/* to FastAPI.
// Set NEXT_PUBLIC_API_BASE=http://localhost:8000 to bypass the proxy
// (workaround for SSE buffering through some Next dev setups).
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export async function startRun(query: string, maxIterations?: number): Promise<StartRunResponse> {
  const r = await fetch(`${API_BASE}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, max_iterations: maxIterations }),
  });
  if (!r.ok) {
    throw new Error(`POST /api/runs failed: ${r.status} ${await r.text()}`);
  }
  return r.json();
}

export function streamUrl(threadId: string): string {
  return `${API_BASE}/api/runs/${threadId}/stream`;
}
