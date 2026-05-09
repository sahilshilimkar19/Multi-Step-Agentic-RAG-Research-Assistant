import { RunView } from "./RunView";

export default async function RunPage({ params }: { params: { threadId: string } }) {
  return (
    <main className="space-y-4">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Run</h1>
        <code className="text-xs text-zinc-500">{params.threadId}</code>
      </header>
      <RunView threadId={params.threadId} />
    </main>
  );
}
