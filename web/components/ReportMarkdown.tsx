"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function ReportMarkdown({ markdown }: { markdown: string }) {
  return (
    <div className="prose prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (props) => <h1 className="text-2xl font-bold mt-2 mb-3" {...props} />,
          h2: (props) => <h2 className="text-xl font-semibold mt-4 mb-2" {...props} />,
          h3: (props) => <h3 className="text-lg font-semibold mt-3 mb-2" {...props} />,
          p: (props) => <p className="my-2 leading-relaxed text-zinc-200" {...props} />,
          ul: (props) => <ul className="list-disc pl-6 my-2 text-zinc-200" {...props} />,
          ol: (props) => <ol className="list-decimal pl-6 my-2 text-zinc-200" {...props} />,
          a: (props) => (
            <a
              {...props}
              className="text-sky-400 underline hover:text-sky-300"
              target="_blank"
              rel="noopener noreferrer"
            />
          ),
          code: (props) => (
            <code className="rounded bg-zinc-800 px-1 py-0.5 text-sm" {...props} />
          ),
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
