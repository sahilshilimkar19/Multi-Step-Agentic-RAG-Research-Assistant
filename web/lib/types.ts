// Shared types mirroring the Python Pydantic schemas.
// Hand-written for PR-1; consider codegen later if these drift.

export type GradedDocument = {
  content: string;
  url: string;
  source: string;
  sub_question: string;
  relevance_score: number;
  is_grounded: boolean;
  rationale: string;
};

export type SerializedMessage = {
  type: string;
  content: string;
};

export type NodeUpdate = {
  research_plan?: string[];
  search_queries?: string[];
  raw_documents?: unknown[];
  graded_documents?: GradedDocument[];
  iteration_count?: number;
  next_action?: string;
  final_report?: string;
  messages?: SerializedMessage[];
  token_usage?: { input_tokens: number; output_tokens: number };
};

export type NodeEvent = {
  node: string;
  update: NodeUpdate;
};

export type PauseInfo = {
  reason: string;
  next: string[];
};

export type StartRunResponse = {
  thread_id: string;
};
