export const LLM_MODELS = [
  "GPT3",
  "GPT4_TURBO",
  "CLAUDE_OPUS",
  "CLAUDE_SONNET",
  "MISTRAL_LARGE",
  "MISTRAL_SMALL",
  "LLAMA_3_8B",
  "LLAMA_3_70B",
] as const;

export type LLMModel = (typeof LLM_MODELS)[number];
