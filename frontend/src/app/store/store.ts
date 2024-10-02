import { create } from "zustand";
import { LLMModel } from "../common/constants";

interface SetupState {
  domain: string;
  LLM: LLMModel | "";
  jsonSchema: string;
  setDomain: (domain: string) => void;
  setLLM: (llm: LLMModel) => void;
  setJsonSchema: (jsonSchema: string) => void;
}

export const useSetupStore = create<SetupState>((set) => ({
  domain: "",
  LLM: "",
  jsonSchema: "",
  setDomain: (domain) => set({ domain }),
  setLLM: (llm) => set({ LLM: llm }),
  setJsonSchema: (jsonSchema) => set({ jsonSchema }),
}));
