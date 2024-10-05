import { create } from "zustand";
import { LLMModel } from "./constants";

interface SetupState {
  domain: string;
  LLM: LLMModel | "";
  jsonSchema: string;
  setDomain: (domain: string) => void;
  setLLM: (llm: LLMModel) => void;
  setJsonSchema: (jsonSchema: string) => void;
  outOfDomainCheck: boolean;
  setOutOfDomainCheck: (value: boolean) => void;
}

export const useSetupStore = create<SetupState>((set) => ({
  domain: "",
  LLM: "",
  jsonSchema: "",
  setDomain: (domain) => set({ domain }),
  setLLM: (llm) => set({ LLM: llm }),
  setJsonSchema: (jsonSchema) => set({ jsonSchema }),
  outOfDomainCheck: false,
  setOutOfDomainCheck: (value) => set({ outOfDomainCheck: value }),
}));
