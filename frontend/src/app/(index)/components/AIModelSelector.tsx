import React from "react";
import { useSetupStore } from "../lib/store";
import { LLMModel, LLM_MODELS } from "../lib/constants";

const AIModelSelector: React.FC = () => {
  const LLM = useSetupStore((state) => state.LLM);
  const setLLM = useSetupStore((state) => state.setLLM);

  return (
    <div className="w-full">
      <label
        htmlFor="ai-model"
        className="block text-sm font-medium text-gray-700 mb-2"
      >
        Select LLM
      </label>
      <select
        id="ai-model"
        value={LLM}
        onChange={(e) => setLLM(e.target.value as LLMModel)}
        className="block w-full pl-3 pr-10 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
      >
        <option value="">Select an AI model</option>
        {LLM_MODELS.map((model) => (
          <option key={model} value={model}>
            {model.replace(/_/g, " ")}
          </option>
        ))}
      </select>
    </div>
  );
};

export default AIModelSelector;
