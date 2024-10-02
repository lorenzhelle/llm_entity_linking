"use client";
import React, { useState } from "react";
import axios from "axios";
import { useSetupStore } from "../store/store";

const Inference: React.FC = () => {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<string | null>(null);

  const selectedLLM = useSetupStore((state) => state.LLM);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await axios.post("/api/check_domain", {
        query,
        model: selectedLLM,
      });
      setResult(response.data.inDomain ? "In Domain" : "Out of Domain");
    } catch (error) {
      console.error("Error checking domain:", error);
      setResult("Error checking domain");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full space-y-4">
      <div>
        <label
          htmlFor="query"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          Query
        </label>
        <input
          type="text"
          id="query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="block w-full pl-3 pr-3 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
          placeholder="Enter your query here"
        />
      </div>
      <button
        type="submit"
        className="w-full py-2 px-4 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
      >
        Check Domain
      </button>
      {result && (
        <div className="mt-4 p-4 bg-gray-100 rounded-md">
          <p className="text-sm font-medium text-gray-900">Result: {result}</p>
        </div>
      )}
    </form>
  );
};

export default Inference;
