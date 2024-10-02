"use client";
import React, { useState } from "react";
import axios from "axios";
import { useSetupStore } from "../store/store";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { ExclamationTriangleIcon } from "@radix-ui/react-icons";
import Tag from "./Tag";

const Inference: React.FC = () => {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<{
    status: "success" | "warning" | "error";
    message: string;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const selectedLLM = useSetupStore((state) => state.LLM);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const response = await axios.post("/api/check_domain", {
        query,
        model: selectedLLM,
      });
      setResult({
        status: response.data.inDomain ? "success" : "warning",
        message: response.data.inDomain
          ? "Query is in domain"
          : "Query is out of domain",
      });
    } catch (error) {
      console.error("Error checking domain:", error);
      setResult({ status: "error", message: "Error checking domain" });
    } finally {
      setIsLoading(false);
    }
  };

  if (!selectedLLM) {
    return (
      <Alert variant="destructive">
        <ExclamationTriangleIcon className="h-4 w-4" />
        <AlertTitle>No LLM Selected</AlertTitle>
        <AlertDescription>
          Please select an LLM in the setup page before proceeding with the
          domain check.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="w-full space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold text-gray-700">Domain Check</h2>
        <Tag text={selectedLLM} />
      </div>
      <form onSubmit={handleSubmit} className="space-y-4">
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
          className="w-full py-2 px-4 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors disabled:bg-blue-300"
          disabled={isLoading}
        >
          {isLoading ? "Checking..." : "Check Domain"}
        </button>
        {result && (
          <Alert
            variant={result.status === "success" ? "default" : "destructive"}
          >
            <AlertDescription>{result.message}</AlertDescription>
          </Alert>
        )}
      </form>
    </div>
  );
};

export default Inference;
