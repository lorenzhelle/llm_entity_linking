"use client";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ExclamationTriangleIcon } from "@radix-ui/react-icons";
import axios, { AxiosError } from "axios";
import React, { useState } from "react";
import Tag from "./Tag";
import { EntitiesResult } from "./EntitiesResult";
import { useSetupStore } from "../lib/store";

const Inference: React.FC = () => {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<{
    status: "success" | "warning" | "error";
    message: string;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [entities, setEntities] = useState<Record<string, unknown> | null>(
    null
  );

  const selectedLLM = useSetupStore((state) => state.LLM);
  const selectedDomain = useSetupStore((state) => state.domain);
  const jsonSchema = useSetupStore((state) => state.jsonSchema);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setEntities(null);
    try {
      const response = await axios.post("/api/check_domain", {
        query,
        model: selectedLLM,
        domain: selectedDomain,
      });
      const inDomain = response.data.inDomain;
      setResult({
        status: inDomain ? "success" : "warning",
        message: inDomain ? "Query is in domain" : "Query is out of domain",
      });

      if (inDomain) {
        const entityResponse = await axios.post("/api/recognize-filters", {
          message: query,
          schema: jsonSchema,
          model: selectedLLM,
        });
        setEntities(entityResponse.data);
      }
    } catch (error) {
      console.error("Error checking domain:", error);

      if (error instanceof AxiosError) {
        const message = error.response?.data.detail || error.message;
        setResult({
          status: "error",
          message: "Error checking domain: " + message,
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  if (!selectedLLM || !selectedDomain || !jsonSchema) {
    return (
      <Alert variant="destructive">
        <ExclamationTriangleIcon className="h-4 w-4" />
        <AlertTitle>Incomplete Setup</AlertTitle>
        <AlertDescription>
          Please select an LLM, enter a domain and a JSON Schema in the setup
          page before proceeding with the domain check.
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
      {selectedDomain ? <p>Domain: {selectedDomain}</p> : null}
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
          disabled={isLoading || !query}
        >
          {isLoading ? "Recognizing..." : "Recognize Entities"}
        </button>
        {result && (
          <Alert
            variant={result.status === "success" ? "default" : "destructive"}
          >
            <AlertDescription>{result.message}</AlertDescription>
          </Alert>
        )}
        {entities && <EntitiesResult entities={entities} />}
      </form>
    </div>
  );
};

export default Inference;
