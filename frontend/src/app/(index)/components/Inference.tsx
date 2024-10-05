"use client";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ExclamationTriangleIcon } from "@radix-ui/react-icons";
import axios, { AxiosError } from "axios";
import React, { useState } from "react";
import { useSetupStore } from "../lib/store";
import { EntitiesResult } from "./EntitiesResult";
import RecognizeEntitiesButton from "./RecognizeEntitiesButton";
import Tag from "./Tag";

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
  const [buttonState, setButtonState] = useState<
    "idle" | "checking" | "linking" | "done"
  >("idle");

  const {
    LLM: selectedLLM,
    domain: selectedDomain,
    jsonSchema,
    outOfDomainCheck,
  } = useSetupStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setEntities(null);
    setButtonState("checking");
    let inDomain = false;
    try {
      if (outOfDomainCheck) {
        const response = await axios.post("/api/check_domain", {
          query,
          model: selectedLLM,
          domain: selectedDomain,
        });
        inDomain = response.data.inDomain;
        setResult({
          status: inDomain ? "success" : "warning",
          message: inDomain ? "Query is in domain" : "Query is out of domain",
        });
      }

      if (outOfDomainCheck && inDomain) {
        setButtonState("linking");
        const entityResponse = await axios.post("/api/recognize-filters", {
          message: query,
          schema: jsonSchema,
          model: selectedLLM,
        });
        setEntities(entityResponse.data);
      }
      setButtonState("done");
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
      setTimeout(() => setButtonState("idle"), 1000); // Reset button state after 1 second
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
      <div className="flex justify-start items-center space-x-2">
        <Tag text={selectedLLM} />
        {!outOfDomainCheck ? <Tag text="No OOTD check" /> : null}
      </div>
      {selectedDomain && outOfDomainCheck ? (
        <>
          <p>Domain: {selectedDomain}</p>
        </>
      ) : null}
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
        <RecognizeEntitiesButton
          isLoading={isLoading}
          buttonState={buttonState}
          disabled={isLoading || !query || buttonState !== "idle"}
        />
        {result?.status === "warning" && (
          <Alert variant={"destructive"}>
            <AlertDescription>{result.message}</AlertDescription>
          </Alert>
        )}
        {entities && <EntitiesResult entities={entities} />}
      </form>
    </div>
  );
};

export default Inference;
