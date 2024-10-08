"use client";
import { Alert, AlertDescription } from "@/components/ui/alert";
import React from "react";
import { EntitiesResult } from "../EntitiesResult";
import RecognizeEntitiesButton from "../RecognizeEntitiesButton";
import Tag from "../Tag";
import { MissingSetupAlert } from "./MissingSetupAlert";
import { useInference } from "../../lib/hooks/useInference";

const Inference: React.FC = () => {
  const {
    query,
    setQuery,
    result,
    isLoading,
    entities,
    buttonState,
    handleSubmit,
    selectedLLM,
    selectedDomain,
    jsonSchema,
    outOfDomainCheck,
  } = useInference();

  const showAlert =
    !selectedLLM || !jsonSchema || (outOfDomainCheck && !selectedDomain);

  if (showAlert) {
    return <MissingSetupAlert />;
  }

  return (
    <div className="w-full space-y-4">
      <div className="flex justify-start items-center space-x-2">
        <Tag text={selectedLLM} />
        {!outOfDomainCheck ? <Tag text="No OOTD check" /> : null}
      </div>
      {selectedDomain && outOfDomainCheck && <p>Domain: {selectedDomain}</p>}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label
            htmlFor="query"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            Query
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="block w-full pl-3 pr-3 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            placeholder="Enter your content here"
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
