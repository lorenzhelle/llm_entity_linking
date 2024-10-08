import { useState } from "react";
import axios, { AxiosError } from "axios";
import { useSetupStore } from "../store";

export const useInference = () => {
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

      if (outOfDomainCheck && !inDomain) {
        setButtonState("done");
        return;
      }

      setButtonState("linking");
      const entityResponse = await axios.post("/api/recognize-filters", {
        message: query,
        schema: jsonSchema,
        model: selectedLLM,
      });
      setEntities(entityResponse.data);
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
      setTimeout(() => setButtonState("idle"), 1000);
    }
  };

  return {
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
  };
};
