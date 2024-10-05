"use client";

import React, { useCallback, useState } from "react";
import Monaco from "./Monaco";
import { useSetupStore } from "../lib/store";

const JsonSchemaEditor: React.FC = () => {
  const [jsonInput, setJsonInput] = useState("");

  const { setJsonSchema, jsonSchema } = useSetupStore();

  const transformer = useCallback(
    async (value: string) => {
      try {
        const { run } = await import("json_typegen_wasm");
        const result = run(
          "Root",
          value,
          JSON.stringify({
            output_mode: "json_schema",
          })
        );
        setJsonSchema(result);
      } catch (error) {
        console.error("Error transforming JSON:", error);
        setJsonSchema("Error: Invalid JSON input");
      }
    },
    [setJsonSchema]
  );

  const handleJsonInputChange = (value: string | undefined) => {
    if (value) {
      setJsonInput(value);
      transformer(value);
    }
  };

  return (
    <div className="flex flex-col space-y-4">
      <div className="w-full border border-gray-300 rounded-md p-4">
        <label
          htmlFor="json-input"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          Target JSON
        </label>
        <Monaco
          language="json"
          value={jsonInput}
          onChange={handleJsonInputChange}
          height="300px"
          options={{
            minimap: { enabled: false },
            automaticLayout: true,
          }}
        />
      </div>
      <div className="w-full border border-gray-300 rounded-md p-4">
        <label
          htmlFor="json-schema"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          JSON Schema
        </label>
        <Monaco
          language="json"
          value={jsonSchema}
          onChange={(schema) => setJsonSchema(schema || "")}
          height="300px"
          options={{
            minimap: { enabled: false },
            automaticLayout: true,
          }}
        />
      </div>
    </div>
  );
};

export default JsonSchemaEditor;
