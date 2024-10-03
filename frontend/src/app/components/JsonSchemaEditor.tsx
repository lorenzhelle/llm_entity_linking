"use client";

import React, { useCallback, useState } from "react";
import Monaco from "./Monaco";

const JsonSchemaEditor: React.FC = () => {
  const [jsonInput, setJsonInput] = useState("");
  const [transformedSchema, setTransformedSchema] = useState("");

  const transformer = useCallback(async (value: string) => {
    try {
      const { run } = await import("json_typegen_wasm");
      const result = run(
        "Root",
        value,
        JSON.stringify({
          output_mode: "json_schema",
        })
      );
      setTransformedSchema(result);
    } catch (error) {
      console.error("Error transforming JSON:", error);
      setTransformedSchema("Error: Invalid JSON input");
    }
  }, []);

  const handleJsonInputChange = (value: string | undefined) => {
    if (value) {
      setJsonInput(value);
      transformer(value);
    }
  };

  return (
    <div className="flex flex-col space-y-4">
      <div className="w-full">
        <label
          htmlFor="json-input"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          JSON
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
      <div className="w-full">
        <label
          htmlFor="json-schema"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          JSON Schema
        </label>
        <Monaco
          language="json"
          value={transformedSchema}
          onChange={() => {}} // Read-only
          height="300px"
          options={{
            readOnly: true,
            minimap: { enabled: false },
            automaticLayout: true,
          }}
        />
      </div>
    </div>
  );
};

export default JsonSchemaEditor;
