"use client";

import React, { useCallback, useState } from "react";
import Monaco from "./Monaco";
import { useSetupStore } from "../lib/store";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const defaultJsonSchema = `{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Generated schema for Root",
  "type": "object",
  "properties": {
  },
  "required": [
  ]
}`;

const JsonSchemaEditor: React.FC = () => {
  const [jsonInput, setJsonInput] = useState("");
  const { setJsonSchema, jsonSchema } = useSetupStore();
  const [directSchema, setDirectSchema] = useState(
    jsonSchema !== "" ? jsonSchema : defaultJsonSchema
  );

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

  const handleDirectSchemaChange = (value: string | undefined) => {
    if (value) {
      setDirectSchema(value);
      setJsonSchema(value);
    }
  };

  return (
    <Tabs defaultValue="direct" className="w-full">
      <p className="text-sm text-gray-600 mb-4">
        Choose how you want to create your JSON schema used to identify and link
        entities in your document. You can either infer it from a JSON sample or
        enter it directly.
      </p>
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="direct">Enter Schema Directly</TabsTrigger>
        <TabsTrigger value="infer">Infer from JSON</TabsTrigger>
      </TabsList>
      <TabsContent value="direct">
        <div className="w-full border border-gray-300 rounded-md p-4">
          <label
            htmlFor="direct-schema"
            className="block text-sm font-medium text-gray-700 mb-2"
          >
            JSON Schema
          </label>
          <Monaco
            language="json"
            value={directSchema}
            onChange={handleDirectSchemaChange}
            height="300px"
            options={{
              minimap: { enabled: false },
              automaticLayout: true,
            }}
          />
        </div>
      </TabsContent>
      <TabsContent value="infer">
        <div className="flex flex-col space-y-4">
          <div className="w-full border border-gray-300 rounded-md p-4">
            <label
              htmlFor="json-input"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Source JSON
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
              Inferred JSON Schema
            </label>
            <Monaco
              language="json"
              value={jsonSchema}
              onChange={(schema) => setJsonSchema(schema || "")}
              height="300px"
              options={{
                minimap: { enabled: false },
                automaticLayout: true,
                readOnly: true,
              }}
            />
          </div>
        </div>
      </TabsContent>
    </Tabs>
  );
};

export default JsonSchemaEditor;
