import React from "react";
import { useSetupStore } from "../store/store";
const JsonSchemaEditor: React.FC = () => {
  const { jsonSchema, setJsonSchema } = useSetupStore();

  return (
    <div>
      <label
        htmlFor="json-schema"
        className="block text-sm font-medium text-gray-700 mb-2"
      >
        JSON Schema
      </label>
      <textarea
        id="json-schema"
        value={jsonSchema}
        onChange={(e) => setJsonSchema(e.target.value)}
        className="block w-full pl-3 pr-3 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
        rows={4}
        placeholder="Enter JSON schema"
      />
    </div>
  );
};

export default JsonSchemaEditor;
