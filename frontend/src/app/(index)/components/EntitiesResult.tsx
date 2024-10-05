import React from "react";
import Monaco from "./Monaco";

export const EntitiesResult = ({
  entities,
}: {
  entities: Record<string, unknown>;
}) => {
  return (
    <div className="mt-4 border border-gray-300 rounded-md p-4">
      <h3 className="text-lg font-semibold text-gray-700 mb-2 ">
        Recognized Entities
      </h3>
      <Monaco
        language="json"
        value={JSON.stringify(entities, null, 2)}
        onChange={() => {}}
      />
    </div>
  );
};
