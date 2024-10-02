import React from "react";
import { useSetupStore } from "../store/store";

const DomainInput: React.FC = () => {
  const { domain, setDomain } = useSetupStore();

  return (
    <div>
      <label
        htmlFor="domain"
        className="block text-sm font-medium text-gray-700 mb-2"
      >
        Domain
      </label>
      <input
        type="text"
        id="domain"
        value={domain}
        onChange={(e) => setDomain(e.target.value)}
        className="block w-full pl-3 pr-3 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
        placeholder="Enter domain"
      />
    </div>
  );
};

export default DomainInput;
