import React from "react";
import { useSetupStore } from "../lib/store";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

const DomainInput: React.FC = () => {
  const { domain, setDomain, outOfDomainCheck, setOutOfDomainCheck } =
    useSetupStore();

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-2">
        <Switch
          id="outOfDomainCheck"
          checked={outOfDomainCheck}
          onCheckedChange={setOutOfDomainCheck}
        />
        <Label
          htmlFor="outOfDomainCheck"
          className="text-sm font-medium text-gray-700"
        >
          Enable out-of-domain check
        </Label>
      </div>

      {outOfDomainCheck && (
        <div className="space-y-2">
          <Label htmlFor="domain" className="text-sm font-medium text-gray-700">
            Domain
          </Label>
          <input
            type="text"
            id="domain"
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="block w-full pl-3 pr-3 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            placeholder="Enter domain"
          />
        </div>
      )}
    </div>
  );
};

export default DomainInput;
