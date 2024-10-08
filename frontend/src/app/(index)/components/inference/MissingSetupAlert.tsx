import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ExclamationTriangleIcon } from "@radix-ui/react-icons";

export const MissingSetupAlert = () => (
  <Alert variant="destructive">
    <ExclamationTriangleIcon className="h-4 w-4" />
    <AlertTitle>Incomplete Setup</AlertTitle>
    <AlertDescription>
      Please select an LLM, enter a domain and a JSON Schema in the setup page
      before proceeding with the domain check.
    </AlertDescription>
  </Alert>
);
