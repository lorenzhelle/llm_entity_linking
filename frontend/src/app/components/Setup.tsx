import DomainInput from "./DomainInput";
import JsonSchemaEditor from "./JsonSchemaEditor";
import AIModelSelector from "./AIModelSelector";
import { useSetupStore } from "../store/store";

interface SetupProps {
  onComplete: () => void;
}

const Setup: React.FC<SetupProps> = ({ onComplete }) => {
  const { domain, aiModel, jsonSchema } = useSetupStore();

  const handleComplete = () => {
    // You can add any additional logic here if needed
    onComplete();
  };

  return (
    <div className="w-full space-y-6">
      <DomainInput />
      <AIModelSelector />
      <JsonSchemaEditor />
      <button
        onClick={handleComplete}
        className="w-full py-2 px-4 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
      >
        Continue to Inference
      </button>
    </div>
  );
};

export default Setup;
