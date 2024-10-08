interface StepIndicatorProps {
  step: "setup" | "inference";
  currentStep: "setup" | "inference";
  onClick: () => void;
}

export function StepIndicator({
  step,
  currentStep,
  onClick,
}: StepIndicatorProps) {
  const isActive = step === currentStep;
  return (
    <div
      className="flex flex-col items-center cursor-pointer"
      onClick={onClick}
    >
      <div
        className={`w-3 h-3 rounded-full mb-1 ${
          isActive ? "bg-blue-500" : "bg-gray-300"
        }`}
      />
      <span
        className={`text-sm ${
          isActive ? "text-blue-500 font-semibold" : "text-gray-500"
        }`}
      >
        {step.charAt(0).toUpperCase() + step.slice(1)}
      </span>
    </div>
  );
}
