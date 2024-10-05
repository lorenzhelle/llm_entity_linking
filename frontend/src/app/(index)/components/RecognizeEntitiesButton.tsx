import React from "react";
import { cn } from "@/lib/utils";

type ButtonState = "idle" | "checking" | "linking" | "done";

interface RecognizeEntitiesButtonProps {
  isLoading: boolean;
  buttonState: ButtonState;
  disabled: boolean;
}

const RecognizeEntitiesButton: React.FC<RecognizeEntitiesButtonProps> = ({
  isLoading,
  buttonState,
  disabled,
}) => {
  return (
    <button
      type="submit"
      className={cn(
        "w-full py-2 px-4 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors disabled:bg-blue-300 relative overflow-hidden",
        buttonState !== "idle" && "cursor-not-allowed"
      )}
      disabled={isLoading || disabled || buttonState !== "idle"}
    >
      <span
        className={cn(
          "absolute inset-0 flex items-center justify-center transition-opacity",
          buttonState === "idle" ? "opacity-0" : "opacity-100"
        )}
      >
        <span className="animate-pulse">
          {buttonState === "checking" && "Checking if out-of-domain..."}
          {buttonState === "linking" && "Linking entities..."}
          {buttonState === "done" && "Done!"}
        </span>
      </span>
      <span
        className={cn(
          "transition-opacity",
          buttonState === "idle" ? "opacity-100" : "opacity-0"
        )}
      >
        Recognize Entities
      </span>
    </button>
  );
};

export default RecognizeEntitiesButton;
