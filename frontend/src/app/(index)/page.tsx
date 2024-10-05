"use client";

import { useState } from "react";
import Setup from "./components/Setup";
import Inference from "./components/Inference";

export default function Home() {
  const [currentStep, setCurrentStep] = useState<"setup" | "inference">(
    "setup"
  );

  return (
    <div className="grid grid-rows-[auto_1fr_auto]  justify-items-center min-h-screen p-2  pb-20 gap-16 sm:p-4 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center w-full max-w-3xl">
        <h1 className="text-2xl font-bold mb-4">Entity Linking</h1>

        <div className="w-full flex justify-center mb-4">
          <button
            className={`px-4 py-2 mr-4 ${
              currentStep === "setup" ? "bg-blue-500 text-white" : "bg-gray-200"
            } rounded-md`}
            onClick={() => setCurrentStep("setup")}
          >
            Setup
          </button>
          <button
            className={`px-4 py-2 ${
              currentStep === "inference"
                ? "bg-blue-500 text-white"
                : "bg-gray-200"
            } rounded-md`}
            onClick={() => setCurrentStep("inference")}
          >
            Inference
          </button>
        </div>

        {currentStep === "setup" ? (
          <Setup onComplete={() => setCurrentStep("inference")} />
        ) : (
          <Inference />
        )}
      </main>
    </div>
  );
}
