import React from "react";
import Editor from "@monaco-editor/react";
import * as Progress from "@radix-ui/react-progress";
import { editor } from "monaco-editor";

interface MonacoProps {
  language?: string;
  value?: string;
  width?: number | string;
  height?: number | string;
  options?: editor.IStandaloneEditorConstructionOptions | undefined;
  defaultValue?: string;
  onChange: (value: string | undefined) => void;
}

export const Monaco: React.FC<MonacoProps> = ({
  language,
  value,
  defaultValue,
  height,
  width,
  options,
  onChange,
}) => {
  return (
    <Editor
      defaultLanguage={language}
      defaultValue={defaultValue}
      value={value}
      height={height ?? "300px"}
      width={width ?? "100%"}
      options={options}
      onChange={onChange}
      loading={
        <div className="flex items-center justify-center h-full">
          <Progress.Root className="relative overflow-hidden bg-gray-200 rounded-full w-[300px] h-[25px]">
            <Progress.Indicator
              className="w-full h-full bg-blue-500 transition-transform duration-500 ease-[cubic-bezier(0.65, 0, 0.35, 1)]"
              style={{ transform: "translateX(-100%)" }}
            />
          </Progress.Root>
        </div>
      }
    />
  );
};

export default Monaco;
