import React from 'react';
import Editor from '@monaco-editor/react';

const CodeEditor = ({ 
  value, 
  onChange, 
  language = 'python', 
  theme = 'vs-dark',
  height = '300px',
  readOnly = false,
  options = {}
}) => {
  const defaultOptions = {
    fontSize: 14,
    wordWrap: 'on',
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    automaticLayout: true,
    tabSize: 2,
    insertSpaces: true,
    renderWhitespace: 'selection',
    ...options
  };

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <Editor
        height={height}
        language={language}
        theme={theme}
        value={value}
        onChange={onChange}
        options={{
          ...defaultOptions,
          readOnly
        }}
        loading={
          <div className="flex items-center justify-center h-full">
            <div className="text-muted-foreground">Loading editor...</div>
          </div>
        }
      />
    </div>
  );
};

export default CodeEditor; 