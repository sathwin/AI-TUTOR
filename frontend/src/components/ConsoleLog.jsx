import React, { useEffect, useRef } from 'react';
import { ScrollArea } from './ui/scroll-area';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Terminal } from 'lucide-react';

const ConsoleLog = ({ logs = [], isConnected = false }) => {
  const scrollAreaRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    // Auto-scroll to bottom when new logs arrive
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const formatLogLine = (log, index) => {
    const timestamp = new Date(log.timestamp).toLocaleTimeString();
    let className = 'font-mono text-sm ';
    
    switch (log.type) {
      case 'stdout':
        className += 'text-foreground';
        break;
      case 'stderr':
        className += 'text-red-400';
        break;
      case 'info':
        className += 'text-blue-400';
        break;
      case 'warning':
        className += 'text-yellow-400';
        break;
      case 'error':
        className += 'text-red-500';
        break;
      default:
        className += 'text-muted-foreground';
    }

    return (
      <div key={index} className="flex gap-2 py-1">
        <span className="text-muted-foreground text-xs flex-shrink-0 font-mono">
          {timestamp}
        </span>
        <span className={className}>{log.message}</span>
      </div>
    );
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Terminal className="h-4 w-4" />
          Console Output
          {isConnected && (
            <div className="flex items-center gap-2 ml-auto">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-xs text-muted-foreground">Connected</span>
            </div>
          )}
        </CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 p-0">
        <ScrollArea className="h-full p-4" ref={scrollAreaRef}>
          <div className="bg-black/20 rounded-md p-3 font-mono text-sm">
            {logs.length === 0 ? (
              <div className="text-muted-foreground italic">
                No output yet. Run some code to see results here.
              </div>
            ) : (
              <div className="space-y-1">
                {logs.map((log, index) => formatLogLine(log, index))}
                <div ref={endRef} />
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default ConsoleLog; 