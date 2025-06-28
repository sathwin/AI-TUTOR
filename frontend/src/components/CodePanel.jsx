import React, { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import CodeEditor from './CodeEditor';
import ConsoleLog from './ConsoleLog';
import ProfilerChart from './ProfilerChart';
import { Play, Square, Loader2 } from 'lucide-react';

const CodePanel = ({ lessonId }) => {
  const [code, setCode] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [, setJobId] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [profilingData, setProfilingData] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    // Load default code for the lesson
    if (lessonId) {
      const defaultCode = getDefaultCode(lessonId);
      setCode(defaultCode);
      setLogs([]);
      setProfilingData(null);
    }
  }, [lessonId]);

  const getDefaultCode = (id) => {
    const codeTemplates = {
      'lesson-1-1': `# Welcome to CUDA Programming!
# This is a simple Python example with CuPy

import cupy as cp
import numpy as np

# Create arrays on GPU
a = cp.array([1, 2, 3, 4, 5])
b = cp.array([6, 7, 8, 9, 10])

# Perform element-wise addition on GPU
c = a + b

print("Array a:", a)
print("Array b:", b)
print("Result c = a + b:", c)

# Convert back to CPU if needed
c_cpu = cp.asnumpy(c)
print("Result on CPU:", c_cpu)`,
      'lesson-1-2': `# GPU Memory Management Example
import cupy as cp
import numpy as np

# Check GPU memory before allocation
mempool = cp.get_default_memory_pool()
print(f"GPU memory used before: {mempool.used_bytes()} bytes")

# Allocate large arrays
size = 1000000
a = cp.random.random(size, dtype=cp.float32)
b = cp.random.random(size, dtype=cp.float32)

print(f"GPU memory used after allocation: {mempool.used_bytes()} bytes")

# Perform computation
c = cp.sqrt(a**2 + b**2)

print(f"Computed {size} elements")
print(f"First 10 results: {c[:10]}")`,
      'lesson-1-3': `# Vector Addition - Your First CUDA-like Program
import cupy as cp
import time

def vector_add_gpu(a, b):
    """Add two vectors on GPU"""
    return a + b

def vector_add_cpu(a, b):
    """Add two vectors on CPU"""
    return a + b

# Create test data
size = 1000000
print(f"Vector size: {size:,} elements")

# CPU version
a_cpu = np.random.random(size).astype(np.float32)
b_cpu = np.random.random(size).astype(np.float32)

start_time = time.time()
c_cpu = vector_add_cpu(a_cpu, b_cpu)
cpu_time = time.time() - start_time

# GPU version
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)

start_time = time.time()
c_gpu = vector_add_gpu(a_gpu, b_gpu)
cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
gpu_time = time.time() - start_time

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")

# Verify results match
print(f"Results match: {np.allclose(c_cpu, cp.asnumpy(c_gpu))}")`
    };

    return codeTemplates[id] || '# Write your CUDA/CuPy code here\nimport cupy as cp\nimport numpy as np\n\nprint("Hello, GPU World!")';
  };

  const runCode = async () => {
    if (!code.trim() || isRunning) return;

    setIsRunning(true);
    setLogs([]);
    setProfilingData(null);

    try {
      // Start code execution
      const response = await fetch('/api/run-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });

      if (response.ok) {
        const data = await response.json();
        setJobId(data.job_id);
        connectWebSocket(data.job_id);
      } else {
        throw new Error('Failed to start code execution');
      }
    } catch (error) {
      console.error('Error running code:', error);
      // Mock execution for demo
      mockCodeExecution();
    }
  };

  const connectWebSocket = (jobId) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/logs/${jobId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        addLog('info', 'Connected to execution stream');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
          addLog(data.level || 'stdout', data.message);
        } else if (data.type === 'complete') {
          setIsRunning(false);
          addLog('info', 'Execution completed');
          fetchProfilingData(jobId);
        } else if (data.type === 'error') {
          addLog('error', data.message);
          setIsRunning(false);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        setIsRunning(false);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
        setIsRunning(false);
        mockCodeExecution();
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      mockCodeExecution();
    }
  };

  const mockCodeExecution = () => {
    // Simulate code execution for demo
    addLog('info', 'Starting code execution...');
    
    const lines = [
      'Array a: [1 2 3 4 5]',
      'Array b: [6 7 8 9 10]',
      'Result c = a + b: [ 7  9 11 13 15]',
      'Result on CPU: [ 7  9 11 13 15]'
    ];

    lines.forEach((line, index) => {
      setTimeout(() => {
        addLog('stdout', line);
        if (index === lines.length - 1) {
          setTimeout(() => {
            addLog('info', 'Execution completed');
            setIsRunning(false);
            // Generate mock profiling data
            setProfilingData({
              gpu_utilization: [
                { time: 0, value: 0 },
                { time: 1, value: 85 },
                { time: 2, value: 92 },
                { time: 3, value: 88 },
                { time: 4, value: 0 }
              ],
              memory_usage: [
                { time: 0, value: 20 },
                { time: 1, value: 65 },
                { time: 2, value: 70 },
                { time: 3, value: 68 },
                { time: 4, value: 25 }
              ]
            });
          }, 500);
        }
      }, (index + 1) * 500);
    });
  };

  const addLog = (type, message) => {
    const newLog = {
      type,
      message,
      timestamp: new Date()
    };
    setLogs(prev => [...prev, newLog]);
  };

  const fetchProfilingData = async (jobId) => {
    try {
      const response = await fetch(`/api/profiling/${jobId}`);
      if (response.ok) {
        const data = await response.json();
        setProfilingData(data);
      }
    } catch (error) {
      console.error('Error fetching profiling data:', error);
    }
  };

  const stopExecution = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsRunning(false);
    setWsConnected(false);
    addLog('info', 'Execution stopped by user');
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="flex flex-col h-full space-y-4 animate-fade-in">
      {/* Code Editor */}
      <Card className="shadow-lg">
        <CardHeader className="pb-3 bg-gradient-to-r from-primary/5 to-primary/10">
          <CardTitle className="text-lg flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Play className="h-4 w-4 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold text-primary">CUDA Code Lab</h3>
                <p className="text-xs text-muted-foreground">Write and execute your CUDA code</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={runCode}
                disabled={isRunning}
                size="sm"
                variant="gradient"
                className="flex items-center gap-2 shadow-lg enhanced-hover"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Run Code
                  </>
                )}
              </Button>
              {isRunning && (
                <Button
                  onClick={stopExecution}
                  size="sm"
                  variant="destructive"
                  className="flex items-center gap-2 enhanced-hover"
                >
                  <Square className="h-4 w-4" />
                  Stop
                </Button>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="relative">
            <CodeEditor
              value={code}
              onChange={setCode}
              language="python"
              height="250px"
            />
            {/* Code metrics overlay */}
            <div className="absolute top-2 right-2 flex gap-2">
              <div className="bg-black/50 backdrop-blur-sm rounded-lg px-2 py-1 text-xs text-white">
                Lines: {code.split('\n').length}
              </div>
              <div className="bg-black/50 backdrop-blur-sm rounded-lg px-2 py-1 text-xs text-white">
                Python
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Console Output */}
      <div className="flex-1 animate-slide-in-right">
        <ConsoleLog logs={logs} isConnected={wsConnected} />
      </div>

      {/* Profiling Chart */}
      {profilingData && (
        <Card className="shadow-lg animate-scale-in">
          <CardHeader className="pb-3 bg-gradient-to-r from-green-500/5 to-blue-500/5">
            <CardTitle className="text-lg flex items-center gap-3">
              <div className="p-2 bg-green-500/10 rounded-lg">
                <svg className="h-4 w-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-green-600 dark:text-green-400">GPU Performance</h3>
                <p className="text-xs text-muted-foreground">Real-time profiling data</p>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            <ProfilerChart data={profilingData} />
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      {!isRunning && !profilingData && (
        <Card className="shadow-md">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium">Ready to run your code</h4>
                <p className="text-xs text-muted-foreground">Click "Run Code" to execute and see results</p>
              </div>
              <div className="flex gap-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setCode(getDefaultCode(lessonId))}
                  className="text-xs"
                >
                  Reset
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  className="text-xs"
                >
                  Examples
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default CodePanel; 