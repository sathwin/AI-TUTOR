import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { CheckCircle, XCircle, Loader2, RefreshCw } from 'lucide-react';

const BackendStatus = () => {
  const [status, setStatus] = useState('checking');
  const [backendInfo, setBackendInfo] = useState(null);
  const [lastCheck, setLastCheck] = useState(null);

  const checkBackendHealth = async () => {
    setStatus('checking');
    try {
      const { default: apiClient } = await import('../utils/api');
      const data = await apiClient.healthCheck();
      setBackendInfo(data);
      setStatus('connected');
      setLastCheck(new Date());
    } catch (error) {
      console.error('Backend health check failed:', error);
      setStatus('disconnected');
      setBackendInfo(null);
      setLastCheck(new Date());
    }
  };

  useEffect(() => {
    checkBackendHealth();
    // Check every 30 seconds
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = () => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'disconnected':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'checking':
      default:
        return <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return 'Backend Connected';
      case 'disconnected':
        return 'Backend Offline';
      case 'checking':
      default:
        return 'Checking...';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return 'text-green-600 dark:text-green-400';
      case 'disconnected':
        return 'text-red-600 dark:text-red-400';
      case 'checking':
      default:
        return 'text-yellow-600 dark:text-yellow-400';
    }
  };

  return (
    <Card className="border-l-4 border-l-primary">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h4 className={`text-sm font-medium ${getStatusColor()}`}>
                {getStatusText()}
              </h4>
              {backendInfo && (
                <div className="text-xs text-muted-foreground">
                  <div>Model: {backendInfo.model_ready ? '✅ Ready' : '⏳ Loading'}</div>
                  <div>Active Jobs: {backendInfo.active_jobs || 0}</div>
                </div>
              )}
              {lastCheck && (
                <p className="text-xs text-muted-foreground">
                  Last check: {lastCheck.toLocaleTimeString()}
                </p>
              )}
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={checkBackendHealth}
            disabled={status === 'checking'}
            className="flex items-center gap-2"
          >
            <RefreshCw className={`h-3 w-3 ${status === 'checking' ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
        
        {status === 'disconnected' && (
          <div className="mt-3 p-3 bg-red-50 dark:bg-red-950/20 rounded-lg border border-red-200 dark:border-red-800">
            <p className="text-sm text-red-700 dark:text-red-400">
              <strong>Backend is offline.</strong> Make sure the FastAPI server is running on port 8000.
            </p>
            <p className="text-xs text-red-600 dark:text-red-500 mt-1">
              Run: <code className="bg-red-100 dark:bg-red-900 px-1 rounded">cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload</code>
            </p>
          </div>
        )}
        
        {status === 'connected' && backendInfo && !backendInfo.model_ready && (
          <div className="mt-3 p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <p className="text-sm text-yellow-700 dark:text-yellow-400">
              <strong>AI Model is loading.</strong> This can take 3-5 minutes on first startup.
            </p>
            <p className="text-xs text-yellow-600 dark:text-yellow-500 mt-1">
              Chat and code features will be available once the model is ready.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default BackendStatus;