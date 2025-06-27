import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ProfilerChart = ({ data }) => {
  if (!data || (!data.gpu_utilization && !data.memory_usage)) {
    return (
      <div className="h-48 flex items-center justify-center text-muted-foreground">
        No profiling data available
      </div>
    );
  }

  // Combine GPU utilization and memory data
  const chartData = data.gpu_utilization?.map((point, index) => ({
    time: point.time,
    gpu_util: point.value,
    memory: data.memory_usage?.[index]?.value || 0
  })) || [];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium">{`Time: ${label}s`}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {`${entry.name}: ${entry.value}%`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              dataKey="time" 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              label={{ value: 'Usage (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="gpu_util"
              stroke="#22c55e"
              strokeWidth={2}
              dot={{ fill: '#22c55e', strokeWidth: 2, r: 4 }}
              name="GPU Utilization"
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="memory"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              name="Memory Usage"
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      {/* Performance Summary */}
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div className="bg-muted rounded-lg p-3">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="font-medium">Peak GPU Utilization</span>
          </div>
          <div className="text-lg font-bold mt-1">
            {Math.max(...(data.gpu_utilization?.map(p => p.value) || [0]))}%
          </div>
        </div>
        
        <div className="bg-muted rounded-lg p-3">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="font-medium">Peak Memory Usage</span>
          </div>
          <div className="text-lg font-bold mt-1">
            {Math.max(...(data.memory_usage?.map(p => p.value) || [0]))}%
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilerChart; 