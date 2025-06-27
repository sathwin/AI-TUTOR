import React, { useState, useEffect } from 'react';

import { Card, CardContent } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { HelpCircle, Loader2 } from 'lucide-react';

const LessonContent = ({ lessonId }) => {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedTerm, setSelectedTerm] = useState('');
  const [explanation, setExplanation] = useState(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);

  useEffect(() => {
    if (!lessonId) return;
    
    setLoading(true);
    // Mock lesson content - replace with API call to fetch lesson HTML/markdown
    setTimeout(() => {
      const mockContent = getLessonContent(lessonId);
      setContent(mockContent);
      setLoading(false);
    }, 500);
  }, [lessonId]);

  const getLessonContent = (id) => {
    const lessons = {
      'lesson-1-1': `
        <h1>What is CUDA?</h1>
        <p>CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA for general computing on <span class="highlight-term" data-term="GPU">GPUs</span>.</p>
        
        <h2>Key Concepts</h2>
        <ul>
          <li><span class="highlight-term" data-term="parallel computing">Parallel computing</span> enables simultaneous execution of multiple tasks</li>
          <li><span class="highlight-term" data-term="kernel">Kernels</span> are functions that run on the GPU</li>
          <li><span class="highlight-term" data-term="thread">Threads</span> are the basic unit of execution</li>
        </ul>
        
        <h2>CUDA Programming Model</h2>
        <p>CUDA extends C/C++ with keywords for parallel programming. The main components include:</p>
        <ul>
          <li><strong>Host:</strong> The CPU and its memory</li>
          <li><strong>Device:</strong> The GPU and its memory</li>
          <li><strong>Grid:</strong> A collection of thread blocks</li>
          <li><strong>Block:</strong> A group of threads that can cooperate</li>
        </ul>
      `,
      'lesson-1-2': `
        <h1>GPU Architecture</h1>
        <p>Understanding GPU architecture is crucial for effective CUDA programming.</p>
        
        <h2>GPU vs CPU</h2>
        <p>GPUs are designed for <span class="highlight-term" data-term="throughput">throughput</span>, while CPUs are optimized for <span class="highlight-term" data-term="latency">latency</span>.</p>
        
        <h2>Streaming Multiprocessors (SMs)</h2>
        <p>Each <span class="highlight-term" data-term="SM">SM</span> contains multiple CUDA cores and manages thread execution.</p>
      `,
      'lesson-1-3': `
        <h1>First CUDA Program</h1>
        <p>Let's write our first CUDA program to add two vectors.</p>
        
        <h2>Vector Addition</h2>
        <p>Vector addition is a perfect example of <span class="highlight-term" data-term="data parallelism">data parallelism</span>.</p>
        
        <pre><code>
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
        </code></pre>
      `
    };
    
    return lessons[id] || '<p>Lesson content not found.</p>';
  };

  const handleTermClick = async (term) => {
    if (!term) return;
    
    setSelectedTerm(term);
    setLoadingExplanation(true);
    setExplanation(null);
    
    try {
      // Replace with actual API call
      const response = await fetch(`/api/explain?term=${encodeURIComponent(term)}`);
      if (response.ok) {
        const data = await response.json();
        setExplanation(data);
      } else {
        // Mock explanation for demo
        setExplanation({
          snippets: [
            {
              text: `${term} is a fundamental concept in parallel computing and CUDA programming.`,
              source: 'CUDA Programming Guide'
            }
          ]
        });
      }
    } catch (error) {
      console.error('Error fetching explanation:', error);
      // Mock explanation for demo
      setExplanation({
        snippets: [
          {
            text: `${term} is a fundamental concept in parallel computing and CUDA programming.`,
            source: 'CUDA Programming Guide'
          }
        ]
      });
    } finally {
      setLoadingExplanation(false);
    }
  };

  const renderContent = () => {
    if (!content) return null;
    
    return (
      <div 
        dangerouslySetInnerHTML={{ __html: content }}
        onClick={(e) => {
          if (e.target.classList.contains('highlight-term')) {
            const term = e.target.getAttribute('data-term');
            handleTermClick(term);
          }
        }}
        className="prose prose-invert max-w-none"
      />
    );
  };

  if (!lessonId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-muted-foreground">Select a lesson to begin learning</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-background">
      <ScrollArea className="flex-1 p-6">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            <span>Loading lesson content...</span>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            {renderContent()}
            
            {explanation && (
              <Card className="mt-6 border-blue-200 bg-blue-50 dark:bg-blue-950 dark:border-blue-800">
                <CardContent className="p-4">
                  <div className="flex items-start gap-2">
                    <HelpCircle className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                        Explanation: {selectedTerm}
                      </h4>
                      {explanation.snippets.map((snippet, index) => (
                        <div key={index} className="mb-2">
                          <p className="text-blue-800 dark:text-blue-200">{snippet.text}</p>
                          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                            Source: {snippet.source}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
            
            {loadingExplanation && (
              <Card className="mt-6 border-blue-200 bg-blue-50 dark:bg-blue-950 dark:border-blue-800">
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm">Loading explanation...</span>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </ScrollArea>
      
      <style jsx>{`
        .highlight-term {
          background-color: rgba(59, 130, 246, 0.3);
          padding: 2px 4px;
          border-radius: 4px;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        .highlight-term:hover {
          background-color: rgba(59, 130, 246, 0.5);
        }
        .prose h1 { font-size: 2rem; font-weight: bold; margin-bottom: 1rem; }
        .prose h2 { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.75rem; margin-top: 2rem; }
        .prose p { margin-bottom: 1rem; line-height: 1.6; }
        .prose ul { margin-bottom: 1rem; padding-left: 1.5rem; }
        .prose li { margin-bottom: 0.5rem; }
        .prose pre { background-color: rgba(0, 0, 0, 0.8); padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }
        .prose code { font-family: 'Courier New', monospace; }
      `}</style>
    </div>
  );
};

export default LessonContent; 