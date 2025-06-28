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
        className="prose prose-invert max-w-none animate-fade-in lesson-content"
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
      <ScrollArea className="flex-1">
        {loading ? (
          <div className="flex items-center justify-center py-16">
            <div className="text-center space-y-4">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Loading lesson content...</h3>
                <p className="text-muted-foreground">Preparing your learning materials</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto p-8">
            {/* Lesson header */}
            <div className="mb-8 p-6 bg-gradient-to-r from-primary/5 to-primary/10 rounded-2xl border border-primary/20">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse-slow"></div>
                <span className="text-sm font-medium text-primary">Interactive Lesson</span>
              </div>
              <p className="text-muted-foreground text-sm">
                Click on highlighted terms for detailed explanations
              </p>
            </div>

            {renderContent()}
            
            {explanation && (
              <Card className="mt-8 border-primary/20 bg-gradient-to-r from-primary/5 to-primary/10 shadow-lg animate-scale-in">
                <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0">
                      <HelpCircle className="h-5 w-5 text-primary" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-3">
                        <h4 className="font-bold text-primary text-lg">
                          {selectedTerm}
                        </h4>
                        <div className="px-2 py-1 bg-primary/10 rounded-full text-xs font-medium text-primary">
                          AI Explanation
                        </div>
                      </div>
                      {explanation.snippets.map((snippet, index) => (
                        <div key={index} className="mb-4 last:mb-0">
                          <p className="text-foreground leading-relaxed mb-2">{snippet.text}</p>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <div className="w-1 h-1 bg-primary rounded-full"></div>
                            <span>Source: {snippet.source}</span>
                          </div>
                        </div>
                      ))}
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => setExplanation(null)}
                        className="mt-4"
                      >
                        Close
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
            
            {loadingExplanation && (
              <Card className="mt-8 border-primary/20 bg-gradient-to-r from-primary/5 to-primary/10 shadow-lg animate-fade-in">
                <CardContent className="p-6">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center">
                      <Loader2 className="h-5 w-5 animate-spin text-primary" />
                    </div>
                    <div>
                      <h4 className="font-semibold text-primary mb-1">Generating explanation...</h4>
                      <p className="text-sm text-muted-foreground">AI is analyzing "{selectedTerm}"</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Lesson navigation */}
            <div className="mt-12 flex items-center justify-between p-6 bg-muted/50 rounded-2xl border border-border">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                  <span className="text-sm font-bold text-primary">💡</span>
                </div>
                <div>
                  <h4 className="font-semibold">Ready to practice?</h4>
                  <p className="text-sm text-muted-foreground">Try the interactive features in the right panel</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm">Previous</Button>
                <Button variant="gradient" size="sm">Next Lesson</Button>
              </div>
            </div>
          </div>
        )}
      </ScrollArea>
      
      <style jsx>{`
        .highlight-term {
          background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(59, 130, 246, 0.2));
          padding: 3px 6px;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.3s ease;
          border: 1px solid rgba(34, 197, 94, 0.3);
          font-weight: 500;
          position: relative;
        }
        .highlight-term:hover {
          background: linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(59, 130, 246, 0.3));
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
          border-color: rgba(34, 197, 94, 0.5);
        }
        .highlight-term::before {
          content: '🔍';
          position: absolute;
          right: -2px;
          top: -2px;
          font-size: 10px;
          opacity: 0;
          transition: opacity 0.2s ease;
        }
        .highlight-term:hover::before {
          opacity: 1;
        }
        .lesson-content h1 { 
          font-size: 2.5rem; 
          font-weight: bold; 
          margin-bottom: 1.5rem; 
          background: linear-gradient(135deg, hsl(var(--primary)), hsl(var(--primary)) 50%, hsl(var(--accent)));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .lesson-content h2 { 
          font-size: 1.75rem; 
          font-weight: bold; 
          margin-bottom: 1rem; 
          margin-top: 2.5rem; 
          color: hsl(var(--primary));
          padding-bottom: 0.5rem;
          border-bottom: 2px solid hsl(var(--primary) / 0.2);
        }
        .lesson-content p { 
          margin-bottom: 1.25rem; 
          line-height: 1.7; 
          font-size: 1.1rem;
        }
        .lesson-content ul { 
          margin-bottom: 1.25rem; 
          padding-left: 1.5rem; 
        }
        .lesson-content li { 
          margin-bottom: 0.75rem; 
          line-height: 1.6;
        }
        .lesson-content pre { 
          background: linear-gradient(135deg, rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.8)); 
          padding: 1.5rem; 
          border-radius: 1rem; 
          overflow-x: auto; 
          border: 1px solid hsl(var(--border));
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          margin: 1.5rem 0;
        }
        .lesson-content code { 
          font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace; 
          font-size: 0.9rem;
        }
        .lesson-content strong {
          color: hsl(var(--primary));
          font-weight: 600;
        }
      `}</style>
    </div>
  );
};

export default LessonContent; 