import React, { useState, useEffect } from 'react';
import { ScrollArea } from './ui/scroll-area';
import { Button } from './ui/button';
import { ChevronDown, ChevronRight, BookOpen, Play, CheckCircle, Clock, Lock } from 'lucide-react';

const Sidebar = ({ onSelectLesson, selectedLessonId }) => {
  const [modules, setModules] = useState([]);
  const [expandedModules, setExpandedModules] = useState(new Set());
  const [loading, setLoading] = useState(true);

  // Mock data - replace with API call to your Flask backend
  useEffect(() => {
    // Simulate loading
    setLoading(true);
    setTimeout(() => {
      const mockModules = [
        {
          id: 'module-1',
          title: 'Introduction to CUDA',
          description: 'Learn the fundamentals of CUDA programming',
          progress: 67,
          lessons: [
            { id: 'lesson-1-1', title: 'What is CUDA?', completed: true, duration: '15 min' },
            { id: 'lesson-1-2', title: 'GPU Architecture', completed: true, duration: '20 min' },
            { id: 'lesson-1-3', title: 'First CUDA Program', completed: false, duration: '25 min' },
          ]
        },
        {
          id: 'module-2',
          title: 'Memory Management',
          description: 'Master CUDA memory hierarchy',
          progress: 0,
          lessons: [
            { id: 'lesson-2-1', title: 'Global Memory', completed: false, duration: '18 min' },
            { id: 'lesson-2-2', title: 'Shared Memory', completed: false, duration: '22 min' },
            { id: 'lesson-2-3', title: 'Constant Memory', completed: false, duration: '16 min' },
          ]
        },
        {
          id: 'module-3',
          title: 'Advanced Topics',
          description: 'Optimize and debug CUDA applications',
          progress: 0,
          locked: true,
          lessons: [
            { id: 'lesson-3-1', title: 'Streams and Events', completed: false, duration: '30 min' },
            { id: 'lesson-3-2', title: 'Optimization Techniques', completed: false, duration: '35 min' },
            { id: 'lesson-3-3', title: 'Profiling and Debugging', completed: false, duration: '28 min' },
          ]
        }
      ];
      setModules(mockModules);
      setExpandedModules(new Set(['module-1']));
      setLoading(false);
    }, 800);
  }, []);

  const toggleModule = (moduleId) => {
    const newExpanded = new Set(expandedModules);
    if (newExpanded.has(moduleId)) {
      newExpanded.delete(moduleId);
    } else {
      newExpanded.add(moduleId);
    }
    setExpandedModules(newExpanded);
  };

  const getLessonIcon = (lesson) => {
    if (lesson.completed) return <CheckCircle className="h-3 w-3 text-green-500" />;
    return <Play className="h-3 w-3 text-muted-foreground" />;
  };

  const getModuleIcon = (module) => {
    if (module.locked) return <Lock className="h-4 w-4 text-muted-foreground" />;
    if (module.progress === 100) return <CheckCircle className="h-4 w-4 text-green-500" />;
    if (module.progress > 0) return <Clock className="h-4 w-4 text-blue-500" />;
    return <BookOpen className="h-4 w-4 text-muted-foreground" />;
  };

  if (loading) {
    return (
      <div className="w-80 bg-card border-r border-border h-full flex flex-col">
        <div className="p-4 border-b border-border">
          <div className="skeleton h-6 w-32"></div>
        </div>
        <div className="p-4 space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-2">
              <div className="skeleton h-12 w-full"></div>
              <div className="ml-6 space-y-2">
                <div className="skeleton h-8 w-full"></div>
                <div className="skeleton h-8 w-full"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-card border-r border-border h-full flex flex-col animate-fade-in">
      <div className="p-6 border-b border-border bg-gradient-to-r from-primary/5 to-primary/10">
        <h2 className="text-xl font-bold flex items-center gap-3 text-primary">
          <div className="p-2 bg-primary/10 rounded-lg">
            <BookOpen className="h-5 w-5" />
          </div>
          Course Content
        </h2>
        <p className="text-sm text-muted-foreground mt-1">Interactive CUDA Learning</p>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="p-3">
          {modules.map((module, index) => (
            <div 
              key={module.id} 
              className={`mb-4 animate-fade-in`}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <Button
                variant="ghost"
                className={`w-full justify-start p-4 h-auto text-left rounded-xl hover:bg-primary/5 transition-all duration-300 ${
                  module.locked ? 'opacity-60 cursor-not-allowed' : ''
                }`}
                onClick={() => !module.locked && toggleModule(module.id)}
                disabled={module.locked}
              >
                <div className="flex items-center gap-3 w-full">
                  <div className="transition-transform duration-200">
                    {expandedModules.has(module.id) ? (
                      <ChevronDown className="h-4 w-4 text-primary" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      {getModuleIcon(module)}
                      <span className="font-semibold text-sm truncate">{module.title}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mb-2">{module.description}</p>
                    
                    {!module.locked && (
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-muted rounded-full h-1.5">
                          <div 
                            className="bg-primary h-1.5 rounded-full transition-all duration-500"
                            style={{ width: `${module.progress}%` }}
                          />
                        </div>
                        <span className="text-xs font-medium text-primary">{module.progress}%</span>
                      </div>
                    )}
                  </div>
                </div>
              </Button>
              
              {expandedModules.has(module.id) && !module.locked && (
                <div className="ml-8 mt-2 space-y-1 animate-slide-in-right">
                  {module.lessons.map((lesson, lessonIndex) => (
                    <Button
                      key={lesson.id}
                      variant={selectedLessonId === lesson.id ? "secondary" : "ghost"}
                      className={`w-full justify-start p-3 h-auto text-left text-sm rounded-lg transition-all duration-200 hover:bg-primary/5 ${
                        selectedLessonId === lesson.id ? 'bg-primary/10 border border-primary/20' : ''
                      }`}
                      onClick={() => onSelectLesson(lesson.id)}
                      style={{ animationDelay: `${lessonIndex * 50}ms` }}
                    >
                      <div className="flex items-center gap-3 w-full">
                        {getLessonIcon(lesson)}
                        <div className="flex-1 min-w-0">
                          <p className="font-medium truncate">{lesson.title}</p>
                          <p className="text-xs text-muted-foreground">{lesson.duration}</p>
                        </div>
                        {lesson.completed && (
                          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse-slow" />
                        )}
                      </div>
                    </Button>
                  ))}
                </div>
              )}
              
              {module.locked && (
                <div className="ml-8 mt-2 p-3 bg-muted/50 rounded-lg border border-dashed border-border">
                  <p className="text-xs text-muted-foreground flex items-center gap-2">
                    <Lock className="h-3 w-3" />
                    Complete previous modules to unlock
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};

export default Sidebar; 