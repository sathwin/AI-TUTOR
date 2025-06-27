import React, { useState, useEffect } from 'react';
import { ScrollArea } from './ui/scroll-area';
import { Button } from './ui/button';
import { ChevronDown, ChevronRight, BookOpen, Play } from 'lucide-react';

const Sidebar = ({ onSelectLesson, selectedLessonId }) => {
  const [modules, setModules] = useState([]);
  const [expandedModules, setExpandedModules] = useState(new Set());

  // Mock data - replace with API call
  useEffect(() => {
    const mockModules = [
      {
        id: 'module-1',
        title: 'Introduction to CUDA',
        lessons: [
          { id: 'lesson-1-1', title: 'What is CUDA?', completed: true },
          { id: 'lesson-1-2', title: 'GPU Architecture', completed: true },
          { id: 'lesson-1-3', title: 'First CUDA Program', completed: false },
        ]
      },
      {
        id: 'module-2',
        title: 'Memory Management',
        lessons: [
          { id: 'lesson-2-1', title: 'Global Memory', completed: false },
          { id: 'lesson-2-2', title: 'Shared Memory', completed: false },
          { id: 'lesson-2-3', title: 'Constant Memory', completed: false },
        ]
      },
      {
        id: 'module-3',
        title: 'Advanced Topics',
        lessons: [
          { id: 'lesson-3-1', title: 'Streams and Events', completed: false },
          { id: 'lesson-3-2', title: 'Optimization Techniques', completed: false },
          { id: 'lesson-3-3', title: 'Profiling and Debugging', completed: false },
        ]
      }
    ];
    setModules(mockModules);
    setExpandedModules(new Set(['module-1']));
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

  return (
    <div className="w-80 bg-card border-r border-border h-full flex flex-col">
      <div className="p-4 border-b border-border">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          Course Content
        </h2>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="p-2">
          {modules.map((module) => (
            <div key={module.id} className="mb-2">
              <Button
                variant="ghost"
                className="w-full justify-start p-3 h-auto text-left"
                onClick={() => toggleModule(module.id)}
              >
                {expandedModules.has(module.id) ? (
                  <ChevronDown className="h-4 w-4 mr-2 flex-shrink-0" />
                ) : (
                  <ChevronRight className="h-4 w-4 mr-2 flex-shrink-0" />
                )}
                <span className="font-medium">{module.title}</span>
              </Button>
              
              {expandedModules.has(module.id) && (
                <div className="ml-6 mt-1 space-y-1">
                  {module.lessons.map((lesson) => (
                    <Button
                      key={lesson.id}
                      variant={selectedLessonId === lesson.id ? "secondary" : "ghost"}
                      className="w-full justify-start p-2 h-auto text-left text-sm"
                      onClick={() => onSelectLesson(lesson.id)}
                    >
                      <div className="flex items-center gap-2 w-full">
                        <Play className="h-3 w-3 flex-shrink-0" />
                        <span className="flex-1">{lesson.title}</span>
                        {lesson.completed && (
                          <div className="h-2 w-2 bg-green-500 rounded-full flex-shrink-0" />
                        )}
                      </div>
                    </Button>
                  ))}
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