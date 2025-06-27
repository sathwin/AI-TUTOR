import React, { useState, useEffect } from 'react';
import './index.css';
import Sidebar from './components/Sidebar';
import LessonContent from './components/LessonContent';
import InteractivePane from './components/InteractivePane';
import ProgressTracker from './components/ProgressTracker';
import { Button } from './components/ui/button';
import { Card } from './components/ui/card';
import { Menu, X, BarChart3 } from 'lucide-react';

function App() {
  const [selectedLessonId, setSelectedLessonId] = useState('lesson-1-1');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showProgress, setShowProgress] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkScreenSize = () => {
      setIsMobile(window.innerWidth < 1024);
      if (window.innerWidth < 1024) {
        setSidebarCollapsed(true);
      }
    };

    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  const handleSelectLesson = (lessonId) => {
    setSelectedLessonId(lessonId);
    if (isMobile) {
      setSidebarCollapsed(true);
    }
  };

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  const toggleProgress = () => {
    setShowProgress(!showProgress);
  };

  return (
    <div className="h-screen bg-background text-foreground dark">
      <div className="flex h-full">
        {/* Mobile overlay */}
        {!sidebarCollapsed && isMobile && (
          <div 
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
            onClick={() => setSidebarCollapsed(true)}
          />
        )}

        {/* Sidebar */}
        <div className={`${
          sidebarCollapsed ? '-translate-x-full lg:translate-x-0 lg:w-16' : 'translate-x-0 w-80'
        } fixed lg:relative h-full transition-all duration-300 z-50 lg:z-auto`}>
          {sidebarCollapsed ? (
            <Card className="h-full w-16 rounded-none border-r border-border flex flex-col items-center py-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleSidebar}
                className="mb-4"
              >
                <Menu className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleProgress}
                className={showProgress ? "bg-accent" : ""}
              >
                <BarChart3 className="h-5 w-5" />
              </Button>
            </Card>
          ) : (
            <div className="relative h-full">
              <Sidebar 
                onSelectLesson={handleSelectLesson}
                selectedLessonId={selectedLessonId}
              />
              {/* Close button for mobile */}
              {isMobile && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleSidebar}
                  className="absolute top-4 right-4 z-10"
                >
                  <X className="h-5 w-5" />
                </Button>
              )}
            </div>
          )}
        </div>

        {/* Main content area */}
        <div className="flex-1 flex">
          {/* Progress tracker (when sidebar is collapsed) */}
          {sidebarCollapsed && showProgress && (
            <div className="w-80 border-r border-border overflow-auto">
              <div className="p-4">
                <ProgressTracker selectedLessonId={selectedLessonId} />
              </div>
            </div>
          )}

          {/* Lesson content */}
          <div className="flex-1 flex">
            <div className="flex-1">
              {/* Mobile header */}
              {isMobile && sidebarCollapsed && (
                <div className="bg-card border-b border-border p-4 flex items-center justify-between">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={toggleSidebar}
                  >
                    <Menu className="h-5 w-5" />
                  </Button>
                  <h1 className="text-lg font-semibold">AI Tutor</h1>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={toggleProgress}
                    className={showProgress ? "bg-accent" : ""}
                  >
                    <BarChart3 className="h-5 w-5" />
                  </Button>
                </div>
              )}
              
              <LessonContent lessonId={selectedLessonId} />
            </div>

            {/* Interactive pane */}
            <div className={`${isMobile ? 'hidden lg:block' : ''}`}>
              <InteractivePane lessonId={selectedLessonId} />
            </div>
          </div>
        </div>
      </div>

      {/* Mobile bottom navigation for interactive features */}
      {isMobile && (
        <div className="fixed bottom-0 left-0 right-0 bg-card border-t border-border lg:hidden">
          <div className="flex">
            <Button
              variant="ghost"
              className="flex-1 h-12 rounded-none"
              onClick={() => {
                // Show chat modal or navigate to chat view
                console.log('Open chat');
              }}
            >
              Chat
            </Button>
            <Button
              variant="ghost"
              className="flex-1 h-12 rounded-none"
              onClick={() => {
                // Show code lab modal or navigate to code view
                console.log('Open code lab');
              }}
            >
              Code Lab
            </Button>
            <Button
              variant="ghost"
              className="flex-1 h-12 rounded-none"
              onClick={() => {
                // Show quiz modal or navigate to quiz view
                console.log('Open quiz');
              }}
            >
              Quiz
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App; 