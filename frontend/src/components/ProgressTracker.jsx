import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { CheckCircle, Trophy, Star, Target } from 'lucide-react';

const ProgressTracker = ({ selectedLessonId }) => {
  const [progress, setProgress] = useState({
    completedLessons: [],
    completedQuizzes: [],
    badges: [],
    totalLessons: 9,
    totalModules: 3
  });

  useEffect(() => {
    loadProgress();
  }, []);

  const loadProgress = async () => {
    try {
      // Replace with actual API call
      const response = await fetch('/api/progress');
      if (response.ok) {
        const data = await response.json();
        setProgress(data);
      } else {
        throw new Error('Failed to load progress');
      }
    } catch (error) {
      console.error('Error loading progress:', error);
      // Mock progress data for demo
      setProgress({
        completedLessons: ['lesson-1-1', 'lesson-1-2'],
        completedQuizzes: ['lesson-1-1'],
        badges: [
          { id: 'first-steps', name: 'First Steps', description: 'Completed your first lesson', icon: 'star' },
          { id: 'quiz-master', name: 'Quiz Master', description: 'Passed your first quiz', icon: 'trophy' }
        ],
        totalLessons: 9,
        totalModules: 3
      });
    }
  };

  const saveProgress = async (lessonId, type = 'lesson') => {
    try {
      const response = await fetch('/api/progress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lesson_id: lessonId,
          type: type,
          completed: true
        }),
      });
      
      if (response.ok) {
        loadProgress(); // Reload progress
      }
    } catch (error) {
      console.error('Error saving progress:', error);
    }
  };

  const markLessonComplete = (lessonId) => {
    if (!progress.completedLessons.includes(lessonId)) {
      setProgress(prev => ({
        ...prev,
        completedLessons: [...prev.completedLessons, lessonId]
      }));
      saveProgress(lessonId, 'lesson');
    }
  };

  const markQuizComplete = (lessonId) => {
    if (!progress.completedQuizzes.includes(lessonId)) {
      setProgress(prev => ({
        ...prev,
        completedQuizzes: [...prev.completedQuizzes, lessonId]
      }));
      saveProgress(lessonId, 'quiz');
    }
  };

  const getCompletionPercentage = () => {
    return Math.round((progress.completedLessons.length / progress.totalLessons) * 100);
  };

  const getModuleProgress = () => {
    const modules = [
      { id: 'module-1', lessons: ['lesson-1-1', 'lesson-1-2', 'lesson-1-3'] },
      { id: 'module-2', lessons: ['lesson-2-1', 'lesson-2-2', 'lesson-2-3'] },
      { id: 'module-3', lessons: ['lesson-3-1', 'lesson-3-2', 'lesson-3-3'] }
    ];

    return modules.map(module => {
      const completed = module.lessons.filter(lesson => 
        progress.completedLessons.includes(lesson)
      ).length;
      return {
        ...module,
        completed,
        total: module.lessons.length,
        percentage: Math.round((completed / module.lessons.length) * 100)
      };
    });
  };

  const getBadgeIcon = (iconType) => {
    switch (iconType) {
      case 'star':
        return <Star className="h-4 w-4" />;
      case 'trophy':
        return <Trophy className="h-4 w-4" />;
      case 'target':
        return <Target className="h-4 w-4" />;
      default:
        return <CheckCircle className="h-4 w-4" />;
    }
  };

  const moduleProgress = getModuleProgress();

  return (
    <div className="space-y-4">
      {/* Overall Progress */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Target className="h-5 w-5" />
            Your Progress
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span>Overall Completion</span>
              <span>{getCompletionPercentage()}%</span>
            </div>
            <Progress value={getCompletionPercentage()} className="h-2" />
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-primary">
                {progress.completedLessons.length}
              </div>
              <div className="text-xs text-muted-foreground">Lessons</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-500">
                {progress.completedQuizzes.length}
              </div>
              <div className="text-xs text-muted-foreground">Quizzes</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-yellow-500">
                {progress.badges.length}
              </div>
              <div className="text-xs text-muted-foreground">Badges</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Module Progress */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Module Progress</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {moduleProgress.map((module, index) => (
            <div key={module.id} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Module {index + 1}</span>
                <span className="text-xs text-muted-foreground">
                  {module.completed}/{module.total}
                </span>
              </div>
              <Progress value={module.percentage} className="h-1" />
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Badges */}
      {progress.badges.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Trophy className="h-5 w-5" />
              Achievements
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {progress.badges.map((badge) => (
                <div key={badge.id} className="flex items-center gap-3 p-2 rounded-lg bg-muted">
                  <div className="flex-shrink-0 w-8 h-8 bg-primary rounded-full flex items-center justify-center text-primary-foreground">
                    {getBadgeIcon(badge.icon)}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-sm">{badge.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {badge.description}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Quick Actions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <button
            onClick={() => markLessonComplete(selectedLessonId)}
            disabled={!selectedLessonId || progress.completedLessons.includes(selectedLessonId)}
            className="w-full text-left p-2 rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          >
            Mark Current Lesson Complete
          </button>
          <button
            onClick={() => markQuizComplete(selectedLessonId)}
            disabled={!selectedLessonId || progress.completedQuizzes.includes(selectedLessonId)}
            className="w-full text-left p-2 rounded hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          >
            Mark Quiz Complete
          </button>
        </CardContent>
      </Card>
    </div>
  );
};

export default ProgressTracker; 