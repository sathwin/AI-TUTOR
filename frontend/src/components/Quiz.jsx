import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { CheckCircle, XCircle, Loader2 } from 'lucide-react';

const Quiz = ({ lessonId }) => {
  const [quizData, setQuizData] = useState(null);
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (lessonId) {
      loadQuiz(lessonId);
    }
  }, [lessonId]);

  const loadQuiz = (id) => {
    // Mock quiz data - replace with API call
    const quizzes = {
      'lesson-1-1': {
        title: 'CUDA Basics Quiz',
        questions: [
          {
            id: 'q1',
            type: 'multiple-choice',
            question: 'What does CUDA stand for?',
            options: [
              'Compute Unified Device Architecture',
              'Central Unit Data Architecture',
              'Computer Universal Device Application',
              'Core Unified Data Architecture'
            ],
            correct: 0
          },
          {
            id: 'q2',
            type: 'multiple-choice',
            question: 'Which company developed CUDA?',
            options: ['AMD', 'Intel', 'NVIDIA', 'Qualcomm'],
            correct: 2
          },
          {
            id: 'q3',
            type: 'fill-in',
            question: 'Complete the sentence: CUDA kernels are functions that run on the ____.',
            correct: 'GPU'
          }
        ]
      },
      'lesson-1-2': {
        title: 'GPU Architecture Quiz',
        questions: [
          {
            id: 'q1',
            type: 'multiple-choice',
            question: 'What are GPUs optimized for?',
            options: ['Latency', 'Throughput', 'Power efficiency', 'Memory capacity'],
            correct: 1
          },
          {
            id: 'q2',
            type: 'fill-in',
            question: 'A Streaming Multiprocessor is abbreviated as ____.',
            correct: 'SM'
          }
        ]
      },
      'lesson-1-3': {
        title: 'First CUDA Program Quiz',
        questions: [
          {
            id: 'q1',
            type: 'multiple-choice',
            question: 'What type of parallelism does vector addition demonstrate?',
            options: ['Task parallelism', 'Data parallelism', 'Pipeline parallelism', 'Instruction parallelism'],
            correct: 1
          },
          {
            id: 'q2',
            type: 'fill-in',
            question: 'In CUDA, thread ID is calculated as: blockIdx.x * blockDim.x + ____',
            correct: 'threadIdx.x'
          }
        ]
      }
    };

    setQuizData(quizzes[id] || null);
    setAnswers({});
    setSubmitted(false);
    setResults(null);
  };

  const handleAnswerChange = (questionId, answer) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
  };

  const submitQuiz = async () => {
    if (!quizData) return;

    setLoading(true);
    
    try {
      // Replace with actual API call
      const response = await fetch('/api/quiz', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lesson_id: lessonId,
          answers
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        throw new Error('Failed to submit quiz');
      }
    } catch (error) {
      console.error('Error submitting quiz:', error);
      // Mock results for demo
      generateMockResults();
    } finally {
      setLoading(false);
      setSubmitted(true);
    }
  };

  const generateMockResults = () => {
    const mockResults = {
      correct: 0,
      total: quizData.questions.length,
      feedback: {},
      passed: false
    };

    quizData.questions.forEach(question => {
      const userAnswer = answers[question.id];
      let isCorrect = false;

      if (question.type === 'multiple-choice') {
        isCorrect = userAnswer === question.correct;
      } else if (question.type === 'fill-in') {
        isCorrect = userAnswer?.toLowerCase().trim() === question.correct.toLowerCase().trim();
      }

      if (isCorrect) {
        mockResults.correct++;
      }

      mockResults.feedback[question.id] = {
        correct: isCorrect,
        explanation: isCorrect 
          ? 'Correct! Well done.' 
          : `Incorrect. The correct answer is: ${
              question.type === 'multiple-choice' 
                ? question.options[question.correct]
                : question.correct
            }`
      };
    });

    mockResults.passed = mockResults.correct >= Math.ceil(mockResults.total * 0.7);
    setResults(mockResults);
  };

  const resetQuiz = () => {
    setAnswers({});
    setSubmitted(false);
    setResults(null);
  };

  if (!quizData) {
    return (
      <Card className="h-full flex items-center justify-center">
        <CardContent>
          <p className="text-muted-foreground">No quiz available for this lesson</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle className="text-lg">{quizData.title}</CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 overflow-auto">
        <div className="space-y-6">
          {quizData.questions.map((question, index) => (
            <div key={question.id} className="space-y-3">
              <h3 className="font-medium">
                {index + 1}. {question.question}
              </h3>
              
              {question.type === 'multiple-choice' && (
                <div className="space-y-2">
                  {question.options.map((option, optionIndex) => (
                    <label
                      key={optionIndex}
                      className={`flex items-center space-x-2 p-2 rounded cursor-pointer transition-colors ${
                        answers[question.id] === optionIndex
                          ? 'bg-primary/10 border border-primary'
                          : 'hover:bg-muted'
                      } ${
                        submitted && results?.feedback[question.id]
                          ? optionIndex === question.correct
                            ? 'bg-green-100 dark:bg-green-900/20 border-green-500'
                            : answers[question.id] === optionIndex && optionIndex !== question.correct
                            ? 'bg-red-100 dark:bg-red-900/20 border-red-500'
                            : ''
                          : ''
                      }`}
                    >
                      <input
                        type="radio"
                        name={question.id}
                        value={optionIndex}
                        checked={answers[question.id] === optionIndex}
                        onChange={() => handleAnswerChange(question.id, optionIndex)}
                        disabled={submitted}
                        className="text-primary"
                      />
                      <span className="flex-1">{option}</span>
                      {submitted && results?.feedback[question.id] && (
                        <>
                          {optionIndex === question.correct && (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          )}
                          {answers[question.id] === optionIndex && optionIndex !== question.correct && (
                            <XCircle className="h-4 w-4 text-red-500" />
                          )}
                        </>
                      )}
                    </label>
                  ))}
                </div>
              )}
              
              {question.type === 'fill-in' && (
                <div>
                  <Input
                    value={answers[question.id] || ''}
                    onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                    placeholder="Type your answer here..."
                    disabled={submitted}
                    className={
                      submitted && results?.feedback[question.id]
                        ? results.feedback[question.id].correct
                          ? 'border-green-500'
                          : 'border-red-500'
                        : ''
                    }
                  />
                </div>
              )}
              
              {submitted && results?.feedback[question.id] && (
                <div className={`p-3 rounded-lg text-sm ${
                  results.feedback[question.id].correct
                    ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200'
                    : 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200'
                }`}>
                  {results.feedback[question.id].explanation}
                </div>
              )}
            </div>
          ))}
        </div>
        
        {!submitted && (
          <div className="mt-6 pt-4 border-t border-border">
            <Button
              onClick={submitQuiz}
              disabled={loading || Object.keys(answers).length !== quizData.questions.length}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Submitting...
                </>
              ) : (
                'Submit Quiz'
              )}
            </Button>
          </div>
        )}
        
        {submitted && results && (
          <div className="mt-6 pt-4 border-t border-border">
            <div className={`p-4 rounded-lg text-center ${
              results.passed
                ? 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200'
                : 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200'
            }`}>
              <h3 className="font-semibold mb-2">
                {results.passed ? '🎉 Congratulations!' : '📚 Keep Learning!'}
              </h3>
              <p>
                You scored {results.correct} out of {results.total} questions correctly.
              </p>
              {results.passed ? (
                <p className="text-sm mt-1">You've mastered this lesson!</p>
              ) : (
                <p className="text-sm mt-1">Review the material and try again.</p>
              )}
            </div>
            
            <Button
              onClick={resetQuiz}
              variant="outline"
              className="w-full mt-4"
            >
              Retake Quiz
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default Quiz; 