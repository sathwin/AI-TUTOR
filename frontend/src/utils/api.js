// API utility functions for backend communication

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiClient {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Chat with AI tutor
  async chat(prompt, context = null, conversationHistory = []) {
    return this.request('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        prompt,
        context,
        conversation_history: conversationHistory
      }),
    });
  }

  // Execute code
  async runCode(code) {
    return this.request('/api/run-code', {
      method: 'POST',
      body: JSON.stringify({ code }),
    });
  }

  // Get job status
  async getJobStatus(jobId) {
    return this.request(`/api/jobs/${jobId}/status`);
  }

  // Get profiling data
  async getProfilingData(jobId) {
    return this.request(`/api/profiling/${jobId}`);
  }

  // Get course modules
  async getModules() {
    return this.request('/api/modules');
  }

  // Get lesson content
  async getLesson(lessonId) {
    return this.request(`/api/lessons/${lessonId}`);
  }

  // Get quiz
  async getQuiz(lessonId) {
    return this.request(`/api/quiz/${lessonId}`);
  }

  // Submit quiz
  async submitQuiz(lessonId, answers) {
    return this.request('/api/quiz', {
      method: 'POST',
      body: JSON.stringify({
        lesson_id: lessonId,
        answers
      }),
    });
  }

  // Get progress
  async getProgress(user = 'demo') {
    return this.request(`/api/progress?user=${user}`);
  }

  // Update progress
  async updateProgress(lessonId, type, completed, user = 'demo') {
    return this.request(`/api/progress?user=${user}`, {
      method: 'POST',
      body: JSON.stringify({
        lesson_id: lessonId,
        type,
        completed
      }),
    });
  }

  // Explain term
  async explainTerm(term) {
    return this.request(`/api/explain?term=${encodeURIComponent(term)}`);
  }

  // Health check
  async healthCheck() {
    return this.request('/api/health');
  }

  // Create WebSocket connection for logs
  createLogWebSocket(jobId) {
    const wsURL = `${this.baseURL.replace('http', 'ws')}/ws/logs/${jobId}`;
    return new WebSocket(wsURL);
  }
}

// Create and export singleton instance
const apiClient = new ApiClient();
export default apiClient;

// Export individual functions for convenience
export const {
  chat,
  runCode,
  getJobStatus,
  getProfilingData,
  getModules,
  getLesson,
  getQuiz,
  submitQuiz,
  getProgress,
  updateProgress,
  explainTerm,
  healthCheck,
  createLogWebSocket
} = apiClient;