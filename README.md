# AI Tutor - Interactive CUDA Learning Platform

A comprehensive AI-powered tutoring system for learning CUDA programming with interactive lessons, code execution, and real-time feedback.

## 🚀 **Project Status**

- ✅ **Frontend**: Complete React application with full UI/UX
- 🚧 **Backend**: In development (FastAPI on Sol server)

## 📁 **Project Structure**

```
ai-tutor/
├── frontend/           # React frontend application
│   ├── src/
│   │   ├── components/ # React components (Sidebar, ChatPanel, CodeEditor, etc.)
│   │   └── ...
│   ├── public/
│   └── package.json
├── backend/            # FastAPI backend (coming soon)
└── README.md          # This file
```

## 🎯 **Features**

### ✅ **Frontend (Complete)**
- **Interactive Learning**: Three-pane layout with lesson content, navigation, and interactive features
- **AI Chat Interface**: Real-time chat with AI tutor for questions and explanations
- **Code Editor**: Monaco-based editor with Python/CUDA syntax highlighting
- **Live Code Execution**: Real-time code execution with streaming output
- **GPU Profiling**: Performance visualization with charts
- **Interactive Quizzes**: Multiple-choice and fill-in-the-blank questions
- **Progress Tracking**: Gamified learning with badges and achievements
- **Responsive Design**: Mobile-friendly interface with dark theme

### 🚧 **Backend (In Development)**
- **FastAPI**: Modern Python web framework
- **CUDA Code Execution**: Remote code execution on GPU servers
- **AI Integration**: LLM-powered tutoring and explanations
- **Real-time Communication**: WebSocket support for live features
- **Progress Persistence**: User progress and achievement tracking

## 🚀 **Getting Started**

### **Frontend Only (Available Now)**

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install --legacy-peer-deps
   ```

3. **Start development server:**
   ```bash
   npm start
   ```

4. **Open in browser:**
   ```
   http://localhost:3000
   ```

The frontend works standalone with mock data - you can explore all features immediately!

### **Full System (When Backend is Ready)**

The backend will provide:
- Real AI tutoring responses
- Actual CUDA code execution on GPU
- Persistent user progress
- Real-time collaborative features

## 🛠 **Tech Stack**

### **Frontend**
- **React 18** - Modern React with hooks
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - High-quality component library
- **Monaco Editor** - VS Code editor for web
- **Recharts** - Data visualization
- **WebSocket** - Real-time communication

### **Backend** (Planned)
- **FastAPI** - Modern Python web framework
- **CUDA/CuPy** - GPU computing
- **WebSocket** - Real-time features
- **LLM Integration** - AI tutoring capabilities

## 🌟 **Current Features Demo**

Even without the backend, you can experience:
- ✅ **Course Navigation** - Browse modules and lessons
- ✅ **Interactive Content** - Click highlighted terms for explanations
- ✅ **AI Chat** - Chat interface (with mock responses)
- ✅ **Code Editor** - Write and "execute" CUDA/Python code
- ✅ **Performance Charts** - GPU utilization visualization
- ✅ **Quizzes** - Test your knowledge
- ✅ **Progress Tracking** - See your learning progress

## 📋 **API Contract**

The frontend is ready to connect to the backend via these endpoints:

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/api/chat` | POST | AI tutor conversations |
| `/api/run-code` | POST | Execute CUDA/Python code |
| `/ws/logs/{job_id}` | WebSocket | Stream execution output |
| `/api/profiling/{job_id}` | GET | GPU performance data |
| `/api/explain?term=XYZ` | GET | Term explanations |
| `/api/quiz` | POST | Quiz submissions |
| `/api/progress` | GET/POST | Progress tracking |

## 🔄 **Development Workflow**

1. **Frontend Development**: Complete ✅
2. **Backend Development**: In Progress 🚧
3. **Integration**: Next Step 🔜
4. **Testing & Optimization**: Future 📝
5. **Deployment**: Final Step 🚀

## 🤝 **Contributing**

This is a learning platform project. The frontend is production-ready and the backend is being developed on Sol server.

## 📞 **Status Updates**

- Frontend: Fully functional with mock data
- Backend: In development on Sol server
- Integration: Ready when backend is complete

---

**🎯 Ready to start learning CUDA? Launch the frontend and explore!** 