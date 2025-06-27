# AI Tutor Frontend

A modern React-based frontend for an interactive AI tutor application focused on CUDA programming education.

## Features

- **Interactive Learning**: Three-pane layout with lesson content, sidebar navigation, and interactive features
- **Code Editor**: Monaco-based editor with syntax highlighting for Python/CUDA code
- **Real-time Chat**: AI tutor chat interface with context-aware responses
- **Code Execution**: Live code execution with WebSocket streaming of output
- **GPU Profiling**: Real-time visualization of GPU utilization and memory usage
- **Quizzes**: Interactive multiple-choice and fill-in-the-blank quizzes
- **Progress Tracking**: Track completion status with badges and achievements
- **Responsive Design**: Mobile-friendly layout with collapsible sidebar
- **Dark Theme**: Modern dark theme with Tailwind CSS and shadcn/ui components

## Tech Stack

- **React 18** - Modern React with hooks
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - High-quality component library built on Radix UI
- **Monaco Editor** - VS Code editor for web with Python syntax highlighting
- **Recharts** - Composable charting library for GPU profiling visualization
- **Lucide React** - Beautiful icon library
- **WebSocket** - Real-time communication for code execution streaming

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── ui/                 # shadcn/ui components
│   │   │   ├── button.jsx
│   │   │   ├── card.jsx
│   │   │   ├── input.jsx
│   │   │   ├── tabs.jsx
│   │   │   ├── progress.jsx
│   │   │   ├── scroll-area.jsx
│   │   │   └── badge.jsx
│   │   ├── Sidebar.jsx         # Course navigation
│   │   ├── LessonContent.jsx   # Lesson display with term highlighting
│   │   ├── InteractivePane.jsx # Tabbed interface container
│   │   ├── ChatPanel.jsx       # AI tutor chat
│   │   ├── CodePanel.jsx       # Code editor and execution
│   │   ├── CodeEditor.jsx      # Monaco editor wrapper
│   │   ├── ConsoleLog.jsx      # Live output display
│   │   ├── ProfilerChart.jsx   # GPU performance charts
│   │   ├── Quiz.jsx            # Interactive quizzes
│   │   └── ProgressTracker.jsx # Progress and achievements
│   ├── lib/
│   │   └── utils.js            # Utility functions
│   ├── App.jsx                 # Main application component
│   ├── index.js                # React entry point
│   └── index.css               # Global styles and CSS variables
├── package.json
├── tailwind.config.js
└── postcss.config.js
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-tutor/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm start
   ```

   The application will open at [http://localhost:3000](http://localhost:3000)

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm run build` - Builds the app for production
- `npm test` - Launches the test runner
- `npm run eject` - Ejects from Create React App (not recommended)

## API Integration

The frontend is designed to integrate with a FastAPI backend. Currently, it includes mock data and fallback functionality for development purposes.

### Expected API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send chat messages to AI tutor |
| `/api/run-code` | POST | Execute Python/CUDA code |
| `/ws/logs/{job_id}` | WebSocket | Stream execution output |
| `/api/profiling/{job_id}` | GET | Get GPU profiling data |
| `/api/explain?term=XYZ` | GET | Get term explanations |
| `/api/quiz` | POST | Submit quiz answers |
| `/api/progress` | GET/POST | Track user progress |

## Component Details

### Sidebar
- Displays course modules and lessons
- Expandable/collapsible sections
- Progress indicators for completed lessons
- Responsive design with mobile support

### LessonContent
- Renders HTML/Markdown lesson content
- Interactive term highlighting
- Click-to-explain functionality
- Smooth loading states

### CodePanel
- Monaco editor with Python syntax highlighting
- Run/Stop execution controls
- Live console output via WebSocket
- GPU profiling charts after execution

### ChatPanel
- Real-time chat with AI tutor
- Message history
- Typing indicators
- Context-aware responses

### Quiz
- Multiple-choice questions
- Fill-in-the-blank questions
- Instant feedback after submission
- Retake functionality

### ProgressTracker
- Overall course progress
- Module-specific progress
- Achievement badges
- Quick action buttons

## Styling

The application uses Tailwind CSS with a custom design system:

- **Dark theme by default**
- **CSS variables** for easy theme customization
- **Responsive design** with mobile-first approach
- **Consistent spacing** and typography
- **Accessible colors** and contrast ratios

## Development

### Adding New Components

1. Create the component in `src/components/`
2. Use Tailwind CSS classes for styling
3. Import and use shadcn/ui components where appropriate
4. Follow the existing patterns for API integration

### Customizing Theme

Modify the CSS variables in `src/index.css` to customize the theme:

```css
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  /* ... more variables */
}
```

### Adding New API Endpoints

1. Add the endpoint to the relevant component
2. Include error handling and loading states
3. Provide mock data for development
4. Update the API documentation above

## Build and Deployment

1. **Build for production**
   ```bash
   npm run build
   ```

2. **Serve the built files**
   The `build` folder contains the optimized production build that can be served by any static file server.

## Browser Support

- Chrome/Chromium 60+
- Firefox 60+
- Safari 12+
- Edge 79+

## Contributing

1. Follow the existing code style and component patterns
2. Use TypeScript for new components when possible
3. Ensure responsive design for all new features
4. Add appropriate error handling and loading states
5. Test with mock data before backend integration

## License

This project is part of the AI Tutor application suite. 