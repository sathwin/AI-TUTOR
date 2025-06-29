# main.py
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import asyncio
import logging

# Import our modules
from run_mixtral import load_model, ask
from store import COURSE, LESSON_HTML, QUIZZES, PROGRESS, EXEC_JOBS
from jobs import run_code_async, stream_logs, get_job_status, get_job_profiling, cleanup_old_jobs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# Load Mixtral at startup for fast responses
###############################################################################
tokenizer, model = None, None
model_loading = False

def get_model():
    """Get the preloaded model or load if not available"""
    global tokenizer, model, model_loading
    if model is None and not model_loading:
        model_loading = True
        logger.info("🔄 Loading Mixtral model...")
        try:
            tokenizer, model = load_model()
            logger.info("✅ Mixtral model loaded and ready!")
        except Exception as e:
            logger.error(f"❌ Failed to load Mixtral model: {e}")
            model_loading = False
            raise e
        model_loading = False
    return tokenizer, model

async def preload_model():
    """Preload model in background during startup"""
    try:
        get_model()
    except Exception as e:
        logger.error(f"Model preloading failed: {e}")

# Start model loading immediately
import asyncio
import threading

def load_model_background():
    """Load model in background thread"""
    try:
        get_model()
    except Exception as e:
        logger.error(f"Background model loading failed: {e}")

# Start loading model immediately
logger.info("🚀 Starting model preload...")
model_thread = threading.Thread(target=load_model_background, daemon=True)
model_thread.start()

###############################################################################
# FastAPI init & CORS
###############################################################################
app = FastAPI(
    title="AI GPU‑Tutor Backend",
    description="Backend API for AI-powered GPU programming tutor",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for port forwarding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# Pydantic Models / Schemas
###############################################################################
class ChatRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    conversation_history: Optional[List[Dict]] = None

class CodeRequest(BaseModel):
    code: str

class QuizSubmission(BaseModel):
    lesson_id: str
    answers: Dict[str, int]

class ProgressUpdate(BaseModel):
    lesson_id: str
    type: str  # "lesson" or "quiz"
    completed: bool

###############################################################################
# Health check endpoint
###############################################################################
@app.get("/")
def root():
    return {
        "message": "AI GPU-Tutor Backend API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else ("loading" if model_loading else "not_loaded"),
        "model_ready": model is not None,
        "active_jobs": len(EXEC_JOBS)
    }

@app.get("/api/model-status")
def model_status():
    """Detailed model loading status"""
    return {
        "loaded": model is not None,
        "loading": model_loading,
        "ready_for_inference": model is not None and tokenizer is not None,
        "estimated_load_time": "3-5 minutes" if model is None else "Ready now"
    }

###############################################################################
# 1. Course structure endpoints
###############################################################################
@app.get("/api/modules")
def get_modules():
    """Get all course modules with their lessons."""
    return COURSE

@app.get("/api/modules/{module_id}")
def get_module(module_id: str):
    """Get a specific module by ID."""
    for module in COURSE["modules"]:
        if module["id"] == module_id:
            return module
    raise HTTPException(status_code=404, detail="Module not found")

###############################################################################
# 2. Lesson content endpoints
###############################################################################
@app.get("/api/lessons/{lesson_id}")
def get_lesson(lesson_id: str):
    """Get lesson content by ID."""
    # Find lesson title from course structure
    lesson_title = None
    lesson_duration = None
    
    for module in COURSE["modules"]:
        for lesson in module["lessons"]:
            if lesson["id"] == lesson_id:
                lesson_title = lesson["title"]
                lesson_duration = lesson["duration"]
                break
        if lesson_title:
            break
    
    if not lesson_title:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    return {
        "id": lesson_id,
        "title": lesson_title,
        "content": LESSON_HTML.get(lesson_id, "<p>Content coming soon...</p>"),
        "duration": lesson_duration,
    }

###############################################################################
# 3. Term explanation (RAG-style with Mixtral)
###############################################################################
@app.get("/api/explain")
def explain_term(term: str):
    """Explain a GPU/CUDA term using the Mixtral model."""
    global model, model_loading
    if model is None:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please wait...")
        else:
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        tokenizer_local, model_local = get_model()
        prompt = f"Explain the term '{term}' in the context of GPU programming and CUDA. Be concise and technical."
        answer = ask(model_local, tokenizer_local, prompt, max_new_tokens=200)
        
        # Clean up the response (remove the original prompt if it's included)
        if prompt in answer:
            answer = answer.replace(prompt, "").strip()
        
        return {
            "term": term,
            "snippets": [
                {
                    "text": answer,
                    "source": "Mixtral 8×7B",
                    "confidence": 0.85
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error explaining term '{term}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to explain term: {str(e)}")

###############################################################################
# 4. Chat endpoint
###############################################################################
@app.post("/api/chat")
def chat(request: ChatRequest):
    """Chat with the AI tutor using Mixtral."""
    global model, model_loading
    if model is None:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please wait...")
        else:
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        tokenizer_local, model_local = get_model()
        
        # Build context-aware prompt
        prompt = request.prompt
        if request.context:
            prompt = f"Context: {request.context}\n\nQuestion: {request.prompt}\n\nAnswer:"
        
        response = ask(
            model_local, 
            tokenizer_local, 
            prompt, 
            max_new_tokens=300,
            temperature=0.7
        )
        
        # Clean up response
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif prompt in response:
            response = response.replace(prompt, "").strip()
        
        import time
        return {
            "response": response,
            "model": "Mixtral 8×7B",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

###############################################################################
# 5. Code execution endpoints
###############################################################################
@app.post("/api/run-code")
async def run_code(request: CodeRequest):
    """Execute code and return a job ID for tracking."""
    try:
        job_id = await run_code_async(request.code)
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Code execution started"
        }
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start code execution: {str(e)}")

@app.get("/api/jobs/{job_id}/status")
def get_job_status_endpoint(job_id: str):
    """Get the status of a running or completed job."""
    status = get_job_status(job_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return status

@app.websocket("/ws/logs/{job_id}")
async def websocket_logs(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for streaming job logs in real-time."""
    await websocket.accept()
    
    try:
        if job_id not in EXEC_JOBS:
            await websocket.send_json({"type": "error", "message": "Job not found"})
            return
        
        # Stream logs as they become available
        async for line in stream_logs(job_id):
            await websocket.send_json({
                "type": "log",
                "data": line,
                "timestamp": asyncio.get_event_loop().time()
            })
        
        # Send completion signal
        await websocket.send_json({
            "type": "complete",
            "message": "Job completed"
        })
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass

@app.get("/api/profiling/{job_id}")
def get_profiling_data(job_id: str):
    """Get profiling data for a completed job."""
    profiling = get_job_profiling(job_id)
    if "error" in profiling:
        raise HTTPException(status_code=404, detail=profiling["error"])
    return profiling

###############################################################################
# 6. Quiz endpoints
###############################################################################
@app.get("/api/quiz/{lesson_id}")
def get_quiz(lesson_id: str):
    """Get quiz questions for a lesson."""
    if lesson_id not in QUIZZES:
        raise HTTPException(status_code=404, detail="Quiz not found for this lesson")
    
    # Return quiz without correct answers for security
    quiz = QUIZZES[lesson_id].copy()
    for question in quiz["questions"]:
        question.pop("correct", None)  # Remove correct answer from response
    
    return quiz

@app.post("/api/quiz")
def submit_quiz(submission: QuizSubmission):
    """Submit quiz answers and get results."""
    if submission.lesson_id not in QUIZZES:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = QUIZZES[submission.lesson_id]
    correct_count = 0
    total_questions = len(quiz["questions"])
    feedback = {}
    
    for question in quiz["questions"]:
        question_id = question["id"]
        user_answer = submission.answers.get(question_id)
        correct_answer = question["correct"]
        
        is_correct = user_answer == correct_answer
        correct_count += int(is_correct)
        
        feedback[question_id] = {
            "correct": is_correct,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "explanation": "Great job!" if is_correct else "Review the lesson material for this topic."
        }
    
    passed = correct_count == total_questions
    
    return {
        "correct": correct_count,
        "total": total_questions,
        "score": (correct_count / total_questions) * 100,
        "passed": passed,
        "feedback": feedback,
        "lesson_id": submission.lesson_id
    }

###############################################################################
# 7. Progress tracking endpoints
###############################################################################
@app.get("/api/progress")
def get_progress(user: str = "demo"):
    """Get user progress data."""
    default_progress = {
        "completedLessons": [],
        "completedQuizzes": [],
        "totalLessons": sum(len(module["lessons"]) for module in COURSE["modules"]),
        "totalQuizzes": len(QUIZZES)
    }
    
    user_progress = PROGRESS.get(user, default_progress)
    
    # Ensure we return the total counts
    user_progress["totalLessons"] = default_progress["totalLessons"]
    user_progress["totalQuizzes"] = default_progress["totalQuizzes"]
    
    return user_progress

@app.post("/api/progress")
def update_progress(update: ProgressUpdate, user: str = "demo"):
    """Update user progress."""
    if user not in PROGRESS:
        PROGRESS[user] = {"completedLessons": [], "completedQuizzes": []}
    
    progress_key = "completedLessons" if update.type == "lesson" else "completedQuizzes"
    
    if update.completed:
        if update.lesson_id not in PROGRESS[user][progress_key]:
            PROGRESS[user][progress_key].append(update.lesson_id)
    else:
        if update.lesson_id in PROGRESS[user][progress_key]:
            PROGRESS[user][progress_key].remove(update.lesson_id)
    
    return {
        "status": "success",
        "message": f"Progress updated for {update.type}: {update.lesson_id}"
    }

###############################################################################
# 8. Utility endpoints
###############################################################################
@app.post("/api/cleanup")
def cleanup_jobs():
    """Clean up old job data (admin endpoint)."""
    cleanup_old_jobs(max_jobs=50)
    return {"status": "success", "message": "Old jobs cleaned up"}

###############################################################################
# Startup event
###############################################################################
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AI GPU-Tutor Backend starting up...")
    logger.info(f"📊 Loaded {len(COURSE['modules'])} modules")
    logger.info(f"📚 Loaded {sum(len(m['lessons']) for m in COURSE['modules'])} lessons")
    logger.info(f"❓ Loaded {len(QUIZZES)} quizzes")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔄 AI GPU-Tutor Backend shutting down...")
    cleanup_old_jobs(max_jobs=0)  # Clean up all jobs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)