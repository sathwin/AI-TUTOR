# jobs.py
import asyncio
import subprocess
import tempfile
import uuid
import os
import json
import time
from typing import AsyncGenerator, Dict
from store import EXEC_JOBS

async def run_code_async(code: str) -> str:
    """
    Execute Python code asynchronously and return a job ID for tracking.
    For demo we'll execute locally rather than SLURM; swap in SBATCH later.
    """
    job_id = str(uuid.uuid4())
    EXEC_JOBS[job_id] = {"logs": [], "profiling": None, "done": False, "error": None}
    
    # Create temporary file with the code
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w")
    tmp.write(code)
    tmp.flush()
    tmp.close()

    async def _worker():
        """Background worker that executes the code and collects logs."""
        try:
            cmd = ["python", tmp.name]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            start_time = time.time()
            
            # Read output line by line
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode().rstrip()
                EXEC_JOBS[job_id]["logs"].append(decoded)

            # Wait for process to complete
            await proc.wait()
            end_time = time.time()
            duration = end_time - start_time
            
            # Generate fake profiling data (replace with nvidia-smi polling or nvprof)
            EXEC_JOBS[job_id]["profiling"] = {
                "execution_time": duration,
                "gpu_utilization": [
                    {"time": 0, "value": 10}, 
                    {"time": duration * 0.3, "value": 75},
                    {"time": duration * 0.7, "value": 85},
                    {"time": duration, "value": 20}
                ],
                "memory_usage": [
                    {"time": 0, "value": 1200}, 
                    {"time": duration * 0.3, "value": 2500},
                    {"time": duration * 0.7, "value": 3200},
                    {"time": duration, "value": 1800}
                ],
                "temperature": [
                    {"time": 0, "value": 45},
                    {"time": duration * 0.5, "value": 72},
                    {"time": duration, "value": 65}
                ]
            }
            
            # Check if process succeeded
            if proc.returncode != 0:
                EXEC_JOBS[job_id]["error"] = f"Process exited with code {proc.returncode}"
            
        except Exception as e:
            EXEC_JOBS[job_id]["error"] = str(e)
            EXEC_JOBS[job_id]["logs"].append(f"ERROR: {str(e)}")
        
        finally:
            EXEC_JOBS[job_id]["done"] = True
            # Clean up temporary file
            try:
                os.unlink(tmp.name)
            except:
                pass

    # Start the background worker
    asyncio.create_task(_worker())
    return job_id

async def stream_logs(job_id: str) -> AsyncGenerator[str, None]:
    """
    Stream log lines from a running job until completion.
    Yields new log lines as they become available.
    """
    if job_id not in EXEC_JOBS:
        return
    
    idx = 0
    while True:
        job_data = EXEC_JOBS[job_id]
        logs = job_data["logs"]
        
        # Yield any new log lines
        while idx < len(logs):
            yield logs[idx]
            idx += 1
        
        # Check if job is done
        if job_data["done"]:
            break
            
        # Wait a bit before checking for more logs
        await asyncio.sleep(0.5)

def get_job_status(job_id: str) -> Dict:
    """Get the current status of a job."""
    if job_id not in EXEC_JOBS:
        return {"error": "Job not found"}
    
    job_data = EXEC_JOBS[job_id]
    return {
        "job_id": job_id,
        "done": job_data["done"],
        "error": job_data.get("error"),
        "log_count": len(job_data["logs"]),
        "has_profiling": job_data["profiling"] is not None
    }

def get_job_profiling(job_id: str) -> Dict:
    """Get profiling data for a completed job."""
    if job_id not in EXEC_JOBS:
        return {"error": "Job not found"}
    
    job_data = EXEC_JOBS[job_id]
    if not job_data["done"]:
        return {"error": "Job not completed yet"}
    
    return job_data["profiling"] or {"error": "No profiling data available"}

def cleanup_old_jobs(max_jobs: int = 100):
    """Clean up old job data to prevent memory leaks."""
    if len(EXEC_JOBS) > max_jobs:
        # Keep only the most recent jobs
        job_ids = list(EXEC_JOBS.keys())
        old_jobs = job_ids[:-max_jobs]
        for job_id in old_jobs:
            del EXEC_JOBS[job_id]