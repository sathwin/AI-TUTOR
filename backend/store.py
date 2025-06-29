# store.py
from typing import Dict, List
from datetime import datetime
import uuid

###############################################################################
# Fake data – replace with real DB later
###############################################################################
COURSE = {
    "modules": [
        {
            "id": "mod1",
            "title": "Intro to CUDA",
            "description": "GPU basics",
            "lessons": [
                {"id": "mod1-les1", "title": "What is CUDA?", "duration": "10 min"},
                {"id": "mod1-les2", "title": "Thread Hierarchy", "duration": "15 min"},
                {"id": "mod1-les3", "title": "Memory Management", "duration": "20 min"},
            ],
        },
        {
            "id": "mod2",
            "title": "CUDA Programming",
            "description": "Hands-on CUDA programming",
            "lessons": [
                {"id": "mod2-les1", "title": "Your First Kernel", "duration": "25 min"},
                {"id": "mod2-les2", "title": "Optimizing Performance", "duration": "30 min"},
            ],
        },
        {
            "id": "mod3",
            "title": "Advanced GPU Computing",
            "description": "Advanced techniques and optimization",
            "lessons": [
                {"id": "mod3-les1", "title": "Shared Memory", "duration": "35 min"},
                {"id": "mod3-les2", "title": "Multi-GPU Programming", "duration": "40 min"},
            ],
        },
    ]
}

LESSON_HTML = {
    "mod1-les1": """
        <h2>What is CUDA?</h2>
        <p>CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA.</p>
        <h3>Key Features:</h3>
        <ul>
            <li>Parallel computing on NVIDIA GPUs</li>
            <li>C/C++ programming extensions</li>
            <li>Massive parallelism capabilities</li>
            <li>Memory hierarchy optimization</li>
        </ul>
        <p>CUDA enables developers to harness the power of GPU parallel processing for general-purpose computing tasks beyond graphics rendering.</p>
    """,
    "mod1-les2": """
        <h2>Thread Hierarchy</h2>
        <p>CUDA organizes threads in a hierarchical structure for efficient parallel execution.</p>
        <h3>Hierarchy Levels:</h3>
        <ul>
            <li><strong>Grid:</strong> Collection of thread blocks</li>
            <li><strong>Block:</strong> Group of threads that can cooperate</li>
            <li><strong>Thread:</strong> Individual execution unit</li>
        </ul>
        <p>Understanding this hierarchy is crucial for writing efficient CUDA kernels and optimizing memory access patterns.</p>
    """,
    "mod1-les3": """
        <h2>Memory Management</h2>
        <p>CUDA provides various memory types with different characteristics and access patterns.</p>
        <h3>Memory Types:</h3>
        <ul>
            <li><strong>Global Memory:</strong> Large but slow, accessible by all threads</li>
            <li><strong>Shared Memory:</strong> Fast, accessible within a block</li>
            <li><strong>Constant Memory:</strong> Read-only, cached</li>
            <li><strong>Texture Memory:</strong> Optimized for spatial locality</li>
        </ul>
    """,
    "mod2-les1": """
        <h2>Your First Kernel</h2>
        <p>Learn to write and launch your first CUDA kernel function.</p>
        <h3>Kernel Basics:</h3>
        <pre><code>__global__ void myKernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}</code></pre>
        <p>Kernels are functions that run on the GPU and are executed by many threads in parallel.</p>
    """,
    "mod2-les2": """
        <h2>Optimizing Performance</h2>
        <p>Techniques to maximize CUDA application performance.</p>
        <h3>Optimization Strategies:</h3>
        <ul>
            <li>Memory coalescing</li>
            <li>Occupancy optimization</li>
            <li>Bank conflict avoidance</li>
            <li>Instruction throughput optimization</li>
        </ul>
    """,
    "mod3-les1": """
        <h2>Shared Memory</h2>
        <p>Leverage shared memory for high-performance data sharing within thread blocks.</p>
        <h3>Shared Memory Benefits:</h3>
        <ul>
            <li>Low latency access</li>
            <li>High bandwidth</li>
            <li>Thread collaboration</li>
            <li>Data reuse optimization</li>
        </ul>
    """,
    "mod3-les2": """
        <h2>Multi-GPU Programming</h2>
        <p>Scale your applications across multiple GPUs for maximum performance.</p>
        <h3>Multi-GPU Techniques:</h3>
        <ul>
            <li>CUDA streams</li>
            <li>Peer-to-peer transfers</li>
            <li>Load balancing</li>
            <li>Unified memory</li>
        </ul>
    """,
}

QUIZZES = {
    "mod1-les1": {
        "title": "CUDA Basics Quiz",
        "questions": [
            {
                "id": "q1",
                "type": "multiple-choice",
                "question": "CUDA stands for:",
                "options": [
                    "Compute Unified Device Architecture",
                    "Central Universal Device Architecture", 
                    "Compute Universal Data Architecture",
                    "Central Unified Data Architecture"
                ],
                "correct": 0,
            },
            {
                "id": "q2",
                "type": "multiple-choice", 
                "question": "CUDA is primarily used for:",
                "options": [
                    "CPU optimization",
                    "GPU parallel computing",
                    "Network programming",
                    "Database management"
                ],
                "correct": 1,
            }
        ],
    },
    "mod1-les2": {
        "title": "Thread Hierarchy Quiz",
        "questions": [
            {
                "id": "q1",
                "type": "multiple-choice",
                "question": "What is the highest level in CUDA thread hierarchy?",
                "options": ["Thread", "Block", "Grid", "Warp"],
                "correct": 2,
            },
            {
                "id": "q2",
                "type": "multiple-choice",
                "question": "Threads within the same block can:",
                "options": [
                    "Access global memory only",
                    "Cooperate and share data",
                    "Execute independently only", 
                    "Access CPU memory directly"
                ],
                "correct": 1,
            }
        ],
    },
    "mod1-les3": {
        "title": "Memory Management Quiz",
        "questions": [
            {
                "id": "q1",
                "type": "multiple-choice",
                "question": "Which memory type is fastest in CUDA?",
                "options": ["Global", "Shared", "Constant", "Texture"],
                "correct": 1,
            }
        ],
    },
    "mod2-les1": {
        "title": "First Kernel Quiz",
        "questions": [
            {
                "id": "q1",
                "type": "multiple-choice",
                "question": "The __global__ qualifier indicates:",
                "options": [
                    "A CPU function",
                    "A GPU kernel function",
                    "A global variable",
                    "A shared memory declaration"
                ],
                "correct": 1,
            }
        ],
    },
    "mod2-les2": {
        "title": "Performance Optimization Quiz",
        "questions": [
            {
                "id": "q1",
                "type": "multiple-choice",
                "question": "Memory coalescing improves:",
                "options": [
                    "CPU performance",
                    "Memory access efficiency",
                    "Kernel compilation time",
                    "Thread synchronization"
                ],
                "correct": 1,
            }
        ],
    },
    "mod3-les1": {
        "title": "Shared Memory Quiz",
        "questions": [
            {
                "id": "q1",
                "type": "multiple-choice",
                "question": "Shared memory is accessible by:",
                "options": [
                    "All threads in the grid",
                    "Threads within the same block",
                    "Threads within the same warp",
                    "Only the main thread"
                ],
                "correct": 1,
            }
        ],
    },
    "mod3-les2": {
        "title": "Multi-GPU Programming Quiz",
        "questions": [
            {
                "id": "q1",
                "type": "multiple-choice",
                "question": "CUDA streams are used for:",
                "options": [
                    "Memory allocation",
                    "Overlapping computation and communication",
                    "Thread synchronization",
                    "Error handling"
                ],
                "correct": 1,
            }
        ],
    },
}

###############################################################################
# Per‑user progress & jobs (RAM only for demo)
###############################################################################
PROGRESS: Dict[str, dict] = {}
EXEC_JOBS: Dict[str, dict] = {}   # job_id → {code, logs, profiling, done}