# ProjectChimera

Project Chimera is a multi-agentic AI system that can autonomously build software applications, learn from its experiences, and continuously improve. It sees what you see, listens to your voice commands, and can take control of your mouse and keyboard to help you with any task. What makes it unique is its ability to spawn specialist agents to handle different aspects of software development — from coding to code review to business development.

## Architecture

### Core System

- **Orchestrator** (`orchestrator.py`) — Central coordinator that manages agent lifecycles, routes tasks, and facilitates inter-agent communication. Integrates the memory system for persistent knowledge across sessions.
- **Prometheus Agent** (`agent.py`) — The primary user-facing agent that listens for voice commands, analyzes the screen, generates action plans, and executes them. Can delegate tasks to specialist agents.

### Specialist Agents

- **Builder Agent** (`agents/builder_agent.py`) — Builds complete software projects in any language (Python, Rust, JavaScript, HTML/CSS, Go, Java). Plans the project architecture, generates all files, self-reviews, and improves code that doesn't meet quality thresholds.
- **Business Agent** (`agents/business_agent.py`) — Manages business operations including proposal generation, revenue tracking, and opportunity identification. Targets $25k/month revenue.
- **Review Agent** (`agents/review_agent.py`) — Provides quality assurance by reviewing code outputs from other agents, scoring them on multiple criteria, and triggering improvements.
- **Athena Agent** (`core/tutor_agent.py`) — A learning agent that researches topics online, synthesizes knowledge, and saves reusable skills.

### Core Modules

- **Memory System** (`core/memory.py`) — Persistent JSON-backed memory store organized by category (skills, feedback, projects, business, errors, improvements). Supports keyword search, importance scoring, and access tracking. Ensures the system never forgets and continuously improves.
- **Code Builder** (`core/code_builder.py`) — LLM-powered code generation engine that creates software projects in multiple languages and frameworks. Supports Python CLI/web apps, Rust CLI, static websites, React apps, and Node.js APIs.
- **Self-Improvement** (`core/self_improvement.py`) — Evaluates agent outputs on correctness, completeness, code quality, error handling, and documentation. Generates improvement plans based on feedback patterns.
- **Project Manager** (`core/project_manager.py`) — Tracks project lifecycle from planning through completion. Handles task decomposition, progress tracking, and project persistence.
- **Business Manager** (`core/business_manager.py`) — Manages client projects, generates proposals with pricing, tracks revenue, and uses LLM to identify growth opportunities.
- **Cognition** (`core/cognition.py`) — LLM-based reasoning engine that generates structured action plans.
- **Perception** (`core/perception.py`) — Screen capture and VLM-based visual analysis.
- **Action** (`core/action.py`) — PC automation (keyboard, mouse, shell commands).
- **Voice Interface** (`core/voice_interface.py`) — Hotword detection, speech-to-text (Whisper), and text-to-speech.

### Supported Project Types

| Template | Description | Language |
|----------|-------------|----------|
| `python_cli` | Python command-line application | Python |
| `python_web` | Flask/FastAPI web application | Python |
| `rust_cli` | Rust command-line application | Rust |
| `website` | Static HTML/CSS/JS website | HTML |
| `react_app` | React web application | JavaScript |
| `node_api` | Node.js REST API | JavaScript |

### Self-Improvement Loop

1. **Build** — Agent generates code for a project
2. **Evaluate** — Self-improvement module scores output on 5 criteria
3. **Improve** — Files scoring below threshold are automatically improved
4. **Learn** — Feedback and lessons are stored in persistent memory
5. **Apply** — Future tasks benefit from accumulated knowledge

## Getting Started

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for local LLM inference)
- Microphone and speakers (for voice interaction)

### Quick Start

```bash
# Start Ollama and Project Chimera
docker-compose up -d

# Pull required models
docker exec ollama ollama pull llama3:8b-instruct
docker exec ollama ollama pull llava
```

### Running Tests

```bash
cd "Project Chimera"
pip install pytest ollama
python -m pytest tests/ -v
```

## Configuration

Edit `config.py` to customize:

- `COGNITIVE_MODEL` — LLM model for reasoning (default: `llama3:8b-instruct`)
- `VISION_MODEL` — VLM model for screen analysis (default: `llava`)
- `WHISPER_MODEL` — ASR model for voice input (default: `base`)
- `HOTWORD` — Wake word for voice activation (default: `hey chimera`)
- `IMPROVEMENT_SCORE_THRESHOLD` — Quality threshold for auto-improvement (default: `0.7`)
- `MONTHLY_REVENUE_TARGET` — Business revenue target in USD (default: `25000`)

## License

See [LICENSE.txt](LICENSE.txt) for details.

