#/config.py

# --- System ---
LOG_LEVEL = "INFO"

# --- Models ---
# Cognitive models: llama3:8b-instruct, mixtral, etc.
COGNITIVE_MODEL = 'llama3:8b-instruct'
# Vision model
VISION_MODEL = 'llava'
# Whisper model: tiny, base, small, medium, large
WHISPER_MODEL = 'base'

# --- Voice Interface ---
HOTWORD = "hey chimera"
# Path to whisper.cpp - for advanced performance
# WHISPER_CPP_PATH = "path/to/whisper.cpp/main"
TTS_ENGINE_RATE = 180 # Words per minute

# --- Perception ---
SCREENSHOT_INTERVAL = 0.5 # seconds

# --- Agent Configuration ---
PROMETHEUS_ROLE = "A top-tier AI assistant that can see, hear, and control the user's computer to accomplish any task."
ATHENA_ROLE = "A specialized 'Tutor' agent that learns new skills by researching online and synthesizes the knowledge into actionable plans or scripts."
BUILDER_ROLE = "A specialist software engineer agent that builds complete applications in any language or framework based on project descriptions."
BUSINESS_ROLE = "A business development agent that generates proposals, tracks revenue, and identifies growth opportunities to reach $25k/month."
REVIEW_ROLE = "A code review specialist that evaluates code quality, identifies issues, and triggers improvements through feedback loops."

# --- Self-Improvement ---
IMPROVEMENT_SCORE_THRESHOLD = 0.7  # Files scoring below this are auto-improved

# --- Business ---
MONTHLY_REVENUE_TARGET = 25000  # Target monthly revenue in USD
