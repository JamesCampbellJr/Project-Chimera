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
