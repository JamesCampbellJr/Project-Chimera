#/config.py

import os

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

# ==========================================================================
# Solana Paper-Trading Bot
# ==========================================================================

# --- Solana RPC ---
# Public mainnet-beta endpoint (no API key needed for basic use).
# Set SOLANA_RPC_URL in your .env to override with a private node / Helius / QuickNode.
SOLANA_RPC_URL     = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
SOLANA_RPC_TIMEOUT = int(os.getenv("SOLANA_RPC_TIMEOUT", "30"))  # seconds

# Delay (seconds) between individual RPC calls to respect rate limits.
RPC_RATE_LIMIT_DELAY = float(os.getenv("RPC_RATE_LIMIT_DELAY", "0.5"))

# --- Database ---
TRADING_DB_PATH = os.getenv("TRADING_DB_PATH", "trading/chimera_trading.db")

# --- Paper Trading ---
PAPER_TRADING_STARTING_BALANCE = float(
    os.getenv("PAPER_TRADING_STARTING_BALANCE", "10000.0")  # USD
)
PAPER_POSITION_SIZE_PCT  = float(os.getenv("PAPER_POSITION_SIZE_PCT",  "5.0"))   # % of cash per trade
PAPER_MAX_POSITION_USD   = float(os.getenv("PAPER_MAX_POSITION_USD",  "500.0"))  # hard cap per trade
PAPER_MIN_TRADE_USD      = float(os.getenv("PAPER_MIN_TRADE_USD",      "10.0"))  # ignore tiny signals
PAPER_TAKE_PROFIT_PCT    = float(os.getenv("PAPER_TAKE_PROFIT_PCT",    "20.0"))  # close at +20 %
PAPER_STOP_LOSS_PCT      = float(os.getenv("PAPER_STOP_LOSS_PCT",      "10.0"))  # close at -10 %

# --- Signals & Patterns ---
MIN_SIGNAL_SCORE          = float(os.getenv("MIN_SIGNAL_SCORE",         "0.60"))  # 0–1
TARGET_WIN_RATE_PCT       = float(os.getenv("TARGET_WIN_RATE_PCT",      "90.0"))  # % target
MIN_TRADES_FOR_RANKING    = int(os.getenv("MIN_TRADES_FOR_RANKING",     "5"))

# Pattern detector thresholds
PATTERN_MOMENTUM_WALLETS  = int(os.getenv("PATTERN_MOMENTUM_WALLETS",  "3"))
PATTERN_MOMENTUM_WINDOW   = int(os.getenv("PATTERN_MOMENTUM_WINDOW",   "300"))   # seconds
PATTERN_QUICK_FLIP_SECS   = int(os.getenv("PATTERN_QUICK_FLIP_SECS",   "600"))   # 10 min
PATTERN_CONSISTENCY_WIN_RATE = float(
    os.getenv("PATTERN_CONSISTENCY_WIN_RATE", "70.0")
)

# --- Risk Management ---
RISK_KELLY_SAFETY_MULTIPLIER = float(os.getenv("RISK_KELLY_SAFETY_MULTIPLIER", "0.5"))   # half-Kelly
RISK_TRAILING_STOP_PCT       = float(os.getenv("RISK_TRAILING_STOP_PCT",       "8.0"))   # trail 8% from peak
RISK_MAX_DRAWDOWN_PCT        = float(os.getenv("RISK_MAX_DRAWDOWN_PCT",       "20.0"))   # circuit breaker at 20%
RISK_MAX_CONSECUTIVE_LOSSES  = int(os.getenv("RISK_MAX_CONSECUTIVE_LOSSES",    "5"))      # circuit breaker

# --- Sentiment Analysis ---
SENTIMENT_BEARISH_THRESHOLD     = float(os.getenv("SENTIMENT_BEARISH_THRESHOLD",     "-0.3"))  # block if token < this
SENTIMENT_MARKET_FEAR_THRESHOLD = float(os.getenv("SENTIMENT_MARKET_FEAR_THRESHOLD", "-0.5"))  # block all trading

# --- Manipulation Detection ---
MANIPULATION_BLOCK_SEVERITY     = float(os.getenv("MANIPULATION_BLOCK_SEVERITY",     "0.5"))   # block if severity >= this
WASH_TRADE_WINDOW_SECS          = int(os.getenv("WASH_TRADE_WINDOW_SECS",            "60"))    # seconds
PUMP_MIN_WALLETS                = int(os.getenv("PUMP_MIN_WALLETS",                  "5"))     # min wallets for pump
PUMP_WINDOW_SECS                = int(os.getenv("PUMP_WINDOW_SECS",                 "180"))   # seconds
COORDINATED_MIN_COOCCURRENCES   = int(os.getenv("COORDINATED_MIN_COOCCURRENCES",    "3"))     # min co-trades

# --- Technical Indicators ---
RSI_OVERSOLD            = float(os.getenv("RSI_OVERSOLD",            "30.0"))
RSI_OVERBOUGHT          = float(os.getenv("RSI_OVERBOUGHT",          "70.0"))
MOMENTUM_BUY_THRESHOLD  = float(os.getenv("MOMENTUM_BUY_THRESHOLD",  "5.0"))   # % ROC
MOMENTUM_SELL_THRESHOLD = float(os.getenv("MOMENTUM_SELL_THRESHOLD", "-5.0"))   # % ROC

# --- Bot Loop ---
TRADING_LOOP_INTERVAL = int(os.getenv("TRADING_LOOP_INTERVAL", "300"))  # seconds between iterations

# --- Seed Wallets ---
# Well-known Solana DeFi power-users used as starting points for analysis.
# Add or replace entries via the SEED_WALLETS env var (comma-separated).
_seed_env = os.getenv("SEED_WALLETS", "")
SEED_WALLETS: list = (
    [w.strip() for w in _seed_env.split(",") if w.strip()]
    if _seed_env
    else [
        # Example publicly-discussed high-activity wallets (mainnet-beta).
        # These are NOT financial recommendations — replace with your own research.
        "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
        "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
        "GUfCR9mK6azb9vcpsxgXyj7XRPAKJd4KMHTTVvtncGgp",
    ]
)
