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

# --- Bot Loop ---
TRADING_LOOP_INTERVAL = int(os.getenv("TRADING_LOOP_INTERVAL", "300"))  # seconds between iterations

# ==========================================================================
# Advanced Trading Modules
# ==========================================================================

# --- Risk Management ---
KELLY_FRACTION           = float(os.getenv("KELLY_FRACTION",           "0.25"))  # fractional Kelly safety
MAX_PORTFOLIO_CONCENTRATION = float(os.getenv("MAX_PORTFOLIO_CONCENTRATION", "20.0"))  # % max in one token
MAX_DAILY_LOSS_PCT       = float(os.getenv("MAX_DAILY_LOSS_PCT",       "5.0"))   # % of portfolio
MAX_OPEN_POSITIONS       = int(os.getenv("MAX_OPEN_POSITIONS",         "10"))
ATR_STOP_MULTIPLIER      = float(os.getenv("ATR_STOP_MULTIPLIER",     "2.0"))
TRAILING_STOP_PCT        = float(os.getenv("TRAILING_STOP_PCT",        "5.0"))
MAX_HOLD_HOURS           = int(os.getenv("MAX_HOLD_HOURS",             "72"))
CIRCUIT_BREAKER_DROP_PCT = float(os.getenv("CIRCUIT_BREAKER_DROP_PCT", "5.0"))   # flash crash threshold
CIRCUIT_BREAKER_WINDOW   = int(os.getenv("CIRCUIT_BREAKER_WINDOW",     "60"))    # seconds
VAR_CONFIDENCE_95        = float(os.getenv("VAR_CONFIDENCE_95",        "0.95"))
VAR_CONFIDENCE_99        = float(os.getenv("VAR_CONFIDENCE_99",        "0.99"))

# --- Execution ---
DEFAULT_SLIPPAGE_BPS     = float(os.getenv("DEFAULT_SLIPPAGE_BPS",     "50.0"))  # 0.5 %
JUPITER_API_URL          = os.getenv("JUPITER_API_URL", "https://quote-api.jup.ag/v6")
MAX_RETRIES              = int(os.getenv("MAX_RETRIES",                "3"))
RETRY_BACKOFF_BASE       = float(os.getenv("RETRY_BACKOFF_BASE",       "1.5"))   # seconds
TX_CONFIRM_TIMEOUT       = int(os.getenv("TX_CONFIRM_TIMEOUT",        "60"))     # seconds
JITO_TIP_LAMPORTS        = int(os.getenv("JITO_TIP_LAMPORTS",         "10000"))  # tip for MEV protection

# --- Sentiment ---
SENTIMENT_WEIGHT_NEWS    = float(os.getenv("SENTIMENT_WEIGHT_NEWS",    "0.3"))
SENTIMENT_WEIGHT_SOCIAL  = float(os.getenv("SENTIMENT_WEIGHT_SOCIAL",  "0.4"))
SENTIMENT_WEIGHT_ONCHAIN = float(os.getenv("SENTIMENT_WEIGHT_ONCHAIN", "0.3"))
SENTIMENT_EMA_PERIOD     = int(os.getenv("SENTIMENT_EMA_PERIOD",       "14"))
SENTIMENT_EXTREME_THRESHOLD = float(os.getenv("SENTIMENT_EXTREME_THRESHOLD", "0.8"))

# --- Anti-Manipulation ---
ANOMALY_CONTAMINATION    = float(os.getenv("ANOMALY_CONTAMINATION",    "0.1"))   # isolation forest
VOLUME_ZSCORE_THRESHOLD  = float(os.getenv("VOLUME_ZSCORE_THRESHOLD",  "3.0"))
PUMP_SCHEME_WALLET_BURST = int(os.getenv("PUMP_SCHEME_WALLET_BURST",   "10"))    # new wallets in window
MIN_HHI_SAFE             = float(os.getenv("MIN_HHI_SAFE",            "0.25"))   # ownership concentration
RUG_RISK_THRESHOLD       = float(os.getenv("RUG_RISK_THRESHOLD",       "0.7"))   # 0-1, above = risky

# --- ML Models ---
LSTM_HIDDEN_SIZE         = int(os.getenv("LSTM_HIDDEN_SIZE",           "64"))
LSTM_NUM_LAYERS          = int(os.getenv("LSTM_NUM_LAYERS",            "2"))
LSTM_DROPOUT             = float(os.getenv("LSTM_DROPOUT",             "0.2"))
LSTM_SEQUENCE_LENGTH     = int(os.getenv("LSTM_SEQUENCE_LENGTH",       "30"))
GBT_N_ESTIMATORS         = int(os.getenv("GBT_N_ESTIMATORS",          "100"))
GBT_LEARNING_RATE        = float(os.getenv("GBT_LEARNING_RATE",        "0.1"))
GBT_MAX_DEPTH            = int(os.getenv("GBT_MAX_DEPTH",             "6"))
RL_REPLAY_BUFFER_SIZE    = int(os.getenv("RL_REPLAY_BUFFER_SIZE",      "10000"))
RL_BATCH_SIZE            = int(os.getenv("RL_BATCH_SIZE",              "64"))
RL_GAMMA                 = float(os.getenv("RL_GAMMA",                 "0.99"))
RL_EPSILON_START          = float(os.getenv("RL_EPSILON_START",         "1.0"))
RL_EPSILON_END            = float(os.getenv("RL_EPSILON_END",           "0.01"))
RL_EPSILON_DECAY          = float(os.getenv("RL_EPSILON_DECAY",         "0.995"))

# --- Backtesting ---
BACKTEST_SLIPPAGE_BPS    = float(os.getenv("BACKTEST_SLIPPAGE_BPS",    "10.0"))
BACKTEST_FAILURE_RATE    = float(os.getenv("BACKTEST_FAILURE_RATE",     "0.02"))
BACKTEST_COMMISSION_BPS  = float(os.getenv("BACKTEST_COMMISSION_BPS",  "5.0"))

# --- Strategies ---
ARB_FEE_PCT              = float(os.getenv("ARB_FEE_PCT",              "0.3"))   # per swap
ARB_MIN_PROFIT_PCT       = float(os.getenv("ARB_MIN_PROFIT_PCT",       "0.1"))   # minimum to act
ARB_OPPORTUNITY_TTL      = int(os.getenv("ARB_OPPORTUNITY_TTL",        "10"))    # seconds
SNIPE_MIN_LIQUIDITY_USD  = float(os.getenv("SNIPE_MIN_LIQUIDITY_USD",  "5000.0"))
SNIPE_MAX_RUG_RISK       = float(os.getenv("SNIPE_MAX_RUG_RISK",       "0.5"))
MM_GAMMA                 = float(os.getenv("MM_GAMMA",                 "0.1"))   # risk aversion
MM_MAX_INVENTORY         = float(os.getenv("MM_MAX_INVENTORY",         "1000.0"))

# --- Monitoring ---
METRICS_RETENTION_HOURS  = int(os.getenv("METRICS_RETENTION_HOURS",    "168"))   # 7 days
ALERT_WEBHOOK_URL        = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_COOLDOWN_SECONDS   = int(os.getenv("ALERT_COOLDOWN_SECONDS",     "300"))

# --- Learning ---
RETRAIN_INTERVAL_HOURS   = int(os.getenv("RETRAIN_INTERVAL_HOURS",    "24"))
DRIFT_PSI_THRESHOLD      = float(os.getenv("DRIFT_PSI_THRESHOLD",      "0.2"))
DRIFT_KS_THRESHOLD       = float(os.getenv("DRIFT_KS_THRESHOLD",       "0.1"))
GA_POPULATION_SIZE       = int(os.getenv("GA_POPULATION_SIZE",         "20"))
GA_MUTATION_RATE         = float(os.getenv("GA_MUTATION_RATE",          "0.1"))
GA_GENERATIONS           = int(os.getenv("GA_GENERATIONS",              "50"))

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
