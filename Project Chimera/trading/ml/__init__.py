"""Machine learning models for price prediction and signal generation."""

from .feature_engineering import FeatureEngineer
from .tree_models import GradientBoostedTrees, RandomForestModel
from .ensemble import ModelEnsemble

# PyTorch-dependent modules are imported lazily so the rest of the package
# remains usable in environments where torch is not installed.
def __getattr__(name: str):
    _torch_modules = {
        "LSTMPredictor": (".lstm_predictor", "LSTMPredictor"),
        "SentimentTransformer": (".transformer_model", "SentimentTransformer"),
        "WalletGNN": (".gnn_model", "WalletGNN"),
        "TradingRLAgent": (".rl_agent", "TradingRLAgent"),
    }
    if name in _torch_modules:
        import importlib
        mod_path, attr = _torch_modules[name]
        module = importlib.import_module(mod_path, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FeatureEngineer", "LSTMPredictor", "GradientBoostedTrees",
    "RandomForestModel", "SentimentTransformer", "WalletGNN",
    "TradingRLAgent", "ModelEnsemble",
]
