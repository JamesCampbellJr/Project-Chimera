"""
Feature engineering pipeline for Solana trading ML models.

All technical-indicator calculations are implemented in pure NumPy — no
external TA libraries are required.  The module produces a feature matrix
ready for consumption by downstream predictors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(data: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (vectorised, forward-fill safe)."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(data, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(data: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average.  First *window-1* values are NaN."""
    kernel = np.ones(window) / window
    sma = np.convolve(data, kernel, mode="full")[: len(data)]
    sma[: window - 1] = np.nan
    return sma


# ---------------------------------------------------------------------------
# Technical indicator functions (pure numpy)
# ---------------------------------------------------------------------------

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi_vals = 100.0 - 100.0 / (1.0 + rs)
    return np.concatenate([[np.nan], rsi_vals])


def macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, and histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    close: np.ndarray,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands (upper, middle, lower)."""
    middle = _sma(close, window)
    rolling_std = np.array([
        np.std(close[max(0, i - window + 1): i + 1])
        if i >= window - 1 else np.nan
        for i in range(len(close))
    ])
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return upper, middle, lower


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    return _ema(tr, period)


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    direction = np.sign(np.diff(close))
    direction = np.concatenate([[0], direction])
    return np.cumsum(direction * volume)


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """%K and %D of the stochastic oscillator."""
    k_vals = np.full(len(close), np.nan)
    for i in range(k_period - 1, len(close)):
        window_high = np.max(high[i - k_period + 1: i + 1])
        window_low = np.min(low[i - k_period + 1: i + 1])
        denom = window_high - window_low
        k_vals[i] = ((close[i] - window_low) / denom * 100.0) if denom != 0 else 50.0
    d_vals = _sma(k_vals, d_period)
    return k_vals, d_vals


def williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Williams %R indicator."""
    wr = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        window_high = np.max(high[i - period + 1: i + 1])
        window_low = np.min(low[i - period + 1: i + 1])
        denom = window_high - window_low
        wr[i] = ((window_high - close[i]) / denom * -100.0) if denom != 0 else -50.0
    return wr


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def z_score_normalize(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score normalisation.  Returns (normalised, mean, std)."""
    if mean is None:
        mean = np.nanmean(data, axis=0)
    if std is None:
        std = np.nanstd(data, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (data - mean) / std, mean, std


def min_max_normalize(
    data: np.ndarray,
    mn: Optional[np.ndarray] = None,
    mx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise min-max normalisation to [0, 1]."""
    if mn is None:
        mn = np.nanmin(data, axis=0)
    if mx is None:
        mx = np.nanmax(data, axis=0)
    rng = mx - mn
    rng = np.where(rng == 0, 1.0, rng)
    return (data - mn) / rng, mn, mx


# ---------------------------------------------------------------------------
# Feature Engineer
# ---------------------------------------------------------------------------

@dataclass
class FeatureEngineer:
    """Build a feature matrix from raw OHLCV + on-chain + social data.

    Parameters
    ----------
    normalization : str
        ``"zscore"`` or ``"minmax"``.
    include_cross_corr : bool
        Whether to append the cross-token correlation matrix to features.
    lookback : int
        Rolling window used for cross-token correlation.
    """

    normalization: str = "zscore"
    include_cross_corr: bool = True
    lookback: int = 30
    _norm_params: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    # Technical indicators
    # ------------------------------------------------------------------ #
    def compute_technical_features(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Return an (N, F_tech) array of technical indicator features."""
        rsi_vals = rsi(close)
        macd_line, macd_sig, macd_hist = macd(close)
        bb_upper, bb_mid, bb_lower = bollinger_bands(close)
        atr_vals = atr(high, low, close)
        obv_vals = obv(close, volume)
        stoch_k, stoch_d = stochastic(high, low, close)
        wr = williams_r(high, low, close)
        ema_12 = _ema(close, 12)
        ema_26 = _ema(close, 26)
        sma_20 = _sma(close, 20)
        sma_50 = _sma(close, 50)

        # Derived: Bollinger %B
        bb_range = bb_upper - bb_lower
        bb_range = np.where(bb_range == 0, 1.0, bb_range)
        bb_pct_b = (close - bb_lower) / bb_range

        features = np.column_stack([
            rsi_vals,
            macd_line, macd_sig, macd_hist,
            bb_upper, bb_mid, bb_lower, bb_pct_b,
            atr_vals,
            obv_vals,
            stoch_k, stoch_d,
            wr,
            ema_12, ema_26,
            sma_20, sma_50,
            close, volume,
        ])
        return features

    # ------------------------------------------------------------------ #
    # On-chain features
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_onchain_features(
        active_addresses: np.ndarray,
        tx_count: np.ndarray,
        fees_total: np.ndarray,
    ) -> np.ndarray:
        """Normalise and return on-chain metrics."""
        features = np.column_stack([active_addresses, tx_count, fees_total])
        norms = np.linalg.norm(features, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return features / norms

    # ------------------------------------------------------------------ #
    # Social features
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_social_features(
        sentiment_score: np.ndarray,
        mention_velocity: np.ndarray,
        influencer_engagement: np.ndarray,
    ) -> np.ndarray:
        """Stack social sentiment signals into a feature matrix."""
        return np.column_stack([
            sentiment_score,
            mention_velocity,
            influencer_engagement,
        ])

    # ------------------------------------------------------------------ #
    # Market micro-structure
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_microstructure_features(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """Estimate bid-ask spread and volume imbalance."""
        # Corwin-Schultz bid-ask spread estimator (simplified)
        log_hl = np.log(high / np.maximum(low, 1e-12))
        spread_estimate = np.sqrt(2.0) * log_hl - np.sqrt(np.log(2.0)) * log_hl
        spread_estimate = np.clip(spread_estimate, 0.0, None)

        # Volume imbalance (normalised diff of consecutive bars)
        vol_diff = np.diff(volume, prepend=volume[0])
        vol_sum = volume + np.abs(vol_diff)
        vol_sum = np.where(vol_sum == 0, 1.0, vol_sum)
        volume_imbalance = vol_diff / vol_sum

        return np.column_stack([spread_estimate, volume_imbalance])

    # ------------------------------------------------------------------ #
    # Cross-token correlation
    # ------------------------------------------------------------------ #
    def compute_cross_correlation(
        self,
        token_returns: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Rolling correlation matrix between tokens.

        Parameters
        ----------
        token_returns : dict
            ``{token_name: 1-D return series}``.  All series must be the
            same length.

        Returns
        -------
        np.ndarray
            Flattened upper-triangle of the correlation matrix at each
            time step, shape (T, n_pairs).
        """
        names = sorted(token_returns.keys())
        if len(names) < 2:
            logger.warning("Need ≥ 2 tokens for cross-correlation; returning zeros.")
            length = len(next(iter(token_returns.values())))
            return np.zeros((length, 1))

        series = np.column_stack([token_returns[n] for n in names])
        n_tokens = series.shape[1]
        n_pairs = n_tokens * (n_tokens - 1) // 2
        T = series.shape[0]
        corr_features = np.zeros((T, n_pairs))

        for t in range(self.lookback, T):
            window = series[t - self.lookback: t]
            # Correlation matrix for this window
            cov = np.cov(window, rowvar=False)
            std = np.sqrt(np.diag(cov))
            std = np.where(std == 0, 1.0, std)
            corr = cov / np.outer(std, std)
            # Upper triangle (excluding diagonal)
            idx = 0
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    corr_features[t, idx] = corr[i, j]
                    idx += 1
        return corr_features

    # ------------------------------------------------------------------ #
    # Unified pipeline
    # ------------------------------------------------------------------ #
    def build_feature_matrix(
        self,
        *,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        active_addresses: Optional[np.ndarray] = None,
        tx_count: Optional[np.ndarray] = None,
        fees_total: Optional[np.ndarray] = None,
        sentiment_score: Optional[np.ndarray] = None,
        mention_velocity: Optional[np.ndarray] = None,
        influencer_engagement: Optional[np.ndarray] = None,
        token_returns: Optional[Dict[str, np.ndarray]] = None,
        fit: bool = True,
    ) -> np.ndarray:
        """Build the complete feature matrix.

        Parameters
        ----------
        fit : bool
            If ``True``, compute and store normalisation parameters.
            If ``False``, re-use previously stored parameters (inference).
        """
        parts: List[np.ndarray] = []

        # Technical
        tech = self.compute_technical_features(open_, high, low, close, volume)
        parts.append(tech)

        # On-chain (optional)
        if active_addresses is not None and tx_count is not None and fees_total is not None:
            onchain = self.compute_onchain_features(active_addresses, tx_count, fees_total)
            parts.append(onchain)

        # Social (optional)
        if sentiment_score is not None and mention_velocity is not None and influencer_engagement is not None:
            social = self.compute_social_features(sentiment_score, mention_velocity, influencer_engagement)
            parts.append(social)

        # Micro-structure
        micro = self.compute_microstructure_features(high, low, close, volume)
        parts.append(micro)

        # Cross-token correlation (optional)
        if self.include_cross_corr and token_returns is not None:
            corr = self.compute_cross_correlation(token_returns)
            parts.append(corr)

        combined = np.concatenate(parts, axis=1)

        # Replace NaN from look-back periods with 0
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalise
        if self.normalization == "zscore":
            if fit:
                combined, mean, std = z_score_normalize(combined)
                self._norm_params = {"mean": mean, "std": std}
            else:
                combined, _, _ = z_score_normalize(
                    combined,
                    self._norm_params.get("mean"),
                    self._norm_params.get("std"),
                )
        elif self.normalization == "minmax":
            if fit:
                combined, mn, mx = min_max_normalize(combined)
                self._norm_params = {"min": mn, "max": mx}
            else:
                combined, _, _ = min_max_normalize(
                    combined,
                    self._norm_params.get("min"),
                    self._norm_params.get("max"),
                )
        else:
            raise ValueError(f"Unknown normalization: {self.normalization!r}")

        logger.info(
            "Feature matrix built: shape=%s, norm=%s",
            combined.shape, self.normalization,
        )
        return combined
