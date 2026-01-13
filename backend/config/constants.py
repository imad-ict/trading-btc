"""
INSTITUTIONAL RISK CONSTANTS (NON-NEGOTIABLE)

These constants are HARD-CODED and must NEVER be modified at runtime.
They represent the core principles of capital preservation.
"""
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class RiskConstants:
    """Hard-coded risk management rules. Capital preservation first."""
    
    # Position Risk
    RISK_PER_TRADE: Final[float] = 0.003  # 0.3% of account per trade
    
    # Daily Limits
    MAX_TRADES_PER_DAY: Final[int] = 10
    MAX_LOSSES_PER_DAY: Final[int] = 4
    DAILY_LOSS_CAP_USD: Final[float] = 100.0
    DAILY_PROFIT_LOCK_USD: Final[float] = 75.0
    
    # Leverage & Margin
    MAX_LEVERAGE: Final[int] = 5
    MARGIN_TYPE: Final[str] = "ISOLATED"  # Always isolated, never cross
    
    # Stop Loss Bounds (percentage from entry)
    MIN_SL_DISTANCE_PCT: Final[float] = 0.15
    MAX_SL_DISTANCE_PCT: Final[float] = 0.35
    
    # Spread Filter
    MAX_SPREAD_PCT: Final[float] = 0.05  # Reject if spread > 0.05%
    
    # Adaptive Risk
    LOSS_STREAK_PENALTY: Final[float] = 0.25  # Reduce risk by 25% per consecutive loss
    MAX_RISK_REDUCTION: Final[float] = 0.5    # Never reduce below 50% of base risk
    NEVER_INCREASE_AFTER_WIN: Final[bool] = True  # Risk never increases after wins


@dataclass(frozen=True)
class SessionConstants:
    """Trading session time gates (UTC)."""
    
    # London Session
    LONDON_START_UTC: Final[int] = 8   # 08:00 UTC
    LONDON_END_UTC: Final[int] = 12    # 12:00 UTC
    
    # New York Session  
    NY_START_UTC: Final[int] = 13      # 13:00 UTC
    NY_END_UTC: Final[int] = 17        # 17:00 UTC
    
    # Asian Session (Optional - lower priority)
    ASIAN_START_UTC: Final[int] = 0    # 00:00 UTC
    ASIAN_END_UTC: Final[int] = 4      # 04:00 UTC
    ASIAN_ENABLED: Final[bool] = False  # Disabled by default
    
    # Volume Requirements
    MIN_VOLUME_MULTIPLIER: Final[float] = 0.7  # At least 70% of average volume


@dataclass(frozen=True)
class TradingConstants:
    """Trading execution constants."""
    
    # Symbols
    MAX_CONCURRENT_SYMBOLS: Final[int] = 3
    DEFAULT_SYMBOLS: Final[tuple] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    
    # Timeframes
    MARKET_STATE_TF: Final[str] = "15m"   # Market state classification
    LIQUIDITY_TF: Final[str] = "5m"       # Liquidity mapping
    ENTRY_TF: Final[str] = "1m"           # Entry signals
    
    # Entry Delay (Anti-Algo)
    ENTRY_DELAY_CANDLES: Final[int] = 1   # Wait 1-2 candles after signal
    
    # Take Profit Scaling
    TP1_PCT: Final[float] = 0.40  # 40% at TP1
    TP2_PCT: Final[float] = 0.30  # 30% at TP2
    TP3_PCT: Final[float] = 0.30  # 30% runner
    
    # Stagnation Exit
    STAGNATION_CANDLES: Final[int] = 15  # Exit if no progress in 15 candles
    
    # Liquidity Engine
    MIN_LIQUIDITY_TOUCHES: Final[int] = 2  # Minimum touches for valid level
    LIQUIDITY_RECENCY_HOURS: Final[int] = 48  # Prioritize recent levels


@dataclass(frozen=True)
class MarketStateThresholds:
    """Thresholds for market state classification."""
    
    # VWAP Slope Thresholds
    VWAP_EXPANSION_SLOPE: Final[float] = 0.001  # 0.1% slope for expansion
    VWAP_MANIPULATION_SLOPE: Final[float] = 0.0003  # Below this = manipulation
    
    # Volume Regime
    HIGH_VOLUME_MULTIPLIER: Final[float] = 1.5
    LOW_VOLUME_MULTIPLIER: Final[float] = 0.5
    
    # Candle Efficiency (body/range ratio)
    MIN_CANDLE_EFFICIENCY: Final[float] = 0.6
    
    # Range Compression
    COMPRESSION_THRESHOLD: Final[float] = 0.6  # Range < 60% of average


# Validate constants at import time
def _validate_constants() -> None:
    """Ensure constants are within safe bounds."""
    assert RiskConstants.RISK_PER_TRADE <= 0.01, "Risk per trade must not exceed 1%"
    assert RiskConstants.MAX_LEVERAGE <= 10, "Leverage must not exceed 10x"
    assert RiskConstants.MIN_SL_DISTANCE_PCT < RiskConstants.MAX_SL_DISTANCE_PCT
    assert TradingConstants.TP1_PCT + TradingConstants.TP2_PCT + TradingConstants.TP3_PCT == 1.0


_validate_constants()
