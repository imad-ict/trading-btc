"""
Market State Engine (15M) - Classifies market conditions.

Classifications:
- EXPANSION: Trending, tradeable
- MANIPULATION: Liquidity hunt in progress
- NO_TRADE: Choppy/unclear

Inputs:
- VWAP slope
- Volume regime
- Candle efficiency
- Range compression
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from config import MarketStateThresholds
from core.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


class MarketState(str, Enum):
    """Market state classification."""
    EXPANSION = "EXPANSION"
    MANIPULATION = "MANIPULATION"
    NO_TRADE = "NO_TRADE"


@dataclass
class MarketStateAnalysis:
    """Complete market state analysis."""
    state: MarketState
    confidence: float  # 0.0 - 1.0
    
    # Component scores
    vwap_score: float
    volume_score: float
    efficiency_score: float
    compression_score: float
    
    # Raw values
    vwap_slope: Optional[float]
    volume_ratio: Optional[float]
    candle_efficiency: Optional[float]
    range_compression: Optional[float]
    
    @property
    def is_tradeable(self) -> bool:
        """Check if state allows trading."""
        return self.state == MarketState.EXPANSION and self.confidence >= 0.6
    
    def to_explanation(self) -> str:
        """Format for trade explanation."""
        parts = [self.state.value]
        
        if self.vwap_slope is not None:
            sign = "+" if self.vwap_slope >= 0 else ""
            parts.append(f"VWAP {sign}{self.vwap_slope*100:.2f}%")
        
        if self.volume_ratio is not None:
            regime = "HIGH" if self.volume_ratio >= 1.5 else "LOW" if self.volume_ratio <= 0.5 else "NORMAL"
            parts.append(f"{regime} vol ({self.volume_ratio:.1f}x)")
        
        if self.candle_efficiency is not None:
            parts.append(f"eff {self.candle_efficiency:.0%}")
        
        parts.append(f"conf {self.confidence:.0%}")
        
        return " | ".join(parts)


class MarketStateEngine:
    """
    Classifies market state using multiple inputs on 15M timeframe.
    
    Principle: Trade only in EXPANSION. Avoid MANIPULATION and NO_TRADE.
    """
    
    def __init__(self, market_data: MarketDataEngine):
        """
        Initialize with market data engine.
        
        Args:
            market_data: MarketDataEngine instance
        """
        self.market_data = market_data
        self.thresholds = MarketStateThresholds()
        self._last_state: Optional[MarketState] = None
    
    def analyze(self) -> MarketStateAnalysis:
        """
        Perform complete market state analysis.
        
        Returns:
            MarketStateAnalysis with state and component scores
        """
        # Get component values
        vwap_slope = self._get_vwap_slope()
        volume_ratio = self._get_volume_ratio()
        candle_efficiency = self._get_candle_efficiency()
        range_compression = self._get_range_compression()
        
        # Score each component (0.0 - 1.0)
        vwap_score = self._score_vwap(vwap_slope)
        volume_score = self._score_volume(volume_ratio)
        efficiency_score = self._score_efficiency(candle_efficiency)
        compression_score = self._score_compression(range_compression)
        
        # Aggregate scores with weights
        weights = {
            "vwap": 0.3,
            "volume": 0.25,
            "efficiency": 0.25,
            "compression": 0.2,
        }
        
        # Valid components only
        valid_scores = []
        if vwap_score is not None:
            valid_scores.append(("vwap", vwap_score))
        if volume_score is not None:
            valid_scores.append(("volume", volume_score))
        if efficiency_score is not None:
            valid_scores.append(("efficiency", efficiency_score))
        if compression_score is not None:
            valid_scores.append(("compression", compression_score))
        
        if not valid_scores:
            # Insufficient data
            return MarketStateAnalysis(
                state=MarketState.NO_TRADE,
                confidence=0.0,
                vwap_score=0.0,
                volume_score=0.0,
                efficiency_score=0.0,
                compression_score=0.0,
                vwap_slope=None,
                volume_ratio=None,
                candle_efficiency=None,
                range_compression=None,
            )
        
        # Calculate weighted score
        total_weight = sum(weights[name] for name, _ in valid_scores)
        weighted_score = sum(
            score * weights[name] / total_weight
            for name, score in valid_scores
        )
        
        # Determine state based on aggregated score
        state = self._classify_state(weighted_score, vwap_slope, volume_ratio)
        
        # Track state transitions
        if state != self._last_state:
            logger.info(f"Market state transition: {self._last_state} -> {state}")
            self._last_state = state
        
        return MarketStateAnalysis(
            state=state,
            confidence=weighted_score,
            vwap_score=vwap_score or 0.0,
            volume_score=volume_score or 0.0,
            efficiency_score=efficiency_score or 0.0,
            compression_score=compression_score or 0.0,
            vwap_slope=vwap_slope,
            volume_ratio=volume_ratio,
            candle_efficiency=candle_efficiency,
            range_compression=range_compression,
        )
    
    def _get_vwap_slope(self) -> Optional[float]:
        """Get VWAP slope from market data."""
        vwap = self.market_data.get_vwap()
        return vwap.slope if vwap else None
    
    def _get_volume_ratio(self) -> Optional[float]:
        """Get volume ratio from market data."""
        regime = self.market_data.get_volume_regime("15m")
        return regime.volume_ratio if regime else None
    
    def _get_candle_efficiency(self) -> Optional[float]:
        """Get candle efficiency from market data."""
        return self.market_data.get_candle_efficiency("15m")
    
    def _get_range_compression(self) -> Optional[float]:
        """Get range compression from market data."""
        return self.market_data.get_range_compression(20, "15m")
    
    def _score_vwap(self, slope: Optional[float]) -> Optional[float]:
        """Score VWAP slope (higher = more directional)."""
        if slope is None:
            return None
        
        abs_slope = abs(slope)
        
        if abs_slope >= self.thresholds.VWAP_EXPANSION_SLOPE:
            return 1.0  # Strong trend
        elif abs_slope >= self.thresholds.VWAP_MANIPULATION_SLOPE:
            # Linear interpolation between manipulation and expansion
            return (abs_slope - self.thresholds.VWAP_MANIPULATION_SLOPE) / \
                   (self.thresholds.VWAP_EXPANSION_SLOPE - self.thresholds.VWAP_MANIPULATION_SLOPE)
        else:
            return 0.0  # Choppy/manipulation
    
    def _score_volume(self, ratio: Optional[float]) -> Optional[float]:
        """Score volume ratio (higher = more conviction)."""
        if ratio is None:
            return None
        
        if ratio >= self.thresholds.HIGH_VOLUME_MULTIPLIER:
            return 1.0  # High volume = good
        elif ratio >= 1.0:
            return 0.7  # Normal volume
        elif ratio >= self.thresholds.LOW_VOLUME_MULTIPLIER:
            return 0.4  # Below average
        else:
            return 0.1  # Very low volume = bad
    
    def _score_efficiency(self, efficiency: Optional[float]) -> Optional[float]:
        """Score candle efficiency (higher = directional)."""
        if efficiency is None:
            return None
        
        if efficiency >= self.thresholds.MIN_CANDLE_EFFICIENCY:
            return 1.0  # Directional candle
        else:
            return efficiency / self.thresholds.MIN_CANDLE_EFFICIENCY
    
    def _score_compression(self, compression: Optional[float]) -> Optional[float]:
        """Score range compression (lower = more compressed = potential)."""
        if compression is None:
            return None
        
        # Invert: low compression = high score (breakout potential)
        if compression <= self.thresholds.COMPRESSION_THRESHOLD:
            return 0.3  # Compressed = wait for breakout
        elif compression >= 1.0:
            return 1.0  # Expanded = trending
        else:
            return (compression - self.thresholds.COMPRESSION_THRESHOLD) / \
                   (1.0 - self.thresholds.COMPRESSION_THRESHOLD)
    
    def _classify_state(
        self,
        weighted_score: float,
        vwap_slope: Optional[float],
        volume_ratio: Optional[float],
    ) -> MarketState:
        """Classify market state based on aggregated score."""
        
        # EXPANSION: High score + directional VWAP + good volume
        if weighted_score >= 0.7:
            if vwap_slope is not None and abs(vwap_slope) >= self.thresholds.VWAP_EXPANSION_SLOPE:
                if volume_ratio is None or volume_ratio >= 0.7:
                    return MarketState.EXPANSION
        
        # MANIPULATION: Low VWAP slope with high volume (stop hunt)
        if vwap_slope is not None and abs(vwap_slope) < self.thresholds.VWAP_MANIPULATION_SLOPE:
            if volume_ratio is not None and volume_ratio >= self.thresholds.HIGH_VOLUME_MULTIPLIER:
                return MarketState.MANIPULATION
        
        # Medium score could still be expansion
        if weighted_score >= 0.5:
            return MarketState.EXPANSION
        
        # Default: Not tradeable
        return MarketState.NO_TRADE
    
    def is_tradeable(self) -> bool:
        """Quick check if market is currently tradeable."""
        analysis = self.analyze()
        return analysis.is_tradeable
