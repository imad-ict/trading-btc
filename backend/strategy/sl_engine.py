"""
Stop Loss Engine - Anti-Hunt placement.

Principle: SL hunts are structural, not random. Place SL where institutions place it.

Logic:
- Wick midpoint OR structure-based
- ATR-aware buffer
- Adaptive to volatility regime
- HARD BOUNDS: 0.15% - 0.35% from entry
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple

from config import RiskConstants
from core.market_data_engine import MarketDataEngine
from strategy.liquidity_engine import LiquidityEngine, SweepEvent

logger = logging.getLogger(__name__)


class InvalidStopLossError(Exception):
    """Raised when SL is outside valid bounds."""
    pass


@dataclass
class StopLossResult:
    """Calculated stop loss with explanation."""
    price: Decimal
    distance_pct: float
    sl_type: str  # "WICK_MIDPOINT" or "STRUCTURE"
    atr_buffer: Decimal
    explanation: str


class StopLossEngine:
    """
    Anti-hunt stop loss placement.
    
    Places SL where retail stops WON'T be, using wick midpoints
    or structure-based levels with ATR buffers.
    """
    
    def __init__(
        self,
        market_data: MarketDataEngine,
        liquidity_engine: LiquidityEngine,
    ):
        """
        Initialize SL engine.
        
        Args:
            market_data: MarketDataEngine instance
            liquidity_engine: LiquidityEngine instance
        """
        self.market_data = market_data
        self.liquidity = liquidity_engine
    
    def calculate_stop_loss(
        self,
        direction: str,
        entry_price: Decimal,
        sweep: SweepEvent,
    ) -> StopLossResult:
        """
        Calculate optimal stop loss placement.
        
        Args:
            direction: "LONG" or "SHORT"
            entry_price: Entry price
            sweep: The sweep event that triggered entry
            
        Returns:
            StopLossResult with price and explanation
            
        Raises:
            InvalidStopLossError: If SL distance is outside bounds
        """
        # Get ATR for buffer calculation
        atr = self.market_data.get_atr(14, "5m")
        atr_buffer = atr * Decimal("0.3") if atr else Decimal("0")
        
        # Calculate both options
        wick_sl = self._calculate_wick_midpoint_sl(direction, sweep, entry_price, atr_buffer)
        structure_sl = self._calculate_structure_sl(direction, entry_price, atr_buffer)
        
        # Select optimal SL
        sl_price, sl_type = self._select_optimal_sl(
            direction, entry_price, wick_sl, structure_sl
        )
        
        # Validate bounds
        distance_pct = self._calculate_distance_pct(entry_price, sl_price)
        
        if not self._validate_bounds(distance_pct):
            raise InvalidStopLossError(
                f"SL distance {distance_pct:.2f}% outside bounds "
                f"({RiskConstants.MIN_SL_DISTANCE_PCT}% - {RiskConstants.MAX_SL_DISTANCE_PCT}%)"
            )
        
        explanation = (
            f"{sl_type} at {sl_price} "
            f"({distance_pct:.2f}% from entry, ATR buffer: {atr_buffer})"
        )
        
        return StopLossResult(
            price=sl_price,
            distance_pct=distance_pct,
            sl_type=sl_type,
            atr_buffer=atr_buffer,
            explanation=explanation,
        )
    
    def _calculate_wick_midpoint_sl(
        self,
        direction: str,
        sweep: SweepEvent,
        entry_price: Decimal,
        atr_buffer: Decimal,
    ) -> Decimal:
        """
        Calculate SL at wick midpoint of sweep candle.
        
        Wick midpoint is where retail stops get hunted but price doesn't
        usually return to after a genuine move.
        """
        # Get the candle that caused the sweep
        candles = self.market_data.candles_5m.get_last_n(10)
        
        if not candles:
            # Fallback to ATR-based
            if direction == "LONG":
                return entry_price - (atr_buffer * 3)
            else:
                return entry_price + (atr_buffer * 3)
        
        # Find the sweep candle (highest/lowest depending on direction)
        if direction == "SHORT":
            # High sweep - SL above
            sweep_candle = max(candles, key=lambda c: float(c.high))
            body_top = max(float(sweep_candle.open), float(sweep_candle.close))
            wick_extreme = float(sweep_candle.high)
            wick_midpoint = (body_top + wick_extreme) / 2
            return Decimal(str(wick_midpoint)) + atr_buffer
        else:
            # Low sweep - SL below
            sweep_candle = min(candles, key=lambda c: float(c.low))
            body_bottom = min(float(sweep_candle.open), float(sweep_candle.close))
            wick_extreme = float(sweep_candle.low)
            wick_midpoint = (body_bottom + wick_extreme) / 2
            return Decimal(str(wick_midpoint)) - atr_buffer
    
    def _calculate_structure_sl(
        self,
        direction: str,
        entry_price: Decimal,
        atr_buffer: Decimal,
    ) -> Decimal:
        """
        Calculate SL behind recent structure.
        
        Places SL behind swing high/low that institutions would defend.
        """
        candles = self.market_data.candles_5m.get_last_n(20)
        
        if not candles:
            # Fallback
            if direction == "LONG":
                return entry_price - (atr_buffer * 3)
            else:
                return entry_price + (atr_buffer * 3)
        
        if direction == "LONG":
            # SL below recent swing low
            recent_low = min(float(c.low) for c in candles[-10:])
            return Decimal(str(recent_low)) - atr_buffer
        else:
            # SL above recent swing high
            recent_high = max(float(c.high) for c in candles[-10:])
            return Decimal(str(recent_high)) + atr_buffer
    
    def _select_optimal_sl(
        self,
        direction: str,
        entry_price: Decimal,
        wick_sl: Decimal,
        structure_sl: Decimal,
    ) -> Tuple[Decimal, str]:
        """
        Select optimal SL from calculated options.
        
        Prefers tighter SL that's still within bounds.
        """
        wick_dist = self._calculate_distance_pct(entry_price, wick_sl)
        structure_dist = self._calculate_distance_pct(entry_price, structure_sl)
        
        wick_valid = self._validate_bounds(wick_dist)
        structure_valid = self._validate_bounds(structure_dist)
        
        # Prefer wick midpoint (more precise anti-hunt)
        if wick_valid:
            return wick_sl, "WICK_MIDPOINT"
        elif structure_valid:
            return structure_sl, "STRUCTURE"
        else:
            # Force to within bounds
            forced_sl = self._force_within_bounds(direction, entry_price)
            return forced_sl, "FORCED_BOUNDS"
    
    def _force_within_bounds(self, direction: str, entry_price: Decimal) -> Decimal:
        """Force SL to middle of allowed range."""
        mid_pct = (RiskConstants.MIN_SL_DISTANCE_PCT + RiskConstants.MAX_SL_DISTANCE_PCT) / 2
        distance = entry_price * Decimal(str(mid_pct / 100))
        
        if direction == "LONG":
            return entry_price - distance
        else:
            return entry_price + distance
    
    def _calculate_distance_pct(self, entry: Decimal, sl: Decimal) -> float:
        """Calculate SL distance as percentage."""
        distance = abs(float(entry) - float(sl))
        return (distance / float(entry)) * 100
    
    def _validate_bounds(self, distance_pct: float) -> bool:
        """Check if distance is within allowed bounds."""
        return RiskConstants.MIN_SL_DISTANCE_PCT <= distance_pct <= RiskConstants.MAX_SL_DISTANCE_PCT
    
    def adjust_to_breakeven(
        self,
        entry_price: Decimal,
        direction: str,
    ) -> Decimal:
        """
        Adjust SL to breakeven + small buffer.
        
        Used after TP1 is hit.
        """
        # Small buffer to avoid exact breakeven wicks
        buffer_pct = Decimal("0.02")  # 0.02%
        buffer = entry_price * (buffer_pct / 100)
        
        if direction == "LONG":
            return entry_price + buffer
        else:
            return entry_price - buffer
    
    def trail_to_level(
        self,
        current_price: Decimal,
        direction: str,
        trail_pct: float = 0.2,
    ) -> Decimal:
        """
        Calculate trailing stop level.
        
        Args:
            current_price: Current market price
            direction: Trade direction
            trail_pct: Trail distance as percentage
        """
        trail_distance = current_price * Decimal(str(trail_pct / 100))
        
        if direction == "LONG":
            return current_price - trail_distance
        else:
            return current_price + trail_distance
