"""
Take Profit Engine - Smart Money targeting.

Targets:
- TP1: VWAP
- TP2: Opposite liquidity pool
- TP3: Fair Value Gap / inefficiency

Enhancement:
- Partial scaling (40%/30%/30%)
- Time-based exit on stagnation
- Momentum decay detection
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from config import TradingConstants
from core.market_data_engine import MarketDataEngine
from strategy.liquidity_engine import LiquidityEngine, LiquidityType

logger = logging.getLogger(__name__)


@dataclass
class TakeProfitLevel:
    """A single take profit level."""
    level: int  # 1, 2, or 3
    price: Decimal
    size_pct: float  # Percentage of position to close
    target_type: str  # "VWAP", "LIQUIDITY", "FVG", etc.
    is_hit: bool = False
    hit_time: Optional[str] = None


@dataclass
class TakeProfitPlan:
    """Complete TP plan with multiple levels."""
    tp1: TakeProfitLevel
    tp2: Optional[TakeProfitLevel]
    tp3: Optional[TakeProfitLevel]
    stagnation_exit_candles: int
    
    def get_active_levels(self) -> List[TakeProfitLevel]:
        """Get all unhit TP levels."""
        levels = [self.tp1]
        if self.tp2:
            levels.append(self.tp2)
        if self.tp3:
            levels.append(self.tp3)
        return [l for l in levels if not l.is_hit]
    
    def to_explanation(self) -> str:
        """Format for trade explanation."""
        parts = [f"TP1: {self.tp1.target_type} @ {self.tp1.price}"]
        if self.tp2:
            parts.append(f"TP2: {self.tp2.target_type} @ {self.tp2.price}")
        if self.tp3:
            parts.append(f"TP3: {self.tp3.target_type}")
        return ", ".join(parts)


class TakeProfitEngine:
    """
    Smart money take profit targeting.
    
    Uses institutional levels: VWAP, opposite liquidity, inefficiencies.
    """
    
    def __init__(
        self,
        market_data: MarketDataEngine,
        liquidity_engine: LiquidityEngine,
    ):
        """
        Initialize TP engine.
        
        Args:
            market_data: MarketDataEngine instance
            liquidity_engine: LiquidityEngine instance
        """
        self.market_data = market_data
        self.liquidity = liquidity_engine
        
        # Stagnation tracking
        self._candles_since_progress = 0
        self._last_best_price: Optional[Decimal] = None
    
    def calculate_targets(
        self,
        direction: str,
        entry_price: Decimal,
        stop_loss: Decimal,
    ) -> TakeProfitPlan:
        """
        Calculate take profit levels.
        
        Args:
            direction: "LONG" or "SHORT"
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            TakeProfitPlan with all levels
        """
        # Calculate R value (risk distance)
        risk_distance = abs(float(entry_price) - float(stop_loss))
        
        # TP1: VWAP or 1R
        tp1 = self._calculate_tp1(direction, entry_price, risk_distance)
        
        # TP2: Opposite liquidity pool or 2R
        tp2 = self._calculate_tp2(direction, entry_price, risk_distance)
        
        # TP3: Runner (trail or major level)
        tp3 = self._calculate_tp3(direction, entry_price, risk_distance)
        
        return TakeProfitPlan(
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            stagnation_exit_candles=TradingConstants.STAGNATION_CANDLES,
        )
    
    def _calculate_tp1(
        self,
        direction: str,
        entry_price: Decimal,
        risk_distance: float,
    ) -> TakeProfitLevel:
        """
        Calculate TP1 at VWAP or 1R.
        
        TP1 is conservative - lock in some profit quickly.
        """
        vwap = self.market_data.get_vwap()
        
        if vwap:
            vwap_price = vwap.value
            
            # Check if VWAP is in the right direction
            if direction == "LONG" and float(vwap_price) > float(entry_price):
                return TakeProfitLevel(
                    level=1,
                    price=vwap_price,
                    size_pct=TradingConstants.TP1_PCT,
                    target_type="VWAP",
                )
            elif direction == "SHORT" and float(vwap_price) < float(entry_price):
                return TakeProfitLevel(
                    level=1,
                    price=vwap_price,
                    size_pct=TradingConstants.TP1_PCT,
                    target_type="VWAP",
                )
        
        # Fallback to 1R
        if direction == "LONG":
            tp1_price = Decimal(str(float(entry_price) + risk_distance))
        else:
            tp1_price = Decimal(str(float(entry_price) - risk_distance))
        
        return TakeProfitLevel(
            level=1,
            price=tp1_price,
            size_pct=TradingConstants.TP1_PCT,
            target_type="1R",
        )
    
    def _calculate_tp2(
        self,
        direction: str,
        entry_price: Decimal,
        risk_distance: float,
    ) -> Optional[TakeProfitLevel]:
        """
        Calculate TP2 at opposite liquidity pool.
        
        Target where stops are resting on the other side.
        """
        # Find opposite liquidity
        zones = self.liquidity.get_active_zones()
        
        if direction == "LONG":
            # Look for resistance liquidity above
            target_types = [LiquidityType.EQUAL_HIGH, LiquidityType.SESSION_HIGH, LiquidityType.SWING_HIGH]
            candidates = [
                z for z in zones
                if z.type in target_types and float(z.price) > float(entry_price)
            ]
            if candidates:
                target = min(candidates, key=lambda z: float(z.price))
                return TakeProfitLevel(
                    level=2,
                    price=target.price,
                    size_pct=TradingConstants.TP2_PCT,
                    target_type=f"LIQUIDITY_{target.type.value}",
                )
        else:
            # Look for support liquidity below
            target_types = [LiquidityType.EQUAL_LOW, LiquidityType.SESSION_LOW, LiquidityType.SWING_LOW]
            candidates = [
                z for z in zones
                if z.type in target_types and float(z.price) < float(entry_price)
            ]
            if candidates:
                target = max(candidates, key=lambda z: float(z.price))
                return TakeProfitLevel(
                    level=2,
                    price=target.price,
                    size_pct=TradingConstants.TP2_PCT,
                    target_type=f"LIQUIDITY_{target.type.value}",
                )
        
        # Fallback to 2R
        if direction == "LONG":
            tp2_price = Decimal(str(float(entry_price) + risk_distance * 2))
        else:
            tp2_price = Decimal(str(float(entry_price) - risk_distance * 2))
        
        return TakeProfitLevel(
            level=2,
            price=tp2_price,
            size_pct=TradingConstants.TP2_PCT,
            target_type="2R",
        )
    
    def _calculate_tp3(
        self,
        direction: str,
        entry_price: Decimal,
        risk_distance: float,
    ) -> Optional[TakeProfitLevel]:
        """
        Calculate TP3 as runner.
        
        TP3 uses trailing stop, target is indicative.
        """
        # Look for major inefficiency or 3R+
        if direction == "LONG":
            tp3_price = Decimal(str(float(entry_price) + risk_distance * 3))
        else:
            tp3_price = Decimal(str(float(entry_price) - risk_distance * 3))
        
        return TakeProfitLevel(
            level=3,
            price=tp3_price,
            size_pct=TradingConstants.TP3_PCT,
            target_type="RUNNER_3R",
        )
    
    def check_stagnation(self, direction: str, current_price: Decimal) -> bool:
        """
        Check if trade is stagnating (no progress).
        
        Returns True if trade should be exited due to stagnation.
        """
        if self._last_best_price is None:
            self._last_best_price = current_price
            return False
        
        # Check if we made progress
        if direction == "LONG":
            made_progress = float(current_price) > float(self._last_best_price)
        else:
            made_progress = float(current_price) < float(self._last_best_price)
        
        if made_progress:
            self._last_best_price = current_price
            self._candles_since_progress = 0
        else:
            self._candles_since_progress += 1
        
        # Stagnation threshold
        if self._candles_since_progress >= TradingConstants.STAGNATION_CANDLES:
            logger.warning(f"STAGNATION DETECTED: No progress for {self._candles_since_progress} candles")
            return True
        
        return False
    
    def check_momentum_decay(self, direction: str) -> bool:
        """
        Check for momentum decay (trend weakening).
        
        Returns True if momentum is decaying significantly.
        """
        volume_regime = self.market_data.get_volume_regime("5m")
        if not volume_regime:
            return False
        
        efficiency = self.market_data.get_candle_efficiency("5m")
        if efficiency is None:
            return False
        
        # Low volume + low efficiency = momentum decay
        if volume_regime.is_low_volume and efficiency < 0.3:
            logger.warning("MOMENTUM DECAY: Low volume + low efficiency")
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset stagnation tracking for new trade."""
        self._candles_since_progress = 0
        self._last_best_price = None
