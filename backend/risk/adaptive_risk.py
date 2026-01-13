"""
Adaptive Risk Manager - Dynamic risk scaling.

Principle: Risk reduces after losses. Risk NEVER increases after wins.

Dynamic Scaling Based On:
- Daily drawdown
- Win/loss streak
- Volatility regime
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from config import RiskConstants
from core.market_data_engine import MarketDataEngine

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveRiskState:
    """Current adaptive risk state."""
    base_risk_pct: float
    adjusted_risk_pct: float
    scaling_factor: float
    
    # Reasons for adjustment
    streak_penalty: float
    volatility_adjustment: float
    drawdown_adjustment: float
    
    def to_string(self) -> str:
        """Format for logging."""
        return (
            f"Risk: {self.base_risk_pct*100:.2f}% → {self.adjusted_risk_pct*100:.2f}% "
            f"(scale: {self.scaling_factor:.2f})"
        )


class AdaptiveRiskManager:
    """
    Dynamically adjusts risk based on performance and conditions.
    
    CRITICAL PRINCIPLE: Risk NEVER increases after wins.
    """
    
    def __init__(self, market_data: Optional[MarketDataEngine] = None):
        """
        Initialize adaptive risk manager.
        
        Args:
            market_data: Optional MarketDataEngine for volatility
        """
        self.market_data = market_data
        
        # Base risk from constants
        self.base_risk = RiskConstants.RISK_PER_TRADE
        
        # Current state
        self._current_streak: int = 0
        self._daily_drawdown_pct: float = 0.0
        self._peak_daily_pnl: Decimal = Decimal("0")
        self._current_daily_pnl: Decimal = Decimal("0")
    
    def get_adjusted_risk(self) -> AdaptiveRiskState:
        """
        Calculate adjusted risk based on current conditions.
        
        Returns:
            AdaptiveRiskState with adjusted risk percentage
        """
        scaling_factor = 1.0
        streak_penalty = 0.0
        volatility_adj = 0.0
        drawdown_adj = 0.0
        
        # 1. STREAK PENALTY (losses reduce risk)
        if self._current_streak < 0:
            # Each consecutive loss reduces risk by 25%
            loss_count = abs(self._current_streak)
            streak_penalty = loss_count * RiskConstants.LOSS_STREAK_PENALTY
            scaling_factor *= (1.0 - min(streak_penalty, 1 - RiskConstants.MAX_RISK_REDUCTION))
        
        # 2. VOLATILITY ADJUSTMENT
        if self.market_data:
            vol_adj = self._calculate_volatility_adjustment()
            volatility_adj = vol_adj
            scaling_factor *= (1.0 - vol_adj)
        
        # 3. DRAWDOWN ADJUSTMENT
        if self._daily_drawdown_pct > 0:
            # Reduce risk proportionally to drawdown
            dd_penalty = self._daily_drawdown_pct / 100  # Convert to 0-1
            drawdown_adj = min(dd_penalty, 0.5)  # Cap at 50% reduction
            scaling_factor *= (1.0 - drawdown_adj)
        
        # 4. NEVER INCREASE ABOVE BASE (even after wins)
        scaling_factor = min(scaling_factor, 1.0)
        
        # 5. NEVER REDUCE BELOW MINIMUM
        scaling_factor = max(scaling_factor, RiskConstants.MAX_RISK_REDUCTION)
        
        adjusted_risk = self.base_risk * scaling_factor
        
        return AdaptiveRiskState(
            base_risk_pct=self.base_risk,
            adjusted_risk_pct=adjusted_risk,
            scaling_factor=scaling_factor,
            streak_penalty=streak_penalty,
            volatility_adjustment=volatility_adj,
            drawdown_adjustment=drawdown_adj,
        )
    
    def _calculate_volatility_adjustment(self) -> float:
        """
        Calculate risk adjustment based on volatility.
        
        Higher volatility = reduce risk.
        
        Returns:
            Reduction percentage (0.0 - 0.3)
        """
        if not self.market_data:
            return 0.0
        
        atr = self.market_data.get_atr(14, "5m")
        if not atr:
            return 0.0
        
        current_price = self.market_data.current_price
        if not current_price or current_price == 0:
            return 0.0
        
        # ATR as percentage of price
        atr_pct = float(atr) / float(current_price) * 100
        
        # Normal ATR for crypto is ~0.5-1.5%
        # High volatility: > 2%
        if atr_pct > 2.5:
            return 0.3  # Reduce 30%
        elif atr_pct > 2.0:
            return 0.2  # Reduce 20%
        elif atr_pct > 1.5:
            return 0.1  # Reduce 10%
        
        return 0.0
    
    def update_streak(self, is_win: bool) -> None:
        """
        Update win/loss streak.
        
        Args:
            is_win: True if trade was profitable
        """
        if is_win:
            if self._current_streak >= 0:
                self._current_streak += 1
            else:
                self._current_streak = 1
        else:
            if self._current_streak <= 0:
                self._current_streak -= 1
            else:
                self._current_streak = -1
        
        logger.info(f"Streak updated: {self._current_streak}")
    
    def update_daily_pnl(self, current_pnl: Decimal) -> None:
        """
        Update daily P&L for drawdown calculation.
        
        Args:
            current_pnl: Current daily P&L
        """
        self._current_daily_pnl = current_pnl
        
        # Track peak
        if current_pnl > self._peak_daily_pnl:
            self._peak_daily_pnl = current_pnl
        
        # Calculate drawdown from peak
        if self._peak_daily_pnl > 0:
            drawdown = self._peak_daily_pnl - current_pnl
            self._daily_drawdown_pct = float(drawdown / self._peak_daily_pnl * 100)
        else:
            self._daily_drawdown_pct = 0.0
    
    def reset_daily(self) -> None:
        """Reset daily tracking for new trading day."""
        self._daily_drawdown_pct = 0.0
        self._peak_daily_pnl = Decimal("0")
        self._current_daily_pnl = Decimal("0")
        # Streak persists across days
        logger.info("Daily adaptive risk state reset")
    
    @property
    def current_streak(self) -> int:
        """Get current streak value."""
        return self._current_streak
