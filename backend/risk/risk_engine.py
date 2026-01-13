"""
Risk Engine - Pre-validated risk management.

Principle: Risk is pre-validated. Trades are invalidated BEFORE entry, not after loss.

HARD-CODED LIMITS (NON-NEGOTIABLE):
- Risk per trade: 0.3%
- Max trades/day: 10
- Max losses/day: 4
- Daily loss cap: $100
- Daily profit lock: $75
- Max leverage: 5×
- Isolated margin only
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from config import RiskConstants

logger = logging.getLogger(__name__)


@dataclass
class RiskValidation:
    """Result of risk validation."""
    is_valid: bool
    rejection_reason: Optional[str] = None
    
    # Validated values
    position_size: Optional[Decimal] = None
    risk_usd: Optional[Decimal] = None
    risk_pct: Optional[float] = None


@dataclass
class DailyState:
    """Daily trading state for risk tracking."""
    trades_today: int = 0
    losses_today: int = 0
    daily_pnl: Decimal = Decimal("0")
    last_trade_time: Optional[datetime] = None
    
    def reset(self) -> None:
        """Reset for new trading day."""
        self.trades_today = 0
        self.losses_today = 0
        self.daily_pnl = Decimal("0")
        self.last_trade_time = None


class RiskEngine:
    """
    Pre-trade risk validation engine.
    
    CRITICAL: Risk engine executes BEFORE strategy.
    If risk is invalid, trade is REJECTED before execution.
    """
    
    def __init__(self, account_balance: Decimal):
        """
        Initialize risk engine.
        
        Args:
            account_balance: Current account balance in USDT
        """
        self.balance = account_balance
        self.daily_state = DailyState()
        
        # Streak tracking for adaptive risk
        self.current_streak: int = 0  # Positive = wins, negative = losses
        
        logger.info(f"RiskEngine initialized with balance: ${self.balance}")
    
    def update_balance(self, new_balance: Decimal) -> None:
        """Update account balance."""
        self.balance = new_balance
    
    def validate_trade(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
    ) -> RiskValidation:
        """
        Pre-validate a trade before execution.
        
        This is the GATE that must pass before any order is placed.
        
        Args:
            entry_price: Proposed entry price
            stop_loss: Proposed stop loss price
            
        Returns:
            RiskValidation with either approval or rejection reason
        """
        # 1. MAX TRADES PER DAY
        if self.daily_state.trades_today >= RiskConstants.MAX_TRADES_PER_DAY:
            return RiskValidation(
                is_valid=False,
                rejection_reason=f"Max trades reached: {self.daily_state.trades_today}/{RiskConstants.MAX_TRADES_PER_DAY}",
            )
        
        # 2. MAX LOSSES PER DAY
        if self.daily_state.losses_today >= RiskConstants.MAX_LOSSES_PER_DAY:
            return RiskValidation(
                is_valid=False,
                rejection_reason=f"Max losses reached: {self.daily_state.losses_today}/{RiskConstants.MAX_LOSSES_PER_DAY}",
            )
        
        # 3. DAILY LOSS CAP
        if self.daily_state.daily_pnl <= Decimal(str(-RiskConstants.DAILY_LOSS_CAP_USD)):
            return RiskValidation(
                is_valid=False,
                rejection_reason=f"Daily loss cap hit: ${self.daily_state.daily_pnl}",
            )
        
        # 4. DAILY PROFIT LOCK
        if self.daily_state.daily_pnl >= Decimal(str(RiskConstants.DAILY_PROFIT_LOCK_USD)):
            return RiskValidation(
                is_valid=False,
                rejection_reason=f"Daily profit locked: ${self.daily_state.daily_pnl} (protecting gains)",
            )
        
        # 5. CALCULATE POSITION SIZE
        try:
            position_size, risk_usd = self.calculate_position_size(entry_price, stop_loss)
        except ValueError as e:
            return RiskValidation(
                is_valid=False,
                rejection_reason=str(e),
            )
        
        # 6. VALIDATE SL DISTANCE
        sl_distance_pct = abs(float(entry_price) - float(stop_loss)) / float(entry_price) * 100
        if sl_distance_pct < RiskConstants.MIN_SL_DISTANCE_PCT:
            return RiskValidation(
                is_valid=False,
                rejection_reason=f"SL too tight: {sl_distance_pct:.2f}% < {RiskConstants.MIN_SL_DISTANCE_PCT}%",
            )
        if sl_distance_pct > RiskConstants.MAX_SL_DISTANCE_PCT:
            return RiskValidation(
                is_valid=False,
                rejection_reason=f"SL too wide: {sl_distance_pct:.2f}% > {RiskConstants.MAX_SL_DISTANCE_PCT}%",
            )
        
        # ALL CHECKS PASSED
        return RiskValidation(
            is_valid=True,
            position_size=position_size,
            risk_usd=risk_usd,
            risk_pct=RiskConstants.RISK_PER_TRADE * 100,
        )
    
    def calculate_position_size(
        self,
        entry_price: Decimal,
        stop_loss: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate position size from risk parameters.
        
        Formula: Position Size = (Balance × Risk%) / |Entry - SL|
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Tuple of (position_size, risk_usd)
        """
        # Risk amount in USD
        risk_usd = self.balance * Decimal(str(RiskConstants.RISK_PER_TRADE))
        
        # SL distance per unit
        sl_distance = abs(entry_price - stop_loss)
        
        if sl_distance == 0:
            raise ValueError("Invalid SL: distance is zero")
        
        # Position size (in base currency, e.g., BTC)
        position_size = risk_usd / sl_distance
        
        # Apply leverage cap
        max_position = self._max_position_for_leverage(entry_price)
        if position_size > max_position:
            position_size = max_position
            # Recalculate actual risk
            risk_usd = position_size * sl_distance
        
        return position_size, risk_usd
    
    def _max_position_for_leverage(self, entry_price: Decimal) -> Decimal:
        """Calculate max position size allowed by leverage limit."""
        # Max notional = Balance × Leverage
        max_notional = self.balance * Decimal(str(RiskConstants.MAX_LEVERAGE))
        # Max position = Max notional / Entry price
        return max_notional / entry_price
    
    def record_trade_start(self) -> None:
        """Record that a new trade has started."""
        self.daily_state.trades_today += 1
        self.daily_state.last_trade_time = datetime.now(timezone.utc)
        logger.info(f"Trade started: {self.daily_state.trades_today}/{RiskConstants.MAX_TRADES_PER_DAY} today")
    
    def record_trade_result(self, pnl: Decimal, is_win: bool) -> None:
        """
        Record trade result for daily tracking.
        
        Args:
            pnl: Trade P&L in USD
            is_win: True if trade was profitable
        """
        self.daily_state.daily_pnl += pnl
        
        if is_win:
            self.current_streak = max(1, self.current_streak + 1)
        else:
            self.daily_state.losses_today += 1
            self.current_streak = min(-1, self.current_streak - 1)
        
        logger.info(
            f"Trade result: {'WIN' if is_win else 'LOSS'} ${pnl:.2f} | "
            f"Daily: ${self.daily_state.daily_pnl:.2f} | "
            f"Streak: {self.current_streak}"
        )
    
    def check_daily_reset(self) -> bool:
        """
        Check if daily state should be reset (new UTC day).
        
        Returns:
            True if reset was performed
        """
        now = datetime.now(timezone.utc)
        last_trade = self.daily_state.last_trade_time
        
        if last_trade and last_trade.date() < now.date():
            logger.info("New trading day - resetting daily state")
            self.daily_state.reset()
            return True
        
        return False
    
    def get_risk_state_string(self) -> str:
        """Get human-readable risk state for UI/logs."""
        return (
            f"Trades: {self.daily_state.trades_today}/{RiskConstants.MAX_TRADES_PER_DAY} | "
            f"Losses: {self.daily_state.losses_today}/{RiskConstants.MAX_LOSSES_PER_DAY} | "
            f"Daily P&L: ${self.daily_state.daily_pnl:.2f} | "
            f"Streak: {self.current_streak}"
        )
    
    def to_explanation(self, validation: RiskValidation) -> str:
        """Format risk validation for trade explanation."""
        if not validation.is_valid:
            return f"REJECTED: {validation.rejection_reason}"
        
        return (
            f"{validation.risk_pct:.1f}% risk (${validation.risk_usd:.2f}), "
            f"{self.daily_state.trades_today} trades today, "
            f"{self.daily_state.losses_today} losses, "
            f"streak: {'+' if self.current_streak > 0 else ''}{self.current_streak}"
        )
