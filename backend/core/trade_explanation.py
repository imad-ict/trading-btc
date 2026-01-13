"""
Trade Explanation Builder - Every trade must be explainable.

Institutional principle: No explanation = No trade.
Each trade stores WHY it was taken, not just WHAT happened.
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass
class TradeExplanation:
    """
    Complete explanation for a trade decision.
    
    This is a mandatory requirement for institutional compliance.
    Every field must be populated before entry execution.
    """
    # Liquidity Event
    liquidity_event: str = ""
    # Example: "Swept session high at 67,450 with 2.3x volume spike"
    
    # Market State
    market_state: str = ""
    # Example: "EXPANSION - VWAP slope +0.23%, high volume regime"
    
    # Entry Logic
    entry_logic: str = ""
    # Example: "Reclaim candle after sweep, 2.1x volume expansion, VWAP aligned"
    
    # Stop Loss Logic  
    stop_logic: str = ""
    # Example: "Wick midpoint at 67,320 (0.19% from entry), ATR buffer applied"
    
    # Target Logic
    target_logic: str = ""
    # Example: "TP1: VWAP 67,580, TP2: Prior high 67,890, TP3: Runner"
    
    # Risk Validation
    risk_validation: str = ""
    # Example: "0.3% risk ($15.40), 2 trades today, streak: +1"
    
    @property
    def is_complete(self) -> bool:
        """Check if all explanation fields are populated."""
        return all([
            self.liquidity_event,
            self.market_state,
            self.entry_logic,
            self.stop_logic,
            self.target_logic,
            self.risk_validation,
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "liquidity_event": self.liquidity_event,
            "market_state": self.market_state,
            "entry_logic": self.entry_logic,
            "stop_logic": self.stop_logic,
            "target_logic": self.target_logic,
            "risk_validation": self.risk_validation,
        }
    
    def format_log(self) -> str:
        """Format explanation for logging."""
        return f"""
══════════════════════════════════════════════════════════════
  TRADE EXPLANATION
══════════════════════════════════════════════════════════════
  LIQUIDITY EVENT : {self.liquidity_event}
  MARKET STATE    : {self.market_state}
  ENTRY LOGIC     : {self.entry_logic}
  STOP LOGIC      : {self.stop_logic}
  TARGET LOGIC    : {self.target_logic}
  RISK VALIDATION : {self.risk_validation}
══════════════════════════════════════════════════════════════
"""


class TradeExplanationBuilder:
    """
    Builder for constructing trade explanations incrementally.
    
    Usage:
        builder = TradeExplanationBuilder()
        builder.set_liquidity_event("Swept equal highs at 67,450")
        builder.set_market_state("EXPANSION", vwap_slope=0.0023, volume_regime="HIGH")
        builder.set_entry_logic(...)
        
        if builder.is_valid():
            explanation = builder.build()
    """
    
    def __init__(self):
        self._explanation = TradeExplanation()
        self._timestamp = datetime.utcnow()
    
    def set_liquidity_event(
        self,
        event_type: str,
        level_price: Decimal,
        volume_multiple: Optional[float] = None,
    ) -> "TradeExplanationBuilder":
        """
        Set the liquidity event that triggered the trade.
        
        Args:
            event_type: Type of liquidity taken (e.g., "Swept session high")
            level_price: Price level that was swept
            volume_multiple: Volume spike multiplier
        """
        volume_str = f" with {volume_multiple:.1f}x volume" if volume_multiple else ""
        self._explanation.liquidity_event = f"{event_type} at {level_price}{volume_str}"
        return self
    
    def set_market_state(
        self,
        state: str,
        vwap_slope: Optional[float] = None,
        volume_regime: Optional[str] = None,
        candle_efficiency: Optional[float] = None,
    ) -> "TradeExplanationBuilder":
        """
        Set the market state classification.
        
        Args:
            state: EXPANSION, MANIPULATION, or NO_TRADE
            vwap_slope: VWAP slope percentage
            volume_regime: HIGH, NORMAL, or LOW
            candle_efficiency: Body/range ratio
        """
        parts = [state]
        
        if vwap_slope is not None:
            sign = "+" if vwap_slope >= 0 else ""
            parts.append(f"VWAP slope {sign}{vwap_slope*100:.2f}%")
        
        if volume_regime:
            parts.append(f"{volume_regime.lower()} volume")
        
        if candle_efficiency is not None:
            parts.append(f"efficiency {candle_efficiency:.0%}")
        
        self._explanation.market_state = " - ".join(parts)
        return self
    
    def set_entry_logic(
        self,
        reclaim_confirmed: bool,
        volume_expansion: float,
        vwap_aligned: bool,
        spread_ok: bool,
        delay_applied: bool = False,
    ) -> "TradeExplanationBuilder":
        """
        Set the entry logic explanation.
        
        Args:
            reclaim_confirmed: Whether reclaim candle was confirmed
            volume_expansion: Volume expansion multiplier
            vwap_aligned: Whether price aligned with VWAP
            spread_ok: Whether spread was acceptable
            delay_applied: Whether entry delay was used
        """
        parts = []
        
        if reclaim_confirmed:
            parts.append("Reclaim confirmed")
        
        parts.append(f"{volume_expansion:.1f}x volume")
        
        if vwap_aligned:
            parts.append("VWAP aligned")
        
        if spread_ok:
            parts.append("spread OK")
        
        if delay_applied:
            parts.append("1-candle delay applied")
        
        self._explanation.entry_logic = ", ".join(parts)
        return self
    
    def set_stop_logic(
        self,
        sl_type: str,
        sl_price: Decimal,
        sl_distance_pct: float,
        atr_buffer: Optional[Decimal] = None,
    ) -> "TradeExplanationBuilder":
        """
        Set the stop loss logic explanation.
        
        Args:
            sl_type: Type of SL (e.g., "Wick midpoint", "Structure")
            sl_price: Stop loss price
            sl_distance_pct: Distance from entry as percentage
            atr_buffer: ATR buffer applied
        """
        buffer_str = f", ATR buffer {atr_buffer}" if atr_buffer else ""
        self._explanation.stop_logic = f"{sl_type} at {sl_price} ({sl_distance_pct:.2f}% from entry{buffer_str})"
        return self
    
    def set_target_logic(
        self,
        tp1_target: str,
        tp1_price: Decimal,
        tp2_target: Optional[str] = None,
        tp2_price: Optional[Decimal] = None,
        tp3_target: Optional[str] = None,
    ) -> "TradeExplanationBuilder":
        """
        Set the take profit logic explanation.
        """
        parts = [f"TP1: {tp1_target} {tp1_price}"]
        
        if tp2_target and tp2_price:
            parts.append(f"TP2: {tp2_target} {tp2_price}")
        
        if tp3_target:
            parts.append(f"TP3: {tp3_target}")
        
        self._explanation.target_logic = ", ".join(parts)
        return self
    
    def set_risk_validation(
        self,
        risk_pct: float,
        risk_usd: Decimal,
        trades_today: int,
        losses_today: int,
        streak: int,
    ) -> "TradeExplanationBuilder":
        """
        Set the risk validation explanation.
        """
        streak_str = f"+{streak}" if streak >= 0 else str(streak)
        self._explanation.risk_validation = (
            f"{risk_pct:.1%} risk (${risk_usd:.2f}), "
            f"{trades_today} trades today, "
            f"{losses_today} losses, "
            f"streak: {streak_str}"
        )
        return self
    
    def is_valid(self) -> bool:
        """Check if the explanation is complete."""
        return self._explanation.is_complete
    
    def build(self) -> TradeExplanation:
        """
        Build and return the trade explanation.
        
        Raises:
            ValueError: If explanation is incomplete
        """
        if not self.is_valid():
            missing = []
            if not self._explanation.liquidity_event:
                missing.append("liquidity_event")
            if not self._explanation.market_state:
                missing.append("market_state")
            if not self._explanation.entry_logic:
                missing.append("entry_logic")
            if not self._explanation.stop_logic:
                missing.append("stop_logic")
            if not self._explanation.target_logic:
                missing.append("target_logic")
            if not self._explanation.risk_validation:
                missing.append("risk_validation")
            
            raise ValueError(f"Incomplete explanation. Missing: {', '.join(missing)}")
        
        return self._explanation
    
    def reset(self) -> None:
        """Reset the builder for a new trade."""
        self._explanation = TradeExplanation()
        self._timestamp = datetime.utcnow()
