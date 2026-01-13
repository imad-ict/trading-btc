"""
Entry Engine (1M) - Failure-based entry logic.

Principle: First move is often a trap. Trade FAILED breakouts, not breakouts.

Entry Conditions (ALL must be true):
1. Liquidity sweep detected
2. Reclaim candle confirmed  
3. Volume expansion
4. VWAP alignment
5. Spread acceptable
6. Entry delay (anti-algo)
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from config import RiskConstants, TradingConstants
from core.market_data_engine import MarketDataEngine
from strategy.liquidity_engine import LiquidityEngine, SweepEvent

logger = logging.getLogger(__name__)


@dataclass
class EntrySignal:
    """A validated entry signal ready for execution."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: Decimal
    stop_loss: Decimal
    
    # Context from liquidity
    sweep_event: SweepEvent
    
    # Validation scores
    volume_multiple: float
    vwap_aligned: bool
    spread_pct: float
    
    # Timing
    signal_time: datetime
    delay_candles: int = 0
    
    @property
    def is_ready_to_execute(self) -> bool:
        """Check if delay period has passed."""
        return self.delay_candles <= 0


@dataclass
class EntryValidation:
    """Result of entry validation."""
    is_valid: bool
    rejection_reason: Optional[str] = None
    
    # Component checks
    liquidity_check: bool = False
    reclaim_check: bool = False
    volume_check: bool = False
    vwap_check: bool = False
    spread_check: bool = False


class EntryEngine:
    """
    1M Entry logic with failure-based confirmation.
    
    Core principle: NO breakout entries. Only sweep + reclaim confirmations.
    """
    
    def __init__(
        self,
        market_data: MarketDataEngine,
        liquidity_engine: LiquidityEngine,
    ):
        """
        Initialize entry engine.
        
        Args:
            market_data: MarketDataEngine instance
            liquidity_engine: LiquidityEngine instance
        """
        self.market_data = market_data
        self.liquidity = liquidity_engine
        
        # Pending signal (waiting for delay)
        self._pending_signal: Optional[EntrySignal] = None
        self._delay_counter: int = 0
    
    def validate_entry(self) -> EntryValidation:
        """
        Validate if entry conditions are met.
        
        Returns:
            EntryValidation with component check results
        """
        validation = EntryValidation(is_valid=False)
        
        # 1. LIQUIDITY SWEEP DETECTED?
        if not self.liquidity.is_liquidity_taken():
            validation.rejection_reason = "No liquidity sweep detected"
            return validation
        validation.liquidity_check = True
        
        sweep = self.liquidity.get_pending_sweep()
        if not sweep:
            validation.rejection_reason = "No pending sweep event"
            return validation
        
        # 2. RECLAIM CANDLE CONFIRMED?
        if not self._has_reclaim_candle(sweep):
            validation.rejection_reason = "Waiting for reclaim candle"
            return validation
        validation.reclaim_check = True
        
        # 3. VOLUME EXPANSION?
        volume_regime = self.market_data.get_volume_regime("1m")
        if not volume_regime or volume_regime.volume_ratio < 1.2:
            validation.rejection_reason = f"Insufficient volume: {volume_regime.volume_ratio:.1f}x"
            return validation
        validation.volume_check = True
        
        # 4. VWAP ALIGNMENT?
        if not self._vwap_aligned(sweep):
            validation.rejection_reason = "VWAP not aligned with direction"
            return validation
        validation.vwap_check = True
        
        # 5. SPREAD ACCEPTABLE?
        spread_info = self.market_data.get_spread_info()
        if not spread_info or not spread_info.is_acceptable:
            spread_pct = spread_info.spread_pct if spread_info else 0
            validation.rejection_reason = f"Spread too wide: {spread_pct:.3f}%"
            return validation
        validation.spread_check = True
        
        # All checks passed
        validation.is_valid = True
        return validation
    
    def _has_reclaim_candle(self, sweep: SweepEvent) -> bool:
        """
        Check for reclaim candle after sweep.
        
        Reclaim = price returns back through the swept level.
        """
        if sweep.reclaimed:
            return True
        
        candles = self.market_data.candles_1m.get_last_n(5)
        if len(candles) < 2:
            return False
        
        last_candle = candles[-1]
        
        # Determine reclaim direction based on sweep type
        is_high_sweep = "HIGH" in sweep.zone.type.value
        is_low_sweep = "LOW" in sweep.zone.type.value
        
        if is_high_sweep:
            # For high sweep: reclaim means close below the level
            # This suggests SHORT after failed breakout above
            if float(last_candle.close) < float(sweep.zone.price):
                if last_candle.is_closed:
                    self.liquidity.confirm_reclaim(sweep, last_candle.close)
                    return True
        
        elif is_low_sweep:
            # For low sweep: reclaim means close above the level
            # This suggests LONG after failed breakdown below
            if float(last_candle.close) > float(sweep.zone.price):
                if last_candle.is_closed:
                    self.liquidity.confirm_reclaim(sweep, last_candle.close)
                    return True
        
        return False
    
    def _vwap_aligned(self, sweep: SweepEvent) -> bool:
        """
        Check if VWAP aligns with trade direction.
        
        For LONG: price should be at/below VWAP (discount)
        For SHORT: price should be at/above VWAP (premium)
        """
        vwap = self.market_data.get_vwap()
        if not vwap:
            return True  # Allow if no VWAP data yet
        
        current_price = self.market_data.current_price
        if not current_price:
            return False
        
        is_high_sweep = "HIGH" in sweep.zone.type.value
        
        if is_high_sweep:
            # SHORT setup: should be at premium (above VWAP)
            return float(current_price) >= float(vwap.lower_band)
        else:
            # LONG setup: should be at discount (below VWAP)
            return float(current_price) <= float(vwap.upper_band)
    
    def generate_signal(self, sweep: SweepEvent) -> Optional[EntrySignal]:
        """
        Generate entry signal from confirmed sweep.
        
        Args:
            sweep: Confirmed sweep event with reclaim
            
        Returns:
            EntrySignal ready for risk validation, or None if invalid
        """
        current_price = self.market_data.current_price
        if not current_price:
            return None
        
        # Determine direction from sweep type
        is_high_sweep = "HIGH" in sweep.zone.type.value
        direction = "SHORT" if is_high_sweep else "LONG"
        
        # Get volume info
        volume_regime = self.market_data.get_volume_regime("1m")
        volume_multiple = volume_regime.volume_ratio if volume_regime else 1.0
        
        # Get spread info
        spread_info = self.market_data.get_spread_info()
        spread_pct = spread_info.spread_pct if spread_info else 0
        
        # Check VWAP
        vwap = self.market_data.get_vwap()
        vwap_aligned = self._vwap_aligned(sweep)
        
        signal = EntrySignal(
            symbol=self.market_data.symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=Decimal("0"),  # To be set by SL engine
            sweep_event=sweep,
            volume_multiple=volume_multiple,
            vwap_aligned=vwap_aligned,
            spread_pct=spread_pct,
            signal_time=datetime.now(timezone.utc),
            delay_candles=TradingConstants.ENTRY_DELAY_CANDLES,
        )
        
        logger.info(
            f"ENTRY SIGNAL: {direction} {self.market_data.symbol} @ {current_price} "
            f"(sweep: {sweep.zone.type.value}, vol: {volume_multiple:.1f}x)"
        )
        
        return signal
    
    def queue_signal(self, signal: EntrySignal) -> None:
        """Queue signal for delayed execution (anti-algo)."""
        self._pending_signal = signal
        self._delay_counter = signal.delay_candles
        logger.info(f"Signal queued with {self._delay_counter} candle delay")
    
    def on_candle_close(self) -> Optional[EntrySignal]:
        """
        Process candle close for pending signals.
        
        Returns:
            EntrySignal if delay complete and ready to execute
        """
        if not self._pending_signal:
            return None
        
        self._delay_counter -= 1
        self._pending_signal.delay_candles = self._delay_counter
        
        if self._delay_counter <= 0:
            # Re-validate before releasing
            validation = self.validate_entry()
            if validation.is_valid:
                signal = self._pending_signal
                self._pending_signal = None
                logger.info("Signal released after delay")
                return signal
            else:
                logger.warning(f"Signal invalidated after delay: {validation.rejection_reason}")
                self._pending_signal = None
        
        return None
    
    def clear_pending(self) -> None:
        """Clear any pending signals."""
        self._pending_signal = None
        self._delay_counter = 0
    
    @property
    def has_pending_signal(self) -> bool:
        """Check if there's a pending signal."""
        return self._pending_signal is not None
