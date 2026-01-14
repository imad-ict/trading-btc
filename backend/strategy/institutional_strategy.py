"""
FAILURE-BASED LIQUIDITY REVERSAL STRATEGY v5

🚫 THIS IS THE ANTI-RETAIL VERSION 🚫

Core Philosophy:
- The first move is usually a trap
- Pullbacks collect liquidity (don't trade them)
- Breakouts inside balance are fake
- Expansion happens AFTER failure

OPPOSITE BEHAVIOR:
- Old system enters on pullback → New system WAITS
- Old system enters on momentum → New system SKIPS
- Old system protects early (BE) → New system gives room
- Old system trades anywhere → New system requires BALANCE + FAILURE

Entry Hierarchy:
1. REQUIRE balance (VWAP flat, overlap > 65%)
2. REQUIRE failure (push attempt that stalls)
3. REQUIRE sweep (liquidity taken)
4. REQUIRE reclaim (confirmation candle)
5. DELAY 1 candle minimum
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market state classification - BALANCE is tradable, TREND is not."""
    BALANCE = "balance"      # Flat VWAP, overlap > 65% → TRADABLE
    TRENDING = "trending"    # Clear direction → DO NOT TRADE
    UNKNOWN = "unknown"      # Insufficient data → DO NOT TRADE


class FailureType(Enum):
    """Type of failure detected."""
    FAILED_PUSH_UP = "failed_push_up"    # Bulls failed → SHORT setup
    FAILED_PUSH_DOWN = "failed_push_down"  # Bears failed → LONG setup
    NO_FAILURE = "no_failure"


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def body_pct(self) -> float:
        if self.total_range == 0:
            return 0
        return self.body_size / self.total_range
    
    @property
    def is_stalling(self) -> bool:
        """Stalling = small body, wicks on both sides."""
        return self.body_pct < 0.4 and self.upper_wick > 0 and self.lower_wick > 0


@dataclass
class FailedPush:
    """Detected failed push (failure to continue)."""
    direction: str  # "up" or "down"
    start_price: float
    failure_price: float
    timestamp: float
    candle_count: int


@dataclass
class SweepEvent:
    """Liquidity sweep event with reclaim status."""
    level: float
    sweep_type: str  # "low" or "high"
    sweep_candle: Candle
    reclaimed: bool = False
    reclaim_candle: Optional[Candle] = None
    confirmation_pending: bool = True  # Wait 1 candle after reclaim


@dataclass
class TradeSignal:
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    tp1: float  # Opposite side of balance
    tp2: float  # Liquidity pool
    tp3: float  # Expansion target
    failure_type: FailureType
    reason: str
    why_old_system_would_lose: str
    
    @property
    def sl_distance_pct(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price
    
    @property
    def is_valid(self) -> bool:
        # SL must survive noise: 0.2% to 0.4%
        return 0.002 <= self.sl_distance_pct <= 0.004


class FailureReversalStrategy:
    """
    FAILURE-BASED LIQUIDITY REVERSAL
    
    This strategy does the OPPOSITE of retail:
    1. WAIT for balance (flat market)
    2. WAIT for failure (push that stalls)
    3. WAIT for sweep (liquidity taken)
    4. WAIT for reclaim (confirmation)
    5. DELAY 1 candle (avoid trap)
    
    NO MOMENTUM. NO PULLBACKS. NO EARLY ENTRIES.
    """
    
    def __init__(self, max_positions: int = 3):
        # Candle storage
        self.candles_1m: Dict[str, deque] = {}
        self.candles_5m: Dict[str, deque] = {}
        self.candles_15m: Dict[str, deque] = {}
        
        # Market state
        self.market_state: Dict[str, MarketState] = {}
        self.vwap: Dict[str, float] = {}
        self.vwap_data: Dict[str, List[Tuple[float, float]]] = {}
        
        # Failure tracking
        self.failed_pushes: Dict[str, List[FailedPush]] = {}
        self.pending_sweeps: Dict[str, List[SweepEvent]] = {}
        
        # Balance range
        self.balance_high: Dict[str, float] = {}
        self.balance_low: Dict[str, float] = {}
        
        # Configuration
        self.max_positions = max_positions
        self.max_candles = 100
        self.balance_overlap_threshold = 0.65  # 65% overlap = balance
        self.vwap_flat_threshold = 0.001  # 0.1% = flat VWAP
    
    def get_diagnostics(self, symbol: str) -> Dict:
        return {
            "symbol": symbol,
            "state": self.market_state.get(symbol, MarketState.UNKNOWN).value,
            "vwap": self.vwap.get(symbol),
            "balance_high": self.balance_high.get(symbol),
            "balance_low": self.balance_low.get(symbol),
            "pending_sweeps": len(self.pending_sweeps.get(symbol, [])),
            "failed_pushes": len(self.failed_pushes.get(symbol, [])),
        }
    
    def add_candle(self, symbol: str, candle: Candle, timeframe: str = "1m"):
        """Add candle and update state."""
        storage_map = {
            "1m": self.candles_1m,
            "5m": self.candles_5m,
            "15m": self.candles_15m
        }
        
        storage = storage_map.get(timeframe, self.candles_1m)
        
        if symbol not in storage:
            storage[symbol] = deque(maxlen=self.max_candles)
        
        storage[symbol].append(candle)
        
        if timeframe == "1m":
            self._update_vwap(symbol, candle)
        
        if timeframe == "5m":
            self._update_market_state(symbol)
            self._detect_failures(symbol)
            self._detect_sweeps(symbol, candle)
            self._check_reclaims(symbol, candle)
    
    def _update_vwap(self, symbol: str, candle: Candle):
        """Update VWAP."""
        if symbol not in self.vwap_data:
            self.vwap_data[symbol] = []
        
        typical_price = (candle.high + candle.low + candle.close) / 3
        self.vwap_data[symbol].append((typical_price, candle.volume))
        
        if len(self.vwap_data[symbol]) > 500:
            self.vwap_data[symbol] = self.vwap_data[symbol][-500:]
        
        total_pv = sum(p * v for p, v in self.vwap_data[symbol])
        total_v = sum(v for _, v in self.vwap_data[symbol])
        
        if total_v > 0:
            self.vwap[symbol] = total_pv / total_v
    
    def _update_market_state(self, symbol: str):
        """
        Detect BALANCE vs TRENDING.
        
        BALANCE (tradable):
        - VWAP slope is FLAT (< 0.1%)
        - Candle overlap > 65%
        - Multiple failed pushes
        
        TRENDING (do not trade):
        - Clear direction
        - VWAP slope significant
        """
        candles = list(self.candles_5m.get(symbol, []))
        if len(candles) < 10:
            self.market_state[symbol] = MarketState.UNKNOWN
            return
        
        recent = candles[-10:]
        
        # Calculate overlap percentage
        overlapping = 0
        for i in range(1, len(recent)):
            prev = (recent[i-1].low, recent[i-1].high)
            curr = (recent[i].low, recent[i].high)
            overlap = max(0, min(prev[1], curr[1]) - max(prev[0], curr[0]))
            range_size = max(prev[1] - prev[0], curr[1] - curr[0])
            if range_size > 0 and overlap / range_size > 0.5:
                overlapping += 1
        
        overlap_pct = overlapping / (len(recent) - 1)
        
        # Calculate VWAP slope
        vwap_start = (recent[0].high + recent[0].low + recent[0].close) / 3
        vwap_end = (recent[-1].high + recent[-1].low + recent[-1].close) / 3
        vwap_change = abs(vwap_end - vwap_start) / vwap_start if vwap_start > 0 else 0
        
        # BALANCE: flat VWAP + high overlap
        if overlap_pct >= self.balance_overlap_threshold and vwap_change < self.vwap_flat_threshold:
            self.market_state[symbol] = MarketState.BALANCE
            # Update balance range
            self.balance_high[symbol] = max(c.high for c in recent)
            self.balance_low[symbol] = min(c.low for c in recent)
        else:
            self.market_state[symbol] = MarketState.TRENDING
    
    def _detect_failures(self, symbol: str):
        """
        Detect FAILED PUSHES - the heart of this strategy.
        
        FAILED PUSH UP: Attempt to hold above VWAP that stalls
        FAILED PUSH DOWN: Attempt to hold below VWAP that stalls
        """
        if self.market_state.get(symbol) != MarketState.BALANCE:
            return  # Only detect failures in BALANCE
        
        candles = list(self.candles_5m.get(symbol, []))
        if len(candles) < 5:
            return
        
        if symbol not in self.failed_pushes:
            self.failed_pushes[symbol] = []
        
        vwap = self.vwap.get(symbol)
        if not vwap:
            return
        
        recent = candles[-5:]
        
        # Check for FAILED PUSH UP (bulls fail → SHORT setup coming)
        above_vwap = [c for c in recent if c.close > vwap]
        if len(above_vwap) >= 2:
            # Check if stalling (last candles have small bodies)
            stalling = sum(1 for c in recent[-3:] if c.is_stalling)
            if stalling >= 2:
                failure = FailedPush(
                    direction="up",
                    start_price=above_vwap[0].close,
                    failure_price=recent[-1].close,
                    timestamp=recent[-1].timestamp,
                    candle_count=len(above_vwap)
                )
                self.failed_pushes[symbol].append(failure)
        
        # Check for FAILED PUSH DOWN (bears fail → LONG setup coming)
        below_vwap = [c for c in recent if c.close < vwap]
        if len(below_vwap) >= 2:
            stalling = sum(1 for c in recent[-3:] if c.is_stalling)
            if stalling >= 2:
                failure = FailedPush(
                    direction="down",
                    start_price=below_vwap[0].close,
                    failure_price=recent[-1].close,
                    timestamp=recent[-1].timestamp,
                    candle_count=len(below_vwap)
                )
                self.failed_pushes[symbol].append(failure)
        
        # Keep only recent failures
        self.failed_pushes[symbol] = self.failed_pushes[symbol][-5:]
    
    def _detect_sweeps(self, symbol: str, candle: Candle):
        """Detect liquidity sweeps at balance extremes."""
        if self.market_state.get(symbol) != MarketState.BALANCE:
            return
        
        if symbol not in self.pending_sweeps:
            self.pending_sweeps[symbol] = []
        
        balance_high = self.balance_high.get(symbol)
        balance_low = self.balance_low.get(symbol)
        
        if not balance_high or not balance_low:
            return
        
        tolerance = candle.close * 0.001  # 0.1%
        
        # SWEEP LOW (bullish setup forming)
        if candle.low < balance_low - tolerance:
            # Check for rejection wick (lower wick > body)
            if candle.lower_wick > candle.body_size:
                sweep = SweepEvent(
                    level=balance_low,
                    sweep_type="low",
                    sweep_candle=candle
                )
                self.pending_sweeps[symbol].append(sweep)
        
        # SWEEP HIGH (bearish setup forming)
        if candle.high > balance_high + tolerance:
            if candle.upper_wick > candle.body_size:
                sweep = SweepEvent(
                    level=balance_high,
                    sweep_type="high",
                    sweep_candle=candle
                )
                self.pending_sweeps[symbol].append(sweep)
        
        # Clean old sweeps
        self.pending_sweeps[symbol] = self.pending_sweeps[symbol][-5:]
    
    def _check_reclaims(self, symbol: str, candle: Candle):
        """Check if pending sweeps have been reclaimed."""
        for sweep in self.pending_sweeps.get(symbol, []):
            if sweep.reclaimed:
                # Already reclaimed, check if confirmation period passed
                if sweep.confirmation_pending:
                    sweep.confirmation_pending = False  # 1 candle delay done
                continue
            
            # Check for reclaim
            if sweep.sweep_type == "low":
                # LONG: Close back ABOVE the swept level
                if candle.close > sweep.level and candle.is_bullish:
                    sweep.reclaimed = True
                    sweep.reclaim_candle = candle
            
            elif sweep.sweep_type == "high":
                # SHORT: Close back BELOW the swept level
                if candle.close < sweep.level and not candle.is_bullish:
                    sweep.reclaimed = True
                    sweep.reclaim_candle = candle
    
    def generate_signal(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """
        Generate signal using FAILURE-BASED logic.
        
        Requirements (ALL must be true):
        1. Market is in BALANCE
        2. A failure has been detected
        3. Liquidity has been swept
        4. Sweep has been reclaimed
        5. 1 candle delay has passed (confirmation_pending = False)
        6. DO NOT enter on momentum candle
        """
        # RULE 1: Must be BALANCE
        state = self.market_state.get(symbol, MarketState.UNKNOWN)
        if state != MarketState.BALANCE:
            return None
        
        # RULE 2: Check for failures
        failures = self.failed_pushes.get(symbol, [])
        if not failures:
            return None
        
        # RULE 3 & 4 & 5: Check for confirmed sweeps
        for sweep in self.pending_sweeps.get(symbol, []):
            if not sweep.reclaimed:
                continue
            if sweep.confirmation_pending:
                continue  # Wait for 1 candle delay
            
            # RULE 6: Do NOT enter on momentum candle
            if current_candle.body_pct > 0.7:
                continue  # Skip strong candles (momentum)
            
            # Find matching failure
            if sweep.sweep_type == "low":
                # Looking for FAILED_PUSH_DOWN → LONG
                matching_failure = next(
                    (f for f in failures if f.direction == "down"), 
                    None
                )
                if matching_failure:
                    return self._create_long_signal(symbol, sweep, matching_failure, current_candle)
            
            elif sweep.sweep_type == "high":
                # Looking for FAILED_PUSH_UP → SHORT
                matching_failure = next(
                    (f for f in failures if f.direction == "up"),
                    None
                )
                if matching_failure:
                    return self._create_short_signal(symbol, sweep, matching_failure, current_candle)
        
        return None
    
    def _create_long_signal(self, symbol: str, sweep: SweepEvent, 
                            failure: FailedPush, candle: Candle) -> Optional[TradeSignal]:
        """Create LONG signal after bearish failure."""
        vwap = self.vwap.get(symbol, candle.close)
        balance_high = self.balance_high.get(symbol, candle.close + candle.total_range * 3)
        
        # SL: Midpoint of sweep wick (NOT at the low)
        sweep_low = sweep.sweep_candle.low
        wick_midpoint = (sweep_low + sweep.level) / 2
        sl = wick_midpoint * 0.999  # Tiny buffer
        
        # TP: Destination-based
        tp1 = vwap  # Opposite side of balance
        tp2 = balance_high  # Balance high (liquidity)
        tp3 = balance_high + (balance_high - sweep_low)  # Expansion
        
        why_old_loses = f"Old system would enter on VWAP pullback BEFORE failure. Price was at ${failure.start_price:,.0f}, pushed down, and old system would buy the pullback. Instead, we waited for failure confirmation."
        
        signal = TradeSignal(
            symbol=symbol,
            direction=TradeDirection.LONG,
            entry_price=candle.close,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            failure_type=FailureType.FAILED_PUSH_DOWN,
            reason=f"Bears failed at ${failure.failure_price:,.0f} | Sweep @ ${sweep.level:,.0f} reclaimed",
            why_old_system_would_lose=why_old_loses
        )
        
        if signal.is_valid:
            # Clear used sweep and failure
            self.pending_sweeps[symbol] = [s for s in self.pending_sweeps[symbol] if s != sweep]
            self.failed_pushes[symbol] = [f for f in self.failed_pushes[symbol] if f != failure]
            return signal
        return None
    
    def _create_short_signal(self, symbol: str, sweep: SweepEvent,
                             failure: FailedPush, candle: Candle) -> Optional[TradeSignal]:
        """Create SHORT signal after bullish failure."""
        vwap = self.vwap.get(symbol, candle.close)
        balance_low = self.balance_low.get(symbol, candle.close - candle.total_range * 3)
        
        # SL: Midpoint of sweep wick (NOT at the high)
        sweep_high = sweep.sweep_candle.high
        wick_midpoint = (sweep_high + sweep.level) / 2
        sl = wick_midpoint * 1.001  # Tiny buffer
        
        # TP: Destination-based
        tp1 = vwap
        tp2 = balance_low
        tp3 = balance_low - (sweep_high - balance_low)  # Expansion
        
        why_old_loses = f"Old system would enter SHORT on VWAP rejection BEFORE failure. Price was at ${failure.start_price:,.0f}, pushed up, and old system would sell. Instead, we waited for confirmation."
        
        signal = TradeSignal(
            symbol=symbol,
            direction=TradeDirection.SHORT,
            entry_price=candle.close,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            failure_type=FailureType.FAILED_PUSH_UP,
            reason=f"Bulls failed at ${failure.failure_price:,.0f} | Sweep @ ${sweep.level:,.0f} reclaimed",
            why_old_system_would_lose=why_old_loses
        )
        
        if signal.is_valid:
            self.pending_sweeps[symbol] = [s for s in self.pending_sweeps[symbol] if s != sweep]
            self.failed_pushes[symbol] = [f for f in self.failed_pushes[symbol] if f != failure]
            return signal
        return None


@dataclass
class ActivePosition:
    """
    Position with ANTI-RETAIL SL management.
    
    SL moves ONLY when structure changes:
    - LONG: New Higher Low
    - SHORT: New Lower High
    
    NO early breakeven. NO mechanical trailing.
    """
    signal: TradeSignal
    order_id: str
    quantity: float
    entry_time: datetime
    
    current_sl: float = field(init=False)
    tp1_hit: bool = False
    tp2_hit: bool = False
    remaining_qty: float = field(init=False)
    
    def __post_init__(self):
        self.current_sl = self.signal.stop_loss
        self.remaining_qty = self.quantity
    
    def check_structure_for_sl_move(self, candles: List[Candle]) -> bool:
        """
        Move SL ONLY on structural change.
        
        LONG: Higher Low formed
        SHORT: Lower High formed
        """
        if len(candles) < 5:
            return False
        
        if self.signal.direction == TradeDirection.LONG:
            # Find Higher Low
            lows = [c.low for c in candles[-5:]]
            for i in range(1, len(lows) - 1):
                # Check if middle is lower than neighbors (swing low)
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    if lows[i] > self.current_sl:
                        new_sl = lows[i] * 0.998
                        if new_sl > self.current_sl:
                            self.current_sl = new_sl
                            return True
        
        else:  # SHORT
            highs = [c.high for c in candles[-5:]]
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    if highs[i] < self.current_sl:
                        new_sl = highs[i] * 1.002
                        if new_sl < self.current_sl:
                            self.current_sl = new_sl
                            return True
        
        return False
    
    def get_partial_qty_tp1(self) -> float:
        return self.quantity * 0.4
    
    def get_partial_qty_tp2(self) -> float:
        return self.quantity * 0.4
    
    def should_exit(self, current_price: float) -> Tuple[bool, str]:
        if self.signal.direction == TradeDirection.LONG:
            if current_price <= self.current_sl:
                return True, "STOP_LOSS"
        else:
            if current_price >= self.current_sl:
                return True, "STOP_LOSS"
        return False, ""
    
    def check_tp_levels(self, current_price: float) -> Optional[str]:
        if self.signal.direction == TradeDirection.LONG:
            if not self.tp1_hit and current_price >= self.signal.tp1:
                return "TP1"
            if not self.tp2_hit and current_price >= self.signal.tp2:
                return "TP2"
            if current_price >= self.signal.tp3:
                return "TP3"
        else:
            if not self.tp1_hit and current_price <= self.signal.tp1:
                return "TP1"
            if not self.tp2_hit and current_price <= self.signal.tp2:
                return "TP2"
            if current_price <= self.signal.tp3:
                return "TP3"
        return None
