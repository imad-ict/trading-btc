"""
Multi-Position Institutional Strategy

Features:
1. Multi-position support (one per symbol)
2. Multiple entry types: Liquidity Sweep, CHoCH, FVG
3. Multi-timeframe analysis: 15M context, 5M structure, 1M entry
4. Independent SL management per position
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
    ACCUMULATION = "accumulation"
    MANIPULATION = "manipulation"  
    EXPANSION = "expansion"
    DISTRIBUTION = "distribution"
    UNKNOWN = "unknown"


class EntryType(Enum):
    LIQUIDITY_SWEEP = "liquidity_sweep"
    CHOCH = "choch"  # Change of Character
    FVG = "fvg"  # Fair Value Gap
    VWAP_BOUNCE = "vwap_bounce"
    MOMENTUM = "momentum"


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
    def mid_price(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class LiquidityLevel:
    price: float
    level_type: str  # "equal_high" or "equal_low"
    touches: int
    first_touch: float
    last_touch: float
    swept: bool = False
    sweep_time: Optional[float] = None


@dataclass  
class FairValueGap:
    """Fair Value Gap (imbalance zone)"""
    high: float  # Top of gap
    low: float   # Bottom of gap
    direction: str  # "bullish" or "bearish"
    timestamp: float
    filled: bool = False


@dataclass
class TradeSignal:
    symbol: str
    direction: TradeDirection
    entry_type: EntryType
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    liquidity_level: Optional[float]
    sweep_candle_low: float
    sweep_candle_high: float
    vwap: float
    reason: str
    market_state: MarketState
    
    @property
    def risk_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price
    
    @property
    def is_valid(self) -> bool:
        # Valid risk: 0.05% to 1%
        return 0.0005 <= self.risk_distance <= 0.01


class InstitutionalStrategy:
    """
    Multi-Position Institutional Strategy
    
    Entry Types (Priority):
    1. Liquidity Sweep + Reclaim
    2. Change of Character (CHoCH)
    3. Fair Value Gap (FVG) Fill
    4. VWAP Bounce
    """
    
    def __init__(self, max_positions: int = 3):
        # Candle storage
        self.candles_1m: Dict[str, deque] = {}
        self.candles_5m: Dict[str, deque] = {}
        self.candles_15m: Dict[str, deque] = {}
        
        self.liquidity_levels: Dict[str, List[LiquidityLevel]] = {}
        self.fair_value_gaps: Dict[str, List[FairValueGap]] = {}
        
        # Market structure tracking
        self.swing_highs: Dict[str, List[float]] = {}
        self.swing_lows: Dict[str, List[float]] = {}
        self.last_structure_break: Dict[str, Optional[str]] = {}  # "bullish" or "bearish"
        
        self.session_high: Dict[str, float] = {}
        self.session_low: Dict[str, float] = {}
        
        self.vwap: Dict[str, float] = {}
        self.vwap_data: Dict[str, List[Tuple[float, float]]] = {}
        
        # Configuration
        self.max_positions = max_positions
        self.equal_level_tolerance = 0.0008
        self.min_level_touches = 1
        self.sl_buffer_pct = 0.001
        self.max_candles_1m = 120
        self.max_candles_5m = 50
        self.max_candles_15m = 30
    
    def get_diagnostics(self, symbol: str) -> Dict:
        """Get diagnostic info for a symbol."""
        return {
            "symbol": symbol,
            "candles_1m": len(self.candles_1m.get(symbol, [])),
            "candles_5m": len(self.candles_5m.get(symbol, [])),
            "candles_15m": len(self.candles_15m.get(symbol, [])),
            "liquidity_levels": [
                {"price": l.price, "type": l.level_type, "touches": l.touches, "swept": l.swept}
                for l in self.liquidity_levels.get(symbol, [])[:5]
            ],
            "fvg_count": len([f for f in self.fair_value_gaps.get(symbol, []) if not f.filled]),
            "vwap": self.vwap.get(symbol),
            "market_state": self.get_market_state(symbol).value,
            "last_structure": self.last_structure_break.get(symbol, "none"),
        }
    
    def add_candle(self, symbol: str, candle: Candle, timeframe: str = "1m"):
        """Add a candle and update all derived data."""
        storage_map = {
            "1m": (self.candles_1m, self.max_candles_1m),
            "5m": (self.candles_5m, self.max_candles_5m),
            "15m": (self.candles_15m, self.max_candles_15m)
        }
        
        storage, max_len = storage_map.get(timeframe, (self.candles_1m, self.max_candles_1m))
        
        if symbol not in storage:
            storage[symbol] = deque(maxlen=max_len)
        
        storage[symbol].append(candle)
        
        # Update session extremes
        if symbol not in self.session_high or candle.high > self.session_high[symbol]:
            self.session_high[symbol] = candle.high
        if symbol not in self.session_low or candle.low < self.session_low[symbol]:
            self.session_low[symbol] = candle.low
        
        # Update VWAP (1m data)
        if timeframe == "1m":
            self._update_vwap(symbol, candle)
            self._detect_fvg(symbol)
        
        # Detect structure (5m data)
        if timeframe == "5m":
            self._detect_liquidity_levels(symbol)
            self._detect_structure_break(symbol)
    
    def _update_vwap(self, symbol: str, candle: Candle):
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
    
    def _detect_liquidity_levels(self, symbol: str):
        """Detect swing highs/lows from 5M candles."""
        candles = list(self.candles_5m.get(symbol, []))
        if len(candles) < 5:
            return
        
        if symbol not in self.liquidity_levels:
            self.liquidity_levels[symbol] = []
        
        # Swing detection with 1 candle confirmation
        for i in range(1, len(candles) - 1):
            # Swing high
            if candles[i].high >= candles[i-1].high and candles[i].high >= candles[i+1].high:
                self._register_level(symbol, candles[i].high, "equal_high", candles[i].timestamp)
            
            # Swing low  
            if candles[i].low <= candles[i-1].low and candles[i].low <= candles[i+1].low:
                self._register_level(symbol, candles[i].low, "equal_low", candles[i].timestamp)
    
    def _register_level(self, symbol: str, price: float, level_type: str, timestamp: float):
        tolerance = price * self.equal_level_tolerance
        
        for level in self.liquidity_levels[symbol]:
            if level.level_type == level_type and abs(level.price - price) < tolerance:
                level.touches += 1
                level.last_touch = timestamp
                return
        
        self.liquidity_levels[symbol].append(LiquidityLevel(
            price=price, level_type=level_type, touches=1,
            first_touch=timestamp, last_touch=timestamp
        ))
        
        # Keep top 20 levels
        self.liquidity_levels[symbol] = sorted(
            self.liquidity_levels[symbol],
            key=lambda x: x.touches,
            reverse=True
        )[:20]
    
    def _detect_fvg(self, symbol: str):
        """Detect Fair Value Gaps from 1M candles."""
        candles = list(self.candles_1m.get(symbol, []))
        if len(candles) < 3:
            return
        
        if symbol not in self.fair_value_gaps:
            self.fair_value_gaps[symbol] = []
        
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # Bullish FVG: c1 high < c3 low (gap up)
        if c1.high < c3.low:
            self.fair_value_gaps[symbol].append(FairValueGap(
                high=c3.low,
                low=c1.high,
                direction="bullish",
                timestamp=c2.timestamp
            ))
        
        # Bearish FVG: c1 low > c3 high (gap down)
        if c1.low > c3.high:
            self.fair_value_gaps[symbol].append(FairValueGap(
                high=c1.low,
                low=c3.high,
                direction="bearish",
                timestamp=c2.timestamp
            ))
        
        # Keep only recent unfilled gaps
        self.fair_value_gaps[symbol] = [
            f for f in self.fair_value_gaps[symbol][-10:] if not f.filled
        ]
    
    def _detect_structure_break(self, symbol: str):
        """Detect Change of Character (CHoCH) from 5M structure."""
        candles = list(self.candles_5m.get(symbol, []))
        if len(candles) < 10:
            return
        
        if symbol not in self.swing_highs:
            self.swing_highs[symbol] = []
        if symbol not in self.swing_lows:
            self.swing_lows[symbol] = []
        
        # Get recent swing points
        recent_highs = [c.high for c in candles[-10:]]
        recent_lows = [c.low for c in candles[-10:]]
        
        max_high = max(recent_highs)
        min_low = min(recent_lows)
        current_close = candles[-1].close
        prev_close = candles[-2].close
        
        # Bullish CHoCH: Price was making lower lows, now breaks a swing high
        if current_close > max_high * 0.999 and prev_close < max_high:
            self.last_structure_break[symbol] = "bullish"
        
        # Bearish CHoCH: Price was making higher highs, now breaks a swing low
        elif current_close < min_low * 1.001 and prev_close > min_low:
            self.last_structure_break[symbol] = "bearish"
    
    def get_market_state(self, symbol: str) -> MarketState:
        """Determine market state - default to expansion for trading."""
        if symbol not in self.candles_15m or len(self.candles_15m[symbol]) < 5:
            return MarketState.EXPANSION
        
        candles = list(self.candles_15m[symbol])[-10:]
        avg_volume = np.mean([c.volume for c in candles])
        recent = candles[-3:]
        recent_volume = np.mean([c.volume for c in recent])
        
        if recent_volume > avg_volume * 1.5:
            return MarketState.EXPANSION
        elif recent_volume < avg_volume * 0.5:
            return MarketState.ACCUMULATION
        
        return MarketState.EXPANSION
    
    def generate_signal(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """
        Generate trading signal using multiple entry types.
        Priority: Liquidity Sweep > CHoCH > FVG > VWAP Bounce
        """
        vwap = self.vwap.get(symbol)
        if not vwap:
            return None
        
        # 1. Check Liquidity Sweep (highest priority)
        signal = self._check_liquidity_sweep(symbol, current_candle)
        if signal:
            return signal
        
        # 2. Check Change of Character
        signal = self._check_choch_entry(symbol, current_candle)
        if signal:
            return signal
        
        # 3. Check FVG Fill
        signal = self._check_fvg_entry(symbol, current_candle)
        if signal:
            return signal
        
        # 4. Check VWAP Bounce
        signal = self._check_vwap_bounce(symbol, current_candle)
        if signal:
            return signal
        
        return None
    
    def _check_liquidity_sweep(self, symbol: str, candle: Candle) -> Optional[TradeSignal]:
        """Check for liquidity sweep + reclaim."""
        if symbol not in self.liquidity_levels:
            return None
        
        for level in self.liquidity_levels[symbol]:
            if level.swept:
                continue
            
            # LONG: Sweep below equal low, close back above
            if level.level_type == "equal_low":
                if candle.low < level.price and candle.close > level.price:
                    level.swept = True
                    return self._create_signal(
                        symbol, TradeDirection.LONG, EntryType.LIQUIDITY_SWEEP,
                        candle, level.price, candle.low,
                        f"Sweep @ ${level.price:,.0f} → Reclaim"
                    )
            
            # SHORT: Sweep above equal high, close back below
            if level.level_type == "equal_high":
                if candle.high > level.price and candle.close < level.price:
                    level.swept = True
                    return self._create_signal(
                        symbol, TradeDirection.SHORT, EntryType.LIQUIDITY_SWEEP,
                        candle, level.price, candle.high,
                        f"Sweep @ ${level.price:,.0f} → Rejection"
                    )
        
        return None
    
    def _check_choch_entry(self, symbol: str, candle: Candle) -> Optional[TradeSignal]:
        """Check for Change of Character entry."""
        structure = self.last_structure_break.get(symbol)
        if not structure:
            return None
        
        vwap = self.vwap.get(symbol, candle.close)
        candles = list(self.candles_1m.get(symbol, []))
        if len(candles) < 5:
            return None
        
        recent_low = min(c.low for c in candles[-5:])
        recent_high = max(c.high for c in candles[-5:])
        
        # Bullish CHoCH: Enter on pullback after structure break
        if structure == "bullish" and candle.is_bullish:
            if candle.low <= vwap * 1.002 and candle.close > vwap:
                self.last_structure_break[symbol] = None  # Reset
                return self._create_signal(
                    symbol, TradeDirection.LONG, EntryType.CHOCH,
                    candle, vwap, recent_low,
                    f"CHoCH Bullish + VWAP"
                )
        
        # Bearish CHoCH
        if structure == "bearish" and not candle.is_bullish:
            if candle.high >= vwap * 0.998 and candle.close < vwap:
                self.last_structure_break[symbol] = None
                return self._create_signal(
                    symbol, TradeDirection.SHORT, EntryType.CHOCH,
                    candle, vwap, recent_high,
                    f"CHoCH Bearish + VWAP"
                )
        
        return None
    
    def _check_fvg_entry(self, symbol: str, candle: Candle) -> Optional[TradeSignal]:
        """Check for Fair Value Gap fill entry."""
        gaps = self.fair_value_gaps.get(symbol, [])
        if not gaps:
            return None
        
        for gap in gaps:
            if gap.filled:
                continue
            
            gap_mid = (gap.high + gap.low) / 2
            
            # Bullish FVG: Price returns to fill, bounces
            if gap.direction == "bullish":
                if candle.low <= gap_mid and candle.close > gap_mid:
                    gap.filled = True
                    return self._create_signal(
                        symbol, TradeDirection.LONG, EntryType.FVG,
                        candle, gap_mid, gap.low,
                        f"FVG Fill @ ${gap_mid:,.0f}"
                    )
            
            # Bearish FVG
            if gap.direction == "bearish":
                if candle.high >= gap_mid and candle.close < gap_mid:
                    gap.filled = True
                    return self._create_signal(
                        symbol, TradeDirection.SHORT, EntryType.FVG,
                        candle, gap_mid, gap.high,
                        f"FVG Fill @ ${gap_mid:,.0f}"
                    )
        
        return None
    
    def _check_vwap_bounce(self, symbol: str, candle: Candle) -> Optional[TradeSignal]:
        """Check for VWAP bounce entry."""
        vwap = self.vwap.get(symbol)
        if not vwap:
            return None
        
        price = candle.close
        vwap_dist = abs(price - vwap) / vwap
        
        # Within 0.1% of VWAP
        if vwap_dist < 0.001:
            if candle.is_bullish and candle.body_size > candle.total_range * 0.5:
                return self._create_signal(
                    symbol, TradeDirection.LONG, EntryType.VWAP_BOUNCE,
                    candle, vwap, candle.low,
                    f"VWAP Bounce @ ${vwap:,.0f}"
                )
            elif not candle.is_bullish and candle.body_size > candle.total_range * 0.5:
                return self._create_signal(
                    symbol, TradeDirection.SHORT, EntryType.VWAP_BOUNCE,
                    candle, vwap, candle.high,
                    f"VWAP Rejection @ ${vwap:,.0f}"
                )
        
        return None
    
    def _create_signal(self, symbol: str, direction: TradeDirection,
                       entry_type: EntryType, candle: Candle, 
                       level: float, wick: float, reason: str) -> Optional[TradeSignal]:
        """Create a validated trade signal."""
        entry = candle.close
        vwap = self.vwap.get(symbol, entry)
        
        if direction == TradeDirection.LONG:
            sl = wick * (1 - self.sl_buffer_pct)
            risk = entry - sl
            tp1 = entry + risk * 1.0
            tp2 = entry + risk * 2.0
            tp3 = entry + risk * 3.0
        else:
            sl = wick * (1 + self.sl_buffer_pct)
            risk = sl - entry
            tp1 = entry - risk * 1.0
            tp2 = entry - risk * 2.0
            tp3 = entry - risk * 3.0
        
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_type=entry_type,
            entry_price=entry,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            liquidity_level=level,
            sweep_candle_low=candle.low,
            sweep_candle_high=candle.high,
            vwap=vwap,
            reason=reason,
            market_state=self.get_market_state(symbol)
        )
        
        if signal.is_valid:
            return signal
        return None


@dataclass
class ActivePosition:
    """Tracks an active position with independent SL management."""
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
    
    def move_sl_to_breakeven(self):
        self.current_sl = self.signal.entry_price
        self.tp1_hit = True
    
    def move_sl_to_tp1(self):
        self.current_sl = self.signal.tp1
        self.tp2_hit = True
    
    def get_partial_qty_tp1(self) -> float:
        return self.quantity * 0.5
    
    def get_partial_qty_tp2(self) -> float:
        return self.quantity * 0.3
    
    def should_exit(self, current_price: float) -> Tuple[bool, str]:
        if self.signal.direction == TradeDirection.LONG:
            if current_price <= self.current_sl:
                return True, "STOP_LOSS"
        else:
            if current_price >= self.current_sl:
                return True, "STOP_LOSS"
        return False, ""
    
    def check_tp_levels(self, current_price: float) -> Optional[str]:
        direction = self.signal.direction
        if direction == TradeDirection.LONG:
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
