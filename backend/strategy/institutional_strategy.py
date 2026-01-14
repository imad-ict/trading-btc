"""
Fast Institutional Trading Strategy

Optimized for quick profit generation while maintaining quality entries:
1. Uses 1M candles for faster structure detection
2. Multiple entry types: swing reclaim, momentum, VWAP bounce
3. Quick level detection (3 candles instead of 5)
4. Lower minimum candle requirements
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


@dataclass
class LiquidityLevel:
    price: float
    level_type: str
    touches: int
    first_touch: float
    last_touch: float
    swept: bool = False


@dataclass
class TradeSignal:
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    liquidity_level: float
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
        return 0.0005 <= self.risk_distance <= 0.008  # 0.05% to 0.8%


class InstitutionalStrategy:
    """
    Fast trading strategy optimized for quick profit generation.
    """
    
    def __init__(self):
        # Candle storage - using 1M as primary for speed
        self.candles_1m: Dict[str, deque] = {}
        self.candles_5m: Dict[str, deque] = {}
        self.candles_15m: Dict[str, deque] = {}
        
        self.liquidity_levels: Dict[str, List[LiquidityLevel]] = {}
        
        self.session_high: Dict[str, float] = {}
        self.session_low: Dict[str, float] = {}
        
        self.vwap: Dict[str, float] = {}
        self.vwap_data: Dict[str, List[Tuple[float, float]]] = {}
        
        # AGGRESSIVE settings for faster trading
        self.equal_level_tolerance = 0.0008  # 0.08%
        self.min_level_touches = 1
        self.sl_buffer_pct = 0.0008
        self.max_candles_1m = 60
        self.max_candles_5m = 30
        self.max_candles_15m = 20
        
        # Momentum settings
        self.momentum_threshold = 0.0015  # 0.15% momentum for entry
    
    def get_diagnostics(self, symbol: str) -> Dict:
        """Get diagnostic info."""
        return {
            "symbol": symbol,
            "candles_1m": len(self.candles_1m.get(symbol, [])),
            "candles_5m": len(self.candles_5m.get(symbol, [])),
            "candles_15m": len(self.candles_15m.get(symbol, [])),
            "liquidity_levels": [
                {"price": l.price, "type": l.level_type, "touches": l.touches, "swept": l.swept}
                for l in self.liquidity_levels.get(symbol, [])[:5]
            ],
            "vwap": self.vwap.get(symbol),
            "session_high": self.session_high.get(symbol),
            "session_low": self.session_low.get(symbol),
            "market_state": self.get_market_state(symbol).value,
        }
    
    def add_candle(self, symbol: str, candle: Candle, timeframe: str = "1m"):
        """Add a candle and update derived data."""
        if timeframe == "1m":
            storage = self.candles_1m
            max_len = self.max_candles_1m
        elif timeframe == "5m":
            storage = self.candles_5m
            max_len = self.max_candles_5m
        else:
            storage = self.candles_15m
            max_len = self.max_candles_15m
        
        if symbol not in storage:
            storage[symbol] = deque(maxlen=max_len)
        
        storage[symbol].append(candle)
        
        # Update session
        if symbol not in self.session_high or candle.high > self.session_high[symbol]:
            self.session_high[symbol] = candle.high
        if symbol not in self.session_low or candle.low < self.session_low[symbol]:
            self.session_low[symbol] = candle.low
        
        # Update VWAP
        if timeframe == "1m":
            self._update_vwap(symbol, candle)
        
        # Detect levels from 1M for speed (not 5M)
        if timeframe == "1m" and len(self.candles_1m.get(symbol, [])) >= 5:
            self._detect_liquidity_levels_fast(symbol)
    
    def _update_vwap(self, symbol: str, candle: Candle):
        if symbol not in self.vwap_data:
            self.vwap_data[symbol] = []
        
        typical_price = (candle.high + candle.low + candle.close) / 3
        self.vwap_data[symbol].append((typical_price, candle.volume))
        
        if len(self.vwap_data[symbol]) > 120:
            self.vwap_data[symbol] = self.vwap_data[symbol][-120:]
        
        total_pv = sum(p * v for p, v in self.vwap_data[symbol])
        total_v = sum(v for _, v in self.vwap_data[symbol])
        
        if total_v > 0:
            self.vwap[symbol] = total_pv / total_v
    
    def _detect_liquidity_levels_fast(self, symbol: str):
        """Fast level detection from 1M candles."""
        candles = list(self.candles_1m.get(symbol, []))
        if len(candles) < 5:
            return
        
        if symbol not in self.liquidity_levels:
            self.liquidity_levels[symbol] = []
        
        # Use only 1 candle before/after for faster detection
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
        
        # Keep top 15 levels
        self.liquidity_levels[symbol] = sorted(
            self.liquidity_levels[symbol],
            key=lambda x: x.touches,
            reverse=True
        )[:15]
    
    def get_market_state(self, symbol: str) -> MarketState:
        # Default to expansion for trading
        return MarketState.EXPANSION
    
    def generate_signal(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """
        Generate trading signal - MULTIPLE ENTRY TYPES:
        1. Liquidity sweep + reclaim (preferred)
        2. VWAP bounce
        3. Momentum breakout
        """
        vwap = self.vwap.get(symbol)
        if not vwap:
            return None
        
        current_price = current_candle.close
        
        # METHOD 1: Check for liquidity sweep
        sweep_signal = self._check_liquidity_sweep_signal(symbol, current_candle)
        if sweep_signal:
            return sweep_signal
        
        # METHOD 2: VWAP Bounce
        vwap_signal = self._check_vwap_bounce(symbol, current_candle)
        if vwap_signal:
            return vwap_signal
        
        # METHOD 3: Momentum entry
        momentum_signal = self._check_momentum_entry(symbol, current_candle)
        if momentum_signal:
            return momentum_signal
        
        return None
    
    def _check_liquidity_sweep_signal(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """Check for liquidity sweep + reclaim."""
        if symbol not in self.liquidity_levels:
            return None
        
        for level in self.liquidity_levels[symbol]:
            if level.swept:
                continue
            
            # Sell-side sweep (LONG opportunity)
            if level.level_type == "equal_low":
                if current_candle.low < level.price and current_candle.close > level.price:
                    return self._create_signal(
                        symbol, TradeDirection.LONG, current_candle,
                        level.price, current_candle.low,
                        f"Sweep @ ${level.price:,.0f} → Reclaim"
                    )
            
            # Buy-side sweep (SHORT opportunity)
            if level.level_type == "equal_high":
                if current_candle.high > level.price and current_candle.close < level.price:
                    return self._create_signal(
                        symbol, TradeDirection.SHORT, current_candle,
                        level.price, current_candle.high,
                        f"Sweep @ ${level.price:,.0f} → Rejection"
                    )
        
        return None
    
    def _check_vwap_bounce(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """Check for VWAP bounce entry."""
        vwap = self.vwap.get(symbol)
        if not vwap:
            return None
        
        price = current_candle.close
        vwap_distance = abs(price - vwap) / vwap
        
        # Price touched VWAP and bounced
        if vwap_distance < 0.001:  # Within 0.1% of VWAP
            if current_candle.is_bullish and current_candle.close > current_candle.open:
                # Bullish bounce
                return self._create_signal(
                    symbol, TradeDirection.LONG, current_candle,
                    vwap, current_candle.low,
                    f"VWAP Bounce @ ${vwap:,.0f}"
                )
            elif not current_candle.is_bullish and current_candle.close < current_candle.open:
                # Bearish rejection
                return self._create_signal(
                    symbol, TradeDirection.SHORT, current_candle,
                    vwap, current_candle.high,
                    f"VWAP Rejection @ ${vwap:,.0f}"
                )
        
        return None
    
    def _check_momentum_entry(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """Check for momentum breakout entry."""
        candles = list(self.candles_1m.get(symbol, []))
        if len(candles) < 10:
            return None
        
        # Calculate recent momentum
        recent_closes = [c.close for c in candles[-5:]]
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        
        vwap = self.vwap.get(symbol)
        if not vwap:
            return None
        
        price = current_candle.close
        
        # Strong bullish momentum + price above VWAP
        if momentum > self.momentum_threshold and price > vwap:
            recent_low = min(c.low for c in candles[-3:])
            return self._create_signal(
                symbol, TradeDirection.LONG, current_candle,
                vwap, recent_low,
                f"Momentum +{momentum*100:.2f}%"
            )
        
        # Strong bearish momentum + price below VWAP
        if momentum < -self.momentum_threshold and price < vwap:
            recent_high = max(c.high for c in candles[-3:])
            return self._create_signal(
                symbol, TradeDirection.SHORT, current_candle,
                vwap, recent_high,
                f"Momentum {momentum*100:.2f}%"
            )
        
        return None
    
    def _create_signal(self, symbol: str, direction: TradeDirection, 
                       candle: Candle, level: float, wick: float,
                       reason: str) -> Optional[TradeSignal]:
        """Create validated trade signal."""
        entry = candle.close
        vwap = self.vwap.get(symbol, entry)
        
        # Calculate SL
        if direction == TradeDirection.LONG:
            sl = wick * (1 - self.sl_buffer_pct)
            risk = entry - sl
            tp1 = entry + risk * 1.0  # 1R
            tp2 = entry + risk * 2.0  # 2R
            tp3 = entry + risk * 3.0  # 3R
        else:
            sl = wick * (1 + self.sl_buffer_pct)
            risk = sl - entry
            tp1 = entry - risk * 1.0
            tp2 = entry - risk * 2.0
            tp3 = entry - risk * 3.0
        
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
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
            market_state=MarketState.EXPANSION
        )
        
        if signal.is_valid:
            return signal
        return None


@dataclass
class ActivePosition:
    """Tracks an active position."""
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
        return self.quantity * 0.5  # 50% at TP1
    
    def get_partial_qty_tp2(self) -> float:
        return self.quantity * 0.3  # 30% at TP2
    
    def should_exit(self, current_price: float) -> Tuple[bool, str]:
        direction = self.signal.direction
        if direction == TradeDirection.LONG:
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
