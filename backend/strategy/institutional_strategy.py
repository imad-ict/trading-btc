"""
High Win-Rate Institutional Strategy v3

CORE PRINCIPLE: Trade WITH the trend, enter on CONFIRMED pullbacks

Entry Requirements:
1. TREND ALIGNMENT (Higher TF) - 15M trend direction
2. ORDER BLOCK - Price returns to a demand/supply zone
3. RSI CONFIRMATION - Momentum in trade direction
4. STRUCTURE - Clear market structure (HH/HL for long, LH/LL for short)

Win Rate Improvement:
- Only trade with trend (no counter-trend)
- Wait for pullback to Order Block (not just sweep)
- RSI must show reversal (oversold for long, overbought for short)
- Tighter SL at Order Block edge
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(Enum):
    ORDER_BLOCK = "order_block"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    VWAP_PULLBACK = "vwap_pullback"


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


@dataclass
class OrderBlock:
    """Institutional Order Block - zone where institutions entered"""
    top: float
    bottom: float
    direction: str  # "bullish" (demand) or "bearish" (supply)
    timestamp: float
    strength: int = 1  # How many times it's been tested
    valid: bool = True


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
    vwap: float
    trend: TrendDirection
    rsi: float
    reason: str
    
    @property
    def risk_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price
    
    @property
    def is_valid(self) -> bool:
        # Valid risk: 0.1% to 0.8% (tighter for scalping)
        return 0.001 <= self.risk_distance <= 0.008


class InstitutionalStrategy:
    """
    High Win-Rate Strategy
    
    Key Rules:
    1. ONLY trade with 15M trend
    2. Wait for pullback to Order Block/VWAP
    3. RSI must confirm reversal
    4. Enter on confirmation candle
    """
    
    def __init__(self, max_positions: int = 3):
        # Candle storage
        self.candles_1m: Dict[str, deque] = {}
        self.candles_5m: Dict[str, deque] = {}
        self.candles_15m: Dict[str, deque] = {}
        
        self.order_blocks: Dict[str, List[OrderBlock]] = {}
        
        # Technical indicators
        self.vwap: Dict[str, float] = {}
        self.vwap_data: Dict[str, List[Tuple[float, float]]] = {}
        self.rsi: Dict[str, float] = {}
        self.trend: Dict[str, TrendDirection] = {}
        
        # Swing points for structure
        self.swing_highs: Dict[str, List[float]] = {}
        self.swing_lows: Dict[str, List[float]] = {}
        
        # Configuration
        self.max_positions = max_positions
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.sl_buffer_pct = 0.0005  # 0.05% buffer
        self.max_candles = 100
    
    def get_diagnostics(self, symbol: str) -> Dict:
        """Get diagnostic info for a symbol."""
        return {
            "symbol": symbol,
            "candles_1m": len(self.candles_1m.get(symbol, [])),
            "candles_5m": len(self.candles_5m.get(symbol, [])),
            "candles_15m": len(self.candles_15m.get(symbol, [])),
            "trend": self.trend.get(symbol, TrendDirection.RANGING).value,
            "rsi": round(self.rsi.get(symbol, 50), 1),
            "vwap": self.vwap.get(symbol),
            "order_blocks": len([ob for ob in self.order_blocks.get(symbol, []) if ob.valid]),
        }
    
    def add_candle(self, symbol: str, candle: Candle, timeframe: str = "1m"):
        """Add a candle and update all derived data."""
        storage_map = {
            "1m": self.candles_1m,
            "5m": self.candles_5m,
            "15m": self.candles_15m
        }
        
        storage = storage_map.get(timeframe, self.candles_1m)
        
        if symbol not in storage:
            storage[symbol] = deque(maxlen=self.max_candles)
        
        storage[symbol].append(candle)
        
        # Update indicators based on timeframe
        if timeframe == "1m":
            self._update_vwap(symbol, candle)
            self._update_rsi(symbol)
        
        if timeframe == "5m":
            self._detect_order_blocks(symbol)
            self._update_swing_points(symbol)
        
        if timeframe == "15m":
            self._update_trend(symbol)
    
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
    
    def _update_rsi(self, symbol: str):
        """Calculate RSI from 1M closes."""
        candles = list(self.candles_1m.get(symbol, []))
        if len(candles) < self.rsi_period + 1:
            self.rsi[symbol] = 50  # Neutral default
            return
        
        closes = [c.close for c in candles[-(self.rsi_period + 1):]]
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]
        
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period
        
        if avg_loss == 0:
            self.rsi[symbol] = 100
        else:
            rs = avg_gain / avg_loss
            self.rsi[symbol] = 100 - (100 / (1 + rs))
    
    def _update_trend(self, symbol: str):
        """Determine trend from 15M candles."""
        candles = list(self.candles_15m.get(symbol, []))
        if len(candles) < 10:
            self.trend[symbol] = TrendDirection.RANGING
            return
        
        recent = candles[-10:]
        closes = [c.close for c in recent]
        
        # Simple: is recent close above/below average?
        avg = sum(closes) / len(closes)
        current = closes[-1]
        
        # Also check structure: are we making higher lows or lower highs?
        lows = [c.low for c in recent]
        highs = [c.high for c in recent]
        
        recent_low = min(lows[-3:])
        older_low = min(lows[:3])
        recent_high = max(highs[-3:])
        older_high = max(highs[:3])
        
        if current > avg and recent_low > older_low:
            self.trend[symbol] = TrendDirection.BULLISH
        elif current < avg and recent_high < older_high:
            self.trend[symbol] = TrendDirection.BEARISH
        else:
            self.trend[symbol] = TrendDirection.RANGING
    
    def _detect_order_blocks(self, symbol: str):
        """Detect Order Blocks from 5M candles."""
        candles = list(self.candles_5m.get(symbol, []))
        if len(candles) < 5:
            return
        
        if symbol not in self.order_blocks:
            self.order_blocks[symbol] = []
        
        # Look for strong impulse moves with body > 60%
        for i in range(2, len(candles) - 1):
            c = candles[i]
            
            # Bullish Order Block: strong bearish candle followed by bullish move
            if not c.is_bullish and c.body_pct > 0.6:
                next_c = candles[i + 1]
                if next_c.is_bullish and next_c.close > c.open:
                    # This bearish candle is a demand zone
                    ob = OrderBlock(
                        top=c.open,
                        bottom=c.close,
                        direction="bullish",
                        timestamp=c.timestamp
                    )
                    self._add_order_block(symbol, ob)
            
            # Bearish Order Block: strong bullish candle followed by bearish move
            if c.is_bullish and c.body_pct > 0.6:
                next_c = candles[i + 1]
                if not next_c.is_bullish and next_c.close < c.open:
                    # This bullish candle is a supply zone
                    ob = OrderBlock(
                        top=c.close,
                        bottom=c.open,
                        direction="bearish",
                        timestamp=c.timestamp
                    )
                    self._add_order_block(symbol, ob)
    
    def _add_order_block(self, symbol: str, ob: OrderBlock):
        """Add order block, avoid duplicates."""
        for existing in self.order_blocks[symbol]:
            if abs(existing.top - ob.top) < ob.top * 0.001:
                existing.strength += 1
                return
        
        self.order_blocks[symbol].append(ob)
        # Keep only recent 10 order blocks
        self.order_blocks[symbol] = self.order_blocks[symbol][-10:]
    
    def _update_swing_points(self, symbol: str):
        """Track swing highs and lows from 5M."""
        candles = list(self.candles_5m.get(symbol, []))
        if len(candles) < 5:
            return
        
        if symbol not in self.swing_highs:
            self.swing_highs[symbol] = []
        if symbol not in self.swing_lows:
            self.swing_lows[symbol] = []
        
        for i in range(2, len(candles) - 2):
            # Swing high
            if candles[i].high > candles[i-1].high and candles[i].high > candles[i+1].high:
                if candles[i].high not in self.swing_highs[symbol]:
                    self.swing_highs[symbol].append(candles[i].high)
            
            # Swing low
            if candles[i].low < candles[i-1].low and candles[i].low < candles[i+1].low:
                if candles[i].low not in self.swing_lows[symbol]:
                    self.swing_lows[symbol].append(candles[i].low)
        
        # Keep recent swings only
        self.swing_highs[symbol] = self.swing_highs[symbol][-10:]
        self.swing_lows[symbol] = self.swing_lows[symbol][-10:]
    
    def generate_signal(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """
        Generate trading signal with HIGH WIN-RATE requirements.
        
        ONLY enter when:
        1. Trend is clear (not ranging)
        2. Price is at an Order Block
        3. RSI confirms reversal
        4. Current candle shows rejection
        """
        trend = self.trend.get(symbol, TrendDirection.RANGING)
        rsi = self.rsi.get(symbol, 50)
        vwap = self.vwap.get(symbol)
        
        if not vwap:
            return None
        
        # RULE 1: Check for Order Block entry (requires trend)
        if trend != TrendDirection.RANGING:
            signal = self._check_order_block_entry(symbol, current_candle, trend, rsi)
            if signal:
                return signal
        
        # RULE 2: Check for VWAP pullback (works in any market)
        signal = self._check_vwap_pullback(symbol, current_candle, trend, rsi)
        if signal:
            return signal
        
        # RULE 3: Momentum entry (strong candle with RSI confirmation)
        signal = self._check_momentum_entry(symbol, current_candle, rsi)
        if signal:
            return signal
        
        return None
    
    def _check_order_block_entry(self, symbol: str, candle: Candle, 
                                  trend: TrendDirection, rsi: float) -> Optional[TradeSignal]:
        """Check if price is at an Order Block with confirmation."""
        order_blocks = [ob for ob in self.order_blocks.get(symbol, []) if ob.valid]
        
        for ob in order_blocks:
            ob_mid = (ob.top + ob.bottom) / 2
            
            # LONG: Bullish trend + Bullish OB + RSI < 50 + Price at OB + Bullish candle
            if (trend == TrendDirection.BULLISH and 
                ob.direction == "bullish" and
                rsi < 50 and  # RSI below 50 is good enough
                candle.low <= ob.top and candle.close > ob_mid and
                candle.is_bullish and candle.body_pct > 0.3):
                
                ob.valid = False  # Mark as used
                return self._create_signal(
                    symbol, TradeDirection.LONG, EntryType.ORDER_BLOCK,
                    candle.close, ob.bottom, trend, rsi,
                    f"Order Block Long @ ${ob_mid:,.0f}"
                )
            
            # SHORT: Bearish trend + Bearish OB + RSI > 50 + Price at OB + Bearish candle
            if (trend == TrendDirection.BEARISH and 
                ob.direction == "bearish" and
                rsi > 50 and  # RSI above 50 is good enough
                candle.high >= ob.bottom and candle.close < ob_mid and
                not candle.is_bullish and candle.body_pct > 0.3):
                
                ob.valid = False
                return self._create_signal(
                    symbol, TradeDirection.SHORT, EntryType.ORDER_BLOCK,
                    candle.close, ob.top, trend, rsi,
                    f"Order Block Short @ ${ob_mid:,.0f}"
                )
        
        return None
    
    def _check_vwap_pullback(self, symbol: str, candle: Candle,
                              trend: TrendDirection, rsi: float) -> Optional[TradeSignal]:
        """VWAP pullback - works in trending AND ranging markets."""
        vwap = self.vwap.get(symbol)
        if not vwap:
            return None
        
        price = candle.close
        vwap_dist = (price - vwap) / vwap
        
        # LONG: Price near VWAP from below + RSI not overbought + bullish candle
        if (abs(vwap_dist) < 0.003 and  # Within 0.3% of VWAP
            price > vwap and  # Just crossed above
            rsi < 60 and  # Not overbought
            candle.is_bullish and candle.body_pct > 0.35):
            
            sl = min(candle.low, vwap) * (1 - self.sl_buffer_pct)
            return self._create_signal(
                symbol, TradeDirection.LONG, EntryType.VWAP_PULLBACK,
                candle.close, sl, trend, rsi,
                f"VWAP Pullback Long"
            )
        
        # SHORT: Price near VWAP from above + RSI not oversold + bearish candle
        if (abs(vwap_dist) < 0.003 and  # Within 0.3% of VWAP
            price < vwap and  # Just crossed below
            rsi > 40 and  # Not oversold
            not candle.is_bullish and candle.body_pct > 0.35):
            
            sl = max(candle.high, vwap) * (1 + self.sl_buffer_pct)
            return self._create_signal(
                symbol, TradeDirection.SHORT, EntryType.VWAP_PULLBACK,
                candle.close, sl, trend, rsi,
                f"VWAP Pullback Short"
            )
        
        return None
    
    def _check_momentum_entry(self, symbol: str, candle: Candle, rsi: float) -> Optional[TradeSignal]:
        """
        Momentum entry - strong candle with RSI confirmation.
        Works in any market when we see clear momentum.
        """
        vwap = self.vwap.get(symbol)
        if not vwap:
            return None
        
        trend = self.trend.get(symbol, TrendDirection.RANGING)
        
        # Need a strong candle (body > 60% of range)
        if candle.body_pct < 0.6:
            return None
        
        # LONG: Strong bullish candle + RSI rising but not overbought
        if candle.is_bullish and 40 < rsi < 65:
            # SL just below the candle low
            sl = candle.low * (1 - self.sl_buffer_pct)
            return self._create_signal(
                symbol, TradeDirection.LONG, EntryType.LIQUIDITY_SWEEP,  # Use existing type
                candle.close, sl, trend, rsi,
                f"Momentum Long"
            )
        
        # SHORT: Strong bearish candle + RSI falling but not oversold
        if not candle.is_bullish and 35 < rsi < 60:
            sl = candle.high * (1 + self.sl_buffer_pct)
            return self._create_signal(
                symbol, TradeDirection.SHORT, EntryType.LIQUIDITY_SWEEP,
                candle.close, sl, trend, rsi,
                f"Momentum Short"
            )
        
        return None
    
    def _create_signal(self, symbol: str, direction: TradeDirection,
                       entry_type: EntryType, entry: float, sl: float,
                       trend: TrendDirection, rsi: float, reason: str) -> Optional[TradeSignal]:
        """Create a validated trade signal with proper R:R."""
        vwap = self.vwap.get(symbol, entry)
        
        risk = abs(entry - sl)
        
        if direction == TradeDirection.LONG:
            tp1 = entry + risk * 1.5  # 1.5R
            tp2 = entry + risk * 2.5  # 2.5R
            tp3 = entry + risk * 4.0  # 4R
        else:
            tp1 = entry - risk * 1.5
            tp2 = entry - risk * 2.5
            tp3 = entry - risk * 4.0
        
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_type=entry_type,
            entry_price=entry,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            vwap=vwap,
            trend=trend,
            rsi=rsi,
            reason=reason
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
