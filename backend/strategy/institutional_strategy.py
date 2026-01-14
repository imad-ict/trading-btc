"""
Institutional Trading Strategy - Liquidity-Based Entry Logic

Core Principles:
1. NO momentum-based entries - only liquidity sweeps + reclaims
2. SL placed behind liquidity structure, not fixed percentages
3. TP targets: VWAP → Opposite Liquidity → FVG
4. Partial closes with dynamic SL movement
5. Market state filter before any trade

Capital preservation > Trade frequency
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market state classification based on 15M structure."""
    ACCUMULATION = "accumulation"      # Ranging, low volume - NO TRADE
    MANIPULATION = "manipulation"       # Liquidity sweep in progress - WAIT
    EXPANSION = "expansion"             # Trending, high volume - TRADE ALLOWED
    DISTRIBUTION = "distribution"       # Topping/bottoming - EXIT ONLY
    UNKNOWN = "unknown"


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Candle:
    """OHLCV candle data."""
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
    def efficiency(self) -> float:
        """Body/Range ratio - higher = stronger move."""
        if self.total_range == 0:
            return 0
        return self.body_size / self.total_range


@dataclass
class LiquidityLevel:
    """Tracks a liquidity level (equal highs/lows)."""
    price: float
    level_type: str  # "equal_high" or "equal_low"
    touches: int
    first_touch: float  # timestamp
    last_touch: float   # timestamp
    swept: bool = False
    sweep_time: Optional[float] = None
    
    @property
    def age_seconds(self) -> float:
        return datetime.now().timestamp() - self.first_touch
    
    @property
    def strength(self) -> float:
        """Stronger levels = more touches + age."""
        return self.touches * min(self.age_seconds / 3600, 5)  # Cap at 5 hours


@dataclass
class TradeSignal:
    """Validated institutional trade signal."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    tp1: float  # VWAP or first target
    tp2: float  # Opposite liquidity
    tp3: float  # FVG or extended target
    
    # Context
    liquidity_level: float
    sweep_candle_low: float
    sweep_candle_high: float
    vwap: float
    
    # Explanation
    reason: str
    market_state: MarketState
    
    @property
    def risk_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price
    
    @property
    def is_valid(self) -> bool:
        """Validate trade meets institutional criteria."""
        # Risk must be between 0.1% and 0.5% (slightly relaxed for more opportunities)
        if self.risk_distance < 0.001 or self.risk_distance > 0.005:
            return False
        return True


class InstitutionalStrategy:
    """
    Institutional Liquidity-Based Trading Strategy.
    
    Entry Logic:
    1. Detect equal highs/lows (liquidity pools)
    2. Wait for liquidity sweep (wick beyond level)
    3. Confirm reclaim (close back inside structure)
    4. Validate with VWAP and volume
    
    Exit Logic:
    1. TP1: VWAP → move SL to breakeven
    2. TP2: Opposite liquidity → move SL to TP1
    3. TP3: Trail remaining with structure
    """
    
    def __init__(self):
        # Candle storage by timeframe
        self.candles_1m: Dict[str, deque] = {}   # 1-minute for entries
        self.candles_5m: Dict[str, deque] = {}   # 5-minute for liquidity
        self.candles_15m: Dict[str, deque] = {}  # 15-minute for market state
        
        # Liquidity tracking
        self.liquidity_levels: Dict[str, List[LiquidityLevel]] = {}
        
        # Session levels (reset daily)
        self.session_high: Dict[str, float] = {}
        self.session_low: Dict[str, float] = {}
        self.session_start: Optional[float] = None
        
        # VWAP tracking
        self.vwap: Dict[str, float] = {}
        self.vwap_data: Dict[str, List[Tuple[float, float]]] = {}  # (price, volume)
        
        # Configuration - tuned for more opportunities while maintaining quality
        self.equal_level_tolerance = 0.0005  # 0.05% tolerance for equal levels (slightly relaxed)
        self.min_level_touches = 1  # Allow single-touch levels (swing highs/lows)
        self.sl_buffer_pct = 0.0005  # 0.05% buffer beyond wick
        self.max_candles_1m = 120  # 2 hours
        self.max_candles_5m = 60   # 5 hours
        self.max_candles_15m = 40  # 10 hours
    
    def get_diagnostics(self, symbol: str) -> Dict:
        """Get diagnostic info about current market conditions."""
        diagnostics = {
            "symbol": symbol,
            "candles_1m": len(self.candles_1m.get(symbol, [])),
            "candles_5m": len(self.candles_5m.get(symbol, [])),
            "candles_15m": len(self.candles_15m.get(symbol, [])),
            "liquidity_levels": [],
            "vwap": self.vwap.get(symbol),
            "session_high": self.session_high.get(symbol),
            "session_low": self.session_low.get(symbol),
            "market_state": self.get_market_state(symbol).value if symbol in self.candles_15m else "unknown",
        }
        
        if symbol in self.liquidity_levels:
            for level in self.liquidity_levels[symbol][:5]:  # Top 5 levels
                diagnostics["liquidity_levels"].append({
                    "price": level.price,
                    "type": level.level_type,
                    "touches": level.touches,
                    "swept": level.swept
                })
        
        return diagnostics
    
    def add_candle(self, symbol: str, candle: Candle, timeframe: str = "1m"):
        """Add a candle and update all derived data."""
        # Select storage
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
        
        # Update session levels
        if symbol not in self.session_high or candle.high > self.session_high[symbol]:
            self.session_high[symbol] = candle.high
        if symbol not in self.session_low or candle.low < self.session_low[symbol]:
            self.session_low[symbol] = candle.low
        
        # Update VWAP (using 1m candles)
        if timeframe == "1m":
            self._update_vwap(symbol, candle)
        
        # Detect liquidity levels (using 5m candles)
        if timeframe == "5m":
            self._detect_liquidity_levels(symbol)
    
    def _update_vwap(self, symbol: str, candle: Candle):
        """Update Volume Weighted Average Price."""
        if symbol not in self.vwap_data:
            self.vwap_data[symbol] = []
        
        typical_price = (candle.high + candle.low + candle.close) / 3
        self.vwap_data[symbol].append((typical_price, candle.volume))
        
        # Keep last 4 hours of data
        if len(self.vwap_data[symbol]) > 240:
            self.vwap_data[symbol] = self.vwap_data[symbol][-240:]
        
        # Calculate VWAP
        total_pv = sum(p * v for p, v in self.vwap_data[symbol])
        total_v = sum(v for _, v in self.vwap_data[symbol])
        
        if total_v > 0:
            self.vwap[symbol] = total_pv / total_v
    
    def _detect_liquidity_levels(self, symbol: str):
        """Detect equal highs and lows from 5M structure."""
        if symbol not in self.candles_5m or len(self.candles_5m[symbol]) < 10:
            return
        
        candles = list(self.candles_5m[symbol])
        if symbol not in self.liquidity_levels:
            self.liquidity_levels[symbol] = []
        
        # Find swing highs (equal highs)
        for i in range(2, len(candles) - 2):
            # Check if this is a swing high
            if candles[i].high >= candles[i-1].high and candles[i].high >= candles[i-2].high:
                if candles[i].high >= candles[i+1].high and candles[i].high >= candles[i+2].high:
                    self._register_level(symbol, candles[i].high, "equal_high", candles[i].timestamp)
        
        # Find swing lows (equal lows)
        for i in range(2, len(candles) - 2):
            if candles[i].low <= candles[i-1].low and candles[i].low <= candles[i-2].low:
                if candles[i].low <= candles[i+1].low and candles[i].low <= candles[i+2].low:
                    self._register_level(symbol, candles[i].low, "equal_low", candles[i].timestamp)
    
    def _register_level(self, symbol: str, price: float, level_type: str, timestamp: float):
        """Register or update a liquidity level."""
        tolerance = price * self.equal_level_tolerance
        
        for level in self.liquidity_levels[symbol]:
            if level.level_type == level_type and abs(level.price - price) < tolerance:
                # Update existing level
                level.touches += 1
                level.last_touch = timestamp
                return
        
        # Create new level
        self.liquidity_levels[symbol].append(LiquidityLevel(
            price=price,
            level_type=level_type,
            touches=1,
            first_touch=timestamp,
            last_touch=timestamp
        ))
        
        # Prune old levels (keep strongest 10)
        self.liquidity_levels[symbol] = sorted(
            self.liquidity_levels[symbol],
            key=lambda x: x.strength,
            reverse=True
        )[:10]
    
    def get_market_state(self, symbol: str) -> MarketState:
        """Determine market state from 15M structure."""
        if symbol not in self.candles_15m or len(self.candles_15m[symbol]) < 5:
            return MarketState.UNKNOWN
        
        candles = list(self.candles_15m[symbol])[-10:]
        
        # Calculate metrics
        avg_volume = np.mean([c.volume for c in candles])
        avg_efficiency = np.mean([c.efficiency for c in candles])
        avg_range = np.mean([c.total_range for c in candles])
        
        # Recent candle characteristics
        recent = candles[-3:]
        recent_volume = np.mean([c.volume for c in recent])
        recent_efficiency = np.mean([c.efficiency for c in recent])
        
        # VWAP slope (if available)
        vwap_slope = 0
        if symbol in self.vwap and len(recent) >= 2:
            closes = [c.close for c in recent]
            vwap_slope = (closes[-1] - closes[0]) / closes[0]
        
        # State classification
        if recent_volume < avg_volume * 0.7 and recent_efficiency < 0.4:
            return MarketState.ACCUMULATION
        
        if recent_volume > avg_volume * 1.5:
            if recent_efficiency > 0.6:
                return MarketState.EXPANSION
            else:
                return MarketState.MANIPULATION
        
        if avg_efficiency < 0.3:
            return MarketState.DISTRIBUTION
        
        return MarketState.EXPANSION
    
    def check_liquidity_sweep(self, symbol: str, current_candle: Candle) -> Optional[Dict]:
        """
        Check if current candle sweeps a liquidity level.
        
        Returns sweep details if valid, None otherwise.
        """
        if symbol not in self.liquidity_levels:
            return None
        
        for level in self.liquidity_levels[symbol]:
            if level.swept:
                continue
            
            if level.touches < self.min_level_touches:
                continue
            
            # Check for sell-side sweep (price goes below equal lows)
            if level.level_type == "equal_low":
                if current_candle.low < level.price:
                    # Relaxed: Wick needs to be significant but not necessarily > body
                    # Just need to see price go below and come back (the reclaim confirms)
                    if current_candle.lower_wick > current_candle.body_size * 0.3:  # 30% of body
                        return {
                            "type": "sell_side_sweep",
                            "level": level,
                            "direction": TradeDirection.LONG,
                            "sweep_low": current_candle.low,
                            "sweep_high": current_candle.high
                        }
            
            # Check for buy-side sweep (price goes above equal highs)
            if level.level_type == "equal_high":
                if current_candle.high > level.price:
                    # Relaxed: wick needs to be significant but not necessarily > body
                    if current_candle.upper_wick > current_candle.body_size * 0.3:  # 30% of body
                        return {
                            "type": "buy_side_sweep",
                            "level": level,
                            "direction": TradeDirection.SHORT,
                            "sweep_low": current_candle.low,
                            "sweep_high": current_candle.high
                        }
        
        return None
    
    def check_reclaim(self, sweep: Dict, current_candle: Candle) -> bool:
        """
        Check if price has reclaimed the swept level.
        
        For LONG: Close must be back above the swept low
        For SHORT: Close must be back below the swept high
        """
        level = sweep["level"]
        
        if sweep["direction"] == TradeDirection.LONG:
            # After sweeping lows, price must close back above
            return current_candle.close > level.price
        else:
            # After sweeping highs, price must close back below
            return current_candle.close < level.price
    
    def validate_entry(self, symbol: str, direction: TradeDirection, 
                       current_price: float) -> Tuple[bool, str]:
        """
        Validate entry conditions beyond sweep + reclaim.
        
        Returns (is_valid, reason)
        """
        vwap = self.vwap.get(symbol)
        if not vwap:
            return False, "VWAP not available"
        
        # Get recent volume
        if symbol not in self.candles_1m or len(self.candles_1m[symbol]) < 20:
            return False, "Insufficient price history"
        
        recent_candles = list(self.candles_1m[symbol])[-20:]
        avg_volume = np.mean([c.volume for c in recent_candles])
        current_volume = recent_candles[-1].volume
        
        # Volume should be reasonable (not drastically below average) - relaxed from strict requirement
        if current_volume < avg_volume * 0.5:  # Only reject if volume is less than half average
            return False, f"Volume too low ({current_volume:.0f} < {avg_volume*0.5:.0f})"
        
        # VWAP alignment
        if direction == TradeDirection.LONG:
            if current_price < vwap:
                return False, f"Price below VWAP ({current_price:.2f} < {vwap:.2f})"
        else:
            if current_price > vwap:
                return False, f"Price above VWAP ({current_price:.2f} > {vwap:.2f})"
        
        # Market state must allow trading
        market_state = self.get_market_state(symbol)
        if market_state not in [MarketState.EXPANSION, MarketState.MANIPULATION]:
            return False, f"Market state: {market_state.value} (no trade)"
        
        return True, "All conditions met"
    
    def calculate_sl(self, sweep: Dict, direction: TradeDirection) -> float:
        """
        Calculate institutional stop-loss placement.
        
        SL is placed BEHIND the liquidity wick with buffer.
        This is anti-SL-hunt positioning.
        """
        buffer_pct = self.sl_buffer_pct
        
        if direction == TradeDirection.LONG:
            # SL below the sweep wick low
            sl = sweep["sweep_low"] * (1 - buffer_pct)
        else:
            # SL above the sweep wick high
            sl = sweep["sweep_high"] * (1 + buffer_pct)
        
        return sl
    
    def calculate_tp_targets(self, symbol: str, entry: float, 
                            direction: TradeDirection, sl: float) -> Tuple[float, float, float]:
        """
        Calculate multi-target take-profits.
        
        TP1: VWAP (40% close here, move SL to BE)
        TP2: Opposite liquidity pool (40% close here, move SL to TP1)
        TP3: Extended target / FVG (trail remaining 20%)
        """
        vwap = self.vwap.get(symbol, entry)
        risk_distance = abs(entry - sl)
        
        # TP1: VWAP or minimum 1R
        if direction == TradeDirection.LONG:
            tp1 = max(vwap, entry + risk_distance)
        else:
            tp1 = min(vwap, entry - risk_distance)
        
        # TP2: Find opposite liquidity or use 2R
        opposite_liq = self._find_opposite_liquidity(symbol, entry, direction)
        if opposite_liq:
            tp2 = opposite_liq
        else:
            if direction == TradeDirection.LONG:
                tp2 = entry + (risk_distance * 2)
            else:
                tp2 = entry - (risk_distance * 2)
        
        # TP3: Extended target (3R or session level)
        if direction == TradeDirection.LONG:
            session_target = self.session_high.get(symbol, entry + risk_distance * 3)
            tp3 = max(session_target, entry + risk_distance * 3)
        else:
            session_target = self.session_low.get(symbol, entry - risk_distance * 3)
            tp3 = min(session_target, entry - risk_distance * 3)
        
        return tp1, tp2, tp3
    
    def _find_opposite_liquidity(self, symbol: str, entry: float, 
                                  direction: TradeDirection) -> Optional[float]:
        """Find the nearest opposite liquidity pool for TP targeting."""
        if symbol not in self.liquidity_levels:
            return None
        
        levels = self.liquidity_levels[symbol]
        
        if direction == TradeDirection.LONG:
            # Looking for equal highs above current price
            targets = [l for l in levels if l.level_type == "equal_high" and l.price > entry]
            if targets:
                return min(t.price for t in targets)
        else:
            # Looking for equal lows below current price
            targets = [l for l in levels if l.level_type == "equal_low" and l.price < entry]
            if targets:
                return max(t.price for t in targets)
        
        return None
    
    def generate_signal(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """
        Main signal generation - Institutional Entry Gate.
        
        Step 1: Check for liquidity sweep
        Step 2: Confirm reclaim
        Step 3: Validate with VWAP + volume
        Step 4: Calculate structure-based SL
        Step 5: Set multi-target TP
        
        NO SWEEP = NO TRADE
        """
        # Step 1: Check for liquidity sweep
        sweep = self.check_liquidity_sweep(symbol, current_candle)
        if not sweep:
            return None
        
        logger.info(f"🎯 Liquidity sweep detected: {sweep['type']} at {sweep['level'].price:.2f}")
        
        # Step 2: Check reclaim
        if not self.check_reclaim(sweep, current_candle):
            logger.info(f"❌ No reclaim confirmed - waiting")
            return None
        
        logger.info(f"✓ Reclaim confirmed at {current_candle.close:.2f}")
        
        # Step 3: Validate entry conditions
        direction = sweep["direction"]
        is_valid, reason = self.validate_entry(symbol, direction, current_candle.close)
        
        if not is_valid:
            logger.info(f"❌ Entry rejected: {reason}")
            return None
        
        logger.info(f"✓ Entry validated: {reason}")
        
        # Step 4: Calculate SL (behind liquidity wick)
        sl = self.calculate_sl(sweep, direction)
        
        # Step 5: Calculate TP targets
        tp1, tp2, tp3 = self.calculate_tp_targets(
            symbol, current_candle.close, direction, sl
        )
        
        # Mark level as swept
        sweep["level"].swept = True
        sweep["level"].sweep_time = current_candle.timestamp
        
        # Build signal
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=current_candle.close,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            liquidity_level=sweep["level"].price,
            sweep_candle_low=sweep["sweep_low"],
            sweep_candle_high=sweep["sweep_high"],
            vwap=self.vwap.get(symbol, current_candle.close),
            reason=f"{sweep['type'].replace('_', ' ').title()} @ {sweep['level'].price:.2f} → Reclaim",
            market_state=self.get_market_state(symbol)
        )
        
        # Final validation
        if not signal.is_valid:
            logger.info(f"❌ Signal rejected: Risk distance {signal.risk_distance*100:.3f}% out of range")
            return None
        
        return signal


@dataclass
class ActivePosition:
    """Tracks an active position with dynamic SL management."""
    signal: TradeSignal
    order_id: str
    quantity: float
    entry_time: datetime
    
    # Position state
    current_sl: float = field(init=False)
    tp1_hit: bool = False
    tp2_hit: bool = False
    remaining_qty: float = field(init=False)
    
    def __post_init__(self):
        self.current_sl = self.signal.stop_loss
        self.remaining_qty = self.quantity
    
    def move_sl_to_breakeven(self):
        """After TP1, move SL to breakeven."""
        self.current_sl = self.signal.entry_price
        self.tp1_hit = True
    
    def move_sl_to_tp1(self):
        """After TP2, move SL to TP1."""
        self.current_sl = self.signal.tp1
        self.tp2_hit = True
    
    def get_partial_qty_tp1(self) -> float:
        """40% of position for TP1."""
        return self.quantity * 0.4
    
    def get_partial_qty_tp2(self) -> float:
        """40% of position for TP2."""
        return self.quantity * 0.4
    
    def get_trailing_qty(self) -> float:
        """Remaining 20% for trailing."""
        return self.quantity * 0.2
    
    def should_exit(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should exit.
        
        Returns (should_exit, reason)
        """
        direction = self.signal.direction
        
        # Check SL hit
        if direction == TradeDirection.LONG:
            if current_price <= self.current_sl:
                return True, "STOP_LOSS"
        else:
            if current_price >= self.current_sl:
                return True, "STOP_LOSS"
        
        return False, ""
    
    def check_tp_levels(self, current_price: float) -> Optional[str]:
        """
        Check if any TP level is hit.
        
        Returns TP level name or None.
        """
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
