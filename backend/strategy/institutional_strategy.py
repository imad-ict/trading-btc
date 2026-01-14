"""
Institutional Strategy v4 - Liquidity-First

CORE PRINCIPLES:
1. Trade LIQUIDITY, not indicators
2. Market context FIRST (no CHOP trading)
3. Structural SL (move on new HL/LH, not profit)
4. Destination-based TP (VWAP, opposite pool, FVG)

ENTRY HIERARCHY:
1. Market Regime Check (EXPANSION/MANIPULATION only)
2. Liquidity Sweep Detection (MANDATORY)
3. Structure Confirmation (reclaim/rejection)
4. Filters: VWAP alignment, RSI filter
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    EXPANSION = "expansion"      # Trending, tradable
    MANIPULATION = "manipulation"  # Sweep reversal, tradable
    CHOP = "chop"                # Balanced, NO TRADING


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(Enum):
    LIQUIDITY_SWEEP = "liquidity_sweep"
    ORDER_BLOCK = "order_block"


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
class LiquiditySweep:
    """Detected liquidity sweep event."""
    price_level: float
    sweep_type: str  # "low" or "high"
    swept_at: float  # timestamp
    reclaimed: bool = False
    reclaim_candle: Optional[Candle] = None


@dataclass
class SignalQuality:
    """Signal quality scoring for entry validation."""
    sweep_quality: int = 0      # 0-25: depth of sweep, wick rejection
    displacement: int = 0       # 0-25: strength of reclaim candle
    volume_expansion: int = 0   # 0-25: volume vs average
    session_timing: int = 0     # 0-25: London/NY = max, Asian = min
    
    @property
    def total_score(self) -> int:
        return self.sweep_quality + self.displacement + self.volume_expansion + self.session_timing
    
    @property
    def is_valid(self) -> bool:
        return self.total_score >= 60


@dataclass
class TradeSignal:
    symbol: str
    direction: TradeDirection
    entry_type: EntryType
    entry_price: float
    stop_loss: float
    tp1: float  # VWAP/mean
    tp2: float  # Opposite liquidity
    tp3: float  # FVG/range boundary
    vwap: float
    regime: MarketRegime
    quality_score: int
    reason: str
    
    @property
    def risk_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price
    
    @property
    def is_valid(self) -> bool:
        # Valid risk: 0.1% to 1%
        return 0.001 <= self.risk_distance <= 0.01


class InstitutionalStrategy:
    """
    Liquidity-First Institutional Strategy v4
    
    Rules:
    1. ONLY trade EXPANSION or MANIPULATION regimes
    2. REQUIRE liquidity sweep + reclaim before entry
    3. Indicators (VWAP, RSI) are FILTERS only
    4. SL moves on STRUCTURE, not profit
    """
    
    def __init__(self, max_positions: int = 3):
        # Candle storage
        self.candles_1m: Dict[str, deque] = {}
        self.candles_5m: Dict[str, deque] = {}
        self.candles_15m: Dict[str, deque] = {}
        
        # Liquidity tracking
        self.swing_highs: Dict[str, List[float]] = {}
        self.swing_lows: Dict[str, List[float]] = {}
        self.recent_sweeps: Dict[str, List[LiquiditySweep]] = {}
        
        # Session levels
        self.session_high: Dict[str, float] = {}
        self.session_low: Dict[str, float] = {}
        
        # Technical indicators (FILTERS ONLY)
        self.vwap: Dict[str, float] = {}
        self.vwap_data: Dict[str, List[Tuple[float, float]]] = {}
        self.rsi: Dict[str, float] = {}
        
        # Market regime
        self.regime: Dict[str, MarketRegime] = {}
        
        # Configuration
        self.max_positions = max_positions
        self.rsi_period = 14
        self.quality_threshold = 60
        self.max_candles = 100
        self.sl_buffer_pct = 0.001  # 0.1% buffer
    
    def get_diagnostics(self, symbol: str) -> Dict:
        """Get diagnostic info for a symbol."""
        return {
            "symbol": symbol,
            "regime": self.regime.get(symbol, MarketRegime.CHOP).value,
            "candles_5m": len(self.candles_5m.get(symbol, [])),
            "swing_highs": len(self.swing_highs.get(symbol, [])),
            "swing_lows": len(self.swing_lows.get(symbol, [])),
            "recent_sweeps": len([s for s in self.recent_sweeps.get(symbol, []) if not s.reclaimed]),
            "rsi": round(self.rsi.get(symbol, 50), 1),
            "vwap": self.vwap.get(symbol),
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
            self._update_session_levels(symbol, candle)
        
        if timeframe == "5m":
            self._update_swing_points(symbol)
            self._detect_liquidity_sweeps(symbol, candle)
            self._update_market_regime(symbol)
    
    def _update_vwap(self, symbol: str, candle: Candle):
        """Update VWAP calculation."""
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
            self.rsi[symbol] = 50
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
    
    def _update_session_levels(self, symbol: str, candle: Candle):
        """Track session high/low (last 4 hours)."""
        candles = list(self.candles_1m.get(symbol, []))[-240:]  # 4 hours
        if not candles:
            return
        
        self.session_high[symbol] = max(c.high for c in candles)
        self.session_low[symbol] = min(c.low for c in candles)
    
    def _update_swing_points(self, symbol: str):
        """Track swing highs and lows from 5M for liquidity levels."""
        candles = list(self.candles_5m.get(symbol, []))
        if len(candles) < 5:
            return
        
        if symbol not in self.swing_highs:
            self.swing_highs[symbol] = []
        if symbol not in self.swing_lows:
            self.swing_lows[symbol] = []
        
        # Look for swing points (3-candle pattern)
        for i in range(2, len(candles) - 2):
            # Swing high: higher than 2 candles before and after
            if (candles[i].high > candles[i-1].high and 
                candles[i].high > candles[i-2].high and
                candles[i].high > candles[i+1].high and 
                candles[i].high > candles[i+2].high):
                level = candles[i].high
                if level not in self.swing_highs[symbol]:
                    self.swing_highs[symbol].append(level)
            
            # Swing low: lower than 2 candles before and after
            if (candles[i].low < candles[i-1].low and 
                candles[i].low < candles[i-2].low and
                candles[i].low < candles[i+1].low and 
                candles[i].low < candles[i+2].low):
                level = candles[i].low
                if level not in self.swing_lows[symbol]:
                    self.swing_lows[symbol].append(level)
        
        # Keep only recent swing points
        self.swing_highs[symbol] = self.swing_highs[symbol][-15:]
        self.swing_lows[symbol] = self.swing_lows[symbol][-15:]
    
    def _detect_liquidity_sweeps(self, symbol: str, candle: Candle):
        """Detect when price sweeps a liquidity level and reclaims."""
        if symbol not in self.recent_sweeps:
            self.recent_sweeps[symbol] = []
        
        swing_highs = self.swing_highs.get(symbol, [])
        swing_lows = self.swing_lows.get(symbol, [])
        session_high = self.session_high.get(symbol)
        session_low = self.session_low.get(symbol)
        
        tolerance = candle.close * 0.0005  # 0.05% tolerance
        
        # Check for LOW sweep (bullish setup)
        for low_level in swing_lows + ([session_low] if session_low else []):
            if low_level is None:
                continue
            # Price went below level (sweep) but closed above (reclaim)
            if candle.low < low_level - tolerance and candle.close > low_level:
                # Validate rejection wick
                if candle.lower_wick > candle.body_size * 0.5:
                    sweep = LiquiditySweep(
                        price_level=low_level,
                        sweep_type="low",
                        swept_at=candle.timestamp,
                        reclaimed=True,
                        reclaim_candle=candle
                    )
                    self.recent_sweeps[symbol].append(sweep)
        
        # Check for HIGH sweep (bearish setup)
        for high_level in swing_highs + ([session_high] if session_high else []):
            if high_level is None:
                continue
            # Price went above level (sweep) but closed below (reclaim)
            if candle.high > high_level + tolerance and candle.close < high_level:
                # Validate rejection wick
                if candle.upper_wick > candle.body_size * 0.5:
                    sweep = LiquiditySweep(
                        price_level=high_level,
                        sweep_type="high",
                        swept_at=candle.timestamp,
                        reclaimed=True,
                        reclaim_candle=candle
                    )
                    self.recent_sweeps[symbol].append(sweep)
        
        # Clean old sweeps (keep last 10)
        self.recent_sweeps[symbol] = self.recent_sweeps[symbol][-10:]
    
    def _update_market_regime(self, symbol: str):
        """
        Classify market regime.
        
        EXPANSION: Trending, large candles, low overlap
        MANIPULATION: Recent sweep + reversal
        CHOP: Flat VWAP, high candle overlap
        """
        candles = list(self.candles_5m.get(symbol, []))
        if len(candles) < 10:
            self.regime[symbol] = MarketRegime.CHOP
            return
        
        recent = candles[-10:]
        
        # Calculate candle overlap percentage
        overlap_count = 0
        for i in range(1, len(recent)):
            prev_range = (recent[i-1].low, recent[i-1].high)
            curr_range = (recent[i].low, recent[i].high)
            overlap = max(0, min(prev_range[1], curr_range[1]) - max(prev_range[0], curr_range[0]))
            prev_size = prev_range[1] - prev_range[0]
            if prev_size > 0 and overlap / prev_size > 0.7:
                overlap_count += 1
        
        overlap_pct = overlap_count / (len(recent) - 1)
        
        # Calculate VWAP slope (flat = chop)
        vwap_values = []
        for i, c in enumerate(recent):
            tp = (c.high + c.low + c.close) / 3
            vwap_values.append(tp)
        
        if len(vwap_values) >= 2:
            vwap_change = abs(vwap_values[-1] - vwap_values[0]) / vwap_values[0]
        else:
            vwap_change = 0
        
        # Check for recent liquidity sweep (manipulation)
        recent_sweeps = [s for s in self.recent_sweeps.get(symbol, []) if s.reclaimed]
        has_recent_sweep = len(recent_sweeps) > 0
        
        # Regime classification
        if has_recent_sweep:
            self.regime[symbol] = MarketRegime.MANIPULATION
        elif overlap_pct > 0.6 and vwap_change < 0.002:
            self.regime[symbol] = MarketRegime.CHOP
        else:
            self.regime[symbol] = MarketRegime.EXPANSION
    
    def _calculate_signal_quality(self, symbol: str, sweep: LiquiditySweep, 
                                   candle: Candle) -> SignalQuality:
        """Calculate signal quality score."""
        quality = SignalQuality()
        
        # Sweep quality (0-25): depth of sweep + wick rejection
        if sweep.reclaim_candle:
            wick_ratio = (sweep.reclaim_candle.lower_wick if sweep.sweep_type == "low" 
                         else sweep.reclaim_candle.upper_wick) / max(sweep.reclaim_candle.total_range, 0.01)
            quality.sweep_quality = min(25, int(wick_ratio * 50))
        
        # Displacement (0-25): strength of reclaim candle
        if candle.body_pct > 0.6:
            quality.displacement = 25
        elif candle.body_pct > 0.4:
            quality.displacement = 15
        else:
            quality.displacement = 5
        
        # Volume (0-25): simplified - use body size as proxy
        avg_body = np.mean([c.body_size for c in list(self.candles_5m.get(symbol, []))[-20:]])
        if avg_body > 0:
            vol_ratio = candle.body_size / avg_body
            quality.volume_expansion = min(25, int(vol_ratio * 15))
        
        # Session timing (0-25): London/NY overlap = best
        # Simplified: always give 15 for now (can add actual time check)
        quality.session_timing = 15
        
        return quality
    
    def generate_signal(self, symbol: str, current_candle: Candle) -> Optional[TradeSignal]:
        """
        Generate trading signal with LIQUIDITY-FIRST logic.
        
        Entry ONLY when:
        1. Market regime is EXPANSION or MANIPULATION (NOT CHOP)
        2. Liquidity has been swept and reclaimed
        3. Quality score >= 60
        4. VWAP and RSI filters pass
        """
        regime = self.regime.get(symbol, MarketRegime.CHOP)
        vwap = self.vwap.get(symbol)
        rsi = self.rsi.get(symbol, 50)
        
        # RULE 1: NO TRADING IN CHOP
        if regime == MarketRegime.CHOP:
            return None
        
        if not vwap:
            return None
        
        # RULE 2: Find valid liquidity sweep
        recent_sweeps = [s for s in self.recent_sweeps.get(symbol, []) 
                        if s.reclaimed and s.reclaim_candle is not None]
        
        if not recent_sweeps:
            return None  # NO SWEEP = NO TRADE
        
        # Get most recent sweep
        sweep = recent_sweeps[-1]
        
        # RULE 3: Calculate signal quality
        quality = self._calculate_signal_quality(symbol, sweep, current_candle)
        
        if not quality.is_valid:
            return None  # Quality too low
        
        # RULE 4: VWAP and RSI filters
        vwap_dist = (current_candle.close - vwap) / vwap
        
        if sweep.sweep_type == "low":  # LONG setup
            # VWAP filter: price should be near or above VWAP
            if vwap_dist < -0.005:  # Too far below VWAP
                return None
            # RSI filter: not extremely overbought
            if rsi > 75:
                return None
            
            return self._create_long_signal(symbol, sweep, current_candle, 
                                            regime, quality.total_score)
        
        elif sweep.sweep_type == "high":  # SHORT setup
            # VWAP filter: price should be near or below VWAP
            if vwap_dist > 0.005:  # Too far above VWAP
                return None
            # RSI filter: not extremely oversold
            if rsi < 25:
                return None
            
            return self._create_short_signal(symbol, sweep, current_candle,
                                             regime, quality.total_score)
        
        return None
    
    def _create_long_signal(self, symbol: str, sweep: LiquiditySweep,
                            candle: Candle, regime: MarketRegime,
                            quality_score: int) -> Optional[TradeSignal]:
        """Create LONG signal with destination-based TP."""
        vwap = self.vwap.get(symbol, candle.close)
        
        # SL: Inside the wick midpoint (not exact low)
        wick_low = sweep.reclaim_candle.low if sweep.reclaim_candle else candle.low
        sl = (wick_low + candle.low) / 2 * (1 - self.sl_buffer_pct)
        
        # Destination-based TP
        risk = candle.close - sl
        swing_highs = self.swing_highs.get(symbol, [])
        
        # TP1: VWAP or mean (whichever is closer target)
        tp1 = max(vwap, candle.close + risk * 1.5)
        
        # TP2: Nearest swing high (opposite liquidity)
        higher_highs = [h for h in swing_highs if h > candle.close]
        tp2 = min(higher_highs) if higher_highs else candle.close + risk * 2.5
        
        # TP3: Session high or range boundary
        session_high = self.session_high.get(symbol, candle.close + risk * 4)
        tp3 = max(session_high, tp2 + risk)
        
        signal = TradeSignal(
            symbol=symbol,
            direction=TradeDirection.LONG,
            entry_type=EntryType.LIQUIDITY_SWEEP,
            entry_price=candle.close,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            vwap=vwap,
            regime=regime,
            quality_score=quality_score,
            reason=f"Liquidity Sweep LONG | Low swept @ ${sweep.price_level:,.0f} | Score: {quality_score}"
        )
        
        if signal.is_valid:
            # Remove used sweep
            self.recent_sweeps[symbol] = [s for s in self.recent_sweeps[symbol] if s != sweep]
            return signal
        return None
    
    def _create_short_signal(self, symbol: str, sweep: LiquiditySweep,
                             candle: Candle, regime: MarketRegime,
                             quality_score: int) -> Optional[TradeSignal]:
        """Create SHORT signal with destination-based TP."""
        vwap = self.vwap.get(symbol, candle.close)
        
        # SL: Inside the wick midpoint (not exact high)
        wick_high = sweep.reclaim_candle.high if sweep.reclaim_candle else candle.high
        sl = (wick_high + candle.high) / 2 * (1 + self.sl_buffer_pct)
        
        # Destination-based TP
        risk = sl - candle.close
        swing_lows = self.swing_lows.get(symbol, [])
        
        # TP1: VWAP or mean
        tp1 = min(vwap, candle.close - risk * 1.5)
        
        # TP2: Nearest swing low (opposite liquidity)
        lower_lows = [l for l in swing_lows if l < candle.close]
        tp2 = max(lower_lows) if lower_lows else candle.close - risk * 2.5
        
        # TP3: Session low or range boundary
        session_low = self.session_low.get(symbol, candle.close - risk * 4)
        tp3 = min(session_low, tp2 - risk)
        
        signal = TradeSignal(
            symbol=symbol,
            direction=TradeDirection.SHORT,
            entry_type=EntryType.LIQUIDITY_SWEEP,
            entry_price=candle.close,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            vwap=vwap,
            regime=regime,
            quality_score=quality_score,
            reason=f"Liquidity Sweep SHORT | High swept @ ${sweep.price_level:,.0f} | Score: {quality_score}"
        )
        
        if signal.is_valid:
            # Remove used sweep
            self.recent_sweeps[symbol] = [s for s in self.recent_sweeps[symbol] if s != sweep]
            return signal
        return None


@dataclass
class ActivePosition:
    """
    Tracks an active position with STRUCTURAL SL management.
    
    SL moves ONLY on new structure:
    - LONG: New Higher Low formed
    - SHORT: New Lower High formed
    """
    signal: TradeSignal
    order_id: str
    quantity: float
    entry_time: datetime
    
    current_sl: float = field(init=False)
    last_structure_level: float = field(init=False)
    tp1_hit: bool = False
    tp2_hit: bool = False
    remaining_qty: float = field(init=False)
    
    def __post_init__(self):
        self.current_sl = self.signal.stop_loss
        self.last_structure_level = self.signal.stop_loss
        self.remaining_qty = self.quantity
    
    def check_structure_for_sl_move(self, candles: List[Candle]) -> bool:
        """
        Check if new structure formed for SL movement.
        Returns True if SL was moved.
        """
        if len(candles) < 3:
            return False
        
        if self.signal.direction == TradeDirection.LONG:
            # Look for Higher Low
            recent_lows = [c.low for c in candles[-5:]]
            min_low = min(recent_lows[:-1]) if len(recent_lows) > 1 else recent_lows[0]
            current_low = recent_lows[-1]
            
            # If current low is higher than recent lows and above our SL
            if current_low > min_low and current_low > self.current_sl:
                new_sl = current_low * 0.999  # Small buffer
                if new_sl > self.current_sl:
                    self.current_sl = new_sl
                    self.last_structure_level = current_low
                    return True
        
        else:  # SHORT
            # Look for Lower High
            recent_highs = [c.high for c in candles[-5:]]
            max_high = max(recent_highs[:-1]) if len(recent_highs) > 1 else recent_highs[0]
            current_high = recent_highs[-1]
            
            # If current high is lower than recent highs and below our SL
            if current_high < max_high and current_high < self.current_sl:
                new_sl = current_high * 1.001  # Small buffer
                if new_sl < self.current_sl:
                    self.current_sl = new_sl
                    self.last_structure_level = current_high
                    return True
        
        return False
    
    def get_partial_qty_tp1(self) -> float:
        return self.quantity * 0.5
    
    def get_partial_qty_tp2(self) -> float:
        return self.quantity * 0.3
    
    def should_exit(self, current_price: float) -> Tuple[bool, str]:
        """Check if SL hit."""
        if self.signal.direction == TradeDirection.LONG:
            if current_price <= self.current_sl:
                return True, "STOP_LOSS"
        else:
            if current_price >= self.current_sl:
                return True, "STOP_LOSS"
        return False, ""
    
    def check_tp_levels(self, current_price: float) -> Optional[str]:
        """Check TP levels."""
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
