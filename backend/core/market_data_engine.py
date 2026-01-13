"""
Market Data Engine - Real-time OHLCV aggregation and indicators.

Processes incoming WebSocket data and maintains multi-timeframe candle data.
Calculates rolling VWAP, volume regimes, and spread monitoring.
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np

from config import TradingConstants
from exchange.websocket_manager import Candle, BookTicker, MarkPrice

logger = logging.getLogger(__name__)


@dataclass
class CandleBuffer:
    """Rolling buffer of candles for a specific timeframe."""
    interval: str
    max_size: int = 500
    candles: deque = field(default_factory=deque)
    
    def add(self, candle: Candle) -> None:
        """Add or update a candle in the buffer."""
        if self.candles and self.candles[-1].time == candle.time:
            # Update existing candle
            self.candles[-1] = candle
        else:
            # New candle
            self.candles.append(candle)
            if len(self.candles) > self.max_size:
                self.candles.popleft()
    
    def get_closes(self) -> np.ndarray:
        """Get array of close prices."""
        return np.array([float(c.close) for c in self.candles])
    
    def get_highs(self) -> np.ndarray:
        """Get array of high prices."""
        return np.array([float(c.high) for c in self.candles])
    
    def get_lows(self) -> np.ndarray:
        """Get array of low prices."""
        return np.array([float(c.low) for c in self.candles])
    
    def get_volumes(self) -> np.ndarray:
        """Get array of volumes."""
        return np.array([float(c.volume) for c in self.candles])
    
    def get_last_n(self, n: int) -> List[Candle]:
        """Get the last n candles."""
        return list(self.candles)[-n:] if self.candles else []
    
    @property
    def last(self) -> Optional[Candle]:
        """Get the most recent candle."""
        return self.candles[-1] if self.candles else None


@dataclass
class VWAPData:
    """VWAP calculation data."""
    value: Decimal
    upper_band: Decimal  # +1 std dev
    lower_band: Decimal  # -1 std dev
    slope: float  # Rate of change


@dataclass
class VolumeRegime:
    """Volume regime classification."""
    current_volume: Decimal
    average_volume: Decimal
    volume_ratio: float
    is_high_volume: bool  # > 1.5x average
    is_low_volume: bool   # < 0.5x average


@dataclass
class SpreadInfo:
    """Current spread information."""
    bid: Decimal
    ask: Decimal
    spread: Decimal
    spread_pct: float
    is_acceptable: bool  # Below max spread threshold


class MarketDataEngine:
    """
    Aggregates and analyzes real-time market data.
    
    Maintains multi-timeframe candles and calculates:
    - Rolling VWAP with bands
    - Volume regime detection
    - Spread monitoring
    - ATR for volatility
    """
    
    def __init__(self, symbol: str):
        """
        Initialize market data engine.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
        """
        self.symbol = symbol.upper()
        
        # Multi-timeframe candle buffers
        self.candles_1m = CandleBuffer(interval="1m")
        self.candles_5m = CandleBuffer(interval="5m")
        self.candles_15m = CandleBuffer(interval="15m")
        
        # Current price data
        self.mark_price: Optional[Decimal] = None
        self.index_price: Optional[Decimal] = None
        self.book_ticker: Optional[BookTicker] = None
        self.funding_rate: Optional[Decimal] = None
        
        # VWAP tracking
        self._vwap_tp_sum: float = 0.0  # Typical price * volume sum
        self._vwap_vol_sum: float = 0.0  # Volume sum
        self._vwap_history: deque = deque(maxlen=100)
        
        # Volume tracking
        self._volume_ma_period = 20
    
    def on_candle(self, candle: Candle) -> None:
        """Handle incoming candle update."""
        if candle.symbol.upper() != self.symbol:
            return
        
        # Route to appropriate buffer
        if candle.interval == "1m":
            self.candles_1m.add(candle)
            
            # Update VWAP on 1m closed candles
            if candle.is_closed:
                self._update_vwap(candle)
                
        elif candle.interval == "5m":
            self.candles_5m.add(candle)
        elif candle.interval == "15m":
            self.candles_15m.add(candle)
    
    def on_mark_price(self, data: MarkPrice) -> None:
        """Handle mark price update."""
        if data.symbol.upper() != self.symbol:
            return
        
        self.mark_price = data.price
        self.index_price = data.index_price
        self.funding_rate = data.funding_rate
    
    def on_book_ticker(self, data: BookTicker) -> None:
        """Handle book ticker update."""
        if data.symbol.upper() != self.symbol:
            return
        
        self.book_ticker = data
    
    def _update_vwap(self, candle: Candle) -> None:
        """Update rolling VWAP calculation."""
        # Typical price = (High + Low + Close) / 3
        typical_price = (float(candle.high) + float(candle.low) + float(candle.close)) / 3
        volume = float(candle.volume)
        
        self._vwap_tp_sum += typical_price * volume
        self._vwap_vol_sum += volume
        
        if self._vwap_vol_sum > 0:
            vwap = self._vwap_tp_sum / self._vwap_vol_sum
            self._vwap_history.append(vwap)
    
    def get_vwap(self) -> Optional[VWAPData]:
        """
        Calculate current VWAP with bands.
        
        Returns:
            VWAPData or None if insufficient data
        """
        if self._vwap_vol_sum == 0 or len(self._vwap_history) < 10:
            return None
        
        vwap = self._vwap_tp_sum / self._vwap_vol_sum
        
        # Calculate standard deviation for bands
        closes = self.candles_1m.get_closes()
        if len(closes) < 20:
            std = 0.0
        else:
            std = np.std(closes[-20:])
        
        # Calculate slope (rate of change)
        if len(self._vwap_history) >= 5:
            recent = list(self._vwap_history)[-5:]
            slope = (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0
        else:
            slope = 0.0
        
        return VWAPData(
            value=Decimal(str(vwap)),
            upper_band=Decimal(str(vwap + std)),
            lower_band=Decimal(str(vwap - std)),
            slope=slope,
        )
    
    def get_volume_regime(self, timeframe: str = "5m") -> Optional[VolumeRegime]:
        """
        Classify current volume regime.
        
        Args:
            timeframe: Candle timeframe to analyze
            
        Returns:
            VolumeRegime or None if insufficient data
        """
        buffer = self._get_buffer(timeframe)
        if not buffer or len(buffer.candles) < self._volume_ma_period:
            return None
        
        volumes = buffer.get_volumes()
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-self._volume_ma_period:])
        
        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        return VolumeRegime(
            current_volume=Decimal(str(current_vol)),
            average_volume=Decimal(str(avg_vol)),
            volume_ratio=ratio,
            is_high_volume=ratio >= 1.5,
            is_low_volume=ratio <= 0.5,
        )
    
    def get_spread_info(self) -> Optional[SpreadInfo]:
        """
        Get current spread information.
        
        Returns:
            SpreadInfo or None if no book ticker data
        """
        if not self.book_ticker:
            return None
        
        bid = self.book_ticker.bid_price
        ask = self.book_ticker.ask_price
        spread = ask - bid
        
        mid_price = (bid + ask) / 2
        spread_pct = float(spread / mid_price * 100) if mid_price > 0 else 0
        
        from config import RiskConstants
        is_acceptable = spread_pct <= RiskConstants.MAX_SPREAD_PCT
        
        return SpreadInfo(
            bid=bid,
            ask=ask,
            spread=spread,
            spread_pct=spread_pct,
            is_acceptable=is_acceptable,
        )
    
    def get_atr(self, period: int = 14, timeframe: str = "5m") -> Optional[Decimal]:
        """
        Calculate Average True Range.
        
        Args:
            period: ATR period
            timeframe: Candle timeframe
            
        Returns:
            ATR value or None if insufficient data
        """
        buffer = self._get_buffer(timeframe)
        if not buffer or len(buffer.candles) < period + 1:
            return None
        
        candles = buffer.get_last_n(period + 1)
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = float(candles[i].high)
            low = float(candles[i].low)
            prev_close = float(candles[i-1].close)
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        atr = np.mean(true_ranges)
        return Decimal(str(atr))
    
    def get_candle_efficiency(self, timeframe: str = "15m") -> Optional[float]:
        """
        Calculate candle efficiency (body/range ratio).
        
        High efficiency = directional move, low = indecision.
        
        Returns:
            Efficiency ratio (0.0-1.0)
        """
        buffer = self._get_buffer(timeframe)
        if not buffer or not buffer.last:
            return None
        
        candle = buffer.last
        body = abs(float(candle.close) - float(candle.open))
        range_ = float(candle.high) - float(candle.low)
        
        return body / range_ if range_ > 0 else 0.0
    
    def get_range_compression(self, lookback: int = 20, timeframe: str = "15m") -> Optional[float]:
        """
        Detect range compression (consolidation).
        
        Returns:
            Ratio of current range to average range
        """
        buffer = self._get_buffer(timeframe)
        if not buffer or len(buffer.candles) < lookback:
            return None
        
        highs = buffer.get_highs()
        lows = buffer.get_lows()
        
        ranges = highs - lows
        current_range = ranges[-1]
        avg_range = np.mean(ranges[-lookback:])
        
        return current_range / avg_range if avg_range > 0 else 1.0
    
    def _get_buffer(self, timeframe: str) -> Optional[CandleBuffer]:
        """Get candle buffer for timeframe."""
        if timeframe == "1m":
            return self.candles_1m
        elif timeframe == "5m":
            return self.candles_5m
        elif timeframe == "15m":
            return self.candles_15m
        return None
    
    @property
    def current_price(self) -> Optional[Decimal]:
        """Get the best available current price."""
        if self.mark_price:
            return self.mark_price
        if self.book_ticker:
            return (self.book_ticker.bid_price + self.book_ticker.ask_price) / 2
        if self.candles_1m.last:
            return self.candles_1m.last.close
        return None
    
    def reset_session(self) -> None:
        """Reset VWAP and session-specific calculations at session start."""
        self._vwap_tp_sum = 0.0
        self._vwap_vol_sum = 0.0
        self._vwap_history.clear()
        logger.info(f"Session reset for {self.symbol}")
