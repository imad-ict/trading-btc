"""
Liquidity Engine (5M) - Maps liquidity zones where stops reside.

Principle: Price moves to seek liquidity, not to respect indicators.
Trade ONLY after liquidity is consumed.

Mapped Levels:
- Equal highs/lows (swing points)
- Session high/low
- Prior day/week highs/lows
- Resting liquidity clusters

Enhancement:
- Weight by number of touches
- Prioritize recent liquidity (48h recency)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np

from config import TradingConstants
from core.market_data_engine import MarketDataEngine
from exchange.websocket_manager import Candle

logger = logging.getLogger(__name__)


class LiquidityType(str, Enum):
    """Types of liquidity zones."""
    EQUAL_HIGH = "EQUAL_HIGH"
    EQUAL_LOW = "EQUAL_LOW"
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"
    PRIOR_DAY_HIGH = "PDH"
    PRIOR_DAY_LOW = "PDL"
    PRIOR_WEEK_HIGH = "PWH"
    PRIOR_WEEK_LOW = "PWL"
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"


@dataclass
class LiquidityZone:
    """A mapped liquidity zone."""
    type: LiquidityType
    price: Decimal
    touch_count: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_touched_at: Optional[datetime] = None
    is_swept: bool = False
    swept_at: Optional[datetime] = None
    
    @property
    def age_hours(self) -> float:
        """Hours since zone was created."""
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds() / 3600
    
    @property
    def strength(self) -> float:
        """
        Calculate zone strength.
        
        Higher touches + recency = stronger.
        """
        # Base strength from touches
        touch_score = min(self.touch_count / 5, 1.0)  # Max at 5 touches
        
        # Recency decay (exponential)
        recency_hours = TradingConstants.LIQUIDITY_RECENCY_HOURS
        decay = np.exp(-self.age_hours / recency_hours)
        
        return touch_score * 0.6 + decay * 0.4


@dataclass
class SweepEvent:
    """A liquidity sweep event."""
    zone: LiquidityZone
    sweep_price: Decimal
    sweep_time: datetime
    volume_multiple: float
    reclaimed: bool = False
    reclaim_price: Optional[Decimal] = None


class LiquidityEngine:
    """
    Maps and tracks liquidity zones on 5M timeframe.
    
    Core principle: NO TRADE without liquidity being taken first.
    """
    
    def __init__(self, market_data: MarketDataEngine, tolerance_pct: float = 0.05):
        """
        Initialize liquidity engine.
        
        Args:
            market_data: MarketDataEngine instance  
            tolerance_pct: Price tolerance for zone matching (default 0.05%)
        """
        self.market_data = market_data
        self.tolerance_pct = tolerance_pct
        
        # Active liquidity zones
        self.zones: Dict[str, LiquidityZone] = {}  # key = f"{type}_{price}"
        
        # Recent sweep events
        self.pending_sweeps: List[SweepEvent] = []
        self.confirmed_sweeps: List[SweepEvent] = []
        
        # Session tracking
        self._session_high: Optional[Decimal] = None
        self._session_low: Optional[Decimal] = None
        self._session_start: Optional[datetime] = None
        
        # Swing tracking
        self._swing_lookback = 5  # Candles for swing detection
    
    def update(self) -> None:
        """
        Update liquidity zones from latest market data.
        
        Should be called on each 5M candle close.
        """
        self._update_swing_points()
        self._update_equal_levels()
        self._update_session_levels()
        self._check_for_sweeps()
        self._prune_old_zones()
    
    def _update_swing_points(self) -> None:
        """Detect and map swing highs/lows."""
        candles = self.market_data.candles_5m.get_last_n(self._swing_lookback * 2 + 1)
        
        if len(candles) < self._swing_lookback * 2 + 1:
            return
        
        # Check for swing high at middle candle
        mid_idx = self._swing_lookback
        mid_candle = candles[mid_idx]
        
        # Swing high: higher high than surrounding candles
        is_swing_high = all(
            float(mid_candle.high) > float(c.high)
            for i, c in enumerate(candles)
            if i != mid_idx
        )
        
        if is_swing_high:
            self._add_or_update_zone(
                LiquidityType.SWING_HIGH,
                mid_candle.high,
            )
        
        # Swing low: lower low than surrounding candles
        is_swing_low = all(
            float(mid_candle.low) < float(c.low)
            for i, c in enumerate(candles)
            if i != mid_idx
        )
        
        if is_swing_low:
            self._add_or_update_zone(
                LiquidityType.SWING_LOW,
                mid_candle.low,
            )
    
    def _update_equal_levels(self) -> None:
        """
        Detect equal highs/lows (liquidity clusters).
        
        Equal levels = where retail puts stops = where institutions hunt.
        """
        candles = self.market_data.candles_5m.get_last_n(50)
        
        if len(candles) < 10:
            return
        
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        
        # Find clusters of similar highs
        high_clusters = self._find_price_clusters(highs)
        for level, count in high_clusters:
            if count >= TradingConstants.MIN_LIQUIDITY_TOUCHES:
                self._add_or_update_zone(
                    LiquidityType.EQUAL_HIGH,
                    Decimal(str(level)),
                    touch_count=count,
                )
        
        # Find clusters of similar lows
        low_clusters = self._find_price_clusters(lows)
        for level, count in low_clusters:
            if count >= TradingConstants.MIN_LIQUIDITY_TOUCHES:
                self._add_or_update_zone(
                    LiquidityType.EQUAL_LOW,
                    Decimal(str(level)),
                    touch_count=count,
                )
    
    def _find_price_clusters(self, prices: List[float]) -> List[tuple]:
        """Find clusters of similar prices."""
        if not prices:
            return []
        
        sorted_prices = sorted(prices)
        clusters = []
        
        current_cluster = [sorted_prices[0]]
        
        for price in sorted_prices[1:]:
            # Within tolerance of cluster center
            cluster_center = np.mean(current_cluster)
            tolerance = cluster_center * (self.tolerance_pct / 100)
            
            if abs(price - cluster_center) <= tolerance:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= 2:
                    clusters.append((np.mean(current_cluster), len(current_cluster)))
                current_cluster = [price]
        
        # Don't forget last cluster
        if len(current_cluster) >= 2:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
        
        return clusters
    
    def _update_session_levels(self) -> None:
        """Track session high/low."""
        candles = self.market_data.candles_5m.get_last_n(100)
        
        if not candles:
            return
        
        # Simple session detection (last 4 hours of candles)
        session_candles = candles[-48:]  # 48 x 5min = 4 hours
        
        if session_candles:
            session_high = max(float(c.high) for c in session_candles)
            session_low = min(float(c.low) for c in session_candles)
            
            self._add_or_update_zone(
                LiquidityType.SESSION_HIGH,
                Decimal(str(session_high)),
            )
            self._add_or_update_zone(
                LiquidityType.SESSION_LOW,
                Decimal(str(session_low)),
            )
    
    def _add_or_update_zone(
        self,
        zone_type: LiquidityType,
        price: Decimal,
        touch_count: int = 1,
    ) -> None:
        """Add new zone or update existing if within tolerance."""
        # Check for existing zone within tolerance
        for key, zone in list(self.zones.items()):
            if zone.type == zone_type and not zone.is_swept:
                tolerance = float(zone.price) * (self.tolerance_pct / 100)
                if abs(float(price) - float(zone.price)) <= tolerance:
                    # Update existing zone
                    zone.touch_count = max(zone.touch_count, touch_count)
                    zone.last_touched_at = datetime.now(timezone.utc)
                    return
        
        # Add new zone
        key = f"{zone_type.value}_{price}"
        self.zones[key] = LiquidityZone(
            type=zone_type,
            price=price,
            touch_count=touch_count,
        )
    
    def _check_for_sweeps(self) -> None:
        """
        Check if any liquidity zones were swept.
        
        Sweep = price breaks past the level, taking out stops.
        """
        current_price = self.market_data.current_price
        if not current_price:
            return
        
        last_candle = self.market_data.candles_5m.last
        if not last_candle or not last_candle.is_closed:
            return
        
        volume_regime = self.market_data.get_volume_regime("5m")
        volume_multiple = volume_regime.volume_ratio if volume_regime else 1.0
        
        for key, zone in list(self.zones.items()):
            if zone.is_swept:
                continue
            
            # Check for sweep
            is_high_type = zone.type in [
                LiquidityType.EQUAL_HIGH,
                LiquidityType.SESSION_HIGH,
                LiquidityType.SWING_HIGH,
                LiquidityType.PRIOR_DAY_HIGH,
                LiquidityType.PRIOR_WEEK_HIGH,
            ]
            
            is_low_type = zone.type in [
                LiquidityType.EQUAL_LOW,
                LiquidityType.SESSION_LOW,
                LiquidityType.SWING_LOW,
                LiquidityType.PRIOR_DAY_LOW,
                LiquidityType.PRIOR_WEEK_LOW,
            ]
            
            # High sweep: wick above level, close below
            if is_high_type:
                if float(last_candle.high) > float(zone.price) and \
                   float(last_candle.close) < float(zone.price):
                    self._record_sweep(zone, last_candle.high, volume_multiple)
            
            # Low sweep: wick below level, close above
            if is_low_type:
                if float(last_candle.low) < float(zone.price) and \
                   float(last_candle.close) > float(zone.price):
                    self._record_sweep(zone, last_candle.low, volume_multiple)
    
    def _record_sweep(
        self,
        zone: LiquidityZone,
        sweep_price: Decimal,
        volume_multiple: float,
    ) -> None:
        """Record a sweep event."""
        zone.is_swept = True
        zone.swept_at = datetime.now(timezone.utc)
        
        sweep = SweepEvent(
            zone=zone,
            sweep_price=sweep_price,
            sweep_time=datetime.now(timezone.utc),
            volume_multiple=volume_multiple,
        )
        
        self.pending_sweeps.append(sweep)
        logger.info(f"SWEEP DETECTED: {zone.type.value} at {zone.price} (swept to {sweep_price})")
    
    def _prune_old_zones(self) -> None:
        """Remove zones older than recency threshold."""
        max_age_hours = TradingConstants.LIQUIDITY_RECENCY_HOURS * 2
        now = datetime.now(timezone.utc)
        
        for key in list(self.zones.keys()):
            zone = self.zones[key]
            if zone.age_hours > max_age_hours:
                del self.zones[key]
    
    def is_liquidity_taken(self) -> bool:
        """
        Check if any liquidity has been taken (sweep detected).
        
        This is a GATE for entry - no sweep = NO TRADE.
        """
        return len(self.pending_sweeps) > 0
    
    def get_pending_sweep(self) -> Optional[SweepEvent]:
        """Get the most recent pending sweep."""
        if self.pending_sweeps:
            return self.pending_sweeps[-1]
        return None
    
    def confirm_reclaim(self, sweep: SweepEvent, reclaim_price: Decimal) -> None:
        """Confirm that price has reclaimed after sweep."""
        sweep.reclaimed = True
        sweep.reclaim_price = reclaim_price
        
        # Move to confirmed
        self.pending_sweeps.remove(sweep)
        self.confirmed_sweeps.append(sweep)
        
        logger.info(f"RECLAIM CONFIRMED: {sweep.zone.type.value} reclaimed at {reclaim_price}")
    
    def get_active_zones(self) -> List[LiquidityZone]:
        """Get all active (non-swept) zones sorted by strength."""
        active = [z for z in self.zones.values() if not z.is_swept]
        return sorted(active, key=lambda z: z.strength, reverse=True)
    
    def get_nearest_zone(self, price: Decimal, direction: str) -> Optional[LiquidityZone]:
        """
        Get nearest liquidity zone in direction.
        
        Args:
            price: Current price
            direction: "LONG" or "SHORT"
        """
        active = self.get_active_zones()
        
        if direction == "LONG":
            # For longs, look above (targets) or below (stops)
            above = [z for z in active if float(z.price) > float(price)]
            if above:
                return min(above, key=lambda z: float(z.price))
        else:
            # For shorts, look below (targets) or above (stops)
            below = [z for z in active if float(z.price) < float(price)]
            if below:
                return max(below, key=lambda z: float(z.price))
        
        return None
    
    def clear_stale_sweeps(self) -> None:
        """Clear sweeps older than 15 minutes."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=15)
        
        self.pending_sweeps = [
            s for s in self.pending_sweeps
            if s.sweep_time > cutoff
        ]
