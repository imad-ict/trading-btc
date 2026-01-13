"""
Database Models - SQLAlchemy ORM definitions.

All trades and system state are persisted for auditing and recovery.
Every trade includes full explanation fields for institutional compliance.
"""
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, ForeignKey, Integer, 
    Numeric, String, Text, Index, JSON
)
from sqlalchemy.orm import DeclarativeBase, relationship


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ═══════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════

class TradeDirection(str, PyEnum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, PyEnum):
    """Trade lifecycle status."""
    PENDING = "PENDING"           # Signal generated, awaiting execution
    OPEN = "OPEN"                 # Position is open
    TP1_HIT = "TP1_HIT"          # First target hit
    TP2_HIT = "TP2_HIT"          # Second target hit
    CLOSED = "CLOSED"            # Position closed
    CANCELLED = "CANCELLED"       # Entry cancelled pre-execution


class TradeResult(str, PyEnum):
    """Trade outcome."""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BE"


class MarketState(str, PyEnum):
    """Market state classification."""
    EXPANSION = "EXPANSION"
    MANIPULATION = "MANIPULATION"
    NO_TRADE = "NO_TRADE"


class LiquidityType(str, PyEnum):
    """Liquidity zone types."""
    EQUAL_HIGH = "EQUAL_HIGH"
    EQUAL_LOW = "EQUAL_LOW"
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"
    PRIOR_DAY_HIGH = "PRIOR_DAY_HIGH"
    PRIOR_DAY_LOW = "PRIOR_DAY_LOW"
    PRIOR_WEEK_HIGH = "PRIOR_WEEK_HIGH"
    PRIOR_WEEK_LOW = "PRIOR_WEEK_LOW"


class BotStateEnum(str, PyEnum):
    """Bot state machine states."""
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    LIQUIDITY_MAPPED = "LIQUIDITY_MAPPED"
    SWEEP_DETECTED = "SWEEP_DETECTED"
    ENTRY_PENDING = "ENTRY_PENDING"
    POSITION_OPEN = "POSITION_OPEN"
    MANAGING_POSITION = "MANAGING_POSITION"
    EXITING = "EXITING"
    HALTED = "HALTED"             # Emergency stop or daily limit


# ═══════════════════════════════════════════════════════════════════
# CORE TABLES
# ═══════════════════════════════════════════════════════════════════

class Trade(Base):
    """
    Complete trade record with full explanation for institutional compliance.
    
    Every trade must be explainable: WHY was it taken, not just WHAT happened.
    """
    __tablename__ = "trades"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    # ─── Trade Identity ───
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(Enum(TradeDirection), nullable=False)
    
    # ─── Prices ───
    entry_price = Column(Numeric(18, 8), nullable=False)
    exit_price = Column(Numeric(18, 8), nullable=True)
    stop_loss = Column(Numeric(18, 8), nullable=False)
    tp1 = Column(Numeric(18, 8), nullable=True)
    tp2 = Column(Numeric(18, 8), nullable=True)
    tp3 = Column(Numeric(18, 8), nullable=True)
    
    # ─── Position ───
    position_size = Column(Numeric(18, 8), nullable=False)
    leverage = Column(Integer, nullable=False, default=5)
    
    # ─── Status & Result ───
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.PENDING)
    result = Column(Enum(TradeResult), nullable=True)
    
    # ─── P&L ───
    pnl_usd = Column(Numeric(18, 8), nullable=True)
    pnl_pct = Column(Numeric(8, 4), nullable=True)
    fees_paid = Column(Numeric(18, 8), nullable=True)
    
    # ─── EXPLANATION FIELDS (INSTITUTIONAL REQUIREMENT) ───
    liquidity_event = Column(Text, nullable=False)     # What liquidity was taken
    market_state = Column(String(20), nullable=False)  # EXPANSION/MANIPULATION/NO_TRADE
    entry_logic = Column(Text, nullable=False)         # Why entry was valid
    stop_logic = Column(Text, nullable=False)          # How SL was calculated
    target_logic = Column(Text, nullable=False)        # TP targeting logic
    risk_validation = Column(Text, nullable=False)     # Risk engine validation
    
    # ─── Partial Exits ───
    partial_exits = Column(JSON, nullable=True)  # {"tp1": {"time": ..., "price": ...}, ...}
    
    # ─── Technical Context at Entry ───
    vwap_at_entry = Column(Numeric(18, 8), nullable=True)
    atr_at_entry = Column(Numeric(18, 8), nullable=True)
    volume_ratio_at_entry = Column(Numeric(8, 4), nullable=True)
    
    # ─── Timestamps ───
    signal_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    entry_time = Column(DateTime, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # ─── Binance Order IDs ───
    entry_order_id = Column(String(50), nullable=True)
    sl_order_id = Column(String(50), nullable=True)
    tp_order_ids = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index("ix_trades_symbol_created", "symbol", "created_at"),
        Index("ix_trades_status", "status"),
    )


class LiquidityZone(Base):
    """
    Mapped liquidity levels where stops likely reside.
    
    Weighted by touch count and recency for prioritization.
    """
    __tablename__ = "liquidity_zones"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    
    symbol = Column(String(20), nullable=False, index=True)
    level_type = Column(Enum(LiquidityType), nullable=False)
    price = Column(Numeric(18, 8), nullable=False)
    
    # ─── Weighting ───
    touch_count = Column(Integer, nullable=False, default=1)
    strength = Column(Numeric(5, 2), nullable=True)  # Calculated importance
    
    # ─── Lifecycle ───
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_touched_at = Column(DateTime, nullable=True)
    invalidated_at = Column(DateTime, nullable=True)  # When swept/broken
    swept_at = Column(DateTime, nullable=True)        # When liquidity was taken
    
    is_active = Column(Boolean, nullable=False, default=True)
    
    __table_args__ = (
        Index("ix_liquidity_symbol_active", "symbol", "is_active"),
    )


class BotState(Base):
    """
    Persistent bot state for crash recovery and daily tracking.
    
    Only one row exists (id=1), updated continuously.
    """
    __tablename__ = "bot_state"
    
    id = Column(Integer, primary_key=True, default=1)
    
    # ─── State Machine ───
    state = Column(Enum(BotStateEnum), nullable=False, default=BotStateEnum.IDLE)
    active_trade_id = Column(String(36), ForeignKey("trades.id"), nullable=True)
    
    # ─── Daily Counters ───
    trades_today = Column(Integer, nullable=False, default=0)
    losses_today = Column(Integer, nullable=False, default=0)
    daily_pnl = Column(Numeric(18, 8), nullable=False, default=0)
    
    # ─── Streak Tracking (for adaptive risk) ───
    current_streak = Column(Integer, nullable=False, default=0)  # Positive = wins, negative = losses
    
    # ─── Session ───
    current_session = Column(String(20), nullable=True)  # LONDON, NY, ASIAN, NONE
    session_start_time = Column(DateTime, nullable=True)
    
    # ─── Safety ───
    is_halted = Column(Boolean, nullable=False, default=False)
    halt_reason = Column(String(100), nullable=True)
    
    # ─── Timestamps ───
    last_trade_time = Column(DateTime, nullable=True)
    daily_reset_at = Column(DateTime, nullable=True)  # When counters were last reset
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    active_trade = relationship("Trade", foreign_keys=[active_trade_id])


class DailySummary(Base):
    """
    End-of-day aggregated performance metrics.
    
    Used for equity curve and drawdown tracking.
    """
    __tablename__ = "daily_summary"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    
    # ─── Trading Metrics ───
    total_trades = Column(Integer, nullable=False, default=0)
    wins = Column(Integer, nullable=False, default=0)
    losses = Column(Integer, nullable=False, default=0)
    breakevens = Column(Integer, nullable=False, default=0)
    
    # ─── P&L ───
    gross_pnl = Column(Numeric(18, 8), nullable=False, default=0)
    net_pnl = Column(Numeric(18, 8), nullable=False, default=0)
    fees_paid = Column(Numeric(18, 8), nullable=False, default=0)
    
    # ─── Calculated Metrics ───
    win_rate = Column(Numeric(5, 2), nullable=True)
    profit_factor = Column(Numeric(8, 4), nullable=True)
    avg_win = Column(Numeric(18, 8), nullable=True)
    avg_loss = Column(Numeric(18, 8), nullable=True)
    
    # ─── Equity ───
    starting_balance = Column(Numeric(18, 8), nullable=False)
    ending_balance = Column(Numeric(18, 8), nullable=False)
    max_drawdown_pct = Column(Numeric(8, 4), nullable=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AuditLog(Base):
    """
    Complete audit trail of every bot decision.
    
    Used for debugging, compliance, and strategy refinement.
    """
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    event_type = Column(String(50), nullable=False, index=True)
    # Event types: TRADE_SIGNAL, TRADE_REJECTED, ENTRY_EXECUTED, EXIT_EXECUTED,
    #              RISK_VALIDATION, SESSION_CHANGE, STATE_TRANSITION, ERROR, etc.
    
    symbol = Column(String(20), nullable=True)
    trade_id = Column(String(36), nullable=True)
    
    details = Column(JSON, nullable=False)  # Full context as JSON
    
    __table_args__ = (
        Index("ix_audit_type_timestamp", "event_type", "timestamp"),
    )
