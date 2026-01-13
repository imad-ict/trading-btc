"""
Session Controller - Trading session time gates.

Institutional trading respects session timing:
- London: 08:00-12:00 UTC (highest FX volume)
- New York: 13:00-17:00 UTC (highest equities/crypto overlap)
- Asian: 00:00-04:00 UTC (optional, lower volume)

Dead hours are rejected automatically.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from config import SessionConstants

logger = logging.getLogger(__name__)


class TradingSession(str, Enum):
    """Trading session identifiers."""
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    ASIAN = "ASIAN"
    OVERLAP = "OVERLAP"  # London/NY overlap
    DEAD = "DEAD"       # Outside active sessions


@dataclass
class SessionInfo:
    """Current session information."""
    session: TradingSession
    is_active: bool
    hours_until_next: float
    next_session: TradingSession
    session_start_utc: int
    session_end_utc: int


class SessionController:
    """
    Controls trading based on session timing.
    
    Principle: Institutions trade during liquid sessions only.
    Dead hours = NO TRADE by default.
    """
    
    def __init__(self, allow_asian: bool = False):
        """
        Initialize session controller.
        
        Args:
            allow_asian: Whether to allow Asian session trading
        """
        self.allow_asian = allow_asian or SessionConstants.ASIAN_ENABLED
        self._current_session: Optional[TradingSession] = None
    
    def get_current_session(self, current_time: Optional[datetime] = None) -> SessionInfo:
        """
        Determine the current trading session.
        
        Args:
            current_time: Optional datetime, defaults to UTC now
            
        Returns:
            SessionInfo with current session details
        """
        now = current_time or datetime.now(timezone.utc)
        hour = now.hour
        
        # Check London session (08:00 - 12:00 UTC)
        if SessionConstants.LONDON_START_UTC <= hour < SessionConstants.LONDON_END_UTC:
            return SessionInfo(
                session=TradingSession.LONDON,
                is_active=True,
                hours_until_next=0,
                next_session=TradingSession.OVERLAP,
                session_start_utc=SessionConstants.LONDON_START_UTC,
                session_end_utc=SessionConstants.LONDON_END_UTC,
            )
        
        # Check NY session (13:00 - 17:00 UTC)
        if SessionConstants.NY_START_UTC <= hour < SessionConstants.NY_END_UTC:
            return SessionInfo(
                session=TradingSession.NEW_YORK,
                is_active=True,
                hours_until_next=0,
                next_session=TradingSession.ASIAN if self.allow_asian else TradingSession.LONDON,
                session_start_utc=SessionConstants.NY_START_UTC,
                session_end_utc=SessionConstants.NY_END_UTC,
            )
        
        # Check London/NY overlap (12:00 - 13:00 UTC)
        if hour == 12:
            return SessionInfo(
                session=TradingSession.OVERLAP,
                is_active=True,
                hours_until_next=0,
                next_session=TradingSession.NEW_YORK,
                session_start_utc=12,
                session_end_utc=13,
            )
        
        # Check Asian session (00:00 - 04:00 UTC)
        if self.allow_asian and SessionConstants.ASIAN_START_UTC <= hour < SessionConstants.ASIAN_END_UTC:
            return SessionInfo(
                session=TradingSession.ASIAN,
                is_active=True,
                hours_until_next=0,
                next_session=TradingSession.LONDON,
                session_start_utc=SessionConstants.ASIAN_START_UTC,
                session_end_utc=SessionConstants.ASIAN_END_UTC,
            )
        
        # Dead hours - calculate next session
        next_session, hours_until = self._calculate_next_session(hour)
        
        return SessionInfo(
            session=TradingSession.DEAD,
            is_active=False,
            hours_until_next=hours_until,
            next_session=next_session,
            session_start_utc=0,
            session_end_utc=0,
        )
    
    def _calculate_next_session(self, current_hour: int) -> tuple[TradingSession, float]:
        """Calculate the next active session and hours until it starts."""
        
        if self.allow_asian:
            # After NY, next is Asian
            if current_hour >= SessionConstants.NY_END_UTC:
                hours_until = (24 - current_hour) + SessionConstants.ASIAN_START_UTC
                return TradingSession.ASIAN, hours_until
            
            # After Asian, before London
            if SessionConstants.ASIAN_END_UTC <= current_hour < SessionConstants.LONDON_START_UTC:
                return TradingSession.LONDON, SessionConstants.LONDON_START_UTC - current_hour
        
        # Standard: After NY, next is London
        if current_hour >= SessionConstants.NY_END_UTC:
            hours_until = (24 - current_hour) + SessionConstants.LONDON_START_UTC
            return TradingSession.LONDON, hours_until
        
        # Before London
        if current_hour < SessionConstants.LONDON_START_UTC:
            return TradingSession.LONDON, SessionConstants.LONDON_START_UTC - current_hour
        
        # Between London and NY (13:00 UTC)
        if SessionConstants.LONDON_END_UTC < current_hour < SessionConstants.NY_START_UTC:
            return TradingSession.NEW_YORK, SessionConstants.NY_START_UTC - current_hour
        
        return TradingSession.LONDON, 24  # Fallback
    
    def is_trading_allowed(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if trading is allowed at the current time.
        
        Args:
            current_time: Optional datetime, defaults to UTC now
            
        Returns:
            True if within an active session
        """
        session_info = self.get_current_session(current_time)
        return session_info.is_active
    
    def get_session_state_string(self) -> str:
        """Get a human-readable session state for logging/UI."""
        info = self.get_current_session()
        
        if info.is_active:
            return f"🟢 {info.session.value} SESSION ACTIVE"
        else:
            return f"🔴 DEAD HOURS - {info.next_session.value} in {info.hours_until_next:.1f}h"
    
    def validate_entry(self) -> tuple[bool, str]:
        """
        Validate if entry is allowed based on session.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        info = self.get_current_session()
        
        if not info.is_active:
            return False, f"Outside active session. {info.next_session.value} starts in {info.hours_until_next:.1f}h"
        
        return True, f"Session active: {info.session.value}"
