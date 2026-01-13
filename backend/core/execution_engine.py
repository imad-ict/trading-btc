"""
Execution Engine - Order execution with safety controls.

Safety Features:
- Isolated margin only (never cross)
- Max 5× leverage enforcement
- Order reconciliation
- Auto-flatten on crash
- Circuit breaker integration
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from config import RiskConstants
from exchange.binance_client import BinanceClient, OrderResult, PositionInfo
from strategy.entry_engine import EntrySignal

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    success: bool
    entry_order: Optional[OrderResult] = None
    sl_order: Optional[OrderResult] = None
    tp_orders: List[OrderResult] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.tp_orders is None:
            self.tp_orders = []


class CircuitBreaker:
    """
    Circuit breaker for execution safety.
    
    Trips on consecutive failures to prevent runaway losses.
    """
    
    def __init__(self, max_failures: int = 3, reset_minutes: int = 15):
        self.max_failures = max_failures
        self.reset_minutes = reset_minutes
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._is_open = False
    
    def record_failure(self) -> None:
        """Record an execution failure."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        
        if self._failure_count >= self.max_failures:
            self._is_open = True
            logger.error(f"CIRCUIT BREAKER OPEN: {self._failure_count} consecutive failures")
    
    def record_success(self) -> None:
        """Record a successful execution."""
        self._failure_count = 0
        self._is_open = False
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._is_open and self._last_failure_time:
            # Auto-reset after timeout
            elapsed = (datetime.now(timezone.utc) - self._last_failure_time).seconds / 60
            if elapsed >= self.reset_minutes:
                self._is_open = False
                self._failure_count = 0
                logger.info("Circuit breaker auto-reset")
        
        return self._is_open
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self._is_open = False
        self._failure_count = 0
        logger.info("Circuit breaker manually reset")


class ExecutionEngine:
    """
    Handles order execution with safety controls.
    
    Enforces:
    - Isolated margin
    - Max leverage cap
    - Order reconciliation
    - Circuit breakers
    """
    
    def __init__(self, client: BinanceClient):
        """
        Initialize execution engine.
        
        Args:
            client: BinanceClient instance
        """
        self.client = client
        self.circuit_breaker = CircuitBreaker()
        
        # Track active positions
        self._active_positions: Dict[str, PositionInfo] = {}
        
        # Trade in progress flag
        self._execution_in_progress = False
    
    async def prepare_symbol(self, symbol: str) -> bool:
        """
        Prepare symbol for trading.
        
        Sets margin type and leverage.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if successful
        """
        try:
            # Set isolated margin
            await self.client.set_margin_type(symbol, RiskConstants.MARGIN_TYPE)
            logger.info(f"Margin type set to {RiskConstants.MARGIN_TYPE} for {symbol}")
            
            # Set leverage
            await self.client.set_leverage(symbol, RiskConstants.MAX_LEVERAGE)
            logger.info(f"Leverage set to {RiskConstants.MAX_LEVERAGE}x for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare {symbol}: {e}")
            return False
    
    async def execute_trade(
        self,
        signal: EntrySignal,
        position_size: Decimal,
        tp_levels: List[tuple],  # [(price, size_pct), ...]
    ) -> ExecutionResult:
        """
        Execute a complete trade with entry, SL, and TPs.
        
        Args:
            signal: Validated entry signal
            position_size: Position size from risk engine
            tp_levels: List of (price, size_pct) tuples
            
        Returns:
            ExecutionResult with order details
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            return ExecutionResult(
                success=False,
                error="Circuit breaker is open - trading halted",
            )
        
        if self._execution_in_progress:
            return ExecutionResult(
                success=False,
                error="Another execution is in progress",
            )
        
        self._execution_in_progress = True
        
        try:
            # 1. Prepare symbol (margin + leverage)
            prepared = await self.prepare_symbol(signal.symbol)
            if not prepared:
                raise Exception("Failed to prepare symbol")
            
            # 2. Place entry order
            entry_side = "BUY" if signal.direction == "LONG" else "SELL"
            entry_order = await self.client.create_market_order(
                symbol=signal.symbol,
                side=entry_side,
                quantity=position_size,
            )
            
            if entry_order.status != "FILLED":
                raise Exception(f"Entry order not filled: {entry_order.status}")
            
            logger.info(f"ENTRY FILLED: {entry_order.side} {entry_order.quantity} @ {entry_order.avg_price}")
            
            # 3. Place stop loss
            sl_side = "SELL" if signal.direction == "LONG" else "BUY"
            sl_order = await self.client.create_stop_loss(
                symbol=signal.symbol,
                side=sl_side,
                quantity=position_size,
                stop_price=signal.stop_loss,
            )
            
            logger.info(f"SL PLACED: {sl_order.price}")
            
            # 4. Place take profits
            tp_orders = []
            remaining_size = position_size
            
            for tp_price, size_pct in tp_levels:
                tp_size = position_size * Decimal(str(size_pct))
                tp_size = min(tp_size, remaining_size)
                
                if tp_size <= 0:
                    continue
                
                try:
                    tp_order = await self.client.create_take_profit(
                        symbol=signal.symbol,
                        side=sl_side,  # Same as SL side
                        quantity=tp_size,
                        take_profit_price=tp_price,
                    )
                    tp_orders.append(tp_order)
                    remaining_size -= tp_size
                    logger.info(f"TP PLACED: {tp_order.price}")
                except Exception as e:
                    logger.warning(f"Failed to place TP at {tp_price}: {e}")
            
            # Success
            self.circuit_breaker.record_success()
            
            # Update active positions
            self._active_positions[signal.symbol] = PositionInfo(
                symbol=signal.symbol,
                side=signal.direction,
                size=position_size,
                entry_price=entry_order.avg_price or signal.entry_price,
                unrealized_pnl=Decimal("0"),
                leverage=RiskConstants.MAX_LEVERAGE,
                margin_type=RiskConstants.MARGIN_TYPE,
                liquidation_price=None,
            )
            
            return ExecutionResult(
                success=True,
                entry_order=entry_order,
                sl_order=sl_order,
                tp_orders=tp_orders,
            )
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self.circuit_breaker.record_failure()
            
            # Attempt to cancel any partial orders
            try:
                await self.client.cancel_all_orders(signal.symbol)
            except Exception:
                pass
            
            return ExecutionResult(
                success=False,
                error=str(e),
            )
        
        finally:
            self._execution_in_progress = False
    
    async def modify_stop_loss(
        self,
        symbol: str,
        current_sl_order_id: str,
        new_sl_price: Decimal,
        quantity: Decimal,
        direction: str,
    ) -> Optional[OrderResult]:
        """
        Modify stop loss order (cancel and replace).
        
        Used for trailing stop and moving to breakeven.
        """
        try:
            # Cancel existing SL
            await self.client.cancel_order(symbol, current_sl_order_id)
            
            # Place new SL
            side = "SELL" if direction == "LONG" else "BUY"
            new_sl = await self.client.create_stop_loss(
                symbol=symbol,
                side=side,
                quantity=quantity,
                stop_price=new_sl_price,
            )
            
            logger.info(f"SL MODIFIED: {new_sl.price}")
            return new_sl
            
        except Exception as e:
            logger.error(f"Failed to modify SL: {e}")
            return None
    
    async def close_position(
        self,
        symbol: str,
        direction: str,
        quantity: Decimal,
    ) -> Optional[OrderResult]:
        """
        Close a position (partial or full).
        
        Args:
            symbol: Trading symbol
            direction: Current position direction
            quantity: Amount to close
        """
        try:
            side = "SELL" if direction == "LONG" else "BUY"
            order = await self.client.create_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reduce_only=True,
            )
            
            logger.info(f"POSITION CLOSED: {order.side} {order.quantity} @ {order.avg_price}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None
    
    async def emergency_close(self, symbol: str) -> Dict:
        """
        Emergency close all positions and orders for a symbol.
        
        Used by kill-switch.
        """
        logger.warning(f"EMERGENCY CLOSE: {symbol}")
        result = await self.client.emergency_close_all(symbol)
        
        # Clear local tracking
        self._active_positions.pop(symbol, None)
        
        return result
    
    async def emergency_close_all(self) -> Dict[str, Dict]:
        """Emergency close ALL positions across all symbols."""
        logger.warning("EMERGENCY CLOSE ALL POSITIONS")
        results = {}
        
        for symbol in list(self._active_positions.keys()):
            results[symbol] = await self.emergency_close(symbol)
        
        self._active_positions.clear()
        return results
    
    async def reconcile_positions(self) -> Dict[str, PositionInfo]:
        """
        Reconcile local state with exchange reality.
        
        Call this on startup and reconnect.
        """
        logger.info("Reconciling positions with exchange...")
        
        for symbol in list(self._active_positions.keys()):
            position = await self.client.reconcile_position(symbol)
            
            if position:
                self._active_positions[symbol] = position
            else:
                # No position on exchange
                del self._active_positions[symbol]
        
        logger.info(f"Reconciliation complete. Active: {list(self._active_positions.keys())}")
        return self._active_positions
    
    @property
    def has_open_position(self) -> bool:
        """Check if any position is open."""
        return len(self._active_positions) > 0
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for a symbol."""
        return self._active_positions.get(symbol)
