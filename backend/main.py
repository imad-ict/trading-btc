"""
Main Trading Bot Orchestrator

The central hub that coordinates all engines:
- Market data ingestion
- Strategy signals
- Risk validation
- Execution
- Trade journaling
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from config import get_settings, RiskConstants, TradingConstants
from database import init_database, get_db_session, Trade, BotState, AuditLog
from database.models import TradeStatus, TradeResult, BotStateEnum, TradeDirection
from security import KeyVault
from exchange import BinanceClient, WebSocketManager
from exchange.websocket_manager import Candle
from core import SessionController, MarketDataEngine, ExecutionEngine, TradeExplanationBuilder
from strategy import MarketStateEngine, LiquidityEngine, EntryEngine, StopLossEngine, TakeProfitEngine
from risk import RiskEngine, AdaptiveRiskManager
from api import server as api_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log"),
    ]
)
logger = logging.getLogger("TradingBot")


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Coordinates all engines and manages the trading lifecycle.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.is_running = False
        self.is_halted = False
        
        # Initialize components (will be set up in start())
        self.binance: Optional[BinanceClient] = None
        self.ws_manager: Optional[WebSocketManager] = None
        self.session_controller: Optional[SessionController] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.risk_engine: Optional[RiskEngine] = None
        self.adaptive_risk: Optional[AdaptiveRiskManager] = None
        
        # Per-symbol engines
        self.market_data: dict = {}
        self.market_state_engines: dict = {}
        self.liquidity_engines: dict = {}
        self.entry_engines: dict = {}
        self.sl_engines: dict = {}
        self.tp_engines: dict = {}
        
        # State
        self.active_trades: dict = {}
        
        logger.info(f"TradingBot initialized (mode: {self.settings.binance_mode})")
    
    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("═" * 60)
        logger.info("  INSTITUTIONAL TRADING PLATFORM - STARTING")
        logger.info("═" * 60)
        
        try:
            # Initialize database
            await init_database()
            logger.info("✓ Database initialized")
            
            # Initialize API key vault
            if not self.settings.encryption_key:
                logger.warning("No encryption key set - using demo mode")
                # Demo credentials for testnet
                if self.settings.binance_mode == "testnet":
                    api_key = "demo_key"
                    api_secret = "demo_secret"
                else:
                    raise ValueError("Encryption key required for live trading")
            else:
                vault = KeyVault(self.settings.encryption_key)
                creds = vault.get_credentials(
                    self.settings.binance_api_key_encrypted,
                    self.settings.binance_api_secret_encrypted,
                )
                api_key = creds.api_key
                api_secret = creds.api_secret
            
            # Initialize Binance client
            self.binance = BinanceClient(api_key, api_secret)
            logger.info(f"✓ Binance client initialized ({self.settings.binance_mode})")
            
            # Get initial balance
            try:
                balance = await self.binance.get_usdt_balance()
                logger.info(f"✓ Account balance: ${balance:.2f}")
            except Exception as e:
                logger.warning(f"Could not fetch balance: {e}")
                balance = Decimal("1000")  # Demo balance
            
            # Initialize risk engine
            self.risk_engine = RiskEngine(balance)
            self.adaptive_risk = AdaptiveRiskManager()
            logger.info("✓ Risk engine initialized")
            
            # Initialize session controller
            self.session_controller = SessionController()
            logger.info("✓ Session controller initialized")
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(self.binance)
            logger.info("✓ Execution engine initialized")
            
            # Initialize per-symbol engines
            for symbol in self.settings.symbols:
                await self._init_symbol(symbol)
            
            # Initialize WebSocket manager
            self.ws_manager = WebSocketManager(self.settings.symbols)
            self.ws_manager.on_candle = self._on_candle
            self.ws_manager.on_mark_price = self._on_mark_price
            self.ws_manager.on_book_ticker = self._on_book_ticker
            logger.info("✓ WebSocket manager initialized")
            
            # Update API state
            api_server.update_status(
                status="running",
                session=self.session_controller.get_current_session().session.value,
                balance=float(balance),
                daily_pnl=0.0,
                trades_today=0,
                losses_today=0,
                streak=0,
            )
            
            # Start WebSocket streams
            await self.ws_manager.start()
            logger.info("✓ WebSocket streams started")
            
            self.is_running = True
            
            logger.info("═" * 60)
            logger.info(f"  BOT RUNNING - {len(self.settings.symbols)} symbols")
            logger.info(f"  Session: {self.session_controller.get_session_state_string()}")
            logger.info("═" * 60)
            
            # Main loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise
    
    async def _init_symbol(self, symbol: str) -> None:
        """Initialize engines for a symbol."""
        # Market data engine
        self.market_data[symbol] = MarketDataEngine(symbol)
        
        # Strategy engines
        self.market_state_engines[symbol] = MarketStateEngine(self.market_data[symbol])
        self.liquidity_engines[symbol] = LiquidityEngine(self.market_data[symbol])
        self.entry_engines[symbol] = EntryEngine(
            self.market_data[symbol],
            self.liquidity_engines[symbol],
        )
        self.sl_engines[symbol] = StopLossEngine(
            self.market_data[symbol],
            self.liquidity_engines[symbol],
        )
        self.tp_engines[symbol] = TakeProfitEngine(
            self.market_data[symbol],
            self.liquidity_engines[symbol],
        )
        
        # Prepare symbol on exchange
        await self.execution_engine.prepare_symbol(symbol)
        
        logger.info(f"✓ Initialized engines for {symbol}")
    
    def _on_candle(self, candle: Candle) -> None:
        """Handle incoming candle data."""
        symbol = candle.symbol.upper()
        
        if symbol in self.market_data:
            self.market_data[symbol].on_candle(candle)
            
            # Update price in API
            api_server.update_price(symbol, float(candle.close))
            
            # On closed candle, run strategy logic
            if candle.is_closed:
                asyncio.create_task(self._on_candle_close(symbol, candle))
    
    def _on_mark_price(self, data) -> None:
        """Handle mark price update."""
        symbol = data.symbol.upper()
        if symbol in self.market_data:
            self.market_data[symbol].on_mark_price(data)
    
    def _on_book_ticker(self, data) -> None:
        """Handle book ticker update."""
        symbol = data.symbol.upper()
        if symbol in self.market_data:
            self.market_data[symbol].on_book_ticker(data)
    
    async def _on_candle_close(self, symbol: str, candle: Candle) -> None:
        """
        Process closed candle - main strategy logic.
        
        This is where the magic happens.
        """
        if self.is_halted:
            return
        
        try:
            # 1. Check session
            if not self.session_controller.is_trading_allowed():
                return
            
            # 2. Update liquidity on 5M
            if candle.interval == "5m":
                self.liquidity_engines[symbol].update()
            
            # 3. Check for entry signals on 1M
            if candle.interval == "1m":
                await self._check_entry(symbol)
                await self._manage_positions(symbol)
        
        except Exception as e:
            logger.error(f"Candle processing error for {symbol}: {e}")
    
    async def _check_entry(self, symbol: str) -> None:
        """Check for entry signals."""
        # Skip if already in position
        if symbol in self.active_trades:
            return
        
        entry_engine = self.entry_engines[symbol]
        market_state_engine = self.market_state_engines[symbol]
        
        # Check market state (15M)
        state_analysis = market_state_engine.analyze()
        if not state_analysis.is_tradeable:
            return
        
        # Check entry conditions (1M)
        validation = entry_engine.validate_entry()
        if not validation.is_valid:
            return
        
        # Get pending sweep
        sweep = self.liquidity_engines[symbol].get_pending_sweep()
        if not sweep:
            return
        
        # Generate signal
        signal = entry_engine.generate_signal(sweep)
        if not signal:
            return
        
        # Apply entry delay
        entry_engine.queue_signal(signal)
        
        # On next candle close, check if ready
        released_signal = entry_engine.on_candle_close()
        if not released_signal:
            return
        
        # Pre-validate with risk engine
        sl_result = self.sl_engines[symbol].calculate_stop_loss(
            released_signal.direction,
            released_signal.entry_price,
            sweep,
        )
        released_signal.stop_loss = sl_result.price
        
        risk_validation = self.risk_engine.validate_trade(
            released_signal.entry_price,
            released_signal.stop_loss,
        )
        
        if not risk_validation.is_valid:
            logger.warning(f"Trade rejected by risk engine: {risk_validation.rejection_reason}")
            await self._log_audit("TRADE_REJECTED", symbol, {
                "reason": risk_validation.rejection_reason,
            })
            return
        
        # Calculate take profits
        tp_plan = self.tp_engines[symbol].calculate_targets(
            released_signal.direction,
            released_signal.entry_price,
            released_signal.stop_loss,
        )
        
        # Build explanation
        explanation_builder = TradeExplanationBuilder()
        explanation_builder.set_liquidity_event(
            sweep.zone.type.value,
            sweep.zone.price,
            sweep.volume_multiple,
        )
        explanation_builder.set_market_state(
            state_analysis.state.value,
            state_analysis.vwap_slope,
            "HIGH" if state_analysis.volume_score > 0.7 else "NORMAL",
            state_analysis.candle_efficiency,
        )
        explanation_builder.set_entry_logic(
            reclaim_confirmed=True,
            volume_expansion=released_signal.volume_multiple,
            vwap_aligned=released_signal.vwap_aligned,
            spread_ok=True,
            delay_applied=True,
        )
        explanation_builder.set_stop_logic(
            sl_result.sl_type,
            sl_result.price,
            sl_result.distance_pct,
            sl_result.atr_buffer,
        )
        explanation_builder.set_target_logic(
            tp_plan.tp1.target_type,
            tp_plan.tp1.price,
            tp_plan.tp2.target_type if tp_plan.tp2 else None,
            tp_plan.tp2.price if tp_plan.tp2 else None,
            tp_plan.tp3.target_type if tp_plan.tp3 else None,
        )
        explanation_builder.set_risk_validation(
            RiskConstants.RISK_PER_TRADE,
            risk_validation.risk_usd,
            self.risk_engine.daily_state.trades_today,
            self.risk_engine.daily_state.losses_today,
            self.risk_engine.current_streak,
        )
        
        explanation = explanation_builder.build()
        logger.info(explanation.format_log())
        
        # Execute trade
        tp_levels = [
            (tp_plan.tp1.price, tp_plan.tp1.size_pct),
        ]
        if tp_plan.tp2:
            tp_levels.append((tp_plan.tp2.price, tp_plan.tp2.size_pct))
        if tp_plan.tp3:
            tp_levels.append((tp_plan.tp3.price, tp_plan.tp3.size_pct))
        
        result = await self.execution_engine.execute_trade(
            released_signal,
            risk_validation.position_size,
            tp_levels,
        )
        
        if result.success:
            self.risk_engine.record_trade_start()
            
            # Save to database
            trade = Trade(
                symbol=symbol,
                direction=TradeDirection[released_signal.direction],
                entry_price=released_signal.entry_price,
                stop_loss=released_signal.stop_loss,
                tp1=tp_plan.tp1.price,
                tp2=tp_plan.tp2.price if tp_plan.tp2 else None,
                tp3=tp_plan.tp3.price if tp_plan.tp3 else None,
                position_size=risk_validation.position_size,
                leverage=RiskConstants.MAX_LEVERAGE,
                status=TradeStatus.OPEN,
                liquidity_event=explanation.liquidity_event,
                market_state=explanation.market_state,
                entry_logic=explanation.entry_logic,
                stop_logic=explanation.stop_logic,
                target_logic=explanation.target_logic,
                risk_validation=explanation.risk_validation,
                entry_time=datetime.now(timezone.utc),
                entry_order_id=result.entry_order.order_id,
                sl_order_id=result.sl_order.order_id,
            )
            
            async with get_db_session() as session:
                session.add(trade)
            
            self.active_trades[symbol] = {
                "trade": trade,
                "tp_plan": tp_plan,
                "sl_order_id": result.sl_order.order_id,
            }
            
            # Update API
            api_server.update_status(
                trades_today=self.risk_engine.daily_state.trades_today,
            )
            api_server.update_active_trade({
                "symbol": symbol,
                "direction": released_signal.direction,
                "entry_price": float(released_signal.entry_price),
                "stop_loss": float(released_signal.stop_loss),
            })
            
            await self._log_audit("TRADE_OPENED", symbol, {
                "entry_price": float(released_signal.entry_price),
                "direction": released_signal.direction,
            })
        else:
            logger.error(f"Trade execution failed: {result.error}")
            await self._log_audit("TRADE_FAILED", symbol, {"error": result.error})
    
    async def _manage_positions(self, symbol: str) -> None:
        """Manage open positions - trailing, TP hits, etc."""
        if symbol not in self.active_trades:
            return
        
        trade_data = self.active_trades[symbol]
        trade = trade_data["trade"]
        tp_plan = trade_data["tp_plan"]
        
        current_price = self.market_data[symbol].current_price
        if not current_price:
            return
        
        # Check for stagnation
        if self.tp_engines[symbol].check_stagnation(trade.direction.value, current_price):
            logger.warning(f"Stagnation exit for {symbol}")
            await self._close_trade(symbol, "STAGNATION")
    
    async def _close_trade(self, symbol: str, reason: str) -> None:
        """Close an active trade."""
        if symbol not in self.active_trades:
            return
        
        trade_data = self.active_trades.pop(symbol)
        trade = trade_data["trade"]
        
        # Get current position
        position = self.execution_engine.get_position(symbol)
        if position:
            await self.execution_engine.close_position(
                symbol,
                trade.direction.value,
                position.size,
            )
        
        # Update trade record
        async with get_db_session() as session:
            trade.status = TradeStatus.CLOSED
            trade.exit_time = datetime.now(timezone.utc)
            # TODO: Calculate actual PnL
            session.add(trade)
        
        api_server.update_active_trade(None)
        
        await self._log_audit("TRADE_CLOSED", symbol, {"reason": reason})
    
    async def _log_audit(self, event_type: str, symbol: str, details: dict) -> None:
        """Log audit event."""
        async with get_db_session() as session:
            log = AuditLog(
                event_type=event_type,
                symbol=symbol,
                details=details,
            )
            session.add(log)
    
    async def _main_loop(self) -> None:
        """Main bot loop."""
        while self.is_running:
            try:
                # Check for daily reset
                self.risk_engine.check_daily_reset()
                
                # Update session state
                session_info = self.session_controller.get_current_session()
                api_server.update_status(session=session_info.session.value)
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)
    
    async def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self.is_running = False
        
        # Close all positions
        if self.execution_engine:
            await self.execution_engine.emergency_close_all()
        
        # Stop WebSocket
        if self.ws_manager:
            await self.ws_manager.stop()
        
        # Close Binance client
        if self.binance:
            await self.binance.close()
        
        logger.info("Trading bot stopped")


async def main():
    """Main entry point."""
    bot = TradingBot()
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))
    
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
