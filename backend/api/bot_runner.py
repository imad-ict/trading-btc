"""
Bot Runner - Institutional Trading Bot

Implements liquidity-based strategy:
- NO momentum entries
- Only liquidity sweep + reclaim
- Structure-based SL (behind wick)
- Multi-target TP with partial closes
- Dynamic SL movement
"""
import asyncio
import logging
import hmac
import hashlib
import time
from decimal import Decimal
from typing import Dict, Optional, List
from datetime import datetime
import httpx
import websockets
import json

from strategy.institutional_strategy import (
    InstitutionalStrategy, TradeSignal, ActivePosition, 
    Candle, TradeDirection, MarketState
)

logger = logging.getLogger(__name__)


class BotRunner:
    """
    Institutional Trading Bot.
    
    Core Principles:
    1. NO MOMENTUM - Only liquidity sweeps
    2. Capital preservation > Trade frequency
    3. Every trade must be explainable
    """
    
    def __init__(self):
        self.is_running = False
        self.settings: Dict = {}
        self.prices: Dict[str, float] = {}
        
        # Tasks
        self.ws_task: Optional[asyncio.Task] = None
        self.strategy_task: Optional[asyncio.Task] = None
        self.kline_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._price_callback = None
        self._status_callback = None
        self._trade_callback = None
        self._log_callback = None
        
        # Strategy
        self.strategy = InstitutionalStrategy()
        self.active_position: Optional[ActivePosition] = None
        
        # Trading state
        self.trades_today = 0
        self.max_trades_per_day = 10
        self.last_signal_time: Dict[str, float] = {}
        self.signal_cooldown = 300  # 5 minutes
        
        # Candle aggregation
        self.current_candle: Dict[str, Dict] = {}
        self.candle_start_time: Dict[str, float] = {}
    
    def set_callbacks(self, price_callback, status_callback, 
                      trade_callback=None, log_callback=None):
        """Set callbacks for updates."""
        self._price_callback = price_callback
        self._status_callback = status_callback
        self._trade_callback = trade_callback
        self._log_callback = log_callback
    
    def _log(self, message: str):
        """Log to both logger and callback."""
        logger.info(message)
        if self._log_callback:
            self._log_callback(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {message}")
    
    async def start(self, settings: Dict) -> bool:
        """Start the institutional trading bot."""
        if self.is_running:
            return False
        
        self.settings = settings
        self.is_running = True
        self.trades_today = 0
        
        # Start tasks
        self.ws_task = asyncio.create_task(self._stream_prices())
        self.kline_task = asyncio.create_task(self._stream_klines())
        self.strategy_task = asyncio.create_task(self._strategy_loop())
        
        self._log(f"🚀 INSTITUTIONAL BOT STARTED")
        self._log(f"📊 Monitoring: {settings.get('symbols', [])}")
        self._log(f"⚠️ Mode: {settings.get('mode', 'testnet').upper()}")
        self._log(f"🎯 Strategy: Liquidity Sweep + Reclaim ONLY")
        
        if self._status_callback:
            self._status_callback("running")
        
        return True
    
    async def stop(self) -> bool:
        """Stop the bot."""
        if not self.is_running:
            return False
        
        self.is_running = False
        
        # Cancel tasks
        for task in [self.ws_task, self.kline_task, self.strategy_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._log("🛑 Bot stopped")
        
        if self._status_callback:
            self._status_callback("stopped")
        
        return True
    
    async def _stream_prices(self):
        """Stream real-time mark prices."""
        symbols = self.settings.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        mode = self.settings.get("mode", "testnet")
        
        if mode == "testnet":
            ws_base = "wss://stream.binancefuture.com/ws"
        else:
            ws_base = "wss://fstream.binance.com/ws"
        
        streams = [f"{s.lower()}@markPrice@1s" for s in symbols]
        ws_url = f"{ws_base}/{'/'.join(streams)}"
        
        while self.is_running:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._log(f"📡 Price stream connected: {len(symbols)} symbols")
                    
                    while self.is_running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            
                            if "s" in data and "p" in data:
                                symbol = data["s"]
                                price = float(data["p"])
                                self.prices[symbol] = price
                                
                                # Aggregate into candles
                                self._aggregate_candle(symbol, price, float(data.get("r", 0)))
                                
                                if self._price_callback:
                                    self._price_callback(symbol, price)
                        
                        except asyncio.TimeoutError:
                            await ws.ping()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.is_running:
                    await asyncio.sleep(5)
    
    async def _stream_klines(self):
        """Stream 5-minute and 15-minute klines for liquidity and market state detection."""
        symbols = self.settings.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        mode = self.settings.get("mode", "testnet")
        
        if mode == "testnet":
            ws_base = "wss://stream.binancefuture.com/ws"
        else:
            ws_base = "wss://fstream.binance.com/ws"
        
        # Stream both 5M (liquidity) and 15M (market state) klines
        streams = []
        for s in symbols:
            streams.append(f"{s.lower()}@kline_5m")
            streams.append(f"{s.lower()}@kline_15m")
        
        ws_url = f"{ws_base}/{'/'.join(streams)}"
        
        while self.is_running:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._log(f"📊 5M + 15M Kline streams connected")
                    
                    while self.is_running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=60)
                            data = json.loads(msg)
                            
                            if "k" in data:
                                k = data["k"]
                                if k["x"]:  # Candle closed
                                    candle = Candle(
                                        timestamp=k["t"] / 1000,
                                        open=float(k["o"]),
                                        high=float(k["h"]),
                                        low=float(k["l"]),
                                        close=float(k["c"]),
                                        volume=float(k["v"])
                                    )
                                    # Determine timeframe from interval
                                    timeframe = "5m" if k["i"] == "5m" else "15m"
                                    self.strategy.add_candle(k["s"], candle, timeframe)
                        
                        except asyncio.TimeoutError:
                            await ws.ping()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Kline WebSocket error: {e}")
                if self.is_running:
                    await asyncio.sleep(5)
    
    def _aggregate_candle(self, symbol: str, price: float, volume: float):
        """Aggregate tick data into 1-minute candles."""
        current_minute = int(time.time() / 60) * 60
        
        if symbol not in self.current_candle:
            self.current_candle[symbol] = {
                "open": price, "high": price, "low": price, 
                "close": price, "volume": volume
            }
            self.candle_start_time[symbol] = current_minute
            return
        
        # Check if new candle
        if current_minute > self.candle_start_time.get(symbol, 0):
            # Close previous candle
            c = self.current_candle[symbol]
            candle = Candle(
                timestamp=self.candle_start_time[symbol],
                open=c["open"], high=c["high"], low=c["low"],
                close=c["close"], volume=c["volume"]
            )
            self.strategy.add_candle(symbol, candle, "1m")
            
            # Start new candle
            self.current_candle[symbol] = {
                "open": price, "high": price, "low": price,
                "close": price, "volume": volume
            }
            self.candle_start_time[symbol] = current_minute
        else:
            # Update current candle
            self.current_candle[symbol]["high"] = max(self.current_candle[symbol]["high"], price)
            self.current_candle[symbol]["low"] = min(self.current_candle[symbol]["low"], price)
            self.current_candle[symbol]["close"] = price
            self.current_candle[symbol]["volume"] += volume
    
    async def _strategy_loop(self):
        """Main strategy execution loop."""
        self._log("📊 Strategy loop started")
        self._log("⏳ Collecting price data for liquidity mapping...")
        
        # Wait for initial data collection
        await asyncio.sleep(60)
        self._log("📈 Data collection complete - Monitoring for signals")
        
        diagnostic_counter = 0
        
        while self.is_running:
            try:
                await asyncio.sleep(5)
                diagnostic_counter += 1
                
                if not self.prices:
                    continue
                
                # Check active position
                if self.active_position:
                    await self._manage_position()
                    continue
                
                # Check trade limits
                if self.trades_today >= self.max_trades_per_day:
                    continue
                
                # Periodic diagnostics every 30 seconds (6 x 5-second loops)
                if diagnostic_counter % 6 == 0:
                    for symbol in list(self.prices.keys())[:1]:  # Just first symbol
                        diag = self.strategy.get_diagnostics(symbol)
                        levels_count = len(diag.get("liquidity_levels", []))
                        candles_5m = diag.get("candles_5m", 0)
                        market_state = diag.get("market_state", "unknown")
                        vwap = diag.get("vwap")
                        
                        self._log(f"📊 {symbol}: 5M candles={candles_5m}, Levels={levels_count}, State={market_state}")
                        
                        if levels_count > 0:
                            for lv in diag["liquidity_levels"][:3]:
                                status = "✓" if not lv["swept"] else "⊗"
                                self._log(f"   {status} {lv['type']}: ${lv['price']:,.2f} (touches: {lv['touches']})")
                        
                        if vwap:
                            price = self.prices.get(symbol, 0)
                            diff = ((price - vwap) / vwap * 100) if vwap else 0
                            self._log(f"   VWAP: ${vwap:,.2f} | Price: ${price:,.2f} ({diff:+.2f}%)")
                
                # Check market state and look for signals
                for symbol in self.prices.keys():
                    market_state = self.strategy.get_market_state(symbol)
                    
                    if market_state not in [MarketState.EXPANSION, MarketState.MANIPULATION]:
                        continue
                    
                    # Check cooldown
                    if time.time() - self.last_signal_time.get(symbol, 0) < self.signal_cooldown:
                        continue
                    
                    # Build current candle for signal check
                    if symbol in self.current_candle:
                        c = self.current_candle[symbol]
                        current = Candle(
                            timestamp=time.time(),
                            open=c["open"], high=c["high"], low=c["low"],
                            close=self.prices[symbol], volume=c["volume"]
                        )
                        
                        # Generate signal
                        signal = self.strategy.generate_signal(symbol, current)
                        
                        if signal:
                            self._log(f"")
                            self._log(f"{'='*50}")
                            self._log(f"🎯 INSTITUTIONAL SIGNAL DETECTED")
                            self._log(f"{'='*50}")
                            self._log(f"Symbol: {signal.symbol}")
                            self._log(f"Direction: {signal.direction.value}")
                            self._log(f"Entry: ${signal.entry_price:,.2f}")
                            self._log(f"Stop Loss: ${signal.stop_loss:,.2f}")
                            self._log(f"TP1 (VWAP): ${signal.tp1:,.2f}")
                            self._log(f"TP2 (Liq): ${signal.tp2:,.2f}")
                            self._log(f"TP3 (Ext): ${signal.tp3:,.2f}")
                            self._log(f"Reason: {signal.reason}")
                            self._log(f"Market State: {signal.market_state.value}")
                            self._log(f"Risk: {signal.risk_distance*100:.3f}%")
                            self._log(f"{'='*50}")
                            
                            success = await self._execute_trade(signal)
                            
                            if success:
                                self.last_signal_time[symbol] = time.time()
                                break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                await asyncio.sleep(5)
    
    async def _manage_position(self):
        """Manage active position with dynamic SL and partial closes."""
        if not self.active_position:
            return
        
        pos = self.active_position
        symbol = pos.signal.symbol
        current_price = self.prices.get(symbol)
        
        if not current_price:
            return
        
        # Check for exit
        should_exit, reason = pos.should_exit(current_price)
        if should_exit:
            await self._close_full_position(reason, current_price)
            return
        
        # Check TP levels
        tp_hit = pos.check_tp_levels(current_price)
        
        if tp_hit == "TP1" and not pos.tp1_hit:
            self._log(f"🎯 TP1 HIT @ ${current_price:,.2f}")
            await self._partial_close(pos.get_partial_qty_tp1(), "TP1", current_price)
            pos.move_sl_to_breakeven()
            self._log(f"📍 SL moved to breakeven: ${pos.current_sl:,.2f}")
        
        elif tp_hit == "TP2" and not pos.tp2_hit:
            self._log(f"🎯 TP2 HIT @ ${current_price:,.2f}")
            await self._partial_close(pos.get_partial_qty_tp2(), "TP2", current_price)
            pos.move_sl_to_tp1()
            self._log(f"📍 SL moved to TP1: ${pos.current_sl:,.2f}")
        
        elif tp_hit == "TP3":
            self._log(f"🎯 TP3 HIT - Full exit @ ${current_price:,.2f}")
            await self._close_full_position("FULL_TP", current_price)
    
    async def _execute_trade(self, signal: TradeSignal) -> bool:
        """Execute trade on Binance Futures."""
        mode = self.settings.get("mode", "testnet")
        api_key = self.settings.get(f"{mode}_api_key", "")
        api_secret = self.settings.get(f"{mode}_api_secret", "")
        
        if not api_key or not api_secret:
            self._log("❌ Cannot trade: API keys not configured")
            return False
        
        if mode == "testnet":
            base_url = "https://testnet.binancefuture.com"
        else:
            base_url = "https://fapi.binance.com"
        
        try:
            # Calculate position size based on risk
            balance = await self.get_balance()
            if not balance or balance < 100:
                self._log(f"❌ Insufficient balance: ${balance}")
                return False
            
            risk_amount = balance * 0.003  # 0.3% risk per trade
            risk_per_unit = abs(signal.entry_price - signal.stop_loss)
            quantity = round(risk_amount / risk_per_unit, 3)
            
            # Minimum quantity check
            if "BTC" in signal.symbol:
                quantity = max(0.001, quantity)
            else:
                quantity = max(0.01, quantity)
            
            side = "BUY" if signal.direction == TradeDirection.LONG else "SELL"
            
            # Market order
            timestamp = int(time.time() * 1000)
            params = {
                "symbol": signal.symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "timestamp": timestamp,
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                api_secret.encode(), query_string.encode(), hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{base_url}/fapi/v1/order",
                    params=params,
                    headers={"X-MBX-APIKEY": api_key}
                )
                
                if response.status_code == 200:
                    order = response.json()
                    self._log(f"✅ ORDER EXECUTED: {side} {quantity} {signal.symbol}")
                    
                    self.active_position = ActivePosition(
                        signal=signal,
                        order_id=str(order.get("orderId", "")),
                        quantity=quantity,
                        entry_time=datetime.utcnow()
                    )
                    self.trades_today += 1
                    
                    if self._trade_callback:
                        self._trade_callback({
                            "symbol": signal.symbol,
                            "direction": signal.direction.value,
                            "entry": signal.entry_price,
                            "sl": signal.stop_loss,
                            "tp1": signal.tp1,
                            "quantity": quantity
                        })
                    
                    return True
                else:
                    error = response.json().get("msg", response.text)
                    self._log(f"❌ Order failed: {error}")
                    return False
                    
        except Exception as e:
            self._log(f"❌ Trade execution error: {e}")
            return False
    
    async def _partial_close(self, quantity: float, reason: str, exit_price: float):
        """Close partial position."""
        if not self.active_position:
            return
        
        pos = self.active_position
        mode = self.settings.get("mode", "testnet")
        api_key = self.settings.get(f"{mode}_api_key", "")
        api_secret = self.settings.get(f"{mode}_api_secret", "")
        
        if mode == "testnet":
            base_url = "https://testnet.binancefuture.com"
        else:
            base_url = "https://fapi.binance.com"
        
        try:
            side = "SELL" if pos.signal.direction == TradeDirection.LONG else "BUY"
            
            timestamp = int(time.time() * 1000)
            params = {
                "symbol": pos.signal.symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "timestamp": timestamp,
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                api_secret.encode(), query_string.encode(), hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{base_url}/fapi/v1/order",
                    params=params,
                    headers={"X-MBX-APIKEY": api_key}
                )
                
                if response.status_code == 200:
                    pnl_pct = self._calculate_pnl_pct(pos, exit_price)
                    self._log(f"✅ Partial close ({reason}): {quantity} @ ${exit_price:,.2f} | P&L: {pnl_pct:.2f}%")
                    pos.remaining_qty -= quantity
                else:
                    error = response.json().get("msg", response.text)
                    self._log(f"❌ Partial close failed: {error}")
                    
        except Exception as e:
            self._log(f"❌ Partial close error: {e}")
    
    async def _close_full_position(self, reason: str, exit_price: float):
        """Close full remaining position."""
        if not self.active_position:
            return
        
        pos = self.active_position
        mode = self.settings.get("mode", "testnet")
        api_key = self.settings.get(f"{mode}_api_key", "")
        api_secret = self.settings.get(f"{mode}_api_secret", "")
        
        if mode == "testnet":
            base_url = "https://testnet.binancefuture.com"
        else:
            base_url = "https://fapi.binance.com"
        
        try:
            side = "SELL" if pos.signal.direction == TradeDirection.LONG else "BUY"
            
            timestamp = int(time.time() * 1000)
            params = {
                "symbol": pos.signal.symbol,
                "side": side,
                "type": "MARKET",
                "quantity": pos.remaining_qty,
                "timestamp": timestamp,
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                api_secret.encode(), query_string.encode(), hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{base_url}/fapi/v1/order",
                    params=params,
                    headers={"X-MBX-APIKEY": api_key}
                )
                
                if response.status_code == 200:
                    pnl_pct = self._calculate_pnl_pct(pos, exit_price)
                    result = "WIN" if pnl_pct > 0 else "LOSS"
                    emoji = "🟢" if result == "WIN" else "🔴"
                    
                    self._log(f"")
                    self._log(f"{'='*50}")
                    self._log(f"{emoji} POSITION CLOSED - {reason}")
                    self._log(f"{'='*50}")
                    self._log(f"Result: {result}")
                    self._log(f"P&L: {pnl_pct:.2f}%")
                    self._log(f"Entry: ${pos.signal.entry_price:,.2f}")
                    self._log(f"Exit: ${exit_price:,.2f}")
                    self._log(f"{'='*50}")
                    
                    self.active_position = None
                else:
                    error = response.json().get("msg", response.text)
                    self._log(f"❌ Close failed: {error}")
                    
        except Exception as e:
            self._log(f"❌ Close position error: {e}")
    
    def _calculate_pnl_pct(self, pos: ActivePosition, exit_price: float) -> float:
        """Calculate P&L percentage."""
        entry = pos.signal.entry_price
        if pos.signal.direction == TradeDirection.LONG:
            return (exit_price - entry) / entry * 100
        else:
            return (entry - exit_price) / entry * 100
    
    async def get_balance(self) -> Optional[float]:
        """Get USDT balance from Binance."""
        mode = self.settings.get("mode", "testnet")
        api_key = self.settings.get(f"{mode}_api_key", "")
        api_secret = self.settings.get(f"{mode}_api_secret", "")
        
        if not api_key or not api_secret:
            return None
        
        if mode == "testnet":
            base_url = "https://testnet.binancefuture.com"
        else:
            base_url = "https://fapi.binance.com"
        
        try:
            timestamp = int(time.time() * 1000)
            query = f"timestamp={timestamp}"
            signature = hmac.new(
                api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{base_url}/fapi/v2/balance",
                    params={"timestamp": timestamp, "signature": signature},
                    headers={"X-MBX-APIKEY": api_key}
                )
                
                if response.status_code == 200:
                    balances = response.json()
                    usdt = next((b for b in balances if b["asset"] == "USDT"), None)
                    return float(usdt["balance"]) if usdt else 0
        except Exception as e:
            logger.error(f"Balance error: {e}")
        
        return None


# Global bot instance
bot_runner = BotRunner()
