"""
Bot Runner - Trading bot with strategy execution.

Runs as a background task within the API server.
Implements a simple momentum-based strategy for demonstration.
"""
import asyncio
import logging
import hmac
import hashlib
import time
from decimal import Decimal
from typing import Dict, Optional, List
from collections import deque
from datetime import datetime
import httpx
import websockets
import json

logger = logging.getLogger(__name__)


class TradingStrategy:
    """
    Simple momentum-based trading strategy.
    
    Entry Logic:
    - Tracks price changes over a rolling window
    - Enters LONG when price momentum is positive and exceeds threshold
    - Enters SHORT when price momentum is negative and exceeds threshold
    
    Exit Logic:
    - Stop-loss at configured percentage
    - Take-profit at configured R:R ratio
    """
    
    def __init__(self, risk_pct: float = 0.5, rr_ratio: float = 2.0):
        self.risk_pct = risk_pct  # Risk per trade (0.5%)
        self.rr_ratio = rr_ratio  # Risk-reward ratio (2:1)
        self.price_history: Dict[str, deque] = {}
        self.window_size = 60  # 60 seconds of price history
        self.momentum_threshold = 0.001  # 0.1% momentum threshold
        
    def add_price(self, symbol: str, price: float) -> None:
        """Add a price point to history."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.window_size)
        self.price_history[symbol].append({
            "price": price,
            "time": time.time()
        })
    
    def get_signal(self, symbol: str) -> Optional[Dict]:
        """
        Analyze price data and generate trading signal.
        
        Returns:
            Dict with signal details or None if no signal
        """
        if symbol not in self.price_history:
            return None
        
        history = self.price_history[symbol]
        if len(history) < 30:  # Need at least 30 seconds of data
            return None
        
        # Calculate momentum (price change over window)
        current_price = history[-1]["price"]
        oldest_price = history[0]["price"]
        momentum = (current_price - oldest_price) / oldest_price
        
        # Generate signal based on momentum
        if abs(momentum) > self.momentum_threshold:
            direction = "LONG" if momentum > 0 else "SHORT"
            
            # Calculate stop-loss and take-profit
            stop_distance = current_price * (self.risk_pct / 100)
            
            if direction == "LONG":
                stop_loss = current_price - stop_distance
                take_profit = current_price + (stop_distance * self.rr_ratio)
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - (stop_distance * self.rr_ratio)
            
            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "momentum": momentum,
                "reason": f"Momentum: {momentum*100:.3f}% ({direction})"
            }
        
        return None


class BotRunner:
    """
    Trading bot that runs within the API server.
    
    Features:
    - Real-time price streaming from Binance WebSocket
    - Strategy execution with entry signals
    - Order execution on Binance Futures
    - Position management
    """
    
    def __init__(self):
        self.is_running = False
        self.settings: Dict = {}
        self.prices: Dict[str, float] = {}
        self.ws_task: Optional[asyncio.Task] = None
        self.strategy_task: Optional[asyncio.Task] = None
        self._price_callback = None
        self._status_callback = None
        self._trade_callback = None
        self._log_callback = None
        
        # Trading state
        self.strategy = TradingStrategy()
        self.active_position: Optional[Dict] = None
        self.trades_today = 0
        self.max_trades_per_day = 10
        self.last_signal_time: Dict[str, float] = {}
        self.signal_cooldown = 300  # 5 minutes between signals per symbol
    
    def set_callbacks(self, price_callback, status_callback, trade_callback=None, log_callback=None):
        """Set callbacks for updates."""
        self._price_callback = price_callback
        self._status_callback = status_callback
        self._trade_callback = trade_callback
        self._log_callback = log_callback
    
    def _log(self, message: str):
        """Log to both logger and callback."""
        logger.info(message)
        if self._log_callback:
            self._log_callback(message)
    
    async def start(self, settings: Dict) -> bool:
        """Start the trading bot."""
        if self.is_running:
            return False
        
        self.settings = settings
        self.is_running = True
        self.trades_today = 0
        
        # Start price streaming
        self.ws_task = asyncio.create_task(self._stream_prices())
        
        # Start strategy loop
        self.strategy_task = asyncio.create_task(self._strategy_loop())
        
        self._log(f"🚀 Bot started with symbols: {settings.get('symbols', [])}")
        
        if self._status_callback:
            self._status_callback("running")
        
        return True
    
    async def stop(self) -> bool:
        """Stop the trading bot."""
        if not self.is_running:
            return False
        
        self.is_running = False
        
        # Cancel tasks
        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
        
        if self.strategy_task:
            self.strategy_task.cancel()
            try:
                await self.strategy_task
            except asyncio.CancelledError:
                pass
        
        self._log("🛑 Bot stopped")
        
        if self._status_callback:
            self._status_callback("stopped")
        
        return True
    
    async def _stream_prices(self):
        """Stream real-time prices from Binance WebSocket."""
        symbols = self.settings.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        mode = self.settings.get("mode", "testnet")
        
        # Binance WebSocket URL
        if mode == "testnet":
            ws_base = "wss://stream.binancefuture.com/ws"
        else:
            ws_base = "wss://fstream.binance.com/ws"
        
        # Build stream names
        streams = [f"{s.lower()}@markPrice@1s" for s in symbols]
        ws_url = f"{ws_base}/{'/'.join(streams)}"
        
        while self.is_running:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._log(f"📡 Connected to Binance WebSocket: {len(symbols)} streams")
                    
                    while self.is_running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            
                            if "s" in data and "p" in data:
                                symbol = data["s"]
                                price = float(data["p"])
                                self.prices[symbol] = price
                                
                                # Add to strategy price history
                                self.strategy.add_price(symbol, price)
                                
                                # Callback to update API state
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
    
    async def _strategy_loop(self):
        """Main strategy execution loop."""
        self._log("📊 Strategy loop started - analyzing market...")
        
        while self.is_running:
            try:
                # Check every 5 seconds
                await asyncio.sleep(5)
                
                if not self.prices:
                    continue
                
                # Check if we have an active position
                if self.active_position:
                    await self._check_position()
                    continue
                
                # Check trade limits
                if self.trades_today >= self.max_trades_per_day:
                    continue
                
                # Look for entry signals
                for symbol in self.prices.keys():
                    # Check cooldown
                    last_signal = self.last_signal_time.get(symbol, 0)
                    if time.time() - last_signal < self.signal_cooldown:
                        continue
                    
                    signal = self.strategy.get_signal(symbol)
                    
                    if signal:
                        self._log(f"📈 Signal: {signal['direction']} {symbol} @ ${signal['entry_price']:,.2f}")
                        self._log(f"   Reason: {signal['reason']}")
                        self._log(f"   SL: ${signal['stop_loss']:,.2f} | TP: ${signal['take_profit']:,.2f}")
                        
                        # Execute trade
                        success = await self._execute_trade(signal)
                        
                        if success:
                            self.last_signal_time[symbol] = time.time()
                            break  # Only one trade at a time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                await asyncio.sleep(5)
    
    async def _check_position(self):
        """Check and manage active position."""
        if not self.active_position:
            return
        
        symbol = self.active_position["symbol"]
        current_price = self.prices.get(symbol)
        
        if not current_price:
            return
        
        entry = self.active_position["entry_price"]
        sl = self.active_position["stop_loss"]
        tp = self.active_position["take_profit"]
        direction = self.active_position["direction"]
        
        # Check for stop-loss or take-profit
        if direction == "LONG":
            if current_price <= sl:
                await self._close_position("STOP_LOSS", current_price)
            elif current_price >= tp:
                await self._close_position("TAKE_PROFIT", current_price)
        else:  # SHORT
            if current_price >= sl:
                await self._close_position("STOP_LOSS", current_price)
            elif current_price <= tp:
                await self._close_position("TAKE_PROFIT", current_price)
    
    async def _execute_trade(self, signal: Dict) -> bool:
        """Execute a trade on Binance."""
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
            # Calculate position size (simplified - 0.001 BTC for demo)
            quantity = 0.001 if "BTC" in signal["symbol"] else 0.01
            
            # Create order
            side = "BUY" if signal["direction"] == "LONG" else "SELL"
            
            timestamp = int(time.time() * 1000)
            params = {
                "symbol": signal["symbol"],
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "timestamp": timestamp, 
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                api_secret.encode(),
                query_string.encode(),
                hashlib.sha256
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
                    self._log(f"✅ Order executed: {side} {quantity} {signal['symbol']}")
                    
                    # Track position
                    self.active_position = {
                        **signal,
                        "order_id": order.get("orderId"),
                        "quantity": quantity,
                        "entry_time": datetime.utcnow().isoformat(),
                    }
                    self.trades_today += 1
                    
                    if self._trade_callback:
                        self._trade_callback(self.active_position)
                    
                    return True
                else:
                    error = response.json().get("msg", response.text)
                    self._log(f"❌ Order failed: {error}")
                    return False
                    
        except Exception as e:
            self._log(f"❌ Trade execution error: {e}")
            return False
    
    async def _close_position(self, reason: str, exit_price: float):
        """Close the active position."""
        if not self.active_position:
            return
        
        mode = self.settings.get("mode", "testnet")
        api_key = self.settings.get(f"{mode}_api_key", "")
        api_secret = self.settings.get(f"{mode}_api_secret", "")
        
        if mode == "testnet":
            base_url = "https://testnet.binancefuture.com"
        else:
            base_url = "https://fapi.binance.com"
        
        try:
            # Close position (opposite side)
            symbol = self.active_position["symbol"]
            quantity = self.active_position["quantity"]
            side = "SELL" if self.active_position["direction"] == "LONG" else "BUY"
            
            timestamp = int(time.time() * 1000)
            params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "timestamp": timestamp,
            }
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                api_secret.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{base_url}/fapi/v1/order",
                    params=params,
                    headers={"X-MBX-APIKEY": api_key}
                )
                
                if response.status_code == 200:
                    # Calculate P&L
                    entry = self.active_position["entry_price"]
                    if self.active_position["direction"] == "LONG":
                        pnl = (exit_price - entry) / entry * 100
                    else:
                        pnl = (entry - exit_price) / entry * 100
                    
                    result = "WIN" if pnl > 0 else "LOSS"
                    emoji = "🟢" if result == "WIN" else "🔴"
                    
                    self._log(f"{emoji} Position closed ({reason}): {result} {pnl:.2f}%")
                    
                    self.active_position = None
                else:
                    error = response.json().get("msg", response.text)
                    self._log(f"❌ Close failed: {error}")
                    
        except Exception as e:
            self._log(f"❌ Close position error: {e}")
    
    async def get_balance(self) -> Optional[float]:
        """Get current USDT balance from Binance."""
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
                api_secret.encode(),
                query.encode(),
                hashlib.sha256
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
            logger.error(f"Balance fetch error: {e}")
        
        return None


# Global bot instance
bot_runner = BotRunner()

