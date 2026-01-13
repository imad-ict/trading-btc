"""
Bot Runner - Lightweight bot for price streaming and trade execution.

Runs as a background task within the API server.
"""
import asyncio
import logging
import hmac
import hashlib
import time
from decimal import Decimal
from typing import Dict, Optional
import httpx
import websockets
import json

logger = logging.getLogger(__name__)


class BotRunner:
    """
    Lightweight trading bot that runs within the API server.
    
    Features:
    - Real-time price streaming from Binance WebSocket
    - Strategy execution based on configured rules
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
    
    def set_callbacks(self, price_callback, status_callback):
        """Set callbacks for price and status updates."""
        self._price_callback = price_callback
        self._status_callback = status_callback
    
    async def start(self, settings: Dict) -> bool:
        """Start the trading bot."""
        if self.is_running:
            return False
        
        self.settings = settings
        self.is_running = True
        
        # Start price streaming
        self.ws_task = asyncio.create_task(self._stream_prices())
        
        # Start strategy loop
        self.strategy_task = asyncio.create_task(self._strategy_loop())
        
        logger.info(f"Bot started with symbols: {settings.get('symbols', [])}")
        
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
        
        logger.info("Bot stopped")
        
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
                    logger.info(f"Connected to Binance WebSocket: {len(symbols)} streams")
                    
                    while self.is_running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            
                            if "s" in data and "p" in data:
                                symbol = data["s"]
                                price = float(data["p"])
                                self.prices[symbol] = price
                                
                                # Callback to update API state
                                if self._price_callback:
                                    self._price_callback(symbol, price)
                        
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            await ws.ping()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.is_running:
                    await asyncio.sleep(5)  # Reconnect delay
    
    async def _strategy_loop(self):
        """Main strategy execution loop."""
        while self.is_running:
            try:
                # Check for trading opportunities every 5 seconds
                await asyncio.sleep(5)
                
                if not self.prices:
                    continue
                
                # Log price check (strategy would be here)
                for symbol, price in self.prices.items():
                    logger.debug(f"{symbol}: ${price:,.2f}")
                
                # TODO: Implement full strategy logic
                # For now, just monitoring prices
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                await asyncio.sleep(5)
    
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
