"""
WebSocket Manager - Multi-stream connection handler.

Manages concurrent WebSocket streams for:
- Kline data (1m, 5m, 15m)
- Mark price updates
- User data stream (orders/positions)

Features auto-reconnect with exponential backoff.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

from config import get_settings

logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    """WebSocket stream types."""
    KLINE_1M = "kline_1m"
    KLINE_5M = "kline_5m"
    KLINE_15M = "kline_15m"
    MARK_PRICE = "markPrice"
    BOOK_TICKER = "bookTicker"
    USER_DATA = "userData"


@dataclass
class Candle:
    """Candlestick data structure."""
    symbol: str
    interval: str
    time: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: int
    is_closed: bool


@dataclass
class MarkPrice:
    """Mark price update."""
    symbol: str
    price: Decimal
    index_price: Decimal
    funding_rate: Decimal
    next_funding_time: int


@dataclass
class BookTicker:
    """Best bid/ask update."""
    symbol: str
    bid_price: Decimal
    bid_qty: Decimal
    ask_price: Decimal
    ask_qty: Decimal


# Type alias for callbacks
StreamCallback = Callable[[StreamType, Any], None]


class WebSocketManager:
    """
    Manages multiple WebSocket streams with auto-reconnect.
    
    Usage:
        manager = WebSocketManager(symbols=["BTCUSDT", "ETHUSDT"])
        manager.on_candle = handle_candle
        manager.on_price = handle_price
        await manager.start()
    """
    
    # Reconnection settings
    INITIAL_RETRY_DELAY = 1.0
    MAX_RETRY_DELAY = 60.0
    PING_INTERVAL = 180  # 3 minutes
    
    def __init__(self, symbols: List[str]):
        """
        Initialize WebSocket manager.
        
        Args:
            symbols: List of trading symbols to subscribe to
        """
        self.symbols = [s.lower() for s in symbols]
        self.settings = get_settings()
        
        self._connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._running = False
        self._retry_counts: Dict[str, int] = {}
        
        # Callbacks
        self.on_candle: Optional[Callable[[Candle], None]] = None
        self.on_mark_price: Optional[Callable[[MarkPrice], None]] = None
        self.on_book_ticker: Optional[Callable[[BookTicker], None]] = None
        self.on_user_data: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[str], None]] = None
    
    @property
    def base_url(self) -> str:
        """Get WebSocket base URL."""
        return self.settings.binance_ws_url
    
    def _build_stream_url(self, streams: List[str]) -> str:
        """Build combined stream URL."""
        stream_names = "/".join(streams)
        return f"{self.base_url}/stream?streams={stream_names}"
    
    async def start(self) -> None:
        """Start all WebSocket streams."""
        if self._running:
            logger.warning("WebSocket manager already running")
            return
        
        self._running = True
        logger.info(f"Starting WebSocket manager for symbols: {self.symbols}")
        
        # Build stream list for all symbols
        streams = []
        for symbol in self.symbols:
            streams.extend([
                f"{symbol}@kline_1m",
                f"{symbol}@kline_5m",
                f"{symbol}@kline_15m",
                f"{symbol}@markPrice@1s",
                f"{symbol}@bookTicker",
            ])
        
        # Start combined stream
        task = asyncio.create_task(self._run_stream("combined", streams))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
    
    async def stop(self) -> None:
        """Stop all WebSocket streams gracefully."""
        self._running = False
        
        # Close all connections
        for name, ws in self._connections.items():
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")
        
        self._connections.clear()
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("WebSocket manager stopped")
    
    async def _run_stream(self, name: str, streams: List[str]) -> None:
        """Run a WebSocket stream with auto-reconnect."""
        retry_delay = self.INITIAL_RETRY_DELAY
        
        while self._running:
            try:
                url = self._build_stream_url(streams)
                logger.info(f"Connecting to stream: {name}")
                
                async with websockets.connect(
                    url,
                    ping_interval=self.PING_INTERVAL,
                    ping_timeout=30,
                ) as ws:
                    self._connections[name] = ws
                    retry_delay = self.INITIAL_RETRY_DELAY
                    self._retry_counts[name] = 0
                    
                    logger.info(f"Stream connected: {name}")
                    if self.on_connect:
                        self.on_connect()
                    
                    await self._handle_messages(name, ws)
                    
            except ConnectionClosed as e:
                logger.warning(f"Stream {name} closed: {e}")
            except Exception as e:
                logger.error(f"Stream {name} error: {e}")
            
            # Cleanup
            self._connections.pop(name, None)
            
            if self.on_disconnect:
                self.on_disconnect(name)
            
            # Reconnect with backoff
            if self._running:
                self._retry_counts[name] = self._retry_counts.get(name, 0) + 1
                logger.info(f"Reconnecting {name} in {retry_delay:.1f}s (attempt {self._retry_counts[name]})")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.MAX_RETRY_DELAY)
    
    async def _handle_messages(
        self,
        name: str,
        ws: websockets.WebSocketClientProtocol,
    ) -> None:
        """Process incoming WebSocket messages."""
        async for message in ws:
            try:
                data = json.loads(message)
                
                # Combined stream format: {"stream": "btcusdt@kline_1m", "data": {...}}
                if "stream" in data:
                    stream_name = data["stream"]
                    payload = data["data"]
                    await self._dispatch_message(stream_name, payload)
                else:
                    # Single stream format
                    await self._dispatch_message(name, data)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Message handling error: {e}")
    
    async def _dispatch_message(self, stream: str, data: Dict[str, Any]) -> None:
        """Route message to appropriate callback."""
        try:
            if "@kline" in stream:
                candle = self._parse_kline(data)
                if self.on_candle:
                    self.on_candle(candle)
            
            elif "@markPrice" in stream:
                mark = self._parse_mark_price(data)
                if self.on_mark_price:
                    self.on_mark_price(mark)
            
            elif "@bookTicker" in stream:
                ticker = self._parse_book_ticker(data)
                if self.on_book_ticker:
                    self.on_book_ticker(ticker)
            
            elif data.get("e") in ("ORDER_TRADE_UPDATE", "ACCOUNT_UPDATE"):
                if self.on_user_data:
                    self.on_user_data(data)
                    
        except Exception as e:
            logger.error(f"Dispatch error for {stream}: {e}")
    
    def _parse_kline(self, data: Dict[str, Any]) -> Candle:
        """Parse kline/candlestick data."""
        k = data["k"]
        return Candle(
            symbol=k["s"],
            interval=k["i"],
            time=k["t"],
            open=Decimal(k["o"]),
            high=Decimal(k["h"]),
            low=Decimal(k["l"]),
            close=Decimal(k["c"]),
            volume=Decimal(k["v"]),
            close_time=k["T"],
            is_closed=k["x"],
        )
    
    def _parse_mark_price(self, data: Dict[str, Any]) -> MarkPrice:
        """Parse mark price data."""
        return MarkPrice(
            symbol=data["s"],
            price=Decimal(data["p"]),
            index_price=Decimal(data["i"]),
            funding_rate=Decimal(data["r"]),
            next_funding_time=data["T"],
        )
    
    def _parse_book_ticker(self, data: Dict[str, Any]) -> BookTicker:
        """Parse book ticker data."""
        return BookTicker(
            symbol=data["s"],
            bid_price=Decimal(data["b"]),
            bid_qty=Decimal(data["B"]),
            ask_price=Decimal(data["a"]),
            ask_qty=Decimal(data["A"]),
        )
    
    @property
    def is_connected(self) -> bool:
        """Check if main stream is connected."""
        return "combined" in self._connections
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all streams."""
        return {name: True for name in self._connections}
