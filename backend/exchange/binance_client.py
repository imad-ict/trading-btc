"""
Binance Futures Client - Unified REST API wrapper.

Supports both Testnet and Live with automatic URL switching.
Implements rate limiting, retry logic, and order reconciliation.
"""
import asyncio
import hmac
import hashlib
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from config import get_settings, RiskConstants

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    """Result of an order execution."""
    order_id: str
    symbol: str
    side: str
    type: str
    status: str
    price: Optional[Decimal]
    avg_price: Optional[Decimal]
    quantity: Decimal
    executed_qty: Decimal
    timestamp: int


@dataclass
class PositionInfo:
    """Current position information."""
    symbol: str
    side: str  # "LONG", "SHORT", "NONE"
    size: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal
    leverage: int
    margin_type: str
    liquidation_price: Optional[Decimal]


class BinanceClientError(Exception):
    """Base exception for Binance client errors."""
    pass


class BinanceClient:
    """
    Async Binance Futures REST API client.
    
    Features:
    - Testnet/Live URL switching
    - Request signing (HMAC-SHA256)
    - Rate limiting with exponential backoff
    - Order reconciliation
    """
    
    # Rate limiting
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    REQUEST_TIMEOUT = 30.0
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the Binance client.
        
        Args:
            api_key: Decrypted API key
            api_secret: Decrypted API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.settings = get_settings()
        
        self._client: Optional[httpx.AsyncClient] = None
        self._recv_window = 5000
        
        logger.info(f"BinanceClient initialized for mode: {self.settings.binance_mode}")
    
    @property
    def base_url(self) -> str:
        """Get the base URL for the current mode."""
        return self.settings.binance_base_url
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.REQUEST_TIMEOUT,
                headers={"X-MBX-APIKEY": self.api_key},
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    def _sign_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamp and signature to request parameters."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window
        
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        params["signature"] = signature
        return params
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a request to the Binance API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            params: Request parameters
            signed: Whether to sign the request
            
        Returns:
            JSON response as dictionary
        """
        client = await self._get_client()
        params = params or {}
        
        if signed:
            params = self._sign_request(params)
        
        for attempt in range(self.MAX_RETRIES):
            try:
                if method == "GET":
                    response = await client.get(endpoint, params=params)
                elif method == "POST":
                    response = await client.post(endpoint, params=params)
                elif method == "DELETE":
                    response = await client.delete(endpoint, params=params)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                # Raise for other errors
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise BinanceClientError(f"Request failed after {self.MAX_RETRIES} attempts: {e}")
        
        raise BinanceClientError("Max retries exceeded")
    
    # ═══════════════════════════════════════════════════════════════════
    # ACCOUNT ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════
    
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """
        Get account balance for all assets.
        
        Returns:
            Dict mapping asset to available balance
        """
        response = await self._request("GET", "/fapi/v2/balance", signed=True)
        
        return {
            asset["asset"]: Decimal(asset["availableBalance"])
            for asset in response
        }
    
    async def get_usdt_balance(self) -> Decimal:
        """Get USDT available balance."""
        balances = await self.get_account_balance()
        return balances.get("USDT", Decimal("0"))
    
    async def get_position_risk(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """
        Get current position information.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of PositionInfo objects
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        response = await self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)
        
        positions = []
        for pos in response:
            size = Decimal(pos["positionAmt"])
            if size != 0:
                positions.append(PositionInfo(
                    symbol=pos["symbol"],
                    side="LONG" if size > 0 else "SHORT",
                    size=abs(size),
                    entry_price=Decimal(pos["entryPrice"]),
                    unrealized_pnl=Decimal(pos["unRealizedProfit"]),
                    leverage=int(pos["leverage"]),
                    margin_type=pos["marginType"],
                    liquidation_price=Decimal(pos["liquidationPrice"]) if pos["liquidationPrice"] != "0" else None,
                ))
        
        return positions
    
    # ═══════════════════════════════════════════════════════════════════
    # MARKET DATA ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════
    
    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current mark price for a symbol."""
        response = await self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
        return Decimal(response["price"])
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Get historical klines/candlesticks.
        
        Args:
            symbol: Trading symbol
            interval: Kline interval (1m, 5m, 15m, etc.)
            limit: Number of candles (max 1500)
            
        Returns:
            List of candle dictionaries
        """
        response = await self._request("GET", "/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500),
        })
        
        return [
            {
                "time": candle[0],
                "open": Decimal(candle[1]),
                "high": Decimal(candle[2]),
                "low": Decimal(candle[3]),
                "close": Decimal(candle[4]),
                "volume": Decimal(candle[5]),
                "close_time": candle[6],
                "quote_volume": Decimal(candle[7]),
                "trades": candle[8],
            }
            for candle in response
        ]
    
    async def get_book_ticker(self, symbol: str) -> Dict[str, Decimal]:
        """
        Get best bid/ask prices for spread calculation.
        
        Returns:
            Dict with bid_price, bid_qty, ask_price, ask_qty
        """
        response = await self._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        
        return {
            "bid_price": Decimal(response["bidPrice"]),
            "bid_qty": Decimal(response["bidQty"]),
            "ask_price": Decimal(response["askPrice"]),
            "ask_qty": Decimal(response["askQty"]),
        }
    
    async def get_24h_volume(self, symbol: str) -> Decimal:
        """Get 24h quote volume for symbol selection."""
        response = await self._request("GET", "/fapi/v1/ticker/24hr", {"symbol": symbol})
        return Decimal(response["quoteVolume"])
    
    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get exchange trading rules and symbol info."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v1/exchangeInfo", params)
    
    # ═══════════════════════════════════════════════════════════════════
    # ORDER ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        Enforces max leverage from RiskConstants.
        """
        leverage = min(leverage, RiskConstants.MAX_LEVERAGE)
        
        return await self._request("POST", "/fapi/v1/leverage", {
            "symbol": symbol,
            "leverage": leverage,
        }, signed=True)
    
    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
        """
        Set margin type for a symbol.
        
        Always enforces ISOLATED margin (from RiskConstants).
        """
        try:
            return await self._request("POST", "/fapi/v1/marginType", {
                "symbol": symbol,
                "marginType": RiskConstants.MARGIN_TYPE,
            }, signed=True)
        except BinanceClientError as e:
            # Ignore "No need to change margin type" error
            if "No need to change" in str(e):
                return {"msg": "Already set to ISOLATED"}
            raise
    
    async def create_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: Decimal,
        reduce_only: bool = False,
    ) -> OrderResult:
        """
        Create a market order.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Position size
            reduce_only: If true, only reduces position
            
        Returns:
            OrderResult with execution details
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": str(quantity),
        }
        
        if reduce_only:
            params["reduceOnly"] = "true"
        
        response = await self._request("POST", "/fapi/v1/order", params, signed=True)
        
        return OrderResult(
            order_id=str(response["orderId"]),
            symbol=response["symbol"],
            side=response["side"],
            type=response["type"],
            status=response["status"],
            price=Decimal(response["price"]) if response["price"] != "0" else None,
            avg_price=Decimal(response["avgPrice"]) if response.get("avgPrice") else None,
            quantity=Decimal(response["origQty"]),
            executed_qty=Decimal(response["executedQty"]),
            timestamp=response["updateTime"],
        )
    
    async def create_stop_loss(
        self,
        symbol: str,
        side: str,  # "SELL" for long SL, "BUY" for short SL
        quantity: Decimal,
        stop_price: Decimal,
    ) -> OrderResult:
        """
        Create a stop-loss market order.
        
        Args:
            symbol: Trading symbol
            side: Opposite of position side
            quantity: Full position size
            stop_price: Trigger price for SL
            
        Returns:
            OrderResult with order details
        """
        response = await self._request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": str(stop_price),
            "quantity": str(quantity),
            "reduceOnly": "true",
        }, signed=True)
        
        return OrderResult(
            order_id=str(response["orderId"]),
            symbol=response["symbol"],
            side=response["side"],
            type=response["type"],
            status=response["status"],
            price=Decimal(response["stopPrice"]) if response.get("stopPrice") else None,
            avg_price=None,
            quantity=Decimal(response["origQty"]),
            executed_qty=Decimal(response["executedQty"]),
            timestamp=response["updateTime"],
        )
    
    async def create_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        take_profit_price: Decimal,
    ) -> OrderResult:
        """Create a take-profit market order."""
        response = await self._request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": str(take_profit_price),
            "quantity": str(quantity),
            "reduceOnly": "true",
        }, signed=True)
        
        return OrderResult(
            order_id=str(response["orderId"]),
            symbol=response["symbol"],
            side=response["side"],
            type=response["type"],
            status=response["status"],
            price=Decimal(response["stopPrice"]) if response.get("stopPrice") else None,
            avg_price=None,
            quantity=Decimal(response["origQty"]),
            executed_qty=Decimal(response["executedQty"]),
            timestamp=response["updateTime"],
        )
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        return await self._request("DELETE", "/fapi/v1/order", {
            "symbol": symbol,
            "orderId": order_id,
        }, signed=True)
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol."""
        return await self._request("DELETE", "/fapi/v1/allOpenOrders", {
            "symbol": symbol,
        }, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v1/openOrders", params, signed=True)
    
    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order status by ID."""
        return await self._request("GET", "/fapi/v1/order", {
            "symbol": symbol,
            "orderId": order_id,
        }, signed=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # EMERGENCY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    async def emergency_close_all(self, symbol: str) -> Dict[str, Any]:
        """
        Emergency close all positions and cancel all orders for a symbol.
        
        Used by kill-switch.
        """
        results = {"cancelled_orders": None, "closed_position": None}
        
        # Cancel all open orders first
        try:
            results["cancelled_orders"] = await self.cancel_all_orders(symbol)
        except BinanceClientError as e:
            logger.error(f"Failed to cancel orders: {e}")
        
        # Close any open position
        positions = await self.get_position_risk(symbol)
        for pos in positions:
            if pos.size > 0:
                side = "SELL" if pos.side == "LONG" else "BUY"
                try:
                    results["closed_position"] = await self.create_market_order(
                        symbol=symbol,
                        side=side,
                        quantity=pos.size,
                        reduce_only=True,
                    )
                except BinanceClientError as e:
                    logger.error(f"Failed to close position: {e}")
        
        return results
    
    async def reconcile_position(self, symbol: str) -> Optional[PositionInfo]:
        """
        Reconcile local state with exchange reality.
        
        Call this on reconnect to ensure consistency.
        """
        positions = await self.get_position_risk(symbol)
        if positions:
            return positions[0]
        return None
