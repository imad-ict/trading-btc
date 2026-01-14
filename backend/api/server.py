"""
FastAPI Server - Dashboard API (Port 8001)

Provides:
- REST endpoints for status, trades, metrics
- WebSocket /ws/live for real-time updates
- Background tasks for price/position sync
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_settings, RiskConstants
from database import init_database, get_db_session, Trade, BotState, DailySummary
from database.models import TradeStatus

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS FOR API
# ═══════════════════════════════════════════════════════════════════

class StatusResponse(BaseModel):
    """Bot status response."""
    status: str  # running, stopped, halted
    mode: str    # testnet, live
    session: str
    balance: float
    daily_pnl: float
    daily_pnl_pct: float
    trades_today: int
    max_trades: int
    losses_today: int
    max_losses: int
    current_streak: int
    active_symbols: List[str]
    is_halted: bool
    halt_reason: Optional[str]


class TradeResponse(BaseModel):
    """Trade record response."""
    id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    status: str
    result: Optional[str]
    pnl_usd: Optional[float]
    pnl_pct: Optional[float]
    
    # Explanation
    liquidity_event: str
    market_state: str
    entry_logic: str
    
    # Timestamps
    entry_time: Optional[str]
    exit_time: Optional[str]


class MetricsResponse(BaseModel):
    """Performance metrics response."""
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    avg_rr: float
    total_pnl: float
    max_drawdown: float


class EmergencyStopRequest(BaseModel):
    """Emergency stop request."""
    confirm: bool


class SettingsRequest(BaseModel):
    """Settings update request."""
    mode: str
    testnet_api_key: str = ""
    testnet_api_secret: str = ""
    live_api_key: str = ""
    live_api_secret: str = ""
    symbols: List[str]


class ConnectionTestRequest(BaseModel):
    """Connection test request."""
    mode: str
    api_key: str
    api_secret: str


# Settings storage (in-memory, persisted to file)
import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"

def load_settings() -> dict:
    """Load settings from file."""
    default = {
        "mode": "testnet",
        "testnet_api_key": "",
        "testnet_api_secret": "",
        "live_api_key": "",
        "live_api_secret": "",
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    }
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE) as f:
                return {**default, **json.load(f)}
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
    return default

def save_settings(settings: dict) -> None:
    """Save settings to file."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

# Load on startup
app_settings = load_settings()


# ═══════════════════════════════════════════════════════════════════
# WEBSOCKET CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        
        for conn in disconnected:
            self.active_connections.discard(conn)


manager = ConnectionManager()

# Global state (will be set by main bot)
bot_state: Dict[str, Any] = {
    "status": "stopped",
    "session": "NONE",
    "balance": 0.0,
    "daily_pnl": 0.0,
    "trades_today": 0,
    "wins_today": 0,
    "losses_today": 0,
    "streak": 0,
    "is_halted": False,
    "halt_reason": None,
    "active_trade": None,
    "price": {},  # symbol -> price
    "trades": [],  # Trade history list
}


# ═══════════════════════════════════════════════════════════════════
# LIFESPAN AND APP FACTORY
# ═══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting API server...")
    
    # Initialize database
    await init_database()
    
    # Start background tasks
    price_task = asyncio.create_task(price_updater())
    status_task = asyncio.create_task(status_updater())
    
    yield
    
    # Cleanup
    price_task.cancel()
    status_task.cancel()
    
    try:
        await price_task
        await status_task
    except asyncio.CancelledError:
        pass
    
    logger.info("API server stopped")


async def price_updater():
    """Background task to broadcast price updates."""
    while True:
        try:
            if bot_state["price"]:
                await manager.broadcast({
                    "type": "price",
                    "data": bot_state["price"],
                    "timestamp": datetime.utcnow().isoformat(),
                })
            await asyncio.sleep(1)  # 1 second updates
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Price updater error: {e}")
            await asyncio.sleep(5)


async def status_updater():
    """Background task to broadcast status updates."""
    while True:
        try:
            await manager.broadcast({
                "type": "status",
                "data": {
                    "status": bot_state["status"],
                    "session": bot_state["session"],
                    "balance": bot_state["balance"],
                    "daily_pnl": bot_state["daily_pnl"],
                    "trades_today": bot_state["trades_today"],
                    "losses_today": bot_state["losses_today"],
                    "streak": bot_state["streak"],
                    "is_halted": bot_state["is_halted"],
                },
                "timestamp": datetime.utcnow().isoformat(),
            })
            await asyncio.sleep(5)  # 5 second updates
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Status updater error: {e}")
            await asyncio.sleep(5)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Institutional Trading Platform API",
        description="Dashboard API for the crypto trading platform",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS - Allow all origins for Render deployment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ═══════════════════════════════════════════════════════════════════
# STATIC FILE SERVING (Frontend)
# ═══════════════════════════════════════════════════════════════════

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Check if frontend build exists
FRONTEND_DIR = Path(__file__).parent.parent / "static"

if FRONTEND_DIR.exists():
    # Serve static files from /static
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    
    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the frontend index.html."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend not found. API is running at /api/status"}
    
    @app.get("/{path:path}", include_in_schema=False)
    async def serve_frontend(path: str):
        """Serve frontend files or fallback to index.html for SPA routing."""
        # Skip API routes
        if path.startswith("api/") or path.startswith("ws/"):
            return {"detail": "Not found"}
        
        file_path = FRONTEND_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        # SPA fallback
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        
        return {"detail": "Not found"}


# ═══════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current bot status and daily state."""
    settings = get_settings()
    
    return StatusResponse(
        status=bot_state["status"],
        mode=settings.binance_mode,
        session=bot_state["session"],
        balance=bot_state["balance"],
        daily_pnl=bot_state["daily_pnl"],
        daily_pnl_pct=(bot_state["daily_pnl"] / bot_state["balance"] * 100) if bot_state["balance"] > 0 else 0,
        trades_today=bot_state["trades_today"],
        max_trades=RiskConstants.MAX_TRADES_PER_DAY,
        losses_today=bot_state["losses_today"],
        max_losses=RiskConstants.MAX_LOSSES_PER_DAY,
        current_streak=bot_state["streak"],
        active_symbols=settings.symbols,
        is_halted=bot_state["is_halted"],
        halt_reason=bot_state["halt_reason"],
    )


@app.get("/api/trades", response_model=List[TradeResponse])
async def get_trades(limit: int = 50, status: Optional[str] = None):
    """Get trade history."""
    from sqlalchemy import select, desc
    
    async with get_db_session() as session:
        query = select(Trade).order_by(desc(Trade.created_at)).limit(limit)
        
        if status:
            query = query.where(Trade.status == status)
        
        result = await session.execute(query)
        trades = result.scalars().all()
        
        return [
            TradeResponse(
                id=str(trade.id),
                symbol=trade.symbol,
                direction=trade.direction.value,
                entry_price=float(trade.entry_price),
                exit_price=float(trade.exit_price) if trade.exit_price else None,
                stop_loss=float(trade.stop_loss),
                status=trade.status.value,
                result=trade.result.value if trade.result else None,
                pnl_usd=float(trade.pnl_usd) if trade.pnl_usd else None,
                pnl_pct=float(trade.pnl_pct) if trade.pnl_pct else None,
                liquidity_event=trade.liquidity_event,
                market_state=trade.market_state,
                entry_logic=trade.entry_logic,
                entry_time=trade.entry_time.isoformat() if trade.entry_time else None,
                exit_time=trade.exit_time.isoformat() if trade.exit_time else None,
            )
            for trade in trades
        ]


@app.get("/api/trades/{trade_id}", response_model=TradeResponse)
async def get_trade(trade_id: str):
    """Get single trade details."""
    from sqlalchemy import select
    import uuid
    
    async with get_db_session() as session:
        result = await session.execute(
            select(Trade).where(Trade.id == uuid.UUID(trade_id))
        )
        trade = result.scalar_one_or_none()
        
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        return TradeResponse(
            id=str(trade.id),
            symbol=trade.symbol,
            direction=trade.direction.value,
            entry_price=float(trade.entry_price),
            exit_price=float(trade.exit_price) if trade.exit_price else None,
            stop_loss=float(trade.stop_loss),
            status=trade.status.value,
            result=trade.result.value if trade.result else None,
            pnl_usd=float(trade.pnl_usd) if trade.pnl_usd else None,
            pnl_pct=float(trade.pnl_pct) if trade.pnl_pct else None,
            liquidity_event=trade.liquidity_event,
            market_state=trade.market_state,
            entry_logic=trade.entry_logic,
            entry_time=trade.entry_time.isoformat() if trade.entry_time else None,
            exit_time=trade.exit_time.isoformat() if trade.exit_time else None,
        )


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics."""
    from sqlalchemy import select, func
    
    async with get_db_session() as session:
        # Count trades
        result = await session.execute(
            select(
                func.count(Trade.id).label("total"),
                func.sum(Trade.pnl_usd).label("total_pnl"),
            ).where(Trade.status == TradeStatus.CLOSED)
        )
        row = result.one()
        total_trades = row.total or 0
        total_pnl = float(row.total_pnl or 0)
        
        # Count wins/losses
        wins_result = await session.execute(
            select(func.count(Trade.id)).where(
                Trade.result == "WIN"
            )
        )
        wins = wins_result.scalar() or 0
        
        losses_result = await session.execute(
            select(func.count(Trade.id)).where(
                Trade.result == "LOSS"
            )
        )
        losses = losses_result.scalar() or 0
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Simple profit factor approximation
        profit_factor = (wins / losses) if losses > 0 else wins
        
        return MetricsResponse(
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_rr=1.5,  # TODO: Calculate from actual trades
            total_pnl=total_pnl,
            max_drawdown=0.0,  # TODO: Calculate from equity curve
        )


@app.get("/api/trades")
async def get_trades():
    """Get all trades history."""
    return {
        "trades": bot_state["trades"],
        "trades_today": bot_state["trades_today"],
        "wins_today": bot_state["wins_today"],
        "losses_today": bot_state["losses_today"],
        "streak": bot_state["streak"],
        "daily_pnl": bot_state["daily_pnl"]
    }


@app.get("/api/active-trade")
async def get_active_trade():
    """Get current active trade if any."""
    if bot_state["active_trade"]:
        return {"active": True, "trade": bot_state["active_trade"]}
    return {"active": False, "trade": None}


@app.post("/api/emergency-stop")
async def emergency_stop(request: EmergencyStopRequest):
    """Emergency stop - close all positions on Binance."""
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")
    
    # Set halt state
    bot_state["is_halted"] = True
    bot_state["halt_reason"] = "Emergency stop triggered via API"
    bot_state["status"] = "halted"
    
    # Close all positions on Binance
    close_result = await _close_all_binance_positions()
    
    # Stop the bot
    await bot_runner.stop()
    
    # Broadcast immediately
    await manager.broadcast({
        "type": "emergency_stop",
        "data": {"halted": True, "positions_closed": close_result},
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    logger.warning(f"EMERGENCY STOP TRIGGERED VIA API - Positions closed: {close_result}")
    
    return {"success": True, "message": "Emergency stop triggered", "positions_closed": close_result}


async def _close_all_binance_positions():
    """Close all open positions on Binance."""
    mode = app_settings.get("mode", "testnet")
    api_key = app_settings.get(f"{mode}_api_key", "")
    api_secret = app_settings.get(f"{mode}_api_secret", "")
    
    if not api_key or not api_secret:
        return {"error": "No API keys configured"}
    
    if mode == "testnet":
        base_url = "https://testnet.binancefuture.com"
    else:
        base_url = "https://fapi.binance.com"
    
    try:
        import hmac
        import hashlib
        import time
        
        # Get open positions
        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}"
        signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{base_url}/fapi/v2/positionRisk",
                params={"timestamp": timestamp, "signature": signature},
                headers={"X-MBX-APIKEY": api_key}
            )
            
            if response.status_code != 200:
                return {"error": response.text}
            
            positions = response.json()
            closed = []
            
            for pos in positions:
                amt = float(pos.get("positionAmt", 0))
                if amt != 0:
                    symbol = pos["symbol"]
                    side = "SELL" if amt > 0 else "BUY"
                    quantity = abs(amt)
                    
                    # Close position
                    timestamp = int(time.time() * 1000)
                    params = {
                        "symbol": symbol,
                        "side": side,
                        "type": "MARKET",
                        "quantity": quantity,
                        "timestamp": timestamp,
                    }
                    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
                    params["signature"] = signature
                    
                    close_response = await client.post(
                        f"{base_url}/fapi/v1/order",
                        params=params,
                        headers={"X-MBX-APIKEY": api_key}
                    )
                    
                    if close_response.status_code == 200:
                        closed.append(symbol)
                        logger.info(f"Closed position: {symbol} {side} {quantity}")
                    else:
                        logger.error(f"Failed to close {symbol}: {close_response.text}")
            
            return {"closed": closed, "count": len(closed)}
            
    except Exception as e:
        logger.error(f"Error closing positions: {e}")
        return {"error": str(e)}


@app.get("/api/settings")
async def get_settings_endpoint():
    """Get current settings."""
    return app_settings


@app.post("/api/settings")
async def save_settings_endpoint(request: SettingsRequest):
    """Save settings."""
    global app_settings
    
    app_settings = {
        "mode": request.mode,
        "testnet_api_key": request.testnet_api_key,
        "testnet_api_secret": request.testnet_api_secret,
        "live_api_key": request.live_api_key,
        "live_api_secret": request.live_api_secret,
        "symbols": request.symbols[:3],  # Max 3 symbols
    }
    
    save_settings(app_settings)
    logger.info(f"Settings saved: mode={request.mode}, symbols={request.symbols}")
    
    return {"success": True, "message": "Settings saved"}


@app.post("/api/test-connection")
async def test_connection(request: ConnectionTestRequest):
    """Test Binance API connection."""
    import httpx
    import hmac
    import hashlib
    import time
    
    if not request.api_key or not request.api_secret:
        return {"success": False, "message": "API key and secret required"}
    
    # Select base URL
    if request.mode == "testnet":
        base_url = "https://testnet.binancefuture.com"
    else:
        base_url = "https://fapi.binance.com"
    
    try:
        # Create signed request
        timestamp = int(time.time() * 1000)
        query = f"timestamp={timestamp}"
        signature = hmac.new(
            request.api_secret.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{base_url}/fapi/v2/balance",
                params={"timestamp": timestamp, "signature": signature},
                headers={"X-MBX-APIKEY": request.api_key}
            )
            
            if response.status_code == 200:
                balances = response.json()
                usdt = next((b for b in balances if b["asset"] == "USDT"), None)
                balance = float(usdt["balance"]) if usdt else 0
                return {
                    "success": True,
                    "message": f"Connected! Balance: ${balance:.2f}",
                    "balance": balance
                }
            else:
                error = response.json().get("msg", response.text)
                return {"success": False, "message": f"API Error: {error}"}
    
    except httpx.TimeoutException:
        return {"success": False, "message": "Connection timeout"}
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return {"success": False, "message": str(e)}


# ═══════════════════════════════════════════════════════════════════
# BOT CONTROL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

from .bot_runner import bot_runner

# Set up callbacks when module loads
def _update_price(symbol: str, price: float):
    bot_state["price"][symbol] = price

def _update_bot_status(status: str):
    bot_state["status"] = status

def _log_callback(message: str):
    """Broadcast log messages to dashboard."""
    import asyncio
    asyncio.create_task(manager.broadcast({
        "type": "log",
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }))

def _trade_callback(trade: dict):
    """Broadcast trade updates to dashboard."""
    import asyncio
    
    # Handle trade close - persist to history and update metrics
    if trade.get("event") == "close":
        # Add to trade history
        trade_record = {
            "id": str(len(bot_state["trades"]) + 1),
            "symbol": trade.get("symbol"),
            "direction": trade.get("direction"),
            "entry_price": trade.get("entry"),
            "exit_price": trade.get("exit"),
            "stop_loss": trade.get("sl"),
            "status": "closed",
            "result": trade.get("result"),
            "pnl_usd": trade.get("pnl_usd"),
            "pnl_pct": trade.get("pnl_pct"),
            "liquidity_event": trade.get("reason", ""),
            "market_state": "expansion",
            "entry_logic": trade.get("reason", ""),
            "exit_time": datetime.utcnow().isoformat()
        }
        bot_state["trades"].append(trade_record)
        
        # Update counters
        bot_state["trades_today"] += 1
        if trade.get("result") == "WIN":
            bot_state["wins_today"] += 1
            bot_state["streak"] = max(0, bot_state["streak"]) + 1
        else:
            bot_state["losses_today"] += 1
            bot_state["streak"] = min(0, bot_state["streak"]) - 1
        
        # Update daily P&L
        bot_state["daily_pnl"] += trade.get("pnl_usd", 0)
        
        # Clear active trade
        bot_state["active_trade"] = None
    else:
        # Trade opened
        bot_state["active_trade"] = trade
    
    # Broadcast to frontend
    asyncio.create_task(manager.broadcast({
        "type": "trade",
        "data": trade,
        "timestamp": datetime.utcnow().isoformat()
    }))

bot_runner.set_callbacks(_update_price, _update_bot_status, _trade_callback, _log_callback)


@app.post("/api/bot/start")
async def start_bot():
    """Start the trading bot."""
    if not app_settings.get("testnet_api_key") and not app_settings.get("live_api_key"):
        raise HTTPException(status_code=400, detail="API keys not configured. Open Settings first.")
    
    # Get balance first
    balance = await bot_runner.get_balance()
    if balance is not None:
        bot_state["balance"] = balance
    
    success = await bot_runner.start(app_settings)
    
    if success:
        return {"success": True, "message": "Bot started", "balance": balance}
    else:
        raise HTTPException(status_code=400, detail="Bot is already running")


@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot."""
    success = await bot_runner.stop()
    
    if success:
        return {"success": True, "message": "Bot stopped"}
    else:
        raise HTTPException(status_code=400, detail="Bot is not running")


@app.get("/api/bot/prices")
async def get_prices():
    """Get current prices from bot."""
    return {"prices": bot_state["price"]}


@app.get("/api/balance")
async def get_balance():
    """Get real balance from Binance API."""
    import httpx
    import hmac
    import hashlib
    import time
    
    mode = app_settings.get("mode", "testnet")
    api_key = app_settings.get(f"{mode}_api_key", "")
    api_secret = app_settings.get(f"{mode}_api_secret", "")
    
    if not api_key or not api_secret:
        return {"success": False, "balance": 0, "message": "API keys not configured"}
    
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
                balance = float(usdt["balance"]) if usdt else 0
                
                # Update global state
                bot_state["balance"] = balance
                
                return {
                    "success": True,
                    "balance": balance,
                    "mode": mode,
                    "message": f"Balance from {mode.upper()}"
                }
            else:
                error = response.json().get("msg", response.text)
                return {"success": False, "balance": 0, "message": f"API Error: {error}"}
    
    except Exception as e:
        logger.error(f"Balance fetch error: {e}")
        return {"success": False, "balance": 0, "message": str(e)}


# ═══════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════════════════════════

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "init",
            "data": {
                "status": bot_state["status"],
                "balance": bot_state["balance"],
                "daily_pnl": bot_state["daily_pnl"],
            },
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ═══════════════════════════════════════════════════════════════════
# STATE UPDATE FUNCTIONS (called by main bot)
# ═══════════════════════════════════════════════════════════════════

def update_price(symbol: str, price: float) -> None:
    """Update price for a symbol."""
    bot_state["price"][symbol] = price


def update_status(
    status: Optional[str] = None,
    session: Optional[str] = None,
    balance: Optional[float] = None,
    daily_pnl: Optional[float] = None,
    trades_today: Optional[int] = None,
    losses_today: Optional[int] = None,
    streak: Optional[int] = None,
    is_halted: Optional[bool] = None,
    halt_reason: Optional[str] = None,
) -> None:
    """Update bot status."""
    if status is not None:
        bot_state["status"] = status
    if session is not None:
        bot_state["session"] = session
    if balance is not None:
        bot_state["balance"] = balance
    if daily_pnl is not None:
        bot_state["daily_pnl"] = daily_pnl
    if trades_today is not None:
        bot_state["trades_today"] = trades_today
    if losses_today is not None:
        bot_state["losses_today"] = losses_today
    if streak is not None:
        bot_state["streak"] = streak
    if is_halted is not None:
        bot_state["is_halted"] = is_halted
    if halt_reason is not None:
        bot_state["halt_reason"] = halt_reason


def update_active_trade(trade: Optional[Dict[str, Any]]) -> None:
    """Update active trade state."""
    bot_state["active_trade"] = trade


async def broadcast_trade_update(trade_data: Dict[str, Any]) -> None:
    """Broadcast trade update to all connected clients."""
    await manager.broadcast({
        "type": "trade",
        "data": trade_data,
        "timestamp": datetime.utcnow().isoformat(),
    })
