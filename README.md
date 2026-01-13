# Institutional Trading Platform

Production-grade crypto trading platform for Binance Futures USDT-M with institutional-level risk management.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- PostgreSQL (or use Docker Compose)
- Python 3.11+ (for local development)
- Node.js 20+ (for frontend development)

### 1. Environment Setup

```bash
# Copy environment template
cp backend/.env.template backend/.env

# Generate encryption key (for API key security)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 2. Run with Docker Compose

```bash
docker-compose up -d
```

This starts:
- PostgreSQL on port 5432
- Backend API on port 8001
- Frontend Dashboard on port 3000

### 3. Local Development

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Status   │ │ Active   │ │ Trade    │ │ Explanation      │   │
│  │ Header   │ │ Trade    │ │ History  │ │ Log              │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ WebSocket + REST
┌───────────────────────┴─────────────────────────────────────────┐
│                     BACKEND (FastAPI)                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               API Server (Port 8001)                      │   │
│  │  • /api/status  • /api/trades  • /ws/live                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               Main Bot Orchestrator                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │   │
│  │  │ Session │ │ Market  │ │ Risk    │ │Execution│         │   │
│  │  │Controller│ │ Data   │ │ Engine  │ │ Engine  │         │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Strategy Engines                          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │   │
│  │  │ Market  │ │Liquidity│ │  Entry  │ │ SL/TP   │         │   │
│  │  │ State   │ │ Engine  │ │ Engine  │ │ Engine  │         │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
┌───┴───┐         ┌─────┴────┐        ┌─────┴────┐
│Binance│         │PostgreSQL│        │   Logs   │
│Futures│         │   DB     │        │          │
└───────┘         └──────────┘        └──────────┘
```

## Core Principles

1. **Market as Liquidity Auction** - Trade only after liquidity is consumed
2. **Stop-Loss Hunts are Structural** - Place SL at wick midpoint or structure
3. **First Move is a Trap** - Trade failed breakouts, not breakouts
4. **Risk is Pre-Validated** - Risk engine executes BEFORE strategy
5. **Every Trade Must Be Explainable** - Full trade explanation logging

## Risk Parameters (Non-Negotiable)

| Parameter | Value |
|-----------|-------|
| Risk per trade | 0.3% |
| Max trades/day | 10 |
| Max losses/day | 4 |
| Daily loss cap | $100 |
| Daily profit lock | $75 |
| Max leverage | 5× |
| Margin type | Isolated |
| SL distance | 0.15% - 0.35% |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/status | GET | Bot status and daily state |
| /api/trades | GET | Trade history |
| /api/trades/{id} | GET | Single trade details |
| /api/metrics | GET | Performance metrics |
| /api/active-trade | GET | Current active position |
| /api/emergency-stop | POST | Emergency close all |
| /ws/live | WS | Real-time updates |

## License

Proprietary - Internal use only
