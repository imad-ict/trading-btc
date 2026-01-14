import { create } from 'zustand'

interface Trade {
    id: string
    symbol: string
    direction: 'LONG' | 'SHORT'
    entry_price: number
    exit_price?: number
    stop_loss: number
    status: string
    result?: 'WIN' | 'LOSS' | 'BE'
    pnl_usd?: number
    pnl_pct?: number
    liquidity_event: string
    market_state: string
    entry_logic: string
    entry_time?: string
    exit_time?: string
}

interface ActiveTrade {
    symbol: string
    direction: 'LONG' | 'SHORT'
    entry_price: number
    stop_loss: number
    current_price?: number
    unrealized_pnl?: number
    unrealized_pnl_pct?: number
}

interface Metrics {
    total_trades: number
    wins: number
    losses: number
    win_rate: number
    profit_factor: number
    avg_rr: number
    total_pnl: number
    max_drawdown: number
}

interface TradeState {
    // Status
    status: 'running' | 'stopped' | 'halted'
    mode: 'testnet' | 'live'
    session: string
    balance: number
    dailyPnl: number
    dailyPnlPct: number
    tradesToday: number
    maxTrades: number
    lossesToday: number
    maxLosses: number
    currentStreak: number
    isHalted: boolean
    haltReason: string | null

    // Prices
    prices: Record<string, number>

    // Trades
    activePositions: Record<string, ActiveTrade>  // Dict by symbol
    trades: Trade[]

    // Metrics
    metrics: Metrics

    // Explanation logs
    explanationLogs: string[]

    // Actions
    setStatus: (status: 'running' | 'stopped' | 'halted') => void
    setSession: (session: string) => void
    setBalance: (balance: number) => void
    setDailyPnl: (pnl: number, pct: number) => void
    updatePrice: (symbol: string, price: number) => void
    addActivePosition: (symbol: string, trade: ActiveTrade) => void
    removeActivePosition: (symbol: string) => void
    setActiveTrade: (trade: ActiveTrade | null) => void  // Legacy support
    setTrades: (trades: Trade[]) => void
    addTrade: (trade: Trade) => void
    setMetrics: (metrics: Metrics) => void
    recalculateMetrics: () => void
    addExplanationLog: (log: string) => void
    setHalted: (halted: boolean, reason?: string) => void
    updateFromStatus: (data: any) => void
    fetchInitialData: () => Promise<void>
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

export const useTradeStore = create<TradeState>((set, get) => ({
    // Initial state
    status: 'stopped',
    mode: 'testnet',
    session: 'NONE',
    balance: 0,
    dailyPnl: 0,
    dailyPnlPct: 0,
    tradesToday: 0,
    maxTrades: 10,
    lossesToday: 0,
    maxLosses: 4,
    currentStreak: 0,
    isHalted: false,
    haltReason: null,
    prices: {},
    activePositions: {},
    trades: [],
    metrics: {
        total_trades: 0,
        wins: 0,
        losses: 0,
        win_rate: 0,
        profit_factor: 0,
        avg_rr: 0,
        total_pnl: 0,
        max_drawdown: 0,
    },
    explanationLogs: [],

    // Actions
    setStatus: (status) => set({ status }),
    setSession: (session) => set({ session }),
    setBalance: (balance) => set({ balance }),
    setDailyPnl: (pnl, pct) => set({ dailyPnl: pnl, dailyPnlPct: pct }),

    updatePrice: (symbol, price) => set((state) => {
        const newPositions = { ...state.activePositions }
        if (newPositions[symbol]) {
            const pos = newPositions[symbol]
            const pnl = pos.direction === 'LONG'
                ? (price - pos.entry_price)
                : (pos.entry_price - price)
            newPositions[symbol] = {
                ...pos,
                current_price: price,
                unrealized_pnl: pnl,
                unrealized_pnl_pct: (pnl / pos.entry_price) * 100
            }
        }
        return { prices: { ...state.prices, [symbol]: price }, activePositions: newPositions }
    }),

    addActivePosition: (symbol, trade) => set((state) => ({
        activePositions: { ...state.activePositions, [symbol]: trade }
    })),

    removeActivePosition: (symbol) => set((state) => {
        const newPositions = { ...state.activePositions }
        delete newPositions[symbol]
        return { activePositions: newPositions }
    }),

    // Legacy support for single trade
    setActiveTrade: (trade) => set((state) => {
        if (trade === null) {
            return { activePositions: {} }
        }
        return { activePositions: { [trade.symbol]: trade } }
    }),

    setTrades: (trades) => {
        set({ trades })
        get().recalculateMetrics()
    },

    addTrade: (trade) => {
        set((state) => ({ trades: [trade, ...state.trades] }))
        get().recalculateMetrics()
    },

    setMetrics: (metrics) => set({ metrics }),

    recalculateMetrics: () => set((state) => {
        const trades = state.trades.filter(t => t.status === 'closed')
        const wins = trades.filter(t => t.result === 'WIN').length
        const losses = trades.filter(t => t.result === 'LOSS').length
        const total = wins + losses
        const totalPnl = trades.reduce((sum, t) => sum + (t.pnl_usd || 0), 0)

        const winPnl = trades.filter(t => t.result === 'WIN').reduce((sum, t) => sum + (t.pnl_usd || 0), 0)
        const lossPnl = Math.abs(trades.filter(t => t.result === 'LOSS').reduce((sum, t) => sum + (t.pnl_usd || 0), 0))

        return {
            metrics: {
                total_trades: total,
                wins,
                losses,
                win_rate: total > 0 ? (wins / total) * 100 : 0,
                profit_factor: lossPnl > 0 ? winPnl / lossPnl : winPnl > 0 ? Infinity : 0,
                avg_rr: 1.5, // Placeholder
                total_pnl: totalPnl,
                max_drawdown: 0  // Placeholder
            },
            dailyPnl: totalPnl
        }
    }),

    addExplanationLog: (log) => set((state) => ({
        explanationLogs: [log, ...state.explanationLogs].slice(0, 50),
    })),

    setHalted: (halted, reason) => set({
        isHalted: halted,
        haltReason: reason || null,
        status: halted ? 'halted' : 'stopped',
    }),

    updateFromStatus: (data) => set({
        status: data.status || 'stopped',
        session: data.session || 'NONE',
        balance: data.balance || 0,
        dailyPnl: data.daily_pnl || 0,
        dailyPnlPct: data.daily_pnl_pct || 0,
        tradesToday: data.trades_today || 0,
        maxTrades: data.max_trades || 10,
        lossesToday: data.losses_today || 0,
        maxLosses: data.max_losses || 4,
        currentStreak: data.current_streak || 0,
        isHalted: data.is_halted || false,
        haltReason: data.halt_reason || null,
        mode: data.mode || 'testnet',
    }),

    fetchInitialData: async () => {
        try {
            // Fetch real balance from Binance
            const balanceRes = await fetch(`${API_URL}/api/balance`)
            if (balanceRes.ok) {
                const balanceData = await balanceRes.json()
                if (balanceData.success) {
                    set({ balance: balanceData.balance, mode: balanceData.mode })
                }
            }

            // Fetch status
            const statusRes = await fetch(`${API_URL}/api/status`)
            if (statusRes.ok) {
                const statusData = await statusRes.json()
                get().updateFromStatus(statusData)
            }

            // Fetch trades from backend (now includes metrics)
            const tradesRes = await fetch(`${API_URL}/api/trades`)
            if (tradesRes.ok) {
                const data = await tradesRes.json()
                set({
                    trades: data.trades || [],
                    tradesToday: data.trades_today || 0,
                    lossesToday: data.losses_today || 0,
                    currentStreak: data.streak || 0,
                    dailyPnl: data.daily_pnl || 0
                })
                get().recalculateMetrics()
            }

            // Fetch metrics (may override calculated)
            const metricsRes = await fetch(`${API_URL}/api/metrics`)
            if (metricsRes.ok) {
                const metricsData = await metricsRes.json()
                set({ metrics: metricsData })
            }
        } catch (error) {
            console.error('Failed to fetch initial data:', error)
        }
    },
}))
