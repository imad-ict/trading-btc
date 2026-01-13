'use client'

import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'

export default function StatusHeader() {
    const {
        status,
        mode,
        balance,
        dailyPnl,
        dailyPnlPct,
        tradesToday,
        maxTrades,
        lossesToday,
        maxLosses,
        currentStreak,
        prices,
    } = useTradeStore()

    const isProfitable = dailyPnl >= 0

    return (
        <div className="card p-4">
            <div className="flex items-center justify-between gap-6">
                {/* Logo & Status */}
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <div className={clsx(
                            'w-3 h-3 rounded-full',
                            status === 'running' && 'bg-profit pulse-live',
                            status === 'stopped' && 'bg-terminal-muted',
                            status === 'halted' && 'bg-loss pulse-live',
                        )} />
                        <span className="text-lg font-bold tracking-tight">INSTITUTIONAL TRADING</span>
                    </div>
                    <div className={clsx(
                        'status-badge',
                        mode === 'testnet' ? 'bg-accent-yellow/20 text-accent-yellow' : 'bg-profit/20 text-profit',
                    )}>
                        {mode.toUpperCase()}
                    </div>
                </div>

                {/* Key Metrics */}
                <div className="flex items-center gap-8">
                    {/* Balance */}
                    <div className="text-center">
                        <div className="stat-label">Balance</div>
                        <div className="stat-value">${balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                    </div>

                    {/* Daily P&L */}
                    <div className="text-center">
                        <div className="stat-label">Daily P&L</div>
                        <div className={clsx('stat-value', isProfitable ? 'profit-text' : 'loss-text')}>
                            {isProfitable ? '+' : ''}${dailyPnl.toFixed(2)}
                            <span className="text-sm ml-1">({isProfitable ? '+' : ''}{dailyPnlPct.toFixed(2)}%)</span>
                        </div>
                    </div>

                    {/* Trades Today */}
                    <div className="text-center">
                        <div className="stat-label">Trades</div>
                        <div className="stat-value">
                            {tradesToday}<span className="text-terminal-muted">/{maxTrades}</span>
                        </div>
                    </div>

                    {/* Losses Today */}
                    <div className="text-center">
                        <div className="stat-label">Losses</div>
                        <div className={clsx('stat-value', lossesToday > 0 && 'loss-text')}>
                            {lossesToday}<span className="text-terminal-muted">/{maxLosses}</span>
                        </div>
                    </div>

                    {/* Streak */}
                    <div className="text-center">
                        <div className="stat-label">Streak</div>
                        <div className={clsx(
                            'stat-value',
                            currentStreak > 0 ? 'profit-text' : currentStreak < 0 ? 'loss-text' : 'text-terminal-muted',
                        )}>
                            {currentStreak > 0 ? `+${currentStreak}` : currentStreak}
                        </div>
                    </div>

                    {/* Live Prices */}
                    <div className="flex gap-4 border-l border-terminal-border pl-4">
                        {Object.entries(prices).slice(0, 3).map(([symbol, price]) => (
                            <div key={symbol} className="text-center">
                                <div className="stat-label">{symbol.replace('USDT', '')}</div>
                                <div className="text-lg font-semibold tabular-nums">
                                    ${Number(price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}
