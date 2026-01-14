'use client'

import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'

export default function ActiveTradePanel() {
    const { activeTrade, prices } = useTradeStore()

    if (!activeTrade) {
        return (
            <div className="card">
                <div className="panel-header">ACTIVE TRADE</div>
                <div className="p-8 flex flex-col items-center justify-center text-center">
                    <div className="text-4xl mb-2">📊</div>
                    <div className="text-terminal-muted text-sm">No active position</div>
                    <div className="text-terminal-muted text-xs mt-1">Waiting for signal...</div>
                </div>
            </div>
        )
    }
    // Early return if activeTrade doesn't have required properties
    if (!activeTrade.entry_price || !activeTrade.stop_loss) {
        return (
            <div className="card">
                <div className="panel-header">ACTIVE TRADE</div>
                <div className="p-8 flex flex-col items-center justify-center text-center">
                    <div className="text-4xl mb-2">⏳</div>
                    <div className="text-terminal-muted text-sm">Loading trade data...</div>
                </div>
            </div>
        )
    }

    const entryPrice = activeTrade.entry_price || 0
    const stopLoss = activeTrade.stop_loss || 0
    const currentPrice = activeTrade.current_price || prices[activeTrade.symbol] || entryPrice
    const isLong = activeTrade.direction === 'LONG'

    // Calculate P&L
    const priceDiff = currentPrice - entryPrice
    const pnl = isLong ? priceDiff : -priceDiff
    const pnlPct = entryPrice > 0 ? (pnl / entryPrice) * 100 : 0
    const isProfitable = pnl >= 0

    // Calculate distance to SL
    const slDistance = Math.abs(entryPrice - stopLoss) || 1
    const currentRR = Math.abs(pnl) / slDistance
    const displayRR = isProfitable ? currentRR : -currentRR

    return (
        <div className="card">
            <div className={clsx(
                'panel-header flex items-center justify-between',
                isLong ? 'bg-profit/10' : 'bg-loss/10',
            )}>
                <span>ACTIVE TRADE</span>
                <span className={clsx(
                    'status-badge',
                    isLong ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss',
                )}>
                    {activeTrade.direction}
                </span>
            </div>

            <div className="p-4 space-y-3">
                {/* Symbol & Direction */}
                <div className="flex items-center justify-between">
                    <span className="text-xl font-bold">{activeTrade.symbol}</span>
                    <div className={clsx(
                        'text-2xl font-bold tabular-nums',
                        isProfitable ? 'profit-text' : 'loss-text',
                    )}>
                        {isProfitable ? '+' : ''}{pnlPct.toFixed(2)}%
                    </div>
                </div>

                {/* Prices */}
                <div className="grid grid-cols-3 gap-3 py-3 border-y border-terminal-border">
                    <div className="text-center">
                        <div className="stat-label">Entry</div>
                        <div className="text-sm font-medium tabular-nums">
                            ${entryPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="stat-label">Current</div>
                        <div className={clsx(
                            'text-sm font-medium tabular-nums',
                            isProfitable ? 'profit-text' : 'loss-text',
                        )}>
                            ${currentPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="stat-label">Stop Loss</div>
                        <div className="text-sm font-medium tabular-nums text-loss">
                            ${stopLoss.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </div>
                    </div>
                </div>

                {/* Current RR */}
                <div className="flex items-center justify-between">
                    <span className="text-terminal-muted">Current R:R</span>
                    <span className={clsx(
                        'font-semibold',
                        displayRR >= 0 ? 'profit-text' : 'loss-text',
                    )}>
                        {displayRR >= 0 ? '+' : ''}{displayRR.toFixed(2)}R
                    </span>
                </div>

                {/* P&L USD */}
                <div className="flex items-center justify-between">
                    <span className="text-terminal-muted">Unrealized P&L</span>
                    <span className={clsx(
                        'font-semibold',
                        isProfitable ? 'profit-text' : 'loss-text',
                    )}>
                        {isProfitable ? '+' : ''}${(activeTrade.unrealized_pnl || 0).toFixed(2)}
                    </span>
                </div>
            </div>
        </div>
    )
}
