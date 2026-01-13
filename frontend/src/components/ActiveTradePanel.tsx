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

    const currentPrice = activeTrade.current_price || prices[activeTrade.symbol] || activeTrade.entry_price
    const isLong = activeTrade.direction === 'LONG'

    // Calculate P&L
    const priceDiff = currentPrice - activeTrade.entry_price
    const pnl = isLong ? priceDiff : -priceDiff
    const pnlPct = (pnl / activeTrade.entry_price) * 100
    const isProfitable = pnl >= 0

    // Calculate distance to SL
    const slDistance = Math.abs(activeTrade.entry_price - activeTrade.stop_loss)
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
                            ${activeTrade.entry_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
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
                            ${activeTrade.stop_loss.toLocaleString(undefined, { maximumFractionDigits: 2 })}
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
