'use client'

import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'

export default function ActiveTradePanel() {
    const { activePositions, prices } = useTradeStore()

    const positions = Object.entries(activePositions)

    if (positions.length === 0) {
        return (
            <div className="card">
                <div className="panel-header">ACTIVE POSITIONS</div>
                <div className="p-8 flex flex-col items-center justify-center text-center">
                    <div className="text-4xl mb-2">📊</div>
                    <div className="text-terminal-muted text-sm">No active positions</div>
                    <div className="text-terminal-muted text-xs mt-1">Waiting for signal...</div>
                </div>
            </div>
        )
    }

    return (
        <div className="card">
            <div className="panel-header flex items-center justify-between">
                <span>ACTIVE POSITIONS</span>
                <span className="status-badge bg-profit/20 text-profit">
                    {positions.length} OPEN
                </span>
            </div>
            <div className="divide-y divide-terminal-border">
                {positions.map(([symbol, pos]) => {
                    const currentPrice = pos.current_price || prices[symbol] || pos.entry_price
                    const isLong = pos.direction === 'LONG'
                    const priceDiff = currentPrice - pos.entry_price
                    const pnl = isLong ? priceDiff : -priceDiff
                    const pnlPct = pos.entry_price > 0 ? (pnl / pos.entry_price) * 100 : 0
                    const isProfitable = pnl >= 0

                    // Calculate R:R
                    const slDistance = Math.abs(pos.entry_price - pos.stop_loss) || 1
                    const currentRR = Math.abs(pnl) / slDistance
                    const displayRR = isProfitable ? currentRR : -currentRR

                    return (
                        <div key={symbol} className="p-3">
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <span className="font-bold">{symbol}</span>
                                    <span className={clsx(
                                        'status-badge text-xs',
                                        isLong ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss',
                                    )}>
                                        {pos.direction}
                                    </span>
                                </div>
                                <div className={clsx(
                                    'text-lg font-bold tabular-nums',
                                    isProfitable ? 'profit-text' : 'loss-text',
                                )}>
                                    {isProfitable ? '+' : ''}{pnlPct.toFixed(2)}%
                                </div>
                            </div>

                            <div className="grid grid-cols-4 gap-2 text-xs">
                                <div>
                                    <div className="stat-label">Entry</div>
                                    <div className="tabular-nums">${pos.entry_price?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                                </div>
                                <div>
                                    <div className="stat-label">Current</div>
                                    <div className={clsx('tabular-nums', isProfitable ? 'profit-text' : 'loss-text')}>
                                        ${currentPrice?.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                    </div>
                                </div>
                                <div>
                                    <div className="stat-label">Stop Loss</div>
                                    <div className="tabular-nums text-loss">${pos.stop_loss?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                                </div>
                                <div>
                                    <div className="stat-label">R:R</div>
                                    <div className={clsx('tabular-nums font-bold', isProfitable ? 'profit-text' : 'loss-text')}>
                                        {displayRR >= 0 ? '+' : ''}{displayRR.toFixed(2)}R
                                    </div>
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
