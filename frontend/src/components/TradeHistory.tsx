'use client'

import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'

export default function TradeHistory() {
    const { activeTrade, trades, prices, status } = useTradeStore()


    // Get current price for active trade
    const currentPrice = activeTrade
        ? (prices[activeTrade.symbol] || activeTrade.entry_price)
        : 0

    // Calculate unrealized P&L
    const isLong = activeTrade?.direction === 'LONG'
    const unrealizedPnl = activeTrade
        ? (isLong
            ? (currentPrice - activeTrade.entry_price) / activeTrade.entry_price * 100
            : (activeTrade.entry_price - currentPrice) / activeTrade.entry_price * 100)
        : 0

    return (
        <div className="card flex flex-col" style={{ minHeight: '200px', maxHeight: '350px' }}>
            <div className="panel-header flex items-center justify-between flex-shrink-0">
                <span>OPEN POSITIONS</span>
                <span className="text-xs">
                    {activeTrade ? (
                        <span className="text-profit">● 1 ACTIVE</span>
                    ) : (
                        <span className="text-terminal-muted">No positions</span>
                    )}
                </span>
            </div>

            <div className="flex-1 overflow-y-auto">
                {!activeTrade ? (
                    <div className="p-6 text-center">
                        <div className="text-3xl mb-2">📊</div>
                        <div className="text-terminal-muted text-sm">No active positions</div>
                        <div className="text-terminal-muted text-xs mt-1">
                            {status === 'running' ? 'Waiting for signal...' : 'Start bot to trade'}
                        </div>
                    </div>
                ) : (
                    <div className="p-3">
                        {/* Active Position Card */}
                        <div className={clsx(
                            'rounded-lg border p-3',
                            isLong ? 'border-profit/40 bg-profit/5' : 'border-loss/40 bg-loss/5'
                        )}>
                            {/* Header */}
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <span className="font-bold text-lg">{activeTrade.symbol}</span>
                                    <span className={clsx(
                                        'px-2 py-0.5 rounded text-xs font-semibold',
                                        isLong ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'
                                    )}>
                                        {activeTrade.direction}
                                    </span>
                                </div>
                                <div className={clsx(
                                    'text-xl font-bold tabular-nums',
                                    unrealizedPnl >= 0 ? 'text-profit' : 'text-loss'
                                )}>
                                    {unrealizedPnl >= 0 ? '+' : ''}{unrealizedPnl.toFixed(2)}%
                                </div>
                            </div>

                            {/* Price Grid */}
                            <div className="grid grid-cols-3 gap-2 text-center text-xs mb-2">
                                <div className="bg-terminal-bg/50 rounded p-2">
                                    <div className="text-terminal-muted mb-1">Entry</div>
                                    <div className="font-mono font-medium">
                                        ${activeTrade.entry_price.toLocaleString()}
                                    </div>
                                </div>
                                <div className="bg-terminal-bg/50 rounded p-2">
                                    <div className="text-terminal-muted mb-1">Current</div>
                                    <div className={clsx(
                                        'font-mono font-medium',
                                        unrealizedPnl >= 0 ? 'text-profit' : 'text-loss'
                                    )}>
                                        ${currentPrice.toLocaleString()}
                                    </div>
                                </div>
                                <div className="bg-terminal-bg/50 rounded p-2">
                                    <div className="text-terminal-muted mb-1">Stop Loss</div>
                                    <div className="font-mono font-medium text-loss">
                                        ${activeTrade.stop_loss.toLocaleString()}
                                    </div>
                                </div>
                            </div>

                            {/* Status */}
                            <div className="flex items-center justify-between text-xs">
                                <span className="text-terminal-muted">Status</span>
                                <span className="flex items-center gap-1">
                                    <span className="w-2 h-2 rounded-full bg-profit animate-pulse"></span>
                                    <span className="text-profit">LIVE</span>
                                </span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Recent Closed Trades (last 3) */}
                {trades.length > 0 && (
                    <div className="border-t border-terminal-border mt-2 pt-2 px-3">
                        <div className="text-xs text-terminal-muted mb-2">Recent Closed</div>
                        <div className="space-y-1">
                            {trades.slice(-3).reverse().map((trade, idx) => (
                                <div key={idx} className="flex items-center justify-between text-xs py-1">
                                    <div className="flex items-center gap-2">
                                        <span className={clsx(
                                            'w-1.5 h-1.5 rounded-full',
                                            trade.result === 'WIN' ? 'bg-profit' : 'bg-loss'
                                        )}></span>
                                        <span className="font-medium">{trade.symbol}</span>
                                        <span className={clsx(
                                            'text-[10px]',
                                            trade.direction === 'LONG' ? 'text-profit' : 'text-loss'
                                        )}>
                                            {trade.direction}
                                        </span>
                                    </div>
                                    <span className={clsx(
                                        'font-mono',
                                        (trade.pnl_usd || 0) >= 0 ? 'text-profit' : 'text-loss'
                                    )}>
                                        {(trade.pnl_usd || 0) >= 0 ? '+' : ''}${(trade.pnl_usd || 0).toFixed(2)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
