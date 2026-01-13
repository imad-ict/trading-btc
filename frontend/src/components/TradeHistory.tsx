'use client'

import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'

export default function TradeHistory() {
    const { trades } = useTradeStore()

    return (
        <div className="card flex flex-col h-full">
            <div className="panel-header flex items-center justify-between">
                <span>TRADE HISTORY</span>
                <span className="text-xs text-terminal-muted">{trades.length} trades</span>
            </div>

            <div className="flex-1 overflow-auto">
                {trades.length === 0 ? (
                    <div className="p-8 text-center text-terminal-muted">
                        No trades recorded yet
                    </div>
                ) : (
                    <table className="w-full text-sm">
                        <thead className="sticky top-0 bg-terminal-card">
                            <tr className="text-left text-xs text-terminal-muted uppercase">
                                <th className="px-4 py-2">Symbol</th>
                                <th className="px-4 py-2">Dir</th>
                                <th className="px-4 py-2">Entry</th>
                                <th className="px-4 py-2">Exit</th>
                                <th className="px-4 py-2">Result</th>
                                <th className="px-4 py-2">P&L</th>
                                <th className="px-4 py-2">Liquidity</th>
                            </tr>
                        </thead>
                        <tbody>
                            {trades.map((trade) => (
                                <tr key={trade.id} className="border-t border-terminal-border hover:bg-terminal-border/20">
                                    <td className="px-4 py-2 font-medium">{trade.symbol}</td>
                                    <td className="px-4 py-2">
                                        <span className={clsx(
                                            'status-badge',
                                            trade.direction === 'LONG' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss',
                                        )}>
                                            {trade.direction}
                                        </span>
                                    </td>
                                    <td className="px-4 py-2 tabular-nums">${trade.entry_price.toFixed(2)}</td>
                                    <td className="px-4 py-2 tabular-nums">
                                        {trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : '-'}
                                    </td>
                                    <td className="px-4 py-2">
                                        {trade.result && (
                                            <span className={clsx(
                                                'status-badge',
                                                trade.result === 'WIN' && 'bg-profit/20 text-profit',
                                                trade.result === 'LOSS' && 'bg-loss/20 text-loss',
                                                trade.result === 'BE' && 'bg-terminal-muted/20 text-terminal-muted',
                                            )}>
                                                {trade.result}
                                            </span>
                                        )}
                                    </td>
                                    <td className={clsx(
                                        'px-4 py-2 tabular-nums font-medium',
                                        (trade.pnl_usd || 0) >= 0 ? 'profit-text' : 'loss-text',
                                    )}>
                                        {trade.pnl_usd != null
                                            ? `${trade.pnl_usd >= 0 ? '+' : ''}$${trade.pnl_usd.toFixed(2)}`
                                            : '-'}
                                    </td>
                                    <td className="px-4 py-2 text-xs text-terminal-muted max-w-[200px] truncate" title={trade.liquidity_event}>
                                        {trade.liquidity_event}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    )
}
