'use client'

import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'

export default function MetricsPanel() {
    const { metrics } = useTradeStore()

    return (
        <div className="card">
            <div className="panel-header">PERFORMANCE METRICS</div>
            <div className="grid grid-cols-4 lg:grid-cols-8 gap-4 p-4">
                <MetricItem
                    label="Total Trades"
                    value={metrics.total_trades.toString()}
                />
                <MetricItem
                    label="Wins"
                    value={metrics.wins.toString()}
                    color="profit"
                />
                <MetricItem
                    label="Losses"
                    value={metrics.losses.toString()}
                    color="loss"
                />
                <MetricItem
                    label="Win Rate"
                    value={`${metrics.win_rate.toFixed(1)}%`}
                    color={metrics.win_rate >= 50 ? 'profit' : 'loss'}
                />
                <MetricItem
                    label="Profit Factor"
                    value={metrics.profit_factor.toFixed(2)}
                    color={metrics.profit_factor >= 1.5 ? 'profit' : metrics.profit_factor >= 1 ? 'neutral' : 'loss'}
                />
                <MetricItem
                    label="Avg R:R"
                    value={`${metrics.avg_rr.toFixed(1)}R`}
                />
                <MetricItem
                    label="Total P&L"
                    value={`${metrics.total_pnl >= 0 ? '+' : ''}$${metrics.total_pnl.toFixed(2)}`}
                    color={metrics.total_pnl >= 0 ? 'profit' : 'loss'}
                />
                <MetricItem
                    label="Max Drawdown"
                    value={`${metrics.max_drawdown.toFixed(1)}%`}
                    color={metrics.max_drawdown < 5 ? 'profit' : metrics.max_drawdown < 10 ? 'neutral' : 'loss'}
                />
            </div>
        </div>
    )
}

interface MetricItemProps {
    label: string
    value: string
    color?: 'profit' | 'loss' | 'neutral'
}

function MetricItem({ label, value, color }: MetricItemProps) {
    return (
        <div className="text-center">
            <div className="stat-label">{label}</div>
            <div className={clsx(
                'text-lg font-semibold tabular-nums',
                color === 'profit' && 'profit-text',
                color === 'loss' && 'loss-text',
                color === 'neutral' && 'text-accent-yellow',
            )}>
                {value}
            </div>
        </div>
    )
}
