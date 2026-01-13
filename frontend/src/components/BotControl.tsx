'use client'

import { useState } from 'react'
import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

export default function BotControl() {
    const { status, isHalted } = useTradeStore()
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const isRunning = status === 'running'
    const isStopped = status === 'stopped'

    const handleStartStop = async () => {
        setIsLoading(true)
        setError(null)

        try {
            const action = isRunning ? 'stop' : 'start'
            const res = await fetch(`${API_URL}/api/bot/${action}`, {
                method: 'POST',
            })

            const data = await res.json()

            if (!res.ok) {
                setError(data.detail || `Failed to ${action} bot`)
            }
        } catch (err) {
            setError('Connection error')
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="card">
            <div className="panel-header">BOT CONTROL</div>
            <div className="p-4 space-y-4">
                {/* Status Indicator */}
                <div className="flex items-center justify-between">
                    <span className="text-terminal-muted text-sm">Status</span>
                    <div className="flex items-center gap-2">
                        <span className={clsx(
                            'w-3 h-3 rounded-full',
                            isRunning && 'bg-profit pulse-live',
                            isStopped && 'bg-terminal-muted',
                            isHalted && 'bg-loss pulse-live',
                        )} />
                        <span className={clsx(
                            'text-sm font-semibold uppercase',
                            isRunning && 'text-profit',
                            isStopped && 'text-terminal-muted',
                            isHalted && 'text-loss',
                        )}>
                            {status}
                        </span>
                    </div>
                </div>

                {/* Start/Stop Button */}
                <button
                    onClick={handleStartStop}
                    disabled={isLoading || isHalted}
                    className={clsx(
                        'w-full py-3 rounded-lg font-bold text-sm uppercase tracking-wider transition-all',
                        isRunning
                            ? 'bg-loss/20 text-loss border-2 border-loss hover:bg-loss hover:text-white'
                            : 'bg-profit/20 text-profit border-2 border-profit hover:bg-profit hover:text-black',
                        (isLoading || isHalted) && 'opacity-50 cursor-not-allowed',
                    )}
                >
                    {isLoading ? (
                        <span className="flex items-center justify-center gap-2">
                            <span className="animate-spin">⟳</span>
                            {isRunning ? 'Stopping...' : 'Starting...'}
                        </span>
                    ) : isHalted ? (
                        '⛔ HALTED'
                    ) : isRunning ? (
                        '⏹ Stop Bot'
                    ) : (
                        '▶ Start Bot'
                    )}
                </button>

                {error && (
                    <div className="text-loss text-xs p-2 bg-loss/10 rounded">
                        {error}
                    </div>
                )}

                {/* Status Description */}
                <div className="text-xs text-terminal-muted text-center">
                    {isRunning ? (
                        'Bot is actively monitoring markets and looking for trades'
                    ) : isHalted ? (
                        'Bot is halted due to risk limits or emergency stop'
                    ) : (
                        'Click Start to begin auto-trading with your configured strategy'
                    )}
                </div>
            </div>
        </div>
    )
}
