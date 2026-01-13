'use client'

import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'

export default function SessionPanel() {
    const { session, status, isHalted, haltReason } = useTradeStore()

    const getSessionColor = (session: string) => {
        switch (session) {
            case 'LONDON':
                return 'text-accent-blue'
            case 'NEW_YORK':
                return 'text-accent-purple'
            case 'OVERLAP':
                return 'text-profit'
            case 'ASIAN':
                return 'text-accent-yellow'
            default:
                return 'text-terminal-muted'
        }
    }

    const getSessionDescription = (session: string) => {
        switch (session) {
            case 'LONDON':
                return '08:00-12:00 UTC'
            case 'NEW_YORK':
                return '13:00-17:00 UTC'
            case 'OVERLAP':
                return 'London/NY Overlap'
            case 'ASIAN':
                return '00:00-04:00 UTC'
            case 'DEAD':
                return 'Outside Trading Hours'
            default:
                return 'Initializing...'
        }
    }

    return (
        <div className="card">
            <div className="panel-header">SESSION</div>
            <div className="p-4 space-y-4">
                {/* Session Status */}
                <div className="flex items-center justify-between">
                    <span className="text-terminal-muted">Current Session</span>
                    <div className="flex items-center gap-2">
                        <span className={clsx(
                            'w-2 h-2 rounded-full',
                            session !== 'DEAD' && session !== 'NONE' ? 'bg-profit pulse-live' : 'bg-terminal-muted'
                        )} />
                        <span className={clsx('font-semibold', getSessionColor(session))}>
                            {session}
                        </span>
                    </div>
                </div>

                {/* Session Time */}
                <div className="flex items-center justify-between">
                    <span className="text-terminal-muted">Time Window</span>
                    <span className="text-sm">{getSessionDescription(session)}</span>
                </div>

                {/* Bot Status */}
                <div className="flex items-center justify-between">
                    <span className="text-terminal-muted">Bot Status</span>
                    <span className={clsx(
                        'status-badge',
                        status === 'running' && 'status-running',
                        status === 'stopped' && 'status-stopped',
                        status === 'halted' && 'status-halted',
                    )}>
                        {status.toUpperCase()}
                    </span>
                </div>

                {/* Halt Warning */}
                {isHalted && haltReason && (
                    <div className="bg-loss/10 border border-loss/30 rounded p-3">
                        <div className="text-loss text-xs font-semibold mb-1">⚠️ TRADING HALTED</div>
                        <div className="text-sm text-loss/80">{haltReason}</div>
                    </div>
                )}

                {/* Trading Gate */}
                <div className="flex items-center justify-between pt-2 border-t border-terminal-border">
                    <span className="text-terminal-muted">Trading Gate</span>
                    <span className={clsx(
                        'text-sm font-medium',
                        session !== 'DEAD' && session !== 'NONE' && status === 'running' && !isHalted
                            ? 'text-profit'
                            : 'text-loss'
                    )}>
                        {session !== 'DEAD' && session !== 'NONE' && status === 'running' && !isHalted
                            ? '✓ OPEN'
                            : '✗ CLOSED'}
                    </span>
                </div>
            </div>
        </div>
    )
}
