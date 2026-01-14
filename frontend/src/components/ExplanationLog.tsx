'use client'

import { useTradeStore } from '@/stores/tradeStore'
import { useEffect, useRef } from 'react'

export default function ExplanationLog() {
    const { explanationLogs } = useTradeStore()
    const logsEndRef = useRef<HTMLDivElement>(null)

    // Auto-scroll to bottom when new logs arrive
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [explanationLogs])

    return (
        <div className="card flex flex-col" style={{ height: '300px' }}>
            <div className="panel-header flex-shrink-0">TRADE EXPLANATION LOG</div>
            <div
                className="flex-1 overflow-y-auto p-2"
                style={{ maxHeight: '260px' }}
            >
                {explanationLogs.length === 0 ? (
                    <div className="p-4 text-center text-terminal-muted text-xs">
                        Trade explanations will appear here
                    </div>
                ) : (
                    <div className="space-y-1">
                        {explanationLogs.slice(-50).map((log, index) => (
                            <div
                                key={index}
                                className="text-xs p-2 bg-terminal-border/30 rounded font-mono leading-relaxed"
                            >
                                {log}
                            </div>
                        ))}
                        <div ref={logsEndRef} />
                    </div>
                )}
            </div>
        </div>
    )
}

