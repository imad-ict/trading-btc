'use client'

import { useTradeStore } from '@/stores/tradeStore'

export default function ExplanationLog() {
    const { explanationLogs } = useTradeStore()

    return (
        <div className="card flex flex-col h-full">
            <div className="panel-header">TRADE EXPLANATION LOG</div>
            <div className="flex-1 overflow-auto p-2">
                {explanationLogs.length === 0 ? (
                    <div className="p-4 text-center text-terminal-muted text-xs">
                        Trade explanations will appear here
                    </div>
                ) : (
                    <div className="space-y-1">
                        {explanationLogs.map((log, index) => (
                            <div
                                key={index}
                                className="text-xs p-2 bg-terminal-border/30 rounded font-mono leading-relaxed"
                            >
                                {log}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
