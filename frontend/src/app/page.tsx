'use client'

import { useEffect, useState } from 'react'
import { useTradeStore } from '@/stores/tradeStore'
import { useWebSocket } from '@/hooks/useWebSocket'
import StatusHeader from '@/components/StatusHeader'
import ActiveTradePanel from '@/components/ActiveTradePanel'
import SessionPanel from '@/components/SessionPanel'
import TradeHistory from '@/components/TradeHistory'
import MetricsPanel from '@/components/MetricsPanel'
import EmergencyStop from '@/components/EmergencyStop'
import ExplanationLog from '@/components/ExplanationLog'
import SettingsPanel from '@/components/SettingsPanel'
import PriceTicker from '@/components/PriceTicker'
import BotControl from '@/components/BotControl'

export default function Dashboard() {
    const { isConnected, connect, disconnect } = useWebSocket()
    const { fetchInitialData } = useTradeStore()
    const [showSettings, setShowSettings] = useState(false)

    useEffect(() => {
        connect()
        fetchInitialData()

        return () => {
            disconnect()
        }
    }, [])

    return (
        <div className="dashboard-grid">
            {/* Settings Modal */}
            {showSettings && <SettingsPanel onClose={() => setShowSettings(false)} />}

            {/* Header Row - Full Width */}
            <header className="col-span-full">
                <StatusHeader />
            </header>

            {/* Main Content Area */}
            <main className="flex flex-col gap-4 min-h-0">
                {/* Top Row: Session, Active Trade, Prices */}
                <div className="grid grid-cols-3 gap-4">
                    <SessionPanel />
                    <ActiveTradePanel />
                    <PriceTicker />
                </div>

                {/* Trade History */}
                <div className="flex-1 min-h-0">
                    <TradeHistory />
                </div>

                {/* Metrics */}
                <MetricsPanel />
            </main>

            {/* Right Sidebar */}
            <aside className="flex flex-col gap-4">
                {/* Bot Control */}
                <BotControl />

                {/* Settings Button */}
                <button
                    onClick={() => setShowSettings(true)}
                    className="card p-4 text-left hover:bg-terminal-border/30 transition-colors"
                >
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-terminal-muted">⚙️ SETTINGS</span>
                        <span className="text-xs text-accent-blue">Configure →</span>
                    </div>
                </button>

                {/* Connection Status */}
                <div className="card p-4">
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-terminal-muted">WEBSOCKET</span>
                        <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-profit pulse-live' : 'bg-loss'}`} />
                            <span className="text-xs">{isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
                        </div>
                    </div>
                </div>

                {/* Explanation Log */}
                <div className="flex-1 min-h-0">
                    <ExplanationLog />
                </div>

                {/* Emergency Stop */}
                <EmergencyStop />
            </aside>
        </div>
    )
}


