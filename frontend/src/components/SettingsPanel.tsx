'use client'

import { useState, useEffect } from 'react'
import clsx from 'clsx'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

interface Settings {
    mode: 'testnet' | 'live'
    testnet_api_key: string
    testnet_api_secret: string
    live_api_key: string
    live_api_secret: string
    symbols: string[]
}

const AVAILABLE_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT'
]

export default function SettingsPanel({ onClose }: { onClose: () => void }) {
    const [settings, setSettings] = useState<Settings>({
        mode: 'testnet',
        testnet_api_key: '',
        testnet_api_secret: '',
        live_api_key: '',
        live_api_secret: '',
        symbols: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    })

    const [isLoading, setIsLoading] = useState(false)
    const [isTesting, setIsTesting] = useState(false)
    const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)
    const [saveResult, setSaveResult] = useState<{ success: boolean; message: string } | null>(null)

    useEffect(() => {
        fetchSettings()
    }, [])

    const fetchSettings = async () => {
        try {
            const res = await fetch(`${API_URL}/api/settings`)
            if (res.ok) {
                const data = await res.json()
                setSettings(data)
            }
        } catch (error) {
            console.error('Failed to fetch settings:', error)
        }
    }

    const handleSave = async () => {
        setIsLoading(true)
        setSaveResult(null)

        try {
            const res = await fetch(`${API_URL}/api/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings),
            })

            const data = await res.json()

            if (res.ok) {
                setSaveResult({ success: true, message: 'Settings saved successfully!' })
            } else {
                setSaveResult({ success: false, message: data.detail || 'Failed to save settings' })
            }
        } catch (error) {
            setSaveResult({ success: false, message: 'Connection error' })
        } finally {
            setIsLoading(false)
        }
    }

    const handleTestConnection = async () => {
        setIsTesting(true)
        setTestResult(null)

        try {
            const res = await fetch(`${API_URL}/api/test-connection`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: settings.mode,
                    api_key: settings.mode === 'testnet' ? settings.testnet_api_key : settings.live_api_key,
                    api_secret: settings.mode === 'testnet' ? settings.testnet_api_secret : settings.live_api_secret,
                }),
            })

            const data = await res.json()
            setTestResult({
                success: data.success,
                message: data.message || (data.success ? `Connected! Balance: $${data.balance}` : 'Connection failed'),
            })
        } catch (error) {
            setTestResult({ success: false, message: 'Connection error' })
        } finally {
            setIsTesting(false)
        }
    }

    const toggleSymbol = (symbol: string) => {
        if (settings.symbols.includes(symbol)) {
            if (settings.symbols.length > 1) {
                setSettings({ ...settings, symbols: settings.symbols.filter(s => s !== symbol) })
            }
        } else if (settings.symbols.length < 3) {
            setSettings({ ...settings, symbols: [...settings.symbols, symbol] })
        }
    }

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
            <div className="card w-full max-w-2xl max-h-[90vh] overflow-auto">
                <div className="panel-header flex items-center justify-between">
                    <span>⚙️ SETTINGS</span>
                    <button onClick={onClose} className="text-terminal-muted hover:text-white">✕</button>
                </div>

                <div className="p-6 space-y-6">
                    {/* Mode Toggle */}
                    <div>
                        <label className="stat-label mb-2 block">Trading Mode</label>
                        <div className="flex gap-2">
                            <button
                                onClick={() => setSettings({ ...settings, mode: 'testnet' })}
                                className={clsx(
                                    'px-4 py-2 rounded font-medium transition-colors',
                                    settings.mode === 'testnet'
                                        ? 'bg-accent-yellow text-black'
                                        : 'bg-terminal-border text-terminal-muted hover:text-white'
                                )}
                            >
                                🧪 TESTNET
                            </button>
                            <button
                                onClick={() => setSettings({ ...settings, mode: 'live' })}
                                className={clsx(
                                    'px-4 py-2 rounded font-medium transition-colors',
                                    settings.mode === 'live'
                                        ? 'bg-profit text-black'
                                        : 'bg-terminal-border text-terminal-muted hover:text-white'
                                )}
                            >
                                🔴 LIVE
                            </button>
                        </div>
                    </div>

                    {/* API Keys Section */}
                    <div className="space-y-4">
                        <h3 className="text-sm font-semibold uppercase text-terminal-muted">
                            {settings.mode === 'testnet' ? '🧪 Testnet' : '🔴 Live'} API Keys
                        </h3>

                        <div>
                            <label className="stat-label mb-1 block">API Key</label>
                            <input
                                type="password"
                                value={settings.mode === 'testnet' ? settings.testnet_api_key : settings.live_api_key}
                                onChange={(e) => setSettings({
                                    ...settings,
                                    [settings.mode === 'testnet' ? 'testnet_api_key' : 'live_api_key']: e.target.value
                                })}
                                className="w-full bg-terminal-border border border-terminal-border rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent-blue"
                                placeholder="Enter API Key..."
                            />
                        </div>

                        <div>
                            <label className="stat-label mb-1 block">API Secret</label>
                            <input
                                type="password"
                                value={settings.mode === 'testnet' ? settings.testnet_api_secret : settings.live_api_secret}
                                onChange={(e) => setSettings({
                                    ...settings,
                                    [settings.mode === 'testnet' ? 'testnet_api_secret' : 'live_api_secret']: e.target.value
                                })}
                                className="w-full bg-terminal-border border border-terminal-border rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent-blue"
                                placeholder="Enter API Secret..."
                            />
                        </div>

                        {/* Test Connection Button */}
                        <button
                            onClick={handleTestConnection}
                            disabled={isTesting}
                            className="btn-primary w-full flex items-center justify-center gap-2"
                        >
                            {isTesting ? (
                                <>
                                    <span className="animate-spin">⟳</span>
                                    Testing...
                                </>
                            ) : (
                                <>🔌 Test Connection</>
                            )}
                        </button>

                        {testResult && (
                            <div className={clsx(
                                'p-3 rounded text-sm',
                                testResult.success ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'
                            )}>
                                {testResult.success ? '✓' : '✗'} {testResult.message}
                            </div>
                        )}
                    </div>

                    {/* Symbol Selection */}
                    <div>
                        <label className="stat-label mb-2 block">Trading Symbols (Max 3)</label>
                        <div className="grid grid-cols-5 gap-2">
                            {AVAILABLE_SYMBOLS.map(symbol => (
                                <button
                                    key={symbol}
                                    onClick={() => toggleSymbol(symbol)}
                                    className={clsx(
                                        'px-2 py-1 text-xs rounded transition-colors',
                                        settings.symbols.includes(symbol)
                                            ? 'bg-accent-blue text-white'
                                            : 'bg-terminal-border text-terminal-muted hover:text-white'
                                    )}
                                >
                                    {symbol.replace('USDT', '')}
                                </button>
                            ))}
                        </div>
                        <p className="text-xs text-terminal-muted mt-2">
                            Selected: {settings.symbols.join(', ')}
                        </p>
                    </div>

                    {/* Save Button */}
                    <div className="flex gap-3 pt-4 border-t border-terminal-border">
                        <button
                            onClick={handleSave}
                            disabled={isLoading}
                            className="btn-primary flex-1"
                        >
                            {isLoading ? 'Saving...' : '💾 Save Settings'}
                        </button>
                        <button onClick={onClose} className="px-4 py-2 bg-terminal-border rounded hover:bg-terminal-muted/20">
                            Cancel
                        </button>
                    </div>

                    {saveResult && (
                        <div className={clsx(
                            'p-3 rounded text-sm',
                            saveResult.success ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'
                        )}>
                            {saveResult.message}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
