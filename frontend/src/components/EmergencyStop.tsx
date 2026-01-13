'use client'

import { useState } from 'react'
import { useTradeStore } from '@/stores/tradeStore'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

export default function EmergencyStop() {
    const [isConfirming, setIsConfirming] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const { isHalted, setHalted } = useTradeStore()

    const handleEmergencyStop = async () => {
        if (!isConfirming) {
            setIsConfirming(true)
            // Auto-cancel confirmation after 5 seconds
            setTimeout(() => setIsConfirming(false), 5000)
            return
        }

        setIsLoading(true)

        try {
            const response = await fetch(`${API_URL}/api/emergency-stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ confirm: true }),
            })

            if (response.ok) {
                setHalted(true, 'Emergency stop triggered by user')
            }
        } catch (error) {
            console.error('Emergency stop failed:', error)
        } finally {
            setIsLoading(false)
            setIsConfirming(false)
        }
    }

    return (
        <div className="card p-4">
            <button
                onClick={handleEmergencyStop}
                disabled={isLoading || isHalted}
                className={`
          w-full py-4 rounded-lg font-bold text-lg uppercase tracking-wider
          transition-all duration-200
          ${isHalted
                        ? 'bg-terminal-muted text-terminal-bg cursor-not-allowed'
                        : isConfirming
                            ? 'bg-loss text-white animate-pulse'
                            : 'bg-loss/20 text-loss border-2 border-loss hover:bg-loss hover:text-white'
                    }
        `}
            >
                {isLoading ? (
                    <span className="flex items-center justify-center gap-2">
                        <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        Processing...
                    </span>
                ) : isHalted ? (
                    '⛔ HALTED'
                ) : isConfirming ? (
                    '⚠️ CLICK AGAIN TO CONFIRM'
                ) : (
                    '🛑 EMERGENCY STOP'
                )}
            </button>

            {!isHalted && (
                <p className="text-xs text-terminal-muted text-center mt-2">
                    Closes all positions immediately
                </p>
            )}
        </div>
    )
}
