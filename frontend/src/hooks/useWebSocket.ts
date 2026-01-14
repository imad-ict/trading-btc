import { useCallback, useEffect, useRef, useState } from 'react'
import { useTradeStore } from '@/stores/tradeStore'

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001/ws/live'

export function useWebSocket() {
    const wsRef = useRef<WebSocket | null>(null)
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const [isConnected, setIsConnected] = useState(false)
    const [reconnectAttempts, setReconnectAttempts] = useState(0)

    const {
        updateFromStatus,
        updatePrice,
        setActiveTrade,
        addTrade,
        addExplanationLog,
        setHalted,
    } = useTradeStore()

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return
        }

        try {
            const ws = new WebSocket(WS_URL)
            wsRef.current = ws

            ws.onopen = () => {
                console.log('WebSocket connected')
                setIsConnected(true)
                setReconnectAttempts(0)
            }

            ws.onclose = () => {
                console.log('WebSocket disconnected')
                setIsConnected(false)
                scheduleReconnect()
            }

            ws.onerror = (error) => {
                console.error('WebSocket error:', error)
            }

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data)
                    handleMessage(message)
                } catch (error) {
                    console.error('Failed to parse message:', error)
                }
            }
        } catch (error) {
            console.error('Failed to connect:', error)
            scheduleReconnect()
        }
    }, [])

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current)
        }
        if (wsRef.current) {
            wsRef.current.close()
            wsRef.current = null
        }
        setIsConnected(false)
    }, [])

    const scheduleReconnect = useCallback(() => {
        if (reconnectAttempts >= 10) {
            console.error('Max reconnect attempts reached')
            return
        }

        const delay = Math.min(2000 * Math.pow(2, reconnectAttempts), 30000)
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts + 1})`)

        reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts((prev) => prev + 1)
            connect()
        }, delay)
    }, [reconnectAttempts, connect])

    const handleMessage = useCallback((message: any) => {
        const { type, data } = message

        switch (type) {
            case 'init':
            case 'status':
                updateFromStatus(data)
                break

            case 'price':
                if (data && typeof data === 'object') {
                    Object.entries(data).forEach(([symbol, price]) => {
                        updatePrice(symbol, price as number)
                    })
                }
                break

            case 'trade':
                if (data) {
                    if (data.event === 'close') {
                        // Trade closed - add to history and clear active
                        addTrade({
                            id: Date.now().toString(),
                            symbol: data.symbol,
                            direction: data.direction,
                            entry_price: data.entry,
                            exit_price: data.exit,
                            stop_loss: data.sl,
                            status: 'closed',
                            result: data.result,
                            pnl_usd: data.pnl_usd,
                            pnl_pct: data.pnl_pct,
                            liquidity_event: data.reason || '',
                            market_state: 'expansion',
                            entry_logic: data.reason || ''
                        })
                        setActiveTrade(null)
                        addExplanationLog(`${data.result === 'WIN' ? '🟢' : '🔴'} Trade closed: ${data.symbol} | ${data.result} | $${data.pnl_usd?.toFixed(2) || 0}`)
                    } else {
                        // Trade opened
                        setActiveTrade({
                            symbol: data.symbol,
                            direction: data.direction,
                            entry_price: data.entry,
                            stop_loss: data.sl,
                        })
                        addExplanationLog(`✅ Trade opened: ${data.direction} ${data.symbol} @ $${data.entry}`)
                    }
                }
                break

            case 'log':
                // Bot log messages
                if (message.message) {
                    addExplanationLog(message.message)
                }
                break

            case 'emergency_stop':
                setHalted(true, 'Emergency stop triggered')
                break

            default:
                // Ignore unknown message types silently
                break
        }
    }, [updateFromStatus, updatePrice, setActiveTrade, addTrade, addExplanationLog, setHalted])

    // Heartbeat
    useEffect(() => {
        const interval = setInterval(() => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send('ping')
            }
        }, 30000)

        return () => clearInterval(interval)
    }, [])

    return {
        isConnected,
        connect,
        disconnect,
        reconnectAttempts,
    }
}
