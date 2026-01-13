'use client'

import { useTradeStore } from '@/stores/tradeStore'
import clsx from 'clsx'
import { useEffect, useRef, useState } from 'react'

interface PriceData {
    price: number
    prevPrice: number
    change24h: number
}

export default function PriceTicker() {
    const { prices } = useTradeStore()
    const [priceHistory, setPriceHistory] = useState<Record<string, PriceData>>({})
    const prevPricesRef = useRef<Record<string, number>>({})

    useEffect(() => {
        // Track price changes for animation
        const newHistory: Record<string, PriceData> = {}

        Object.entries(prices).forEach(([symbol, price]) => {
            const prevPrice = prevPricesRef.current[symbol] || price
            newHistory[symbol] = {
                price,
                prevPrice,
                change24h: priceHistory[symbol]?.change24h || 0,
            }
        })

        setPriceHistory(newHistory)
        prevPricesRef.current = { ...prices }
    }, [prices])

    const symbols = Object.keys(prices)

    if (symbols.length === 0) {
        return (
            <div className="card">
                <div className="panel-header">LIVE PRICES</div>
                <div className="p-6 text-center text-terminal-muted text-sm">
                    Waiting for price data...
                    <div className="text-xs mt-2">Start the bot to stream prices</div>
                </div>
            </div>
        )
    }

    return (
        <div className="card">
            <div className="panel-header flex items-center justify-between">
                <span>LIVE PRICES</span>
                <span className="w-2 h-2 rounded-full bg-profit pulse-live" title="Live" />
            </div>
            <div className="divide-y divide-terminal-border">
                {symbols.map(symbol => {
                    const data = priceHistory[symbol]
                    if (!data) return null

                    const isUp = data.price >= data.prevPrice
                    const priceChanged = data.price !== data.prevPrice

                    return (
                        <div key={symbol} className="p-4 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-terminal-border flex items-center justify-center text-sm font-bold">
                                    {symbol.replace('USDT', '').slice(0, 3)}
                                </div>
                                <div>
                                    <div className="font-semibold">{symbol.replace('USDT', '')}</div>
                                    <div className="text-xs text-terminal-muted">/ USDT</div>
                                </div>
                            </div>

                            <div className="text-right">
                                <div className={clsx(
                                    'text-xl font-bold tabular-nums transition-colors duration-300',
                                    priceChanged && isUp && 'text-profit',
                                    priceChanged && !isUp && 'text-loss',
                                    !priceChanged && 'text-white',
                                )}>
                                    ${data.price.toLocaleString(undefined, {
                                        minimumFractionDigits: 2,
                                        maximumFractionDigits: 2
                                    })}
                                </div>
                                <div className={clsx(
                                    'text-xs flex items-center justify-end gap-1',
                                    isUp ? 'text-profit' : 'text-loss',
                                )}>
                                    <span>{isUp ? '▲' : '▼'}</span>
                                    <span className="animate-pulse">LIVE</span>
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
