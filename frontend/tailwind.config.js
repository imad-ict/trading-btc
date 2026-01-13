/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
        './src/components/**/*.{js,ts,jsx,tsx,mdx}',
        './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                // Institutional dark theme
                terminal: {
                    bg: '#0a0a0f',
                    card: '#12121a',
                    border: '#1e1e2a',
                    muted: '#6b7280',
                },
                profit: '#00d48a',
                loss: '#ff4757',
                accent: {
                    blue: '#3b82f6',
                    yellow: '#fbbf24',
                    purple: '#8b5cf6',
                },
            },
            fontFamily: {
                mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
            },
        },
    },
    plugins: [],
}
