import './globals.css'
import type { Metadata } from 'next'
import { Navbar } from './components/Navbar'

export const metadata: Metadata = {
    title: 'Natural Medicine Discovery AI',
    description: 'Evidence-backed natural medicine reasoning platform',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body>
                <div className="container">
                    <header>
                        <h1>Natural Medicine Discovery AI</h1>
                        <Navbar />
                    </header>
                    <main>{children}</main>
                </div>
            </body>
        </html>
    )
}
