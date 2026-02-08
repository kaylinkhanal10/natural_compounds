import './globals.css'
import type { Metadata } from 'next'
import { Navbar } from './components/Navbar'

export const metadata: Metadata = {
    title: 'SynerG',
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
                {/* <header style={{ borderBottom: '1px solid #e2e8f0', background: 'white' }}>
                    <div className="container" style={{ paddingBottom: '1rem' }}>
                        <h1>SynerG <span style={{ fontWeight: 'normal', fontSize: '0.6em', opacity: 0.8 }}>Synergistic Graph Intelligence</span></h1>
                        <Navbar />
                    </div>
                </header> */}
                <main>{children}</main>
            </body>
        </html>
    )
}
