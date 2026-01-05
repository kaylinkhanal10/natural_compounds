import Link from 'next/link'

export function Navbar() {
    return (
        <nav className="nav">
            <Link href="/">Dashboard</Link>
            <Link href="/herbs">Herb Explorer</Link>
            <Link href="/combine">Combination Reasoning</Link>
            <Link href="/intent">Intent Search</Link>
            <Link href="/formulas">Formulas</Link>
            <Link href="/diseases">Diseases</Link>
        </nav>
    )
}
