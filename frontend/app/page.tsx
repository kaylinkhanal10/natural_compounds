import Link from 'next/link'

export default function Home() {
    return (
        <div className="card">
            <h2>Welcome to the Platform</h2>
            <p>Select a module to begin:</p>
            <ul>
                <li><Link href="/herbs">Explore Herbs & Compounds</Link></li>
                <li><Link href="/combine">Analyze Formulations</Link></li>
                <li><Link href="/intent">Natural Language Discovery</Link></li>
                <li><Link href="/formulas">Traditional Formulas (Prescriptions)</Link></li>
                <li><Link href="/diseases">Disease Network Explorer</Link></li>
            </ul>
        </div>
    )
}
