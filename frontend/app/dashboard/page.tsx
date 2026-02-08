import Link from 'next/link'

export default function Dashboard() {
    return (
        <div className="container">
            <h1 style={{ marginTop: '2rem', marginBottom: '1rem' }}>Dashboard</h1>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginTop: '2rem' }}>
                <Link href="/herbs" style={{ textDecoration: 'none' }}>
                    <div className="card" style={{ height: '100%', cursor: 'pointer', transition: 'transform 0.2s' }}>
                        <h2 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Herb Explorer</h2>
                        <p style={{ color: '#666' }}>Browse database of medicinal herbs, compounds, and biological effects.</p>
                    </div>
                </Link>

                <Link href="/workspace" style={{ textDecoration: 'none' }}>
                    <div className="card" style={{ height: '100%', cursor: 'pointer', transition: 'transform 0.2s', border: '2px solid var(--primary)' }}>
                        <h2 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Research Workspace</h2>
                        <p style={{ color: '#666' }}>Interactive graph canvas for synergy discovery and hypothesis generation.</p>
                    </div>
                </Link>

                <Link href="/intent" style={{ textDecoration: 'none' }}>
                    <div className="card" style={{ height: '100%', cursor: 'pointer', transition: 'transform 0.2s' }}>
                        <h2 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Intent Search</h2>
                        <p style={{ color: '#666' }}>Find herbs based on desired health outcomes or conditions.</p>
                    </div>
                </Link>

                <Link href="/formulas" style={{ textDecoration: 'none' }}>
                    <div className="card" style={{ height: '100%', cursor: 'pointer', transition: 'transform 0.2s' }}>
                        <h2 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>TCM Formulas</h2>
                        <p style={{ color: '#666' }}>Explore classical formulas and their ingredient compositions.</p>
                    </div>
                </Link>

                <Link href="/diseases" style={{ textDecoration: 'none' }}>
                    <div className="card" style={{ height: '100%', cursor: 'pointer', transition: 'transform 0.2s' }}>
                        <h2 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Disease Network</h2>
                        <p style={{ color: '#666' }}>Analyze disease-target-compound relationships.</p>
                    </div>
                </Link>
            </div>
        </div>
    )
}
