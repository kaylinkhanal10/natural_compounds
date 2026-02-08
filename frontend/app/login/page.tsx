'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function LoginPage() {
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        // Mock authentication
        if (email && password) {
            router.push('/workspace');
        } else {
            alert("Please enter credentials");
        }
    };

    return (
        <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f8fafc' }}>
            <div className="card" style={{ width: '400px', padding: '2rem' }}>
                <h2 style={{ textAlign: 'center', marginBottom: '1.5rem', color: 'var(--primary)' }}>Sign In</h2>
                <form onSubmit={handleLogin}>
                    <div style={{ marginBottom: '1rem' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>Email</label>
                        <input
                            type="email"
                            className="input"
                            style={{ width: '100%' }}
                            value={email}
                            onChange={e => setEmail(e.target.value)}
                            required
                        />
                    </div>
                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>Password</label>
                        <input
                            type="password"
                            className="input"
                            style={{ width: '100%' }}
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    <button type="submit" className="btn" style={{ width: '100%' }}>Login</button>
                </form>
                <div style={{ marginTop: '1rem', textAlign: 'center', fontSize: '0.9rem', color: '#666' }}>
                    <p>Demo Account: any / any</p>
                    <Link href="/" style={{ color: 'var(--primary)', textDecoration: 'none' }}>Back to Home</Link>
                </div>
            </div>
        </div>
    );
}
