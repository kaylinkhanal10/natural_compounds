'use client';
import { useEffect, useState } from 'react';

const MOLECULES = [
    '⬡', '⌬', '⏣', 'C8H10N4O2', 'C15H10O4', 'H2O', 'CH4', 'C6H12O6',
    'InChI=1S/C8H10N4O2', 'O=C(C)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
];

interface Particle {
    id: number;
    text: string;
    x: number;
    y: number;
    size: number;
    duration: number;
    delay: number;
    opacity: number;
}

export function LandingBackground() {
    const [particles, setParticles] = useState<Particle[]>([]);

    useEffect(() => {
        // Generate random particles
        const newParticles = Array.from({ length: 20 }).map((_, i) => ({
            id: i,
            text: MOLECULES[Math.floor(Math.random() * MOLECULES.length)],
            x: Math.random() * 100,
            y: Math.random() * 100,
            size: Math.random() * 1.5 + 0.5, // 0.5rem to 2rem
            duration: Math.random() * 20 + 10, // 10s to 30s
            delay: Math.random() * 5,
            opacity: Math.random() * 0.3 + 0.1 // 0.1 to 0.4 opacity
        }));
        setParticles(newParticles);
    }, []);

    return (
        <div className="absolute inset-0 overflow-hidden pointer-events-none z-0 bg-black">
            {particles.map((p) => (
                <div
                    key={p.id}
                    className="absolute text-emerald-500/30 whitespace-nowrap animate-float"
                    style={{
                        left: `${p.x}%`,
                        top: `${p.y}%`,
                        fontSize: `${p.size}rem`,
                        opacity: p.opacity,
                        animationDuration: `${p.duration}s`,
                        animationDelay: `-${p.delay}s`,
                    }}
                >
                    {p.text}
                </div>
            ))}
            <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-black/80" />
        </div>
    );
}
