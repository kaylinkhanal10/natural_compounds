'use client';

import Link from 'next/link';
import { LandingBackground } from '../components/LandingBackground';
import { Fredoka } from 'next/font/google';
import { ArrowRight, Sparkles } from 'lucide-react';

const fredoka = Fredoka({ subsets: ['latin'], weight: ['400', '600'] });

export default function LandingPage() {
    return (
        <div className="relative min-h-screen bg-black text-white overflow-hidden flex flex-col font-sans">
            {/* Background Animation */}
            <LandingBackground />

            {/* Navbar */}
            <header className="relative z-10 px-6 py-6 flex justify-between items-center max-w-7xl mx-auto w-full">
                <div className="flex items-center gap-1">
                    <span className={`text-xl font-bold tracking-tight ${fredoka.className}`}>osad</span>
                    <span className={`text-xl font-bold tracking-tight bg-emerald-500 text-black px-1 rounded ${fredoka.className}`}>AI</span>
                </div>
            </header>

            {/* Hero Section */}
            <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 text-center">
                <div className="max-w-4xl space-y-8">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-xs text-zinc-400 mb-4 animate-in fade-in slide-in-from-bottom-4 duration-700">
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                        osadai v0.0.1
                    </div>

                    <h1 className={`text-6xl md:text-8xl font-bold tracking-tight leading-tight ${fredoka.className}`}>
                        <span className="text-white">Nature Decoded.</span>
                        <br />
                        <span className="text-emerald-500">Medicine Evolved.</span>
                    </h1>

                    <div className="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed space-y-4 text-justify">
                        <p>
                            OsadAI blends the wisdom of traditional herbal knowledge with advanced AI, streamlining the formulation process by offering data-driven insights to optimize natural compound-based product development.
                        </p>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-8">
                        <Link
                            href="/workspace"
                            className="group relative px-8 py-4 bg-emerald-500 text-black text-lg font-bold rounded-full overflow-hidden transition-transform hover:scale-105"
                        >
                            <span className="relative z-10 flex items-center gap-2">
                                Start Discovery <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
                            </span>
                            <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                        </Link>


                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="relative z-10 py-8 text-center text-zinc-600 text-xs">
            </footer>
        </div>
    );
}
