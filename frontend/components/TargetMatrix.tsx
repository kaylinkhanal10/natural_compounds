"use client"

import { useMemo, useState } from "react"
import { Zap, Key, Lock, Search, Filter } from "lucide-react"

interface SharedEffect {
    name: string
    count: number
    sources: string[]
}

interface TargetMatrixProps {
    sharedEffects: SharedEffect[]
    activeHerbs: string[]
    targetMap?: Record<string, any>
}

const styles = {
    container: {
        width: '100%',
        marginTop: '2rem',
        marginBottom: '4rem',
        borderRadius: '16px',
        backgroundColor: '#ffffff',
        border: '1px solid #e2e8f0', // slate-200
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
        color: '#0f172a', // slate-900
        fontFamily: 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        overflow: 'hidden'
    },
    controls: {
        padding: '1.5rem',
        borderBottom: '1px solid #e2e8f0',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: '#f8fafc' // slate-50
    },
    searchBox: {
        display: 'flex',
        alignItems: 'center',
        backgroundColor: 'white',
        border: '1px solid #cbd5e1',
        borderRadius: '8px',
        padding: '0.5rem 1rem',
        width: '300px',
        gap: '0.5rem'
    },
    title: {
        fontSize: '1.125rem',
        fontWeight: 700,
        color: '#1e293b' // slate-800
    },
    scrollArea: {
        maxHeight: '600px',
        overflowY: 'auto' as const,
        overflowX: 'auto' as const
    },
    matrix: (activeHerbs: number) => ({
        display: 'grid',
        gridTemplateColumns: `200px repeat(${activeHerbs}, minmax(80px, 1fr)) 120px`,
        minWidth: '800px'
    }),
    headerRow: {
        position: 'sticky' as const,
        top: 0,
        backgroundColor: 'white',
        zIndex: 10,
        borderBottom: '2px solid #e2e8f0',
        fontWeight: 600,
        fontSize: '0.875rem',
        color: '#64748b' // slate-500
    },
    headerCell: {
        padding: '1rem',
        display: 'flex',
        alignItems: 'end',
        justifyContent: 'center',
        backgroundColor: 'white' // Ensure opacity
    },
    herbLabel: {
        writingMode: 'vertical-rl' as const,
        transform: 'rotate(180deg)',
        textTransform: 'uppercase' as const,
        fontSize: '0.75rem',
        letterSpacing: '0.05em',
        color: '#334155' // slate-700
    },
    row: {
        borderBottom: '1px solid #f1f5f9', // slate-100
        transition: 'background-color 0.1s'
    },
    cell: {
        padding: '0.75rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
    },
    proteinCell: {
        justifyContent: 'flex-start',
        paddingLeft: '1.5rem',
        fontWeight: 600,
        fontSize: '0.875rem',
        color: '#334155'
    },
    hitMarker: (active: boolean, count: number) => ({
        width: '100%',
        height: '100%',
        minHeight: '2rem',
        borderRadius: '6px',
        transition: 'all 0.2s',
        backgroundColor: active ? `rgba(79, 70, 229, ${0.4 + (count * 0.1)})` : 'transparent', // Indigo scale
        border: active ? '1px solid #4f46e5' : '1px dashed #e2e8f0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
    }),
    badge: {
        padding: '0.25rem 0.75rem',
        borderRadius: '9999px',
        fontSize: '0.75rem',
        fontWeight: 700,
        backgroundColor: '#e0e7ff', // indigo-100
        color: '#4338ca', // indigo-700
        display: 'flex',
        gap: '0.25rem',
        alignItems: 'center'
    }
}

export default function TargetMatrix({ sharedEffects, activeHerbs, targetMap }: TargetMatrixProps) {
    const [searchTerm, setSearchTerm] = useState("")

    // 1. Filter and Sort (Show ALL, no slice)
    const filteredEffects = useMemo(() => {
        if (!sharedEffects) return []

        let data = [...sharedEffects]

        // Filter by search
        if (searchTerm) {
            data = data.filter(e => e.name.toLowerCase().includes(searchTerm.toLowerCase()))
        }

        // Sort by Count (High Synergy first)
        return data.sort((a, b) => b.count - a.count)
    }, [sharedEffects, searchTerm])

    if (!sharedEffects || sharedEffects.length === 0) {
        return <div className="p-8 text-center text-slate-500">No shared targets found.</div>;
    }

    return (
        <div style={styles.container}>
            {/* Control Bar */}
            <div style={styles.controls}>
                <div style={styles.title}>
                    Target Consensus Map ({filteredEffects.length})
                </div>
                <div style={styles.searchBox}>
                    <Search size={16} color="#94a3b8" />
                    <input
                        type="text"
                        placeholder="Search proteins..."
                        style={{ border: 'none', outline: 'none', width: '100%', fontSize: '0.875rem' }}
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>
            </div>

            {/* Scrollable Matrix */}
            <div style={styles.scrollArea}>
                <div style={styles.matrix(activeHerbs.length)}>

                    {/* Sticky Header */}
                    <>
                        <div style={{ ...styles.headerCell, ...styles.headerRow, justifyContent: 'flex-start', paddingLeft: '1.5rem' }}>
                            Target Protein (Gene)
                        </div>
                        {activeHerbs.map(h => (
                            <div key={h} style={{ ...styles.headerCell, ...styles.headerRow }}>
                                <span style={styles.herbLabel}>{h.split(' ')[0]}</span>
                            </div>
                        ))}
                        <div style={{ ...styles.headerCell, ...styles.headerRow }}>
                            Consensus
                        </div>
                    </>

                    {/* Data Rows */}
                    {filteredEffects.map((row) => (
                        <>
                            {/* Protein Name */}
                            <div style={{ ...styles.headerCell, ...styles.proteinCell, ...styles.row, height: 'auto', minHeight: '3rem', padding: '0.5rem 0 0.5rem 1.5rem' }}>
                                {targetMap && targetMap[row.name] ? (
                                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                                        <span style={{ fontSize: '0.875rem', fontWeight: 700 }}>
                                            {targetMap[row.name].full_name}
                                        </span>
                                        <span style={{ fontSize: '0.75rem', fontWeight: 400, color: '#64748b' }}>
                                            {row.name}
                                        </span>
                                    </div>
                                ) : (
                                    <span>{row.name}</span>
                                )}
                            </div>

                            {/* Herb Hits (Cells) */}
                            {activeHerbs.map((herbName) => {
                                const hasHit = row.sources.some(s => s === herbName || s.includes(herbName) || herbName.includes(s));
                                return (
                                    <div key={herbName} style={{ ...styles.cell, ...styles.row }}>
                                        <div style={styles.hitMarker(hasHit, row.count)}>
                                            {hasHit && <Zap size={10} color="white" fill="white" />}
                                        </div>
                                    </div>
                                )
                            })}

                            {/* Score */}
                            <div style={{ ...styles.cell, ...styles.row }}>
                                <div style={styles.badge}>
                                    {row.count} / {activeHerbs.length}
                                </div>
                            </div>
                        </>
                    ))}

                </div>
            </div>
        </div>
    )
}
