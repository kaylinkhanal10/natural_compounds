import React from 'react';

export const ReasoningPanel = ({ result }: { result: any }) => {
    if (!result) return null;

    return (
        <div style={{
            position: 'absolute',
            bottom: 0,
            left: '250px', // Palette width
            right: '300px', // Inspector width
            height: '200px',
            background: 'white',
            borderTop: '1px solid #e2e8f0',
            padding: '1rem',
            overflowY: 'auto',
            boxShadow: '0 -4px 6px -1px rgba(0,0,0,0.1)',
            zIndex: 10
        }}>
            <h3 style={{ marginTop: 0 }}>Synergy Reasoning Code</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                <div>
                    <h4>Shared Targets</h4>
                    <ul style={{ fontSize: '0.9em' }}>
                        {result.shared_targets?.slice(0, 5).map((t: any, i: number) => (
                            <li key={i}>
                                <strong>{t.name}</strong> - Targeted by {t.count} herbs ({t.sources.join(', ')})
                            </li>
                        ))}
                    </ul>
                </div>
                <div>
                    <h4>Chemical Feasibility</h4>
                    <p>Analyzed {Array.isArray(result.chemical_feasibility) ? result.chemical_feasibility.length : 0} compounds for drug-likeness.</p>
                    <div>
                        <strong>Coverage:</strong> {result.counts?.comp_count} Compounds, {result.counts?.prot_count} Targets
                    </div>
                </div>
            </div>
        </div>
    );
};
