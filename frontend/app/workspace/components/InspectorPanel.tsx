import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface InspectorProps {
    selectedNode: any | null;
    modifiers?: { alpha: number, beta: number };
    onModifierChange?: (m: any) => void;
    isAdvancedMode?: boolean;
    onAddCompound?: (parentId: string, compound: any, added: boolean) => void;
}

export const InspectorPanel = ({ selectedNode, modifiers, onModifierChange, isAdvancedMode, onAddCompound }: InspectorProps) => {
    const [compounds, setCompounds] = useState<any[]>([]);
    const [checkedState, setCheckedState] = useState<{ [key: string]: boolean }>({});

    if (!selectedNode) {
        return null; // Don't show anything if nothing selected in floating mode
    }

    const { data, type, id } = selectedNode;

    // Fetch compounds when a Herb is selected (Always fetch for visibility)
    useEffect(() => {
        if (type === 'herb' && data.id) {
            axios.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/herbs/${data.id}`)
                .then(res => {
                    // Use extended_compounds if available, otherwise fallback or empty
                    const comps = res.data.extended_compounds || [];
                    const wrappedComps = comps.map((c: any) => ({
                        type: 'compound',
                        data: { ...c, label: c.name }
                    }));
                    setCompounds(wrappedComps);
                })
                .catch(err => console.error("Failed to fetch inspector compounds", err));
        } else {
            setCompounds([]);
        }
    }, [id, type, data.id]);

    const handleCompoundToggle = (c: any) => {
        if (!isAdvancedMode) return; // Guard logic
        const isChecked = !checkedState[c.data.compoundId];
        setCheckedState(prev => ({ ...prev, [c.data.compoundId]: isChecked }));

        if (onAddCompound) {
            onAddCompound(id, c.data, isChecked);
        }
    };

    const handleChange = (key: string, val: string) => {
        if (onModifierChange && modifiers) {
            onModifierChange({ ...modifiers, [key]: parseFloat(val) });
        }
    };

    return (
        <div style={{ width: '300px', background: 'white', display: 'flex', flexDirection: 'column', maxHeight: '80vh' }}>
            <div style={{ padding: '1rem', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h3 style={{ margin: '0 0 0.5rem 0' }}>{data.label}</h3>
                    <span style={{
                        display: 'inline-block',
                        padding: '2px 8px',
                        borderRadius: '12px',
                        background: '#e2e8f0',
                        fontSize: '0.8em',
                        fontWeight: 600,
                        color: '#475569'
                    }}>
                        {type?.toUpperCase() || 'UNKNOWN'}
                    </span>
                </div>
                {/* Close button could be passed in props if needed, but clicking empty canvas deselects normally */}
            </div>

            <div style={{ padding: '1rem', overflowY: 'auto', flex: 1 }}>
                {/* Compound Composition List */}
                {type === 'herb' && (
                    <div style={{ marginTop: '1.5rem', borderTop: '1px solid #e2e8f0', paddingTop: '1rem' }}>
                        <h4 style={{ margin: '0 0 10px 0', color: isAdvancedMode ? '#b91c1c' : '#334155' }}>
                            {isAdvancedMode ? 'Compound Composition (Hypothesis)' : 'Compound Composition'}
                        </h4>
                        <p style={{ fontSize: '0.75em', color: '#64748b', marginBottom: '10px' }}>
                            {isAdvancedMode
                                ? "Select compounds to explicitly include them in the graph as assumptions."
                                : "List of known compounds. Enable Advanced Mode to select and add them to the graph."}
                        </p>
                        <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '4px' }}>
                            {compounds.length === 0 ? (
                                <div style={{ padding: '0.5rem', fontSize: '0.8em', color: '#94a3b8' }}>No compounds data available.</div>
                            ) : (
                                compounds.map((c, i) => (
                                    <div key={i} style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        padding: '6px 10px',
                                        borderBottom: '1px solid #f1f5f9',
                                        fontSize: '0.85rem',
                                        opacity: isAdvancedMode ? 1 : 0.7
                                    }}>
                                        <input
                                            type="checkbox"
                                            checked={!!checkedState[c.data.compoundId]}
                                            onChange={() => handleCompoundToggle(c)}
                                            disabled={!isAdvancedMode}
                                            style={{ marginRight: '8px', cursor: isAdvancedMode ? 'pointer' : 'not-allowed' }}
                                        />
                                        <div style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                            {c.data.name}
                                        </div>
                                        <div style={{ fontSize: '0.75em', color: '#94a3b8' }}>
                                            {Math.round(c.data.mw)} Da
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                )}

                {['zone', 'result'].includes(type) && modifiers && (
                    <div style={{ marginBottom: '1.5rem', background: '#f8fafc', padding: '10px', borderRadius: '8px', border: '1px solid #cbd5e1' }}>
                        <h4 style={{ margin: '0 0 10px 0', color: '#334155' }}>Assumption Modifiers</h4>
                        <div style={{ marginBottom: '10px' }}>
                            <label style={{ fontSize: '0.8em', color: '#64748b' }}>Biological Weight (Alpha): {modifiers.alpha}</label>
                            <input
                                type="range"
                                min="0"
                                max="2"
                                step="0.1"
                                value={modifiers.alpha}
                                onChange={(e) => handleChange('alpha', e.target.value)}
                                style={{ width: '100%' }}
                            />
                        </div>
                        <div style={{ marginBottom: '10px' }}>
                            <label style={{ fontSize: '0.8em', color: '#64748b' }}>Redundancy Penalty (Beta): {modifiers.beta}</label>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={modifiers.beta}
                                onChange={(e) => handleChange('beta', e.target.value)}
                                style={{ width: '100%' }}
                            />
                        </div>
                        <div style={{ fontSize: '0.75em', color: '#94a3b8', fontStyle: 'italic' }}>
                            Adjusts the neuro-symbolic balance. Re-analyze to apply.
                        </div>
                    </div>
                )}

                <h4>Properties</h4>
                <div style={{ display: 'grid', gap: '0.5rem' }}>
                    {Object.entries(data).map(([key, value]) => {
                        if (['label', 'onExpand', 'onAnalyze', 'id'].includes(key)) return null;
                        if (typeof value === 'object') return null;
                        return (
                            <div key={key}>
                                <div style={{ fontSize: '0.8em', color: '#64748b', textTransform: 'capitalize' }}>{key}</div>
                                <div style={{ fontSize: '0.9em' }}>{String(value)}</div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};
