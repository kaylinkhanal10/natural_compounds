'use client';
import { useState, useEffect } from 'react';
import axios from 'axios';

export default function Combine() {
    const [herbs, setHerbs] = useState<any[]>([]);
    const [selected, setSelected] = useState<string[]>([]);
    const [result, setResult] = useState<any>(null);

    useEffect(() => {
        axios.get('http://localhost:8000/herbs/').then(res => setHerbs(res.data));
    }, []);

    const toggleHerb = (id: string) => {
        if (selected.includes(id)) {
            setSelected(selected.filter(x => x !== id));
        } else {
            setSelected([...selected, id]);
        }
    };

    const analyze = async () => {
        try {
            const res = await axios.post('http://localhost:8000/combine/', { herbs: selected });
            setResult(res.data);
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div>
            <h2>Combination Reasoning</h2>
            <div className="card">
                <h3>Select Herbs</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
                    {herbs.map(h => (
                        <div key={h.herbId}>
                            <label>
                                <input
                                    type="checkbox"
                                    checked={selected.includes(h.herbId)}
                                    onChange={() => toggleHerb(h.herbId)}
                                />
                                {' '}{h.name}
                            </label>
                        </div>
                    ))}
                </div>
                <button className="btn" style={{ marginTop: '1rem' }} onClick={analyze}>Analyze Combination</button>
            </div>

            {result && (
                <div>
                    <div className="card">
                        <h3>Reasoning</h3>
                        <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>{result.explanation}</pre>
                    </div>

                    <div className="card">
                        <h3>Shared Effects</h3>
                        <ul>
                            {result.shared_effects.map((e: any, i: number) => (
                                <li key={i}>{e.e.name} (Shared by: {e.herb_names.join(', ')})</li>
                            ))}
                        </ul>
                    </div>

                    <div className="card">
                        <h3>Interactions</h3>
                        {result.interactions.length === 0 ? <p>No known interactions.</p> : (
                            <ul>
                                {result.interactions.map((inter: any, i: number) => (
                                    <li key={i}>{inter.description} ({inter.type})</li>
                                ))}
                            </ul>
                        )}
                    </div>
                </div>
            )}

            {result && result.extended && (
                <div className="card">
                    <h3>Mechanistic Synergy</h3>
                    <p><b>Shared Targets:</b> {result.extended.shared_targets.length}</p>
                    <ul>
                        {result.extended.shared_targets.map((t: any, i: number) => (
                            <li key={i}>{t.name} (Hit by {t.count} herbs)</li>
                        ))}
                    </ul>
                    <p><b>Network Scope:</b> {result.extended.counts.comp_count} compounds targeting {result.extended.counts.prot_count} proteins.</p>
                </div>
            )}

            {result && result.extended && result.extended.chemical_feasibility && (
                <div className="card">
                    <h3>Chemical Feasibility Details</h3>
                    <p><i>Intrinsic physicochemical properties of key compounds.</i></p>
                    <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                        <table style={{ width: '100%', fontSize: '0.9em' }}>
                            <thead>
                                <tr><th>Compound</th><th>Herb</th><th>MW</th><th>TPSA</th><th>LogP</th></tr>
                            </thead>
                            <tbody>
                                {result.extended.chemical_feasibility.map((c: any, i: number) => {
                                    const isHighlight = (c.mw > 600 || c.tpsa > 140);
                                    return (
                                        <tr key={i} style={{ backgroundColor: isHighlight ? '#fff0f0' : 'transparent' }}>
                                            <td>{c.name}</td>
                                            <td>{c.herb}</td>
                                            <td style={{ color: c.mw > 600 ? 'red' : 'inherit' }}>{c.mw}</td>
                                            <td style={{ color: c.tpsa > 140 ? 'red' : 'inherit' }}>{c.tpsa}</td>
                                            <td>{c.logp}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
