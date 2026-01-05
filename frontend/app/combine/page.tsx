'use client';
import { useState, useEffect } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea, Label, Cell } from 'recharts';
import { Loader2 } from 'lucide-react';
import TargetMatrix from '../../components/TargetMatrix';

export default function Combine() {
    const [herbs, setHerbs] = useState<any[]>([]);
    const [selected, setSelected] = useState<string[]>([]);
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);

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
        setLoading(true);
        setResult(null);
        try {
            const res = await axios.post('http://localhost:8000/combine/', { herbs: selected });
            setResult(res.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
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
                <button
                    className="btn"
                    style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}
                    onClick={analyze}
                    disabled={loading || selected.length === 0}
                >
                    {loading && <Loader2 className="animate-spin" size={16} />}
                    {loading ? 'Analyzing...' : 'Analyze Combination'}
                </button>
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
                    <h3>Mechanistic Synergy & Overlap</h3>
                    <p style={{ fontSize: '0.9em', color: '#555' }}>
                        Scores reflect overlap and complementarity in protein associations, not efficacy.
                    </p>
                    <p><b>Shared Targets:</b> {result.extended.shared_targets.length}</p>

                    {/* Matrix Visualization */}
                    <TargetMatrix
                        sharedEffects={result.extended.shared_targets}
                        activeHerbs={herbs.filter(h => selected.includes(h.herbId)).map(h => h.name)}
                        targetMap={result.extended.target_dictionary}
                    />

                    <p><b>Network Scope:</b> {result.extended.counts.comp_count} compounds targeting {result.extended.counts.prot_count} proteins.</p>
                </div>
            )}

            {result && result.extended && result.extended.chemical_feasibility && (
                <div className="card">
                    <h3>Chemical Feasibility Details</h3>
                    <p style={{ fontSize: '0.9em', color: '#555' }}>
                        <b>The Bioavailability Landscape:</b> Compounds in the <span style={{ color: 'green', fontWeight: 'bold' }}>Green Zone</span> are likely to be well-absorbed (Rule of 5).
                    </p>

                    <div style={{ height: '400px', width: '100%', marginTop: '20px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid />
                                <XAxis
                                    type="number"
                                    dataKey="mw"
                                    name="Molecular Weight"
                                    unit=" Da"
                                    domain={[0, 1000]}
                                    label={{ value: 'Molecular Weight (Daltons)', position: 'insideBottom', offset: -10 }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="tpsa"
                                    name="TPSA"
                                    domain={[0, 300]}
                                    label={{ value: 'Polar Surface Area (TPSA)', angle: -90, position: 'insideLeft' }}
                                />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                        const data = payload[0].payload;
                                        return (
                                            <div style={{ backgroundColor: 'white', padding: '10px', border: '1px solid #ccc', borderRadius: '5px' }}>
                                                <p><b>{data.name}</b></p>
                                                <p>Herb: {data.herb}</p>
                                                <p>MW: {data.mw}</p>
                                                <p>TPSA: {data.tpsa}</p>
                                                {(data.mw > 500 || data.tpsa > 140) ?
                                                    <p style={{ color: 'red' }}>⚠ Violates Rules</p> :
                                                    <p style={{ color: 'green' }}>✔ Drug-like</p>
                                                }
                                            </div>
                                        );
                                    }
                                    return null;
                                }} />

                                {/* Safe Zone: MW < 500, TPSA < 140 */}
                                <ReferenceArea x1={0} x2={500} y1={0} y2={140} fill="green" fillOpacity={0.1} stroke="green" strokeDasharray="3 3">
                                    <Label value="Safe Zone (High Absorption)" position="insideTopRight" fill="green" fontSize={12} />
                                </ReferenceArea>

                                {/* Danger Zone Labels (Implicit outside) */}

                                <Scatter name="Compounds" data={result.extended.chemical_feasibility} fill="#8884d8">
                                    {result.extended.chemical_feasibility.map((entry: any, index: number) => (
                                        <Cell key={`cell-${index}`} fill={(entry.mw > 500 || entry.tpsa > 140) ? 'red' : '#8884d8'} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>

                    <p style={{ fontSize: '0.8em', marginTop: '10px', fontStyle: 'italic' }}>
                        * Points colored <span style={{ color: 'red' }}>Red</span> violate one or more bioavailability rules.
                    </p>

                    {/* Structure Gallery */}
                    <div style={{ marginTop: '2rem', borderTop: '1px solid #eee', paddingTop: '1rem' }}>
                        <h3>Key Compound Structures (2D)</h3>
                        <div style={{ display: 'flex', overflowX: 'auto', gap: '1rem', padding: '1rem 0' }}>
                            {result.extended.chemical_feasibility
                                .filter((c: any) => c.inchikey) // Only if InschKey exists
                                .slice(0, 30) // Limit to 30 to prevent lag
                                .map((c: any, i: number) => (
                                    <div key={i} style={{
                                        minWidth: '180px',
                                        border: '1px solid #e2e8f0',
                                        borderRadius: '8px',
                                        padding: '10px',
                                        backgroundColor: 'white',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center'
                                    }}>
                                        <div style={{ width: '150px', height: '150px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#fff' }}>
                                            <img
                                                src={`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/${c.inchikey}/PNG?record_type=2d&image_size=300x300`}
                                                alt={c.name}
                                                style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                                                loading="lazy"
                                            />
                                        </div>
                                        <div style={{ marginTop: '10px', textAlign: 'center' }}>
                                            <p style={{ fontWeight: 'bold', fontSize: '0.8rem', margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '160px' }}>{c.name}</p>
                                            <p style={{ fontSize: '0.7rem', color: '#64748b', margin: 0 }}>from {c.herb}</p>
                                            <p style={{ fontSize: '0.65rem', color: '#94a3b8', margin: '4px 0 0 0', fontFamily: 'monospace' }}>M.W. {Math.round(c.mw)}</p>
                                        </div>
                                    </div>
                                ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
