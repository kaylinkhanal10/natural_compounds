'use client';
import { useState } from 'react';
import axios from 'axios';

export default function Intent() {
    const [text, setText] = useState('');
    const [result, setResult] = useState<any>(null);

    const search = async () => {
        try {
            const res = await axios.post('http://localhost:8000/intent/', { text });
            setResult(res.data);
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className="container">
            <h2>Intent-Based Discovery</h2>
            <div className="card">
                <label>Describe your goal (e.g., "Natural immunity and stress relief"):</label>
                <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                    <input
                        type="text"
                        value={text}
                        onChange={e => setText(e.target.value)}
                        style={{ flex: 1, padding: '0.5rem' }}
                        placeholder="Enter query..."
                    />
                    <button className="btn" onClick={search}>Search</button>
                </div>
            </div>

            {result && (
                <div className="card">
                    <h3>Analysis</h3>
                    <p>{result.rationale}</p>

                    <h4>Mapped Biological Effects</h4>
                    <ul>
                        {result.mapped_effects.map((e: string) => <li key={e}>{e}</li>)}
                    </ul>

                    <h4>Candidate Herbs</h4>
                    <ul>
                        {result.candidate_herbs.map((h: string) => <li key={h}>{h}</li>)}
                    </ul>
                </div>
            )}
        </div>
    );
}
