'use client';
import { useEffect, useState } from 'react';
import axios from 'axios';

export default function DiseaseDetail({ params }: { params: { id: string } }) {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        axios.get(`http://localhost:8000/diseases/${params.id}`)
            .then(res => setData(res.data))
            .catch(err => console.error(err))
            .finally(() => setLoading(false));
    }, [params.id]);

    if (loading) return <div>Loading...</div>;
    if (!data) return <div>Not found</div>;

    return (
        <div className="container">
            <h2>{data.name}</h2>

            <div className="card">
                <h3>Associated Targets (Proteins)</h3>
                <ul>
                    {data.proteins?.map((p: string, i: number) => <li key={i}>{p}</li>)}
                </ul>
            </div>

            <div className="card">
                <h3>Related Compounds</h3>
                <p>Count: {data.related_compounds_count}</p>
            </div>

            <div className="card">
                <h3>Mechanistically Relevant Herbs & Formulas</h3>
                <p style={{ fontSize: '0.85em', color: '#666', marginBottom: '10px' }}>Ranked based on compound–protein associations reported in disease-related pathways.</p>
                <ul>
                    {data.related_herbs?.map((h: string, i: number) => <li key={i}>{h}</li>)}
                </ul>
            </div>

            <div style={{ marginTop: '2rem', fontSize: '0.8em', color: '#888', borderTop: '1px solid #eee', paddingTop: '10px' }}>
                * Protein–disease associations are literature-derived and do not imply therapeutic efficacy.
            </div>
        </div>
    );
}
