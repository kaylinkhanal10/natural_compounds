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
        <div>
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
                <h3>Potential Treatments (Herbs)</h3>
                <ul>
                    {data.related_herbs?.map((h: string, i: number) => <li key={i}>{h}</li>)}
                </ul>
            </div>
        </div>
    );
}
