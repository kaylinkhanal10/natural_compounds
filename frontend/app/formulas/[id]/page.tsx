'use client';
import { useEffect, useState } from 'react';
import axios from 'axios';

export default function FormulaDetail({ params }: { params: { id: string } }) {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const name = decodeURIComponent(params.id);

    useEffect(() => {
        axios.get(`http://localhost:8000/formulas/${name}`)
            .then(res => setData(res.data))
            .catch(err => console.error(err))
            .finally(() => setLoading(false));
    }, [name]);

    if (loading) return <div>Loading...</div>;
    if (!data) return <div>Not found</div>;

    return (
        <div>
            <h2>{data.name}</h2>
            <p>Source: {data.source_book}</p>

            <div className="card">
                <h3>Ingredients</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Herb (Latin)</th>
                            <th>Dosage</th>
                            <th>Unit</th>
                            <th>Role (Seq)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.ingredients.map((ing: any, i: number) => (
                            <tr key={i}>
                                <td>{ing.name}</td>
                                <td>{ing.dosage}</td>
                                <td>{ing.unit}</td>
                                <td>{ing.role}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="card">
                <h3>Mechanistic Expansion</h3>
                <p>Compounds Involved: {data.mechanistics?.compounds}</p>
                <p>Targets Hit: {data.mechanistics?.targets}</p>
                <h4>Top Associated Diseases</h4>
                <ul>
                    {data.mechanistics?.diseases.map((d: string) => <li key={d}>{d}</li>)}
                </ul>
            </div>

            <button className="btn" onClick={() => {
                const text = JSON.stringify(data, null, 2);
                const blob = new Blob([text], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${data.name}_rationale.json`;
                a.click();
            }}>Export Rationale</button>
        </div>
    );
}
