'use client';
import { useEffect, useState } from 'react';
import axios from 'axios';

export default function HerbDetail({ params }: { params: { id: string } }) {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        axios.get(`http://localhost:8000/herbs/${params.id}`)
            .then(res => setData(res.data))
            .catch(err => console.error(err))
            .finally(() => setLoading(false));
    }, [params.id]);

    if (loading) return <div>Loading...</div>;
    if (!data) return <div>Not found</div>;

    return (
        <div>
            <h2>{data.name} ({data.scientificName})</h2>
            <p><i>{data.sanskritName}</i></p>
            <p>{data.description}</p>

            <div className="card">
                <h3>Compounds</h3>
                {data.extended_compounds ? (
                    <table>
                        <thead>
                            <tr><th>Name</th><th>MW</th><th>LogP</th><th>TPSA</th><th>HBA</th><th>HBD</th><th>RotB</th></tr>
                        </thead>
                        <tbody>
                            {data.extended_compounds.map((c: any, i: number) => (
                                <tr key={i}>
                                    <td>{c.name}</td>
                                    <td>{c.mw}</td>
                                    <td>{c.logp}</td>
                                    <td>{c.tpsa}</td>
                                    <td>{c.hba}</td>
                                    <td>{c.hbd}</td>
                                    <td>{c.rotb}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <ul>
                        {data.compounds?.map((c: any) => (
                            <li key={c.compoundId}>{c.name} ({c.chemicalClass})</li>
                        ))}
                    </ul>
                )}
            </div>

            {data.targets && (
                <div className="card">
                    <h3>Associated Targets (Proteins)</h3>
                    <ul>{data.targets.map((t: string, i: number) => <li key={i}>{t}</li>)}</ul>
                </div>
            )}

            {data.diseases && (
                <div className="card">
                    <h3>Associated Diseases</h3>
                    <ul>{data.diseases.map((d: string, i: number) => <li key={i}>{d}</li>)}</ul>
                </div>
            )}

            <div className="card">
                <h3>Biological Effects</h3>
                <ul>
                    {data.effects.map((e: any) => (
                        <li key={e.effectId}>{e.name} ({e.category})</li>
                    ))}
                </ul>
            </div>

            <div className="card">
                <h3>Evidence</h3>
                <ul>
                    {data.evidence.map((ev: any) => (
                        <li key={ev.evidenceId}>
                            <a href={ev.url} target="_blank" rel="noopener noreferrer">{ev.title}</a> (DOI: {ev.doi})
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}
