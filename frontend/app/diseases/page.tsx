'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import axios from 'axios';

export default function DiseaseList() {
    const [diseases, setDiseases] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        axios.get((process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/diseases/')
            .then(res => setDiseases(res.data))
            .catch(err => console.error(err))
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <div>Loading...</div>;

    return (
        <div className="container">
            <h2>Disease Network</h2>
            <div className="card">
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {diseases.map((d) => (
                            <tr key={d.id}>
                                <td>{d.name}</td>
                                <td>
                                    <Link href={`/diseases/${d.id}`} className="btn">View</Link>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
