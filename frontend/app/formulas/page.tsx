'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import axios from 'axios';

export default function FormulaList() {
    const [formulas, setFormulas] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        axios.get((process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/formulas/')
            .then(res => setFormulas(res.data))
            .catch(err => console.error(err))
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <div>Loading...</div>;

    return (
        <div className="container">
            <h2>Prescription Formulas</h2>
            <div className="card">
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Source</th>
                            <th>Ingredients</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {formulas.map((f, i) => (
                            <tr key={i}>
                                <td>{f.name}</td>
                                <td>{f.source}</td>
                                <td>{f.ingredient_count}</td>
                                <td>
                                    <Link href={`/formulas/${encodeURIComponent(f.name)}`} className="btn">View</Link>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
