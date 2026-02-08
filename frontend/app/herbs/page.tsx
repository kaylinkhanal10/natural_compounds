'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import axios from 'axios';

export default function HerbList() {
    const [herbs, setHerbs] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        axios.get((process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/herbs/')
            .then(res => setHerbs(res.data))
            .catch(err => console.error(err))
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <div>Loading...</div>;

    return (
        <div className="container">
            <h2>Herb Explorer</h2>
            <div className="card">
                <table>
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Scientific Name</th>
                            <th>Sanskrit Name</th>
                            <th>Description</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {herbs.map(herb => (
                            <tr key={herb.herbId}>
                                <td>{herb.name}</td>
                                <td>{herb.scientificName}</td>
                                <td>{herb.sanskritName}</td>
                                <td>{herb.description}</td>
                                <td>
                                    <Link href={`/herbs/${herb.herbId}`} className="btn">View</Link>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
