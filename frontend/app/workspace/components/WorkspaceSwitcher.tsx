'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';

interface Workspace {
    id: string;
    name: string;
}

interface WorkspaceSwitcherProps {
    currentWorkspaceId?: string;
}

export function WorkspaceSwitcher({ currentWorkspaceId }: WorkspaceSwitcherProps) {
    const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
    const router = useRouter();

    useEffect(() => {
        axios.get((process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/workspaces/')
            .then(res => setWorkspaces(res.data))
            .catch(err => console.error("Failed to fetch workspaces", err));
    }, []);

    const handleCreate = async () => {
        const name = prompt("Enter workspace name:", "New Discovery");
        if (!name) return;

        try {
            const res = await axios.post((process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/workspaces/', {
                name,
                description: "Created via Switcher"
            });
            // Refresh list and switch
            setWorkspaces([...workspaces, res.data]);
            router.push(`/workspace/${res.data.id}`);
        } catch (error) {
            console.error(error);
            alert("Failed to create workspace");
        }
    };

    const handleSwitch = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const id = e.target.value;
        if (id === 'create_new') {
            handleCreate();
        } else if (id) {
            router.push(`/workspace/${id}`);
        }
    };

    return (
        <div style={{ display: 'flex', alignItems: 'center', marginRight: '1rem' }}>
            <span style={{ marginRight: '0.5rem', fontWeight: 600, color: '#333' }}>Workspace:</span>
            <select
                value={currentWorkspaceId || ""}
                onChange={handleSwitch}
                style={{
                    padding: '0.4rem',
                    borderRadius: '6px',
                    border: '1px solid #ccc',
                    fontSize: '0.9rem',
                    minWidth: '200px'
                }}
            >
                <option value="" disabled>Select a Workspace</option>
                {workspaces.map(w => (
                    <option key={w.id} value={w.id}>
                        {w.name}
                    </option>
                ))}
                <option disabled>──────────</option>
                <option value="create_new">+ Create New Workspace</option>
            </select>
        </div>
    );
}
