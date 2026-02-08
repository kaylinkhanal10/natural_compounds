'use client';
import ResearchCanvas from './components/ResearchCanvas';

export default function WorkspaceHome() {
    return (
        <div style={{ height: 'calc(100vh - 60px)', display: 'flex', flexDirection: 'column' }}>
            <ResearchCanvas />
        </div>
    );
}
