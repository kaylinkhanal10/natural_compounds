import ResearchCanvas from '../components/ResearchCanvas';

export default function WorkspaceDetail({ params }: { params: { id: string } }) {
    return (
        <div style={{ height: 'calc(100vh - 60px)', display: 'flex', flexDirection: 'column' }}>
            {/* Header could go here */}
            <ResearchCanvas workspaceId={params.id} />
        </div>
    );
}
