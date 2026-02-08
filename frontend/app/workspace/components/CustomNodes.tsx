import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { MoreHorizontal, PlusCircle } from 'lucide-react';

const NodeBase = ({ data, color, typeLabel, children }: any) => {
    return (
        <div style={{
            background: 'white',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            minWidth: '150px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            fontSize: '12px',
            overflow: 'hidden'
        }}>
            <Handle type="target" position={Position.Top} style={{ background: '#94a3b8' }} />

            <div style={{
                background: color,
                padding: '4px 8px',
                color: 'white',
                fontWeight: 600,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <span>{typeLabel}</span>
                <button
                    onClick={(e) => { e.stopPropagation(); data.onExpand(data.id, data.type); }}
                    title="Expand connections"
                    style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer', padding: 0 }}
                >
                    <PlusCircle size={14} />
                </button>
            </div>

            <div style={{ padding: '8px' }}>
                <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{data.label}</div>
                {children}
            </div>

            <Handle type="source" position={Position.Bottom} style={{ background: '#94a3b8' }} />
        </div>
    );
};

export const HerbNode = memo(({ data, selected }: NodeProps) => {
    return (
        <div style={{
            padding: '10px',
            borderRadius: '8px',
            background: 'white',
            border: selected ? '2px solid #059669' : '1px solid #10b981',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            minWidth: '200px',
            textAlign: 'center',
            position: 'relative' // relative to support absolute containment visual if needed
        }}>
            <Handle type="target" position={Position.Top} className="w-16 !bg-emerald-500" />
            <div style={{ fontWeight: 'bold', color: '#065f46' }}>{data.label}</div>
            <div style={{ fontSize: '0.7em', color: '#6ee7b7' }}>{data.scientific || data.scientificName}</div>

            <div style={{
                marginTop: '8px',
                padding: '4px',
                background: '#ecfdf5',
                borderRadius: '4px',
                fontSize: '0.7em',
                color: '#047857',
                border: '1px dashed #6ee7b7'
            }}>
                <span className="font-mono">Contains ~50-200 compounds</span>
                <div style={{ fontSize: '0.9em', opacity: 0.8, fontStyle: 'italic' }}>Collapsed for accuracy</div>
            </div>

            <Handle type="source" position={Position.Bottom} className="w-16 !bg-emerald-500" />
        </div>
    );
});

export const CompoundNode = memo(({ data }: NodeProps) => {
    return (
        <NodeBase data={data} color="#7c3aed" typeLabel="COMPOUND">
            <div style={{ color: '#64748b' }}>MW: {data.mw}</div>
        </NodeBase>
    );
});

export const TargetNode = memo(({ data }: NodeProps) => {
    return (
        <NodeBase data={data} color="#db2777" typeLabel="TARGET">
            <div style={{ color: '#64748b' }}>Protein</div>
        </NodeBase>
    );
});

export const DiseaseNode = memo(({ data }: NodeProps) => {
    return (
        <NodeBase data={data} color="#dc2626" typeLabel="DISEASE">
            <div style={{ color: '#64748b' }}>Condition</div>
        </NodeBase>
    );
});

export const SynergyZoneNode = memo(({ data, id }: NodeProps) => {
    return (
        <div style={{
            width: '400px',
            height: '300px',
            border: '2px dashed #94a3b8',
            borderRadius: '12px',
            background: 'rgba(241, 245, 249, 0.5)',
            padding: '1rem',
            position: 'relative'
        }}>
            <div style={{ position: 'absolute', top: -12, left: 20, background: '#f1f5f9', padding: '0 8px', color: '#64748b', fontWeight: 600 }}>
                COMBINATION ZONE
            </div>
            <div style={{ height: '100%', display: 'flex', alignItems: 'flex-end', justifyContent: 'center' }}>
                <button
                    onClick={(e) => { e.stopPropagation(); data.onAnalyze(id); }}
                    className="nodrag"
                    style={{
                        background: '#0f172a',
                        color: 'white',
                        border: 'none',
                        padding: '8px 16px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontWeight: 600,
                        zIndex: 10
                    }}
                >
                    Analyze Synergy
                </button>
            </div>
        </div>
    );
});

export const ResultNode = memo(({ data }: NodeProps) => {
    return (
        <NodeBase data={data} color="#2563eb" typeLabel="RESULT">
            <div style={{ color: '#1e293b', fontWeight: 'bold' }}>Synergy Score: {data.score}</div>
            <div style={{ color: '#64748b', fontSize: '0.9em' }}>{data.description}</div>
        </NodeBase>
    );
});

export const nodeTypes = {
    herb: HerbNode,
    compound: CompoundNode,
    target: TargetNode,
    disease: DiseaseNode,
    zone: SynergyZoneNode,
    result: ResultNode
};
