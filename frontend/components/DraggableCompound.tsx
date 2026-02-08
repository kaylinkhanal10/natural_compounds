import React from 'react';
import { Beaker, GripVertical } from 'lucide-react';

interface Compound {
    compoundId: string;
    name: string;
    mw?: number;
    logp?: number;
    tpsa?: number;
    sourceHerb?: string;
}

interface Props {
    compound: Compound;
    onDragStart: (e: React.DragEvent, c: Compound) => void;
    onClick: (c: Compound) => void;
    isDragging?: boolean;
}

export const DraggableCompound: React.FC<Props> = ({ compound, onDragStart, onClick, isDragging }) => {
    return (
        <div
            draggable
            onDragStart={(e) => onDragStart(e, compound)}
            onClick={() => onClick(compound)}
            className={`
                group relative bg-zinc-900 border border-zinc-800 rounded-lg p-3 
                hover:border-emerald-500/50 hover:shadow-[0_0_15px_-3px_rgba(16,185,129,0.2)]
                transition-all duration-300 cursor-grab active:cursor-grabbing select-none
                ${isDragging ? 'opacity-50 scale-95' : 'opacity-100 scale-100'}
            `}
        >
            <div className="absolute top-2 right-2 text-zinc-700 opacity-0 group-hover:opacity-100 transition-opacity">
                <GripVertical size={14} />
            </div>

            <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-full bg-zinc-800 flex items-center justify-center text-emerald-500">
                    <Beaker size={16} />
                </div>
                <h4 className="font-medium text-sm text-zinc-100 truncate flex-1" title={compound.name}>
                    {compound.name}
                </h4>
            </div>

            <div className="flex justify-between text-[10px] text-zinc-500 font-mono mt-2">
                <span className="bg-zinc-950 px-1.5 py-0.5 rounded border border-zinc-800">
                    MW: {compound.mw ? Math.round(compound.mw) : '-'}
                </span>
                <span className="bg-zinc-950 px-1.5 py-0.5 rounded border border-zinc-800">
                    LogP: {compound.logp ? compound.logp.toFixed(1) : '-'}
                </span>
            </div>

            {/* Hover Glow Effect */}
            <div className="absolute inset-0 rounded-lg bg-gradient-to-tr from-emerald-500/0 via-emerald-500/0 to-emerald-500/10 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-500" />
        </div>
    );
};
