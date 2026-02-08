import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { FlaskConical, AlertTriangle, Info } from 'lucide-react';

const CompoundNode = ({ data, selected }: NodeProps) => {
    return (
        <div className={`
      relative min-w-[180px] bg-zinc-900/90 backdrop-blur-md 
      border-2 rounded-xl transition-all duration-300 shadow-xl
      ${selected ? 'border-emerald-500 shadow-emerald-500/20' : 'border-zinc-700 hover:border-zinc-500'}
    `}>
            {/* Header / Title */}
            <div className="p-3 border-b border-zinc-800 flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 overflow-hidden">
                    <div className="w-6 h-6 rounded bg-emerald-500/10 flex items-center justify-center text-emerald-500 shrink-0">
                        <FlaskConical size={12} />
                    </div>
                    <span className="text-xs font-bold text-zinc-200 truncate" title={data.label}>
                        {data.label}
                    </span>
                </div>
                {data.toxicity > 0.3 && (
                    <div className="text-yellow-500" title="Potential Toxicity">
                        <AlertTriangle size={12} />
                    </div>
                )}
            </div>

            {/* Body / content */}
            <div className="p-3 space-y-2">
                {data.image && (
                    <div className="w-full h-24 bg-white/5 rounded-lg overflow-hidden flex items-center justify-center mb-2">
                        <img src={data.image} alt="Structure" className="w-full h-full object-contain opacity-80" />
                    </div>
                )}

                <div className="grid grid-cols-2 gap-1 text-[10px] items-center">
                    <div className="text-zinc-500">MW</div>
                    <div className="text-zinc-300 text-right font-mono">{data.mw ? Math.round(data.mw) : '-'}</div>

                    <div className="text-zinc-500">LogP</div>
                    <div className="text-zinc-300 text-right font-mono">{data.logp ? data.logp.toFixed(1) : '-'}</div>
                </div>
            </div>

            {/* Footer / Actions */}
            <div className="px-3 py-2 bg-zinc-950/50 border-t border-zinc-800 rounded-b-xl flex justify-between items-center">
                <span className="text-[9px] text-zinc-600 font-mono uppercase tracking-wider">
                    {data.sourceHerb || 'Unknown Source'}
                </span>
                <button className="text-zinc-500 hover:text-emerald-400 transition-colors">
                    <Info size={12} />
                </button>
            </div>

            {/* Connection Handles */}
            <Handle type="target" position={Position.Top} className="!bg-zinc-500 !w-2 !h-2" />
            <Handle type="source" position={Position.Bottom} className="!bg-emerald-500 !w-2 !h-2" />
        </div>
    );
};

export default memo(CompoundNode);
