'use client';
import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import ReactFlow, {
    Node, Edge, Controls, Background, useNodesState, useEdgesState,
    ReactFlowProvider, Connection, addEdge, MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
    Search, X, Beaker, ArrowLeft, ChevronRight, Hexagon, Pin,
    Zap, Activity, Maximize2, Share2
} from 'lucide-react';
import {
    ResponsiveContainer, RadarChart, PolarGrid,
    PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { DraggableCompound } from '../../../components/DraggableCompound';
import CompoundNode from './CompoundNode';

// Types
interface SearchResult {
    id: string;
    name: string;
    type: string;
    meta?: string;
    count?: number;
}

type Compound = {
    compoundId: string;
    name: string;
    inchikey?: string;
    smiles?: string;
    mw?: number;
    logp?: number;
    tpsa?: number;
    sourceHerb?: string; // or context source
};

type Formulation = Compound[];

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

// Node Types Configuration
const nodeTypes = {
    compound: CompoundNode,
};

interface ResearchCanvasProps {
    workspaceId?: string;
}

export default function ResearchCanvas({ workspaceId }: ResearchCanvasProps) {
    // State
    const [searchTerm, setSearchTerm] = useState('');
    const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
    const [pinnedResults, setPinnedResults] = useState<SearchResult[]>([]); // Pinned items

    // Context & Inventory State
    const [selectedContext, setSelectedContext] = useState<SearchResult | null>(null);
    const [inventoryMode, setInventoryMode] = useState<'compounds' | 'children'>('compounds');
    const [childNodes, setChildNodes] = useState<SearchResult[]>([]); // Intermediate items (e.g. Targets)
    const [compounds, setCompounds] = useState<Compound[]>([]); // Inventory
    const [loadingInventory, setLoadingInventory] = useState(false);

    // ReactFlow State
    const reactFlowWrapper = useRef<HTMLDivElement>(null);
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
    const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

    // Sidebar (Tracing)
    const [inspectedCompound, setInspectedCompound] = useState<Compound | null>(null);
    const [relatedHerbs, setRelatedHerbs] = useState<string[]>([]); // Names of herbs containing inspected compound
    const [loadingTrace, setLoadingTrace] = useState(false);

    // Analysis
    const [analyzing, setAnalyzing] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [toxicityScore, setToxicityScore] = useState<number>(0);

    // search debouncer
    useEffect(() => {
        const delayDebounceFn = setTimeout(async () => {
            if (searchTerm.length > 1) {
                try {
                    const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/search/global?q=${searchTerm}`);
                    setSearchResults(res.data);
                } catch (e) {
                    console.error("Search failed", e);
                }
            } else {
                setSearchResults([]);
            }
        }, 300);

        return () => clearTimeout(delayDebounceFn)
    }, [searchTerm]);

    // Load Inventory when context selected
    useEffect(() => {
        if (!selectedContext) return;

        const loadInventoryData = async () => {
            setLoadingInventory(true);
            setCompounds([]);
            setChildNodes([]);

            try {
                // HIERARCHY LOGIC
                // 1. If Disease -> Load Children (Targets)
                if (selectedContext.type === 'Disease') {
                    const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/search/context/${selectedContext.type}/${selectedContext.id}/children`);
                    if (res.data.length > 0) {
                        setChildNodes(res.data);
                        setInventoryMode('children');
                    } else {
                        // Fallback: try loading compounds directly?
                        const resComp = await axios.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/search/context/${selectedContext.type}/${selectedContext.id}`);
                        setCompounds(resComp.data);
                        setInventoryMode('compounds');
                    }
                }
                // 2. If Target or Herb -> Load Compounds (Leaf)
                else {
                    let url = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/search/context/${selectedContext.type}/${selectedContext.id}`;

                    // Fix for Herb ID
                    if (selectedContext.type === 'Herb') {
                        url = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/search/context/Herb/${selectedContext.id}`;
                    }

                    const res = await axios.get(url);
                    setCompounds(res.data);
                    setInventoryMode('compounds');
                }
            } catch (e) {
                console.error("Failed to load inventory", e);
            } finally {
                setLoadingInventory(false);
            }
        };
        loadInventoryData();
    }, [selectedContext]);

    // Handler for drilling down into a child node (e.g. clicking a Target in the list)
    const onDrillDown = async (child: SearchResult) => {
        setLoadingInventory(true);
        try {
            // Assume we are drilling down to compounds now (Target -> Compounds)
            let url = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/search/context/${child.type}/${child.id}`;
            const res = await axios.get(url);
            setCompounds(res.data);
            setInventoryMode('compounds');
        } catch (e) {
            console.error(e);
        } finally {
            setLoadingInventory(false);
        }
    };

    // Handler for going back
    const onInventoryBack = () => {
        // Simple 1-level back for now: If in compounds mode and we have childNodes, go back to children
        if (inventoryMode === 'compounds' && childNodes.length > 0 && selectedContext?.type === 'Disease') {
            setInventoryMode('children');
        } else {
            // Deselect context?
            setSelectedContext(null);
            setCompounds([]);
            setChildNodes([]);
        }
    };

    // Update Analysis on Node Change
    useEffect(() => {
        // Real-time Tox
        let risk = 0;
        const currentCompounds = nodes.map(n => n.data);
        const total = currentCompounds.length;

        currentCompounds.forEach(c => {
            if ((c.mw && c.mw > 500) || (c.tpsa && c.tpsa > 140)) risk++;
        });
        const tox = total > 0 ? (risk / total) * 10 : 0;
        setToxicityScore(Math.round(tox));
    }, [nodes]);

    // Toggle Pin
    const togglePin = (e: React.MouseEvent, item: SearchResult) => {
        e.stopPropagation(); // Prevent selection
        if (pinnedResults.some(p => p.id === item.id)) {
            setPinnedResults(pinnedResults.filter(p => p.id !== item.id));
        } else {
            setPinnedResults([...pinnedResults, item]);
        }
    };

    // Trace Compound Sources
    useEffect(() => {
        if (inspectedCompound) {
            setLoadingTrace(true);
            // In a real app, this would be a specific API endpoint: GET /compounds/{id}/sources
            // For now, we simulate or iterate heavily if dataset is small. 
            // Better: Just show "Source Herb" from the object if we only care about provenance in this session.
            // But user asked: "show what all herbs has these compounds".
            // Placeholder: Show current source.
            setRelatedHerbs([inspectedCompound.sourceHerb || 'Unknown']);
            setLoadingTrace(false);
        }
    }, [inspectedCompound]);

    // --- Drag and Drop Handlers ---
    const onDragStart = (event: React.DragEvent, compound: Compound) => {
        event.dataTransfer.setData('application/reactflow', JSON.stringify(compound));
        event.dataTransfer.effectAllowed = 'move';
    };

    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event: React.DragEvent) => {
            event.preventDefault();

            if (!reactFlowWrapper.current || !reactFlowInstance) return;

            const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
            const compoundData = event.dataTransfer.getData('application/reactflow');

            if (!compoundData) return;

            const compound: Compound = JSON.parse(compoundData);

            // Check if already exists
            if (nodes.find(n => n.id === compound.compoundId)) {
                // Flash existing node?
                return;
            }

            const position = reactFlowInstance.project({
                x: event.clientX - reactFlowBounds.left,
                y: event.clientY - reactFlowBounds.top,
            });

            const newNode: Node = {
                id: compound.compoundId,
                type: 'compound',
                position,
                data: {
                    ...compound,
                    label: compound.name,
                    image: compound.inchikey ? `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/${compound.inchikey}/PNG?record_type=2d&image_size=300x300` : null
                },
            };

            setNodes((nds) => nds.concat(newNode));

            // Auto-connect Check (Mock Synergy for now)
            // If new node has synergy with existing, draw edge
            // Logic would go here.
        },
        [reactFlowInstance, nodes, setNodes]
    );

    const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

    const onNodeClick = (_: React.MouseEvent, node: Node) => {
        setInspectedCompound(node.data);
    };

    const analyze = async () => {
        setAnalyzing(true);
        setResult(null);
        try {
            const currentCompounds = nodes.map(n => n.data);
            if (currentCompounds.length === 0) return;

            // Backend expects component groups, here we just send one "Formula" group
            const groups: any = { "Formula": currentCompounds };

            // Expected Response: { plausibility_score, metrics, decision, ... }
            const res = await axios.post((process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/combine/', { groups });
            setResult(res.data);
        } catch (err) {
            console.error("Analysis Failed:", err);
            // Optional: Set error state
        } finally {
            setAnalyzing(false);
        }
    };

    return (
        <div className="flex h-full bg-black text-zinc-100 overflow-hidden font-sans border-t border-zinc-900">

            {/* COL 1: Discovery Repository */}
            <div className="w-72 border-r border-zinc-800 flex flex-col bg-zinc-950/50 backdrop-blur-xl">
                <div className="p-4 border-b border-zinc-800">
                    <h2 className="font-bold text-zinc-100 flex items-center gap-2">
                        <Search size={16} className="text-emerald-500" /> Discovery
                    </h2>
                    <div className="relative mt-4">
                        <Search className="absolute left-3 top-2.5 text-zinc-500" size={14} />
                        <input
                            type="text"
                            placeholder="Search Herbs, Diseases, Targets..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full bg-zinc-900 border border-zinc-800 rounded-lg py-2 pl-9 pr-3 text-sm text-zinc-300 focus:outline-none focus:border-emerald-500/50 transition-colors placeholder:text-zinc-600"
                        />
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin scrollbar-thumb-zinc-800">
                    {/* Pinned Section */}
                    {pinnedResults.length > 0 && (
                        <div className="mb-4">
                            <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider px-2 mb-2 flex items-center gap-2">
                                <Pin size={10} className="text-emerald-500" /> Pinned
                            </h3>
                            {pinnedResults.map((result) => (
                                <div
                                    key={`pinned-${result.id}`}
                                    onClick={() => setSelectedContext(result)}
                                    className={`
                                        p-3 rounded-lg cursor-pointer transition-all border relative group
                                        ${selectedContext?.id === result.id
                                            ? 'bg-emerald-900/20 border-emerald-500/30'
                                            : 'bg-zinc-900/40 border-zinc-800 hover:bg-zinc-900'}
                                    `}
                                >
                                    <button
                                        onClick={(e) => togglePin(e, result)}
                                        className="absolute right-2 top-2 text-emerald-500 opacity-100 hover:text-zinc-400 transition-colors"
                                    >
                                        <Pin size={12} fill="currentColor" />
                                    </button>

                                    <div className="flex items-center justify-between mb-1">
                                        <span className={`text-xs font-bold uppercase tracking-wider px-1.5 py-0.5 rounded
                                            ${result.type === 'Herb' ? 'bg-emerald-900/30 text-emerald-400' :
                                                result.type === 'Disease' ? 'bg-red-900/30 text-red-400' :
                                                    'bg-blue-900/30 text-blue-400'}
                                        `}>
                                            {result.type}
                                        </span>
                                        {result.count !== undefined && (
                                            <span className="text-[9px] text-zinc-500 bg-zinc-800 px-1.5 rounded-full">
                                                {result.count}
                                            </span>
                                        )}
                                    </div>
                                    <div className="font-medium text-zinc-200 text-sm">{result.name}</div>
                                </div>
                            ))}
                            <div className="h-px bg-zinc-800 my-2 mx-2"></div>
                        </div>
                    )}

                    {searchResults.length === 0 && searchTerm.length < 2 && pinnedResults.length === 0 && (
                        <div className="text-center mt-10 text-zinc-600 text-xs px-4">
                            Start typing to search the knowledge graph...
                        </div>
                    )}

                    {searchResults.map((result) => {
                        const isPinned = pinnedResults.some(p => p.id === result.id);
                        return (
                            <div
                                key={result.id + result.type}
                                onClick={() => setSelectedContext(result)}
                                className={`
                                p-3 rounded-lg cursor-pointer transition-all border relative group
                                ${selectedContext?.id === result.id
                                        ? 'bg-emerald-900/20 border-emerald-500/30 shadow-lg shadow-emerald-900/10'
                                        : 'bg-transparent border-transparent hover:bg-zinc-900 hover:border-zinc-800'}
                            `}
                            >
                                <button
                                    onClick={(e) => togglePin(e, result)}
                                    className={`absolute right-2 top-2 transition-colors ${isPinned ? 'text-emerald-500 opacity-100' : 'text-zinc-600 opacity-0 group-hover:opacity-100 hover:text-emerald-500'}`}
                                >
                                    <Pin size={12} fill={isPinned ? "currentColor" : "none"} />
                                </button>

                                <div className="flex items-center justify-between mb-1">
                                    <span className={`text-xs font-bold uppercase tracking-wider px-1.5 py-0.5 rounded
                                    ${result.type === 'Herb' ? 'bg-emerald-900/30 text-emerald-400' :
                                            result.type === 'Disease' ? 'bg-red-900/30 text-red-400' :
                                                'bg-blue-900/30 text-blue-400'}
                                `}>
                                        {result.type}
                                    </span>
                                    {result.count !== undefined && (
                                        <span className="text-[9px] text-zinc-500 bg-zinc-800 px-1.5 rounded-full">
                                            {result.count} {result.type === 'Disease' ? 'Targets' : 'Comps'}
                                        </span>
                                    )}
                                </div>
                                <div className="font-medium text-zinc-200 text-sm">{result.name}</div>
                                {result.meta && (
                                    <div className="text-xs text-zinc-500 italic mt-0.5 truncate">{result.meta}</div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* COL 2: Inventory & Drill-Down */}
            <div className="w-64 border-r border-zinc-800 flex flex-col bg-zinc-950/30">
                <div className="p-4 border-b border-zinc-800 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-2 overflow-hidden">
                        {(inventoryMode === 'compounds' && childNodes.length > 0) || (selectedContext && inventoryMode === 'compounds' && childNodes.length > 0) ? (
                            <button
                                onClick={onInventoryBack}
                                className="text-zinc-500 hover:text-emerald-500 transition-colors"
                            >
                                <ArrowLeft size={16} />
                            </button>
                        ) : selectedContext ? (
                            <button
                                onClick={onInventoryBack}
                                className="text-zinc-500 hover:text-rose-500 transition-colors"
                            >
                                <X size={16} />
                            </button>
                        ) : null}
                        <div>
                            <h3 className="font-bold text-zinc-400 text-xs uppercase tracking-wider truncate max-w-[120px]">
                                {inventoryMode === 'children' ? 'Related Targets' : 'Inventory'}
                            </h3>
                            {selectedContext && (
                                <p className="text-[9px] text-zinc-600 truncate max-w-[120px]">{selectedContext.name}</p>
                            )}
                        </div>
                    </div>

                    <span className="text-[10px] bg-zinc-900 text-zinc-500 px-2 py-1 rounded-full border border-zinc-800">
                        {inventoryMode === 'children' ? childNodes.length : compounds.length}
                    </span>
                </div>

                <div className="flex-1 overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-zinc-800">
                    {loadingInventory ? (
                        <div className="p-8 text-center text-zinc-500 text-sm animate-pulse">
                            Loading...
                        </div>
                    ) : (
                        <div className="space-y-2">
                            {/* MODE: CHILDREN (Intermediate Targets) */}
                            {inventoryMode === 'children' && (
                                childNodes.map((child) => (
                                    <div
                                        key={child.id}
                                        onClick={() => onDrillDown(child)}
                                        className="p-3 rounded-lg cursor-pointer bg-zinc-900/50 border border-zinc-800 hover:bg-zinc-800 hover:border-emerald-500/30 transition-all flex items-center justify-between group"
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded bg-blue-900/20 flex items-center justify-center text-blue-500">
                                                <Hexagon size={14} />
                                            </div>
                                            <div>
                                                <div className="font-medium text-zinc-300 text-sm truncate max-w-[120px]" title={child.name}>{child.name}</div>
                                                <div className="flex items-center gap-2">
                                                    <div className="text-[10px] text-zinc-500">{child.type}</div>
                                                    {child.count !== undefined && (
                                                        <span className={`text-[9px] px-1.5 rounded-full ${child.count > 0 ? 'bg-emerald-900/30 text-emerald-500' : 'bg-zinc-800 text-zinc-600'}`}>
                                                            {child.count}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                        <ChevronRight size={14} className="text-zinc-600 group-hover:text-emerald-500" />
                                    </div>
                                ))
                            )}

                            {/* MODE: COMPOUNDS (Leaf Nodes) */}
                            {inventoryMode === 'compounds' && compounds.length === 0 && !loadingInventory && (
                                <div className="p-8 text-center text-zinc-700 text-xs italic">
                                    {selectedContext ? 'No compounds found.' : 'Select a context to browse.'}
                                </div>
                            )}

                            {inventoryMode === 'compounds' && compounds.map((c) => (
                                <div
                                    key={c.compoundId}
                                    onDragStart={(e) => onDragStart(e, c)}
                                    draggable
                                    className="cursor-grab active:cursor-grabbing"
                                >
                                    <div className="bg-zinc-900 border border-zinc-800 p-3 rounded flex items-center justify-between hover:border-emerald-500/50 transition-colors group">
                                        <div className="flex items-center gap-3 overflow-hidden">
                                            <div className="w-8 h-8 rounded bg-black flex items-center justify-center text-zinc-600 font-mono text-[10px]">
                                                {c.mw ? Math.round(c.mw) : 'C'}
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <h4 className="text-sm font-medium text-zinc-300 truncate" title={c.name}>{c.name}</h4>
                                                <div className="flex gap-2 text-[10px] text-zinc-500 font-mono">
                                                    <span>M:{c.mw ? c.mw.toFixed(0) : '-'}</span>
                                                    <span>L:{c.logp ? c.logp.toFixed(1) : '-'}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* COL 3: Infinite Formulation Canvas (ReactFlow) */}
            <div className="flex-1 flex flex-col bg-zinc-950 relative h-full">
                {/* Header & Controls */}
                <div className="absolute top-0 left-0 right-0 h-16 border-b border-zinc-800/0 flex items-center justify-between px-6 z-10 pointer-events-none">
                    <div className="flex items-center gap-3 pointer-events-auto bg-zinc-900/80 backdrop-blur rounded p-2 border border-zinc-800">
                        <div className="w-8 h-8 rounded bg-emerald-500/10 flex items-center justify-center text-emerald-500">
                            <Zap size={18} />
                        </div>
                        <div>
                            <h1 className="font-bold text-sm text-zinc-200">New Formulation</h1>
                            <p className="text-[10px] text-zinc-500">{nodes.length} Compounds</p>
                        </div>
                    </div>

                    <div className="flex items-center gap-4 pointer-events-auto">
                        <div className="flex flex-col items-end bg-zinc-900/80 backdrop-blur rounded p-2 border border-zinc-800 min-w-[140px]">
                            <span className="text-[9px] uppercase text-zinc-500 font-bold tracking-wider">Toxicity Risk</span>
                            <div className="flex items-center gap-2 w-full justify-end">
                                <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full transition-all duration-500 ${toxicityScore > 3 ? (toxicityScore > 7 ? 'bg-red-500' : 'bg-yellow-500') : 'bg-emerald-500'}`}
                                        style={{ width: `${toxicityScore * 10}%` }}
                                    />
                                </div>
                                <span className={`text-xs font-mono font-bold ${toxicityScore > 3 ? 'text-yellow-500' : 'text-emerald-500'}`}>
                                    {toxicityScore}/10
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="flex-1 w-full h-full relative" ref={reactFlowWrapper}>
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        onConnect={onConnect}
                        onInit={setReactFlowInstance}
                        onDrop={onDrop}
                        onDragOver={onDragOver}
                        onNodeClick={onNodeClick}
                        nodeTypes={nodeTypes}
                        proOptions={{ hideAttribution: true }}
                        fitView
                        className="bg-black"
                    >
                        <Background color="#27272a" gap={20} size={1} />
                        <Controls className="!bg-zinc-900 !border-zinc-800 !fill-zinc-400" />
                    </ReactFlow>

                    {/* Bottom Center Action Bar */}
                    <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 z-10">
                        <button
                            onClick={analyze}
                            disabled={analyzing || nodes.length === 0}
                            className="
                                bg-emerald-600 hover:bg-emerald-500 text-white px-8 py-3 rounded-full 
                                font-bold text-sm transition-all shadow-xl shadow-emerald-900/40 
                                disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95
                                flex items-center gap-2
                            "
                        >
                            <Zap size={16} className={analyzing ? "animate-pulse" : ""} />
                            {analyzing ? 'Processing...' : 'Analyze with OSADAI'}
                        </button>
                    </div>
                </div>
            </div>

            {/* COL 4: Inspector (Right Sidebar) */}
            {inspectedCompound && (
                <div className="w-80 border-l border-zinc-800 bg-zinc-950 flex flex-col animate-in slide-in-from-right duration-300 absolute right-0 top-0 bottom-0 z-30 shadow-2xl">
                    <div className="p-4 border-b border-zinc-900 flex items-center justify-between">
                        <h3 className="font-bold text-zinc-400 uppercase tracking-widest text-xs">Compound Details</h3>
                        <div className="flex gap-2">
                            <button className="text-zinc-600 hover:text-emerald-500"><Maximize2 size={14} /></button>
                            <button onClick={() => setInspectedCompound(null)} className="text-zinc-600 hover:text-zinc-300"><X size={16} /></button>
                        </div>
                    </div>

                    <div className="p-6 flex flex-col items-center border-b border-zinc-900 bg-zinc-900/20">
                        {inspectedCompound.inchikey ? (
                            <img
                                src={`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/${inspectedCompound.inchikey}/PNG?record_type=2d&image_size=300x300`}
                                alt="Structure"
                                className="w-48 h-48 object-contain opacity-80 mix-blend-screen"
                            />
                        ) : (
                            <div className="w-32 h-32 bg-zinc-900 rounded-full flex items-center justify-center text-zinc-700 font-mono text-xs">
                                NO STRUCTURE
                            </div>
                        )}
                        <h2 className="mt-4 font-bold text-lg text-center text-zinc-200">{inspectedCompound.name}</h2>

                        {/* InChIKey Display */}
                        {inspectedCompound.inchikey && (
                            <div className="mt-2 text-[9px] font-mono text-zinc-600 break-all text-center px-4 select-all cursor-text">
                                {inspectedCompound.inchikey}
                            </div>
                        )}

                        {/* SMILES Display */}
                        {inspectedCompound.smiles && (
                            <div className="mt-1 text-[9px] font-mono text-zinc-600 break-all text-center px-4 select-all cursor-text opacity-75">
                                {inspectedCompound.smiles}
                            </div>
                        )}

                        <div className="mt-3 flex gap-2 flex-wrap justify-center">
                            <span className="text-[10px] bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded border border-zinc-700">MW {inspectedCompound.mw?.toFixed(0)}</span>
                            <span className="text-[10px] bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded border border-zinc-700">TPSA {inspectedCompound.tpsa?.toFixed(0)}</span>
                            {inspectedCompound.mw && inspectedCompound.mw > 500 && (
                                <span className="text-[10px] bg-yellow-900/30 text-yellow-500 px-2 py-0.5 rounded border border-yellow-900/50 flex items-center gap-1">
                                    <Activity size={10} /> Ro5 Violation
                                </span>
                            )}
                        </div>
                    </div>

                    <div className="flex-1 p-4 overflow-y-auto">
                        <h4 className="text-xs font-bold text-zinc-500 uppercase mb-3">Biological Context</h4>

                        {/* Mock Targets if missing */}
                        <div className="mb-6">
                            <h5 className="text-[10px] text-zinc-400 font-bold mb-2">TARGETS</h5>
                            <div className="flex flex-wrap gap-2">
                                {(inspectedCompound as any).targets && (inspectedCompound as any).targets.length > 0 ? (
                                    (inspectedCompound as any).targets.map((t: string, i: number) => (
                                        <span key={i} className="text-[10px] px-2 py-1 bg-blue-900/20 text-blue-400 border border-blue-900/30 rounded">{t}</span>
                                    ))
                                ) : (
                                    <span className="text-zinc-600 text-xs italic">No direct targets known.</span>
                                )}
                            </div>
                        </div>

                        <h4 className="text-xs font-bold text-zinc-500 uppercase mb-3">Found In Herbs</h4>
                        {loadingTrace ? (
                            <div className="text-zinc-600 text-sm animate-pulse">Tracing sources...</div>
                        ) : (
                            <div className="space-y-2">
                                {relatedHerbs.map((h, i) => (
                                    <div key={i} className="flex items-center gap-3 p-2 rounded bg-zinc-900 border border-zinc-800">
                                        <div className="w-6 h-6 rounded bg-zinc-800 flex items-center justify-center text-zinc-500 font-serif text-xs">H</div>
                                        <span className="text-sm text-zinc-300">{h}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* RESULT MODAL (Overlay) */}
            {result && (
                <div className="absolute inset-x-4 bottom-24 top-20 bg-zinc-900/95 backdrop-blur-xl border border-zinc-800 rounded-2xl shadow-2xl z-50 flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-300">

                    {/* Header */}
                    <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-950/50">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500">
                                <Activity size={20} />
                            </div>
                            <div>
                                <h2 className="text-lg font-bold text-zinc-100">Analysis Results</h2>
                                <p className="text-xs text-zinc-500">Neuro-Symbolic Plausibility Assessment</p>
                            </div>
                        </div>
                        <button onClick={() => setResult(null)} className="p-2 hover:bg-zinc-800 rounded-full transition-colors">
                            <X size={20} className="text-zinc-500" />
                        </button>
                    </div>

                    {/* Content Grid */}
                    <div className="flex-1 overflow-y-auto p-6 grid grid-cols-12 gap-6">

                        {/* LEFT: Score & Decision */}
                        <div className="col-span-4 space-y-6">
                            {/* Score Card */}
                            <div className="bg-black/40 border border-zinc-800 rounded-xl p-6 flex flex-col items-center relative overflow-hidden">
                                <div className={`absolute inset-0 opacity-10 blur-3xl 
                                    ${result.plausibility_score > 0.7 ? 'bg-emerald-500' : result.plausibility_score > 0.4 ? 'bg-yellow-500' : 'bg-red-500'}`}>
                                </div>
                                <span className="text-xs font-bold uppercase tracking-widest text-zinc-500 mb-2">Plausibility Score</span>
                                <div className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-b from-white to-zinc-400">
                                    {result.plausibility_score}
                                </div>
                                <div className={`mt-4 px-3 py-1 rounded-full text-xs font-bold border 
                                    ${result.decision?.band === 'HIGH_PRIORITY' ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400' :
                                        result.decision?.band === 'MODERATE' ? 'bg-yellow-500/20 border-yellow-500 text-yellow-400' :
                                            'bg-red-500/20 border-red-500 text-red-400'}`}>
                                    {result.decision?.band?.replace('_', ' ')}
                                </div>
                            </div>

                            {/* Metrics Grid */}
                            <div className="grid grid-cols-2 gap-3">
                                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
                                    <div className="text-[10px] text-zinc-500 uppercase">Coverage</div>
                                    <div className="text-xl font-mono font-bold text-zinc-200">{result.metrics?.coverage}</div>
                                </div>
                                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
                                    <div className="text-[10px] text-zinc-500 uppercase">Redundancy</div>
                                    <div className="text-xl font-mono font-bold text-zinc-200">{result.metrics?.redundancy_penalty}</div>
                                </div>
                                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
                                    <div className="text-[10px] text-zinc-500 uppercase">Risk Penalty</div>
                                    <div className="text-xl font-mono font-bold text-rose-400">{result.metrics?.risk_penalty}</div>
                                </div>
                                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3">
                                    <div className="text-[10px] text-zinc-500 uppercase">Uncertainty</div>
                                    <div className="text-xl font-mono font-bold text-blue-400">{result.metrics?.uncertainty}</div>
                                </div>
                            </div>

                            {/* Recommendation */}
                            <div className="bg-zinc-900/50 border border-zinc-800 p-4 rounded-xl">
                                <h4 className="text-xs font-bold text-zinc-400 uppercase mb-2">Recommendation</h4>
                                <p className="text-sm text-zinc-300 leading-relaxed">
                                    {result.decision?.recommendation}
                                </p>
                            </div>
                        </div>

                        {/* RIGHT: Detailed Tabs */}
                        <div className="col-span-8 space-y-6">

                            {/* Targets Section */}
                            <div className="bg-zinc-900/30 border border-zinc-800 rounded-xl overflow-hidden">
                                <div className="bg-zinc-900/80 px-4 py-2 border-b border-zinc-800 flex justify-between items-center">
                                    <h3 className="font-bold text-sm text-zinc-300">Target Confidence</h3>
                                    <span className="text-[10px] text-zinc-500">Threshold: {result.target_confidence_summary?.threshold}</span>
                                </div>
                                <div className="p-4 grid grid-cols-2 gap-4">
                                    {result.target_confidence_summary?.max_prob_per_compound &&
                                        Object.entries(result.target_confidence_summary.max_prob_per_compound).map(([name, prob]: any) => (
                                            <div key={name} className="flex items-center justify-between text-sm p-2 bg-black/20 rounded">
                                                <span className="text-zinc-400 truncate w-32" title={name}>{name}</span>
                                                <div className="flex items-center gap-2">
                                                    <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                                        <div className="h-full bg-blue-500" style={{ width: `${prob * 100}%` }}></div>
                                                    </div>
                                                    <span className="font-mono text-xs">{prob}</span>
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </div>

                            {/* ADME & Risk */}
                            <div className="grid grid-cols-2 gap-6">
                                <div className="bg-zinc-900/30 border border-zinc-800 rounded-xl overflow-hidden">
                                    <div className="bg-zinc-900/80 px-4 py-2 border-b border-zinc-800">
                                        <h3 className="font-bold text-sm text-zinc-300">ADME Simulation</h3>
                                    </div>
                                    <div className="p-4 space-y-3">
                                        {result.adme_simulation?.slice(0, 3).map((sim: any, i: number) => (
                                            <div key={i} className="text-xs">
                                                <div className="font-bold text-zinc-400 mb-1">{sim.name}</div>
                                                <div className="grid grid-cols-2 gap-2 text-[10px] text-zinc-500">
                                                    <div className="bg-black/20 p-1 rounded">LogP: {sim.logP_proxy}</div>
                                                    <div className="bg-black/20 p-1 rounded">BBB: {sim.blood_brain_barrier?.split(' ')[0]}</div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div className="bg-zinc-900/30 border border-zinc-800 rounded-xl overflow-hidden">
                                    <div className="bg-zinc-900/80 px-4 py-2 border-b border-zinc-800">
                                        <h3 className="font-bold text-sm text-zinc-300">Risk Assessment</h3>
                                    </div>
                                    <div className="p-4">
                                        <div className="flex items-center gap-2 mb-3">
                                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold border
                                                ${result.biological_risk_assessment?.risk_level === 'LOW' ? 'bg-emerald-900/20 border-emerald-900 text-emerald-500' :
                                                    'bg-rose-900/20 border-rose-900 text-rose-500'}`}>
                                                {result.biological_risk_assessment?.risk_level} Risk
                                            </span>
                                            <span className="text-xs text-zinc-500">Penalty: {result.metrics?.risk_penalty}</span>
                                        </div>
                                        <p className="text-xs text-zinc-400 leading-relaxed italic">
                                            "{result.biological_risk_assessment?.interpretation}"
                                        </p>
                                    </div>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
