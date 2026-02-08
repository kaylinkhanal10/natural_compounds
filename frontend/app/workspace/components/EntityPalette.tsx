import React, { useState, useEffect } from 'react';
import { Search } from 'lucide-react';

interface PaletteProps {
    isAdvanced: boolean;
    orientation?: 'vertical' | 'horizontal';
}

export const EntityPalette = ({ isAdvanced, orientation = 'vertical' }: PaletteProps) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [searchResults, setSearchResults] = useState<any[]>([]);

    // Initial Pinned Items (Static tools + maybe defaults)
    const [pinnedItems, setPinnedItems] = useState<any[]>([
        { type: 'zone', name: 'Zone', id: 'ZONE_1' },
        { type: 'disease', name: 'Inflammation', id: 'DIS_INFLAMMATION' },
        // Default pinned herbs for demo
        // { type: 'herb', name: 'Turmeric', id: 'HERB_TURMERIC', scientific: 'Curcumae Longae Rhizoma' }
    ]);

    // Search Effect
    useEffect(() => {
        if (searchTerm.length < 2) {
            setSearchResults([]);
            return;
        }

        const delayDebounceFn = setTimeout(() => {
            import('axios').then(axios => {
                axios.default.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/search/global?q=${searchTerm}`)
                    .then(res => {
                        const items = res.data.map((item: any) => ({
                            type: item.type.toLowerCase(),
                            name: item.name,
                            id: item.id,
                            scientific: item.meta,
                            count: item.count
                        }));
                        setSearchResults(items);
                    })
                    .catch(err => console.error("Failed to search", err));
            });
        }, 300);

        return () => clearTimeout(delayDebounceFn);
    }, [searchTerm]);

    const onDragStart = (event: React.DragEvent, nodeType: string, payload: any) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.setData('application/reactflow-data', JSON.stringify(payload));
        event.dataTransfer.effectAllowed = 'move';
    };

    const togglePin = (item: any) => {
        const exists = pinnedItems.find(p => p.id === item.id);
        if (exists) {
            setPinnedItems(prev => prev.filter(p => p.id !== item.id));
        } else {
            setPinnedItems(prev => [...prev, item]);
            setSearchTerm(''); // Clear search on pin? Optional.
        }
    };

    // Determine what to display
    let displayList = [];
    let isSearching = searchTerm.length > 0;

    // Static items that are always available for search (and potentially pinned)
    const staticSearchableItems = [
        { type: 'zone', name: 'Zone', id: 'ZONE_1' },
        { type: 'compound', name: 'Curcumin', id: 'CMP_CURCUMIN', mw: 368.38 },
        { type: 'compound', name: 'Glycyrrhizin', id: 'CMP_GLYCYRRHIZIN', mw: 822.9 },
        { type: 'disease', name: 'Inflammation', id: 'DIS_INFLAMMATION' },
        { type: 'target', name: 'TNF-Alpha', id: 'PROT_TNF', gene: 'TNF' },
        { type: 'target', name: 'COX-2', id: 'TGT_COX2' },
    ];

    if (isSearching) {
        // Filter from SEARCH RESULTS + static generic items that match
        const lowerCaseSearchTerm = searchTerm.toLowerCase();
        // searchResults are already filtered by backend more or less, but we can display them directly
        // Backend returns loose matches.

        const matchingStaticItems = staticSearchableItems.filter(s => s.name.toLowerCase().includes(lowerCaseSearchTerm));

        displayList = [...matchingStaticItems, ...searchResults];
    } else {
        displayList = pinnedItems;
    }

    // Filter by advanced mode if needed (though herbs/zones are always allowed)
    const filtered = displayList.filter(e => {
        if (isAdvanced) return true;
        return ['herb', 'zone'].includes(e.type);
    });

    const renderItem = (item: any, i: number) => {
        const isPinned = pinnedItems.some(p => p.id === item.id);
        return (
            <div
                key={item.id}
                draggable
                onDragStart={(event) => onDragStart(event, item.type, item)}
                style={{
                    padding: orientation === 'horizontal' ? '4px 12px' : '0.75rem',
                    background: 'white',
                    border: '1px solid #cbd5e1',
                    borderRadius: '4px',
                    cursor: 'grab',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    whiteSpace: 'nowrap',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '6px',
                    minWidth: orientation === 'horizontal' ? 'auto' : '100%'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span>{item.name}</span>
                    <span style={{ fontSize: '0.7em', color: '#64748b', background: '#f1f5f9', padding: '1px 4px', borderRadius: '4px' }}>
                        {item.type.substr(0, 3).toUpperCase()}
                    </span>
                </div>

                {/* Pin Button - Only show for herbs effectively */}
                {item.type === 'herb' && (
                    <button
                        onClick={(e) => { e.stopPropagation(); togglePin(item); }} // Stop propagation to prevent drag event
                        style={{
                            border: 'none',
                            background: 'none',
                            cursor: 'pointer',
                            color: isPinned ? '#2563eb' : '#94a3b8',
                            padding: '2px',
                            lineHeight: 1
                        }}
                        title={isPinned ? "Unpin" : "Pin to Library"}
                    >
                        {isPinned ? '★' : '☆'}
                    </button>
                )}
            </div>
        );
    };

    if (orientation === 'horizontal') {
        return (
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                padding: '0.5rem 1rem',
                background: '#f8fafc',
                borderBottom: '1px solid #e2e8f0',
                height: '60px',
                overflowX: 'auto'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '1rem' }}>
                    <Search size={16} color="#94a3b8" />
                    <input
                        type="text"
                        // Use placeholder to indicate state
                        placeholder={isSearching ? "Searching DB..." : "Search database..."}
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        style={{
                            border: '1px solid #e2e8f0',
                            borderRadius: '4px',
                            padding: '4px 8px',
                            fontSize: '0.9rem',
                            width: '200px'
                        }}
                    />
                </div>

                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    {filtered.map((item, i) => renderItem(item, i))}
                    {filtered.length === 0 && isSearching && (
                        <span style={{ fontSize: '0.8em', color: '#94a3b8', padding: '4px' }}>No matches found.</span>
                    )}
                </div>
            </div>
        );
    }

    return (
        <aside style={{ width: '250px', borderRight: '1px solid #e2e8f0', padding: '1rem', background: '#f8fafc', display: 'flex', flexDirection: 'column', height: '100%' }}>
            <h3>{isSearching ? 'Search Results' : 'Library'}</h3>
            <div style={{ position: 'relative', marginBottom: '1rem' }}>
                <Search size={16} style={{ position: 'absolute', left: 8, top: 10, color: '#94a3b8' }} />
                <input
                    type="text"
                    placeholder="Search database..."
                    value={searchTerm}
                    onChange={e => setSearchTerm(e.target.value)}
                    style={{
                        width: '100%',
                        padding: '0.5rem 0.5rem 0.5rem 2rem',
                        border: '1px solid #e2e8f0',
                        borderRadius: '6px'
                    }}
                />
            </div>
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {filtered.map((item, i) => renderItem(item, i))}
                {filtered.length === 0 && isSearching && (
                    <span style={{ fontSize: '0.8em', color: '#94a3b8', padding: '4px' }}>No matches found.</span>
                )}
            </div>
        </aside>
    );
};
