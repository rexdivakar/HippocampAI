import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Brain,
  Search,
  Plus,
  SlidersHorizontal,
  LayoutGrid,
  List,
  RefreshCw
} from 'lucide-react';
import { apiClient } from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';
import type { Memory, MemoryFilters } from '../types';
import { MemoryListItem } from '../components/memory/MemoryListItem';
import { MemoryDetailPanel } from '../components/memory/MemoryDetailPanel';
import { MemoryFilterSidebar } from '../components/memory/MemoryFilterSidebar';
import { AddMemoryModal } from '../components/AddMemoryModal';
import { EditMemoryModal } from '../components/EditMemoryModal';
import clsx from 'clsx';

interface MemoriesPageRedesignedProps {
  userId: string;
}

type ViewDensity = 'comfortable' | 'compact' | 'ultra';
type ViewMode = 'list' | 'grid';

export function MemoriesPageRedesigned({ userId }: MemoriesPageRedesignedProps) {
  const queryClient = useQueryClient();
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [density, setDensity] = useState<ViewDensity>('comfortable');
  const [filters, setFilters] = useState<MemoryFilters>({});
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null);
  const [showFilters, setShowFilters] = useState(true);

  // Fetch memories
  const { data: memories = [], isLoading, refetch } = useQuery({
    queryKey: ['memories', userId, filters],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        filters: {
          ...(filters as Record<string, any>),
          session_id: userId, // Pass userId as session_id to match by either field
        },
        limit: 1000, // Load more for scalability
      });
      return result;
    },
  });

  // Auto-select first memory when list changes
  useEffect(() => {
    if (memories.length > 0 && !selectedMemory) {
      setSelectedMemory(memories[0]);
    }
  }, [memories, selectedMemory]);

  // Create mutation
  const createMutation = useMutation({
    mutationFn: (data: {
      text: string;
      type: string;
      importance: number;
      tags: string[];
      sessionId?: string;
    }) =>
      apiClient.createMemory({
        text: data.text,
        user_id: userId,
        type: data.type,
        importance: data.importance,
        tags: data.tags,
        session_id: data.sessionId,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memories', userId] });
    },
  });

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: ({ memoryId, data }: { memoryId: string; data: Partial<Memory> }) =>
      apiClient.updateMemory(memoryId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memories', userId] });
      setEditingMemory(null);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (memoryId: string) => apiClient.deleteMemory(memoryId, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memories', userId] });
      setSelectedMemory(null);
    },
  });

  // WebSocket real-time updates
  const { on, off } = useWebSocket({ userId });

  useEffect(() => {
    on('memory:created', () => {
      queryClient.invalidateQueries({ queryKey: ['memories', userId] });
    });
    on('memory:updated', () => {
      queryClient.invalidateQueries({ queryKey: ['memories', userId] });
    });
    on('memory:deleted', () => {
      queryClient.invalidateQueries({ queryKey: ['memories', userId] });
    });
    return () => {
      off('memory:created');
      off('memory:updated');
      off('memory:deleted');
    };
  }, [on, off, queryClient, userId]);

  // Filter memories
  const filteredMemories = memories.filter((memory) => {
    // Search filter
    if (searchQuery) {
      const search = searchQuery.toLowerCase();
      const matchesText = memory.text.toLowerCase().includes(search);
      const matchesTags = memory.tags.some((tag) => tag.toLowerCase().includes(search));
      if (!matchesText && !matchesTags) return false;
    }

    // Type filter
    if (filters.types && filters.types.length > 0) {
      if (!filters.types.includes(memory.type)) return false;
    }

    // Importance range
    if (filters.minImportance !== undefined && memory.importance < filters.minImportance) {
      return false;
    }
    if (filters.maxImportance !== undefined && memory.importance > filters.maxImportance) {
      return false;
    }

    // Date range
    if (filters.startDate) {
      if (new Date(memory.created_at) < new Date(filters.startDate)) return false;
    }
    if (filters.endDate) {
      if (new Date(memory.created_at) > new Date(filters.endDate)) return false;
    }

    // Tags
    if (filters.tags && filters.tags.length > 0) {
      const hasTags = filters.tags.some((tag) => memory.tags.includes(tag));
      if (!hasTags) return false;
    }

    return true;
  });

  const handleDelete = (memory: Memory) => {
    if (window.confirm(`Are you sure you want to delete this memory?`)) {
      deleteMutation.mutate(memory.id);
    }
  };

  const handleSelectMemory = (memory: Memory) => {
    setSelectedMemory(memory);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!selectedMemory || filteredMemories.length === 0) return;

    const currentIndex = filteredMemories.findIndex((m) => m.id === selectedMemory.id);

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const nextIndex = Math.min(currentIndex + 1, filteredMemories.length - 1);
      setSelectedMemory(filteredMemories[nextIndex]);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prevIndex = Math.max(currentIndex - 1, 0);
      setSelectedMemory(filteredMemories[prevIndex]);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50" onKeyDown={handleKeyDown} tabIndex={0}>
      {/* Sidebar Filters */}
      {showFilters && (
        <MemoryFilterSidebar
          filters={filters}
          onFiltersChange={setFilters}
          memories={memories}
          onClose={() => setShowFilters(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Left: Title & Search */}
            <div className="flex items-center space-x-4 flex-1">
              <div className="flex items-center space-x-2">
                <Brain className="w-6 h-6 text-primary-600" />
                <h1 className="text-xl font-semibold text-gray-900">Memories</h1>
                <span className="text-sm text-gray-500">({filteredMemories.length})</span>
              </div>

              {/* Search */}
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search memories..."
                  className="w-full pl-10 pr-4 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
            </div>

            {/* Right: Actions */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={clsx(
                  'px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                  showFilters
                    ? 'bg-primary-50 text-primary-600'
                    : 'text-gray-600 hover:bg-gray-100'
                )}
              >
                <SlidersHorizontal className="w-4 h-4" />
              </button>

              {/* Density */}
              <select
                value={density}
                onChange={(e) => setDensity(e.target.value as ViewDensity)}
                className="px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="comfortable">Comfortable</option>
                <option value="compact">Compact</option>
                <option value="ultra">Ultra</option>
              </select>

              {/* View Mode */}
              <div className="flex items-center bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setViewMode('list')}
                  className={clsx(
                    'p-1.5 rounded transition-colors',
                    viewMode === 'list'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  )}
                >
                  <List className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('grid')}
                  className={clsx(
                    'p-1.5 rounded transition-colors',
                    viewMode === 'grid'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  )}
                >
                  <LayoutGrid className="w-4 h-4" />
                </button>
              </div>

              <button
                onClick={() => refetch()}
                disabled={isLoading}
                className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
              </button>

              <button
                onClick={() => setShowAddModal(true)}
                className="px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 transition-colors flex items-center space-x-2"
              >
                <Plus className="w-4 h-4" />
                <span>New</span>
              </button>
            </div>
          </div>
        </div>

        {/* Master-Detail Layout */}
        <div className="flex-1 flex min-h-0">
          {/* Memory List */}
          <div className="w-full lg:w-1/4 xl:w-1/5 2xl:w-1/6 min-w-[320px] max-w-[480px] bg-white border-r border-gray-200 flex flex-col">
            {isLoading ? (
              <div className="flex-1 flex items-center justify-center">
                <RefreshCw className="w-8 h-8 text-primary-600 animate-spin" />
              </div>
            ) : filteredMemories.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
                <Brain className="w-16 h-16 text-gray-300 mb-4" />
                <p className="text-gray-600 font-medium">No memories found</p>
                <p className="text-sm text-gray-500 mt-1">
                  Try adjusting your filters or create a new memory
                </p>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto">
                {filteredMemories.map((memory) => (
                  <MemoryListItem
                    key={memory.id}
                    memory={memory}
                    isSelected={selectedMemory?.id === memory.id}
                    density={density}
                    onClick={() => handleSelectMemory(memory)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Detail Panel */}
          <div className="flex-1 bg-white overflow-y-auto">
            {selectedMemory ? (
              <MemoryDetailPanel
                memory={selectedMemory}
                onEdit={setEditingMemory}
                onDelete={handleDelete}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-400">
                <div className="text-center">
                  <Brain className="w-20 h-20 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-medium">Select a memory to view details</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Modals */}
      <AddMemoryModal
        userId={userId}
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onSubmit={async (data) => {
          await createMutation.mutateAsync(data);
        }}
      />

      <EditMemoryModal
        memory={editingMemory}
        isOpen={!!editingMemory}
        onClose={() => setEditingMemory(null)}
        onSubmit={async (memoryId, data) => {
          await updateMutation.mutateAsync({ memoryId, data });
        }}
      />
    </div>
  );
}
