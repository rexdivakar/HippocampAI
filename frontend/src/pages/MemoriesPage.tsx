import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Brain, Grid3x3, List, RefreshCw, Plus } from 'lucide-react';
import { apiClient } from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';
import type { Memory, MemoryFilters } from '../types';
import { MemoryCard } from '../components/MemoryCard';
import { MemoryFiltersComponent } from '../components/MemoryFilters';
import { MemoryDetailDrawer } from '../components/MemoryDetailDrawer';
import { AddMemoryModal } from '../components/AddMemoryModal';
import { EditMemoryModal } from '../components/EditMemoryModal';
import { ShareMemoryModal } from '../components/ShareMemoryModal';
import clsx from 'clsx';

interface MemoriesPageProps {
  userId: string;
}

export function MemoriesPage({ userId }: MemoriesPageProps) {
  const queryClient = useQueryClient();
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filters, setFilters] = useState<MemoryFilters>({});
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null);
  const [sharingMemory, setSharingMemory] = useState<Memory | null>(null);

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
        limit: 100,
      });
      return result;
    },
  });

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
    // Listen for real-time memory updates
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

  // Filter memories client-side
  const filteredMemories = memories.filter((memory) => {
    // Text search
    if (filters.searchText) {
      const searchLower = filters.searchText.toLowerCase();
      const matchesText = memory.text.toLowerCase().includes(searchLower);
      const matchesTags = memory.tags.some((tag) => tag.toLowerCase().includes(searchLower));
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

  const resetFilters = () => {
    setFilters({});
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Brain className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Memories</h1>
            <p className="text-gray-600">
              {filteredMemories.length} of {memories.length} memories
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* View mode toggle */}
          <div className="flex items-center bg-white rounded-lg border border-gray-200 p-1">
            <button
              onClick={() => setViewMode('grid')}
              className={clsx(
                'p-2 rounded transition-all',
                viewMode === 'grid'
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              )}
            >
              <Grid3x3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={clsx(
                'p-2 rounded transition-all',
                viewMode === 'list'
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              )}
            >
              <List className="w-4 h-4" />
            </button>
          </div>

          {/* Refresh button */}
          <button
            onClick={() => refetch()}
            disabled={isLoading}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
            <span>Refresh</span>
          </button>

          {/* Add memory button */}
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Add Memory</span>
          </button>
        </div>
      </div>

      {/* Filters */}
      <MemoryFiltersComponent
        filters={filters}
        onFiltersChange={setFilters}
        onReset={resetFilters}
      />

      {/* Loading state */}
      {isLoading && (
        <div className="card text-center py-12">
          <RefreshCw className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading memories...</p>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && filteredMemories.length === 0 && (
        <div className="card text-center py-12">
          <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-700 mb-2">No memories found</h2>
          <p className="text-gray-500 mb-4">
            {memories.length === 0
              ? 'Start by creating your first memory'
              : 'Try adjusting your filters'}
          </p>
        </div>
      )}

      {/* Memory grid/list */}
      {!isLoading && filteredMemories.length > 0 && (
        <div
          className={clsx(
            viewMode === 'grid'
              ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'
              : 'space-y-4'
          )}
        >
          {filteredMemories.map((memory) => (
            <MemoryCard
              key={memory.id}
              memory={memory}
              onView={setSelectedMemory}
              onEdit={setEditingMemory}
              onShare={setSharingMemory}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}

      {/* Memory detail drawer */}
      <MemoryDetailDrawer
        memory={selectedMemory}
        isOpen={!!selectedMemory}
        onClose={() => setSelectedMemory(null)}
      />

      {/* Add memory modal */}
      <AddMemoryModal
        userId={userId}
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onSubmit={async (data) => {
          await createMutation.mutateAsync(data);
        }}
      />

      {/* Edit memory modal */}
      <EditMemoryModal
        memory={editingMemory}
        isOpen={!!editingMemory}
        onClose={() => setEditingMemory(null)}
        onSubmit={async (memoryId, data) => {
          await updateMutation.mutateAsync({ memoryId, data });
        }}
      />

      {/* Share memory modal */}
      <ShareMemoryModal
        memory={sharingMemory}
        isOpen={!!sharingMemory}
        onClose={() => setSharingMemory(null)}
      />
    </div>
  );
}
