import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Layers,
  RefreshCw,
  Circle,
  AlertCircle,
  Copy,
  Filter,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { Memory } from '../types';
import clsx from 'clsx';

interface ClusterPageProps {
  userId: string;
}

export function ClusterPage({ userId }: ClusterPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);

  // Fetch memories
  const { data: memories = [], isLoading } = useQuery({
    queryKey: ['memories', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        limit: 1000,
      });
      return result;
    },
  });

  // Simple clustering based on tags and types (in production, use UMAP/HDBSCAN on embeddings)
  const clusters = useMemo(() => {
    const clusterMap = new Map<string, Memory[]>();

    memories.forEach((memory: Memory) => {
      // Cluster by primary tag or type
      const key = memory.tags[0] || memory.type;
      if (!clusterMap.has(key)) {
        clusterMap.set(key, []);
      }
      clusterMap.get(key)!.push(memory);
    });

    return Array.from(clusterMap.entries()).map(([name, items], idx) => ({
      id: idx,
      name,
      size: items.length,
      memories: items,
      avgImportance: items.reduce((sum: number, m: Memory) => sum + m.importance, 0) / items.length,
      color: `hsl(${idx * 137.5}, 70%, 60%)`,
    }));
  }, [memories]);

  // Find outliers (memories with unique characteristics)
  const outliers = useMemo(() => {
    return memories.filter((m: Memory) => m.tags.length === 0 || m.importance < 2);
  }, [memories]);

  // Find potential duplicates (simple text similarity)
  const duplicates = useMemo(() => {
    const dups: Memory[][] = [];
    for (let i = 0; i < memories.length; i++) {
      for (let j = i + 1; j < memories.length; j++) {
        const similarity = calculateSimilarity(memories[i].text, memories[j].text);
        if (similarity > 0.7) {
          if (!dups.find((d: Memory[]) => d.some((m: Memory) => m.id === memories[i].id))) {
            dups.push([memories[i], memories[j]]);
          }
        }
      }
    }
    return dups;
  }, [memories]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Layers className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Memory Cluster View</h1>
            <p className="text-gray-600">Topic visualization with themes, clusters, and outliers</p>
          </div>
        </div>

        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="btn-secondary flex items-center space-x-2"
        >
          <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card bg-blue-50 border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 mb-1">Clusters</p>
              <p className="text-2xl font-bold text-blue-900">{clusters.length}</p>
            </div>
            <Layers className="w-10 h-10 text-blue-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-green-50 border border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 mb-1">Total Memories</p>
              <p className="text-2xl font-bold text-green-900">{memories.length}</p>
            </div>
            <Circle className="w-10 h-10 text-green-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-yellow-50 border border-yellow-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-yellow-600 mb-1">Outliers</p>
              <p className="text-2xl font-bold text-yellow-900">{outliers.length}</p>
            </div>
            <AlertCircle className="w-10 h-10 text-yellow-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-red-50 border border-red-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-red-600 mb-1">Duplicates</p>
              <p className="text-2xl font-bold text-red-900">{duplicates.length}</p>
            </div>
            <Copy className="w-10 h-10 text-red-500 opacity-20" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Cluster List */}
        <div className="lg:col-span-1">
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Memory Islands</h2>

            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <RefreshCw className="w-12 h-12 text-primary-600 animate-spin" />
              </div>
            ) : (
              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {clusters.map((cluster) => (
                  <div
                    key={cluster.id}
                    onClick={() => setSelectedCluster(cluster.id)}
                    className={clsx(
                      'p-4 rounded-lg border-2 cursor-pointer transition-all',
                      selectedCluster === cluster.id
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    )}
                  >
                    <div className="flex items-center space-x-3 mb-2">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: cluster.color }}
                      />
                      <span className="font-semibold text-gray-900">{cluster.name}</span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>{cluster.size} memories</span>
                      <span>Avg: {cluster.avgImportance.toFixed(1)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Cluster Details */}
        <div className="lg:col-span-2 space-y-4">
          {selectedCluster !== null ? (
            <>
              {/* Selected Cluster Info */}
              <div
                className="card"
                style={{
                  background: `linear-gradient(135deg, ${clusters[selectedCluster].color}15 0%, ${clusters[selectedCluster].color}30 100%)`,
                }}
              >
                <div className="flex items-center space-x-3 mb-2">
                  <div
                    className="w-6 h-6 rounded-full"
                    style={{ backgroundColor: clusters[selectedCluster].color }}
                  />
                  <h2 className="text-2xl font-bold text-gray-900">
                    {clusters[selectedCluster].name}
                  </h2>
                </div>
                <p className="text-sm text-gray-600">
                  {clusters[selectedCluster].size} memories with average importance of{' '}
                  {clusters[selectedCluster].avgImportance.toFixed(1)}
                </p>
              </div>

              {/* Cluster Memories */}
              <div className="card">
                <h3 className="text-lg font-bold text-gray-900 mb-4">Cluster Memories</h3>
                <div className="space-y-3 max-h-[500px] overflow-y-auto">
                  {clusters[selectedCluster].memories.map((memory: Memory) => (
                    <div key={memory.id} className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm font-medium text-gray-900 mb-2">{memory.text}</p>
                      <div className="flex items-center space-x-3 text-xs text-gray-500">
                        <span className="px-2 py-0.5 bg-white rounded">{memory.type}</span>
                        <span>⭐ {memory.importance.toFixed(1)}</span>
                        <span>🔁 {memory.access_count} uses</span>
                      </div>
                      {memory.tags.length > 0 && (
                        <div className="flex items-center space-x-2 mt-2">
                          {memory.tags.map((tag: string) => (
                            <span
                              key={tag}
                              className="px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded-full"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="card">
              <div className="text-center py-12 text-gray-400">
                <Filter className="w-16 h-16 mx-auto mb-4" />
                <p>Select a cluster to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Outliers Section */}
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
          <AlertCircle className="w-6 h-6 text-yellow-600" />
          <span>Outliers & Uncategorized Memories</span>
        </h2>
        {outliers.length === 0 ? (
          <p className="text-center py-8 text-gray-400">No outliers found</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {outliers.slice(0, 6).map((memory: Memory) => (
              <div key={memory.id} className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-sm font-medium text-gray-900 mb-2">{memory.text}</p>
                <div className="flex items-center space-x-3 text-xs text-gray-500">
                  <span className="px-2 py-0.5 bg-white rounded">{memory.type}</span>
                  <span>⭐ {memory.importance.toFixed(1)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Duplicates Section */}
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
          <Copy className="w-6 h-6 text-red-600" />
          <span>Potential Duplicates</span>
        </h2>
        {duplicates.length === 0 ? (
          <p className="text-center py-8 text-gray-400">No duplicates detected</p>
        ) : (
          <div className="space-y-4">
            {duplicates.slice(0, 3).map((pair, idx) => (
              <div key={idx} className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {pair.map((memory: Memory) => (
                    <div key={memory.id} className="bg-white rounded-lg p-3">
                      <p className="text-sm text-gray-900 mb-2">{memory.text}</p>
                      <div className="flex items-center space-x-2 text-xs text-gray-500">
                        <span>ID: {memory.id.slice(0, 8)}</span>
                        <span>⭐ {memory.importance.toFixed(1)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Simple Jaccard similarity for text comparison
function calculateSimilarity(text1: string, text2: string): number {
  const words1 = new Set(text1.toLowerCase().split(/\s+/));
  const words2 = new Set(text2.toLowerCase().split(/\s+/));
  const intersection = new Set([...words1].filter((x) => words2.has(x)));
  const union = new Set([...words1, ...words2]);
  return intersection.size / union.size;
}
