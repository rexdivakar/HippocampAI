import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
  Play,
  Search,
  RefreshCw,
  TrendingUp,
  Activity,
  Target,
  MessageSquare,
  ChevronRight,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { Memory, RetrievalResult } from '../types';
import clsx from 'clsx';
import { format } from 'date-fns';

interface ReplayPageProps {
  userId: string;
}

export function ReplayPage({ userId }: ReplayPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [recallQuery, setRecallQuery] = useState('');
  const [recallResults, setRecallResults] = useState<RetrievalResult[]>([]);

  // Fetch memories from real API
  const { data: memories = [], isLoading } = useQuery({
    queryKey: ['memories', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        filters: {
          session_id: userId,
        },
        limit: 1000,
      });
      return result;
    },
  });

  // Filter memories by search term
  const filteredMemories = memories.filter((m) =>
    m.text.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  // Recall mutation — calls the real recall endpoint
  const recallMutation = useMutation({
    mutationFn: (query: string) =>
      apiClient.recallMemories({
        query,
        user_id: userId,
        k: 10,
      }),
    onSuccess: (results) => {
      setRecallResults(results);
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      console.error('Recall failed:', msg);
    },
  });

  const handleRecall = () => {
    if (!recallQuery.trim()) return;
    recallMutation.mutate(recallQuery.trim());
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Play className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Memory Replay & Usage Trace</h1>
            <p className="text-gray-600">Track memory usage, retrieval events, and impact</p>
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

      {/* Replay Recall Panel */}
      <div className="card space-y-4">
        <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
          <Target className="w-5 h-5 text-primary-600" />
          Replay Recall
        </h2>
        <p className="text-sm text-gray-600">
          Enter a query to replay retrieval and see which memories would be recalled with ranked
          scores.
        </p>
        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={recallQuery}
              onChange={(e) => setRecallQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleRecall()}
              placeholder="e.g. What does the user prefer for breakfast?"
              className="w-full pl-10 pr-4 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          <button
            onClick={handleRecall}
            disabled={recallMutation.isPending || !recallQuery.trim()}
            className="px-4 py-2 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {recallMutation.isPending ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Recall
          </button>
        </div>

        {/* Recall results */}
        {recallResults.length > 0 && (
          <div className="space-y-2 pt-2">
            <h3 className="text-sm font-semibold text-gray-700">
              {recallResults.length} result{recallResults.length !== 1 ? 's' : ''} for &ldquo;
              {recallQuery}&rdquo;
            </h3>
            {recallResults.map((result) => (
              <div
                key={result.memory.id}
                onClick={() => setSelectedMemory(result.memory)}
                className={clsx(
                  'flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all',
                  selectedMemory?.id === result.memory.id
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300 bg-white'
                )}
              >
                {/* Rank badge */}
                <span className="shrink-0 w-6 h-6 rounded-full bg-primary-100 text-primary-700 text-xs font-bold flex items-center justify-center mt-0.5">
                  {result.rank}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900 line-clamp-2">{result.memory.text}</p>
                  <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                    <span className="px-1.5 py-0.5 bg-gray-100 rounded">{result.memory.type}</span>
                    <span>Score: {result.score.toFixed(3)}</span>
                    <span>Importance: {result.memory.importance.toFixed(1)}</span>
                  </div>
                  {/* Score bar */}
                  <div className="flex items-center gap-2 mt-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                      <div
                        className="bg-primary-500 h-1.5 rounded-full transition-all"
                        style={{ width: `${Math.min(result.score * 100, 100)}%` }}
                      />
                    </div>
                    <span className="text-xs font-semibold text-primary-600 w-10 text-right">
                      {(result.score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <ChevronRight className="shrink-0 w-4 h-4 text-gray-400 mt-1" />
              </div>
            ))}
          </div>
        )}

        {recallMutation.isError && (
          <p className="text-sm text-red-600">
            Recall failed. Ensure the backend is reachable and try again.
          </p>
        )}
      </div>

      {/* Search filter for memory list */}
      <div className="card">
        <div className="flex items-center space-x-4">
          <Search className="w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Filter memories by text..."
            className="flex-1 outline-none text-gray-900"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Memory List */}
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Select a Memory</h2>

          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-12 h-12 text-primary-600 animate-spin" />
            </div>
          ) : filteredMemories.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <p>No memories found</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[600px] overflow-y-auto">
              {filteredMemories.map((memory) => (
                <div
                  key={memory.id}
                  onClick={() => setSelectedMemory(memory)}
                  className={clsx(
                    'p-4 rounded-lg border-2 cursor-pointer transition-all',
                    selectedMemory?.id === memory.id
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300 bg-white'
                  )}
                >
                  <p className="text-sm font-medium text-gray-900 line-clamp-2 mb-2">
                    {memory.text}
                  </p>
                  <div className="flex items-center space-x-3 text-xs text-gray-500">
                    <span className="px-2 py-0.5 bg-gray-100 rounded">{memory.type}</span>
                    <span>{memory.access_count} uses</span>
                    <span>importance: {memory.importance.toFixed(1)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Usage Trace Details */}
        <div className="space-y-4">
          {selectedMemory ? (
            <>
              {/* Selected Memory Info */}
              <div className="card bg-gradient-to-r from-primary-50 to-blue-50">
                <h3 className="text-lg font-bold text-gray-900 mb-2">Selected Memory</h3>
                <p className="text-sm text-gray-700 mb-4">{selectedMemory.text}</p>
                <div className="flex flex-wrap gap-3 text-xs text-gray-600">
                  <span>Created: {format(new Date(selectedMemory.created_at), 'PPp')}</span>
                  {selectedMemory.last_accessed_at && (
                    <span>
                      Last accessed: {format(new Date(selectedMemory.last_accessed_at), 'PPp')}
                    </span>
                  )}
                </div>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="card bg-blue-50 border border-blue-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-blue-600 mb-1">Total Retrievals</p>
                      <p className="text-2xl font-bold text-blue-900">
                        {selectedMemory.access_count}
                      </p>
                    </div>
                    <Activity className="w-10 h-10 text-blue-500 opacity-20" />
                  </div>
                </div>

                <div className="card bg-green-50 border border-green-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-green-600 mb-1">Importance Score</p>
                      <p className="text-2xl font-bold text-green-900">
                        {selectedMemory.importance.toFixed(1)}
                      </p>
                    </div>
                    <TrendingUp className="w-10 h-10 text-green-500 opacity-20" />
                  </div>
                </div>
              </div>

              {/* Impact on Response */}
              <div className="card bg-gradient-to-r from-yellow-50 to-orange-50">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <MessageSquare className="w-5 h-5 text-primary-600" />
                  <span>Memory Profile</span>
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Type</span>
                    <span className="font-medium text-gray-900 capitalize">
                      {selectedMemory.type}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Confidence</span>
                    <span className="font-medium text-gray-900">
                      {(selectedMemory.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Tags</span>
                    <span className="font-medium text-gray-900">
                      {selectedMemory.tags.length > 0 ? selectedMemory.tags.join(', ') : '—'}
                    </span>
                  </div>
                  {selectedMemory.expires_at && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Expires</span>
                      <span className="font-medium text-gray-900">
                        {format(new Date(selectedMemory.expires_at), 'PP')}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Recall results for this memory (if a recall was run) */}
              {recallResults.length > 0 && (
                <div className="card">
                  <h3 className="text-lg font-bold text-gray-900 mb-3 flex items-center gap-2">
                    <Target className="w-5 h-5 text-primary-600" />
                    Recall Rank for This Memory
                  </h3>
                  {(() => {
                    const match = recallResults.find((r) => r.memory.id === selectedMemory.id);
                    return match ? (
                      <div className="space-y-2">
                        <div className="flex items-center gap-3">
                          <span className="text-3xl font-bold text-primary-600">#{match.rank}</span>
                          <div className="flex-1">
                            <p className="text-sm text-gray-600 mb-1">
                              Score: {match.score.toFixed(4)}
                            </p>
                            <div className="flex items-center gap-2">
                              <div className="flex-1 bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-primary-500 h-2 rounded-full"
                                  style={{ width: `${Math.min(match.score * 100, 100)}%` }}
                                />
                              </div>
                              <span className="text-xs font-semibold text-primary-600 w-10 text-right">
                                {(match.score * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-gray-500">
                        This memory was not in the top recall results for &ldquo;{recallQuery}&rdquo;.
                      </p>
                    );
                  })()}
                </div>
              )}
            </>
          ) : (
            <div className="card">
              <div className="text-center py-12 text-gray-400">
                <Play className="w-16 h-16 mx-auto mb-4" />
                <p>Select a memory to view its usage trace</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
