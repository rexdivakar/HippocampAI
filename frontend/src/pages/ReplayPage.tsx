import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Play,
  Search,
  RefreshCw,
  MapPin,
  TrendingUp,
  Users,
  MessageSquare,
  Activity,
  Target,
} from 'lucide-react';
import { apiClient } from '../services/api';
import clsx from 'clsx';
import { format } from 'date-fns';

interface ReplayPageProps {
  userId: string;
}

export function ReplayPage({ userId }: ReplayPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedMemory, setSelectedMemory] = useState<any | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

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

  // Filter memories
  const filteredMemories = memories.filter((m) =>
    m.text.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  // Mock retrieval data - in production, this would come from backend
  const getRetrievalData = (memory: any) => ({
    usageCount: memory.access_count,
    lastRetrieved: memory.last_accessed_at || memory.updated_at,
    retrievalScore: memory.confidence * 10,
    usedInSessions: [
      { id: memory.session_id || 'N/A', timestamp: memory.created_at, score: 9.2 },
    ],
    agentUsage: [
      { agent: memory.agent_id || 'default-agent', count: memory.access_count, avgScore: 8.5 },
    ],
    rerankerContribution: memory.importance * 0.1,
  });

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

      {/* Search */}
      <div className="card">
        <div className="flex items-center space-x-4">
          <Search className="w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search memories to trace their usage..."
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
                    <span>🔁 {memory.access_count} uses</span>
                    <span>⭐ {memory.importance.toFixed(1)}</span>
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
                <div className="flex items-center space-x-4 text-xs text-gray-600">
                  <span>Created: {format(new Date(selectedMemory.created_at), 'PPp')}</span>
                  {selectedMemory.last_accessed_at && (
                    <span>Last Used: {format(new Date(selectedMemory.last_accessed_at), 'PPp')}</span>
                  )}
                </div>
              </div>

              {/* Usage Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="card bg-blue-50 border border-blue-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-blue-600 mb-1">Total Retrievals</p>
                      <p className="text-2xl font-bold text-blue-900">
                        {getRetrievalData(selectedMemory).usageCount}
                      </p>
                    </div>
                    <Activity className="w-10 h-10 text-blue-500 opacity-20" />
                  </div>
                </div>

                <div className="card bg-green-50 border border-green-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-green-600 mb-1">Retrieval Score</p>
                      <p className="text-2xl font-bold text-green-900">
                        {getRetrievalData(selectedMemory).retrievalScore.toFixed(1)}
                      </p>
                    </div>
                    <TrendingUp className="w-10 h-10 text-green-500 opacity-20" />
                  </div>
                </div>
              </div>

              {/* Retrieval Events */}
              <div className="card">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <MapPin className="w-5 h-5 text-primary-600" />
                  <span>Retrieval Events</span>
                </h3>
                <div className="space-y-3">
                  {getRetrievalData(selectedMemory).usedInSessions.map((session, idx) => (
                    <div key={idx} className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-900">
                          Session: {session.id.slice(0, 8)}
                        </span>
                        <span className="text-xs text-gray-500">
                          {format(new Date(session.timestamp), 'PPp')}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${session.score * 10}%` }}
                          />
                        </div>
                        <span className="text-xs font-semibold text-green-600">
                          {session.score.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Agent Usage */}
              <div className="card">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <Users className="w-5 h-5 text-primary-600" />
                  <span>Agent Usage</span>
                </h3>
                <div className="space-y-3">
                  {getRetrievalData(selectedMemory).agentUsage.map((agent, idx) => (
                    <div key={idx} className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-900">{agent.agent}</span>
                        <span className="text-xs text-gray-500">{agent.count} times</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-600">Avg Score:</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${agent.avgScore * 10}%` }}
                          />
                        </div>
                        <span className="text-xs font-semibold text-blue-600">
                          {agent.avgScore.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Reranker Contribution */}
              <div className="card">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <Target className="w-5 h-5 text-primary-600" />
                  <span>Reranker Contribution</span>
                </h3>
                <div className="bg-purple-50 rounded-lg p-4">
                  <p className="text-sm text-gray-700 mb-3">
                    This memory's contribution to final retrieval ranking:
                  </p>
                  <div className="flex items-center space-x-3">
                    <div className="flex-1 bg-purple-200 rounded-full h-4">
                      <div
                        className="bg-purple-600 h-4 rounded-full flex items-center justify-end pr-2"
                        style={{ width: `${getRetrievalData(selectedMemory).rerankerContribution * 10}%` }}
                      >
                        <span className="text-xs font-bold text-white">
                          {(getRetrievalData(selectedMemory).rerankerContribution * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Impact on Response */}
              <div className="card bg-gradient-to-r from-yellow-50 to-orange-50">
                <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center space-x-2">
                  <MessageSquare className="w-5 h-5 text-primary-600" />
                  <span>Impact on Response</span>
                </h3>
                <p className="text-sm text-gray-700">
                  This memory influenced the final response by providing contextual information
                  about {selectedMemory.type} with {(selectedMemory.confidence * 100).toFixed(0)}%
                  confidence. It was ranked with importance score of {selectedMemory.importance.toFixed(1)}/10.
                </p>
              </div>
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
