import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Clock,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ArrowRight,
  Calendar,
  Activity,
  Zap,
} from 'lucide-react';
import { apiClient } from '../services/api';
import clsx from 'clsx';
import { format } from 'date-fns';

interface TimelinePageProps {
  userId: string;
}

export function TimelinePage({ userId }: TimelinePageProps) {
  const [refreshKey, setRefreshKey] = useState(0);

  // Fetch memories
  const { data: memories = [], isLoading } = useQuery({
    queryKey: ['memories', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        filters: {
          session_id: userId, // Pass userId as session_id to match by either field
        },
        limit: 1000,
      });
      return result;
    },
  });

  // Group memories by date
  const timelineData = memories
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .reduce((acc, memory) => {
      const date = format(new Date(memory.created_at), 'yyyy-MM-dd');
      if (!acc[date]) {
        acc[date] = [];
      }
      acc[date].push(memory);
      return acc;
    }, {} as Record<string, typeof memories>);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Clock className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Memory Timeline</h1>
            <p className="text-gray-600">Chronological view of memory evolution</p>
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

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card bg-blue-50 border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 mb-1">Learned Today</p>
              <p className="text-2xl font-bold text-blue-900">
                {timelineData[format(new Date(), 'yyyy-MM-dd')]?.length || 0}
              </p>
            </div>
            <TrendingUp className="w-10 h-10 text-blue-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-orange-50 border border-orange-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-orange-600 mb-1">Reinforced</p>
              <p className="text-2xl font-bold text-orange-900">
                {memories.filter((m) => m.access_count > 3).length}
              </p>
            </div>
            <Zap className="w-10 h-10 text-orange-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-red-50 border border-red-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-red-600 mb-1">Decayed</p>
              <p className="text-2xl font-bold text-red-900">
                {memories.filter((m) => m.importance < 3).length}
              </p>
            </div>
            <TrendingDown className="w-10 h-10 text-red-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-green-50 border border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 mb-1">Active Sessions</p>
              <p className="text-2xl font-bold text-green-900">
                {new Set(memories.filter((m) => m.session_id).map((m) => m.session_id)).size}
              </p>
            </div>
            <Activity className="w-10 h-10 text-green-500 opacity-20" />
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-6">Memory Evolution Timeline</h2>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-12 h-12 text-primary-600 animate-spin" />
          </div>
        ) : Object.keys(timelineData).length === 0 ? (
          <div className="text-center py-12 text-gray-400">
            <Calendar className="w-16 h-16 mx-auto mb-4" />
            <p>No memories found</p>
          </div>
        ) : (
          <div className="space-y-8">
            {Object.entries(timelineData).map(([date, dayMemories]) => (
              <div key={date} className="relative">
                {/* Date Header */}
                <div className="flex items-center mb-4">
                  <div className="bg-primary-600 text-white px-4 py-2 rounded-lg font-semibold">
                    {format(new Date(date), 'MMMM d, yyyy')}
                  </div>
                  <div className="ml-4 text-sm text-gray-500">
                    {dayMemories.length} {dayMemories.length === 1 ? 'memory' : 'memories'}
                  </div>
                </div>

                {/* Timeline Items */}
                <div className="relative pl-8 border-l-2 border-gray-200">
                  <div className="space-y-4">
                    {dayMemories.map((memory) => (
                      <div key={memory.id} className="relative">
                        {/* Timeline Dot */}
                        <div
                          className={clsx(
                            'absolute -left-10 w-4 h-4 rounded-full border-2 border-white',
                            memory.access_count > 3
                              ? 'bg-green-500'
                              : memory.importance > 7
                              ? 'bg-blue-500'
                              : memory.importance < 3
                              ? 'bg-red-500'
                              : 'bg-gray-400'
                          )}
                        />

                        {/* Memory Card */}
                        <div className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex-1">
                              <p className="text-sm font-medium text-gray-900 mb-1">
                                {memory.text}
                              </p>
                              <div className="flex items-center space-x-3 text-xs text-gray-500">
                                <span className="px-2 py-0.5 bg-white rounded">{memory.type}</span>
                                <span>⭐ {memory.importance.toFixed(1)}</span>
                                <span>🔁 {memory.access_count} uses</span>
                                <span>💯 {(memory.confidence * 100).toFixed(0)}%</span>
                              </div>
                            </div>
                            <div className="text-xs text-gray-400 ml-4">
                              {format(new Date(memory.created_at), 'HH:mm')}
                            </div>
                          </div>

                          {/* Tags */}
                          {memory.tags.length > 0 && (
                            <div className="flex items-center space-x-2 mt-2">
                              {memory.tags.map((tag) => (
                                <span
                                  key={tag}
                                  className="px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded-full"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}

                          {/* Session Info */}
                          {memory.session_id && (
                            <div className="flex items-center space-x-2 mt-2 text-xs text-gray-500">
                              <ArrowRight className="w-3 h-3" />
                              <span>Session: {memory.session_id.slice(0, 8)}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Retention Curve */}
      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Memory Retention Curve</h2>
        <div className="bg-gray-50 rounded-lg p-8 text-center">
          <p className="text-gray-500 mb-4">
            Retention curve visualization showing memory strength over time
          </p>
          <div className="grid grid-cols-3 gap-4 max-w-2xl mx-auto">
            <div className="bg-white rounded-lg p-4">
              <p className="text-2xl font-bold text-green-600">
                {((memories.filter((m) => m.access_count > 3).length / memories.length) * 100 || 0).toFixed(0)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">Strong Retention</p>
            </div>
            <div className="bg-white rounded-lg p-4">
              <p className="text-2xl font-bold text-yellow-600">
                {((memories.filter((m) => m.access_count > 0 && m.access_count <= 3).length / memories.length) * 100 || 0).toFixed(0)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">Medium Retention</p>
            </div>
            <div className="bg-white rounded-lg p-4">
              <p className="text-2xl font-bold text-red-600">
                {((memories.filter((m) => m.access_count === 0).length / memories.length) * 100 || 0).toFixed(0)}%
              </p>
              <p className="text-sm text-gray-600 mt-1">Weak Retention</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
