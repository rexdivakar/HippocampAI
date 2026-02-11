import { useQuery } from '@tanstack/react-query';
import {
  MessageCircle,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  CircleDot,
  AlertTriangle,
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { apiClient } from '../services/api';
import clsx from 'clsx';

interface FeedbackPageProps {
  userId: string;
}

const FEEDBACK_COLORS: Record<string, string> = {
  relevant: '#22c55e',
  not_relevant: '#ef4444',
  partially_relevant: '#eab308',
  outdated: '#6b7280',
};

const FEEDBACK_LABELS: Record<string, string> = {
  relevant: 'Relevant',
  not_relevant: 'Not Relevant',
  partially_relevant: 'Partially Relevant',
  outdated: 'Outdated',
};

export function FeedbackPage({ userId }: FeedbackPageProps) {
  const {
    data: stats,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ['feedback-stats', userId],
    queryFn: () => apiClient.getFeedbackStats(userId),
  });

  const feedbackMap = stats?.stats ?? {};
  const totalFeedback = Object.values(feedbackMap).reduce((sum, v) => sum + v, 0);

  const pieData = Object.entries(feedbackMap).map(([key, value]) => ({
    name: FEEDBACK_LABELS[key] || key,
    value,
    color: FEEDBACK_COLORS[key] || '#6b7280',
  }));

  const relevantCount = feedbackMap.relevant ?? 0;
  const notRelevantCount = feedbackMap.not_relevant ?? 0;
  const partialCount = feedbackMap.partially_relevant ?? 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-primary-100 rounded-lg">
            <MessageCircle className="w-6 h-6 text-primary-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Relevance Feedback</h1>
            <p className="text-sm text-gray-500">Track how relevant retrieved memories are to queries</p>
          </div>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isLoading}
          className="btn-secondary flex items-center space-x-2"
        >
          <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Error Banner */}
      {isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
          <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-red-800">Failed to load feedback stats</p>
            <p className="text-xs text-red-600">{error instanceof Error ? error.message : 'Unknown error'}</p>
          </div>
          <button onClick={() => refetch()} className="ml-auto text-sm text-red-700 hover:text-red-900 font-medium">
            Retry
          </button>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <MessageCircle className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Feedback</p>
              <p className="text-2xl font-bold text-gray-900">{totalFeedback}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <ThumbsUp className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Relevant</p>
              <p className="text-2xl font-bold text-green-600">{relevantCount}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-red-100 rounded-lg">
              <ThumbsDown className="w-5 h-5 text-red-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Not Relevant</p>
              <p className="text-2xl font-bold text-red-600">{notRelevantCount}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <CircleDot className="w-5 h-5 text-yellow-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Partially Relevant</p>
              <p className="text-2xl font-bold text-yellow-600">{partialCount}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Pie Chart */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Feedback Distribution</h3>
        {isLoading ? (
          <div className="flex items-center justify-center h-[300px]">
            <RefreshCw className="w-6 h-6 text-primary-600 animate-spin" />
          </div>
        ) : pieData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-[300px] text-gray-400">
            <div className="text-center">
              <AlertTriangle className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No feedback data yet</p>
              <p className="text-sm mt-1">Rate memories on the Memories page to see data here</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
