import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  HeartPulse,
  Brain,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Activity,
  BarChart3,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { apiClient } from '../services/api';
import clsx from 'clsx';

interface HealthPageProps {
  userId: string;
}

const COLORS = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#6B7280'];

export function HealthPage({ userId }: HealthPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);

  // Fetch memories
  const { data: memories = [], isLoading: loadingMemories } = useQuery({
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

  // Fetch health score
  const { data: healthScore, isLoading: loadingHealth } = useQuery({
    queryKey: ['health-score', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getHealthScore(userId, true);
      return result;
    },
  });

  const isLoading = loadingMemories || loadingHealth;

  // Calculate statistics
  const stats = useMemo(() => {
    if (memories.length === 0) {
      return {
        total: 0,
        byType: [],
        byImportance: [],
        byConfidence: [],
        topTags: [],
        avgImportance: 0,
        avgConfidence: 0,
        avgAccessCount: 0,
        recentlyAccessed: [],
      };
    }

    // Type distribution
    const typeCount = memories.reduce((acc, m) => {
      acc[m.type] = (acc[m.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const byType = Object.entries(typeCount).map(([name, value]) => ({
      name,
      value,
    }));

    // Importance distribution
    const importanceRanges = [
      { name: '0-2', min: 0, max: 2, count: 0 },
      { name: '2-4', min: 2, max: 4, count: 0 },
      { name: '4-6', min: 4, max: 6, count: 0 },
      { name: '6-8', min: 6, max: 8, count: 0 },
      { name: '8-10', min: 8, max: 10, count: 0 },
    ];

    memories.forEach((m) => {
      const range = importanceRanges.find((r) => m.importance >= r.min && m.importance <= r.max);
      if (range) range.count++;
    });

    const byImportance = importanceRanges.map((r) => ({
      name: r.name,
      count: r.count,
    }));

    // Confidence distribution
    const confidenceRanges = [
      { name: '0-20%', min: 0, max: 0.2, count: 0 },
      { name: '20-40%', min: 0.2, max: 0.4, count: 0 },
      { name: '40-60%', min: 0.4, max: 0.6, count: 0 },
      { name: '60-80%', min: 0.6, max: 0.8, count: 0 },
      { name: '80-100%', min: 0.8, max: 1, count: 0 },
    ];

    memories.forEach((m) => {
      const range = confidenceRanges.find((r) => m.confidence >= r.min && m.confidence <= r.max);
      if (range) range.count++;
    });

    const byConfidence = confidenceRanges.map((r) => ({
      name: r.name,
      count: r.count,
    }));

    // Top tags
    const tagCount = memories.reduce((acc, m) => {
      m.tags.forEach((tag) => {
        acc[tag] = (acc[tag] || 0) + 1;
      });
      return acc;
    }, {} as Record<string, number>);

    const topTags = Object.entries(tagCount)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([name, count]) => ({ name, count }));

    // Averages
    const avgImportance = memories.reduce((sum, m) => sum + m.importance, 0) / memories.length;
    const avgConfidence = memories.reduce((sum, m) => sum + m.confidence, 0) / memories.length;
    const avgAccessCount = memories.reduce((sum, m) => sum + m.access_count, 0) / memories.length;

    // Recently accessed
    const recentlyAccessed = [...memories]
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      .slice(0, 5);

    return {
      total: memories.length,
      byType,
      byImportance,
      byConfidence,
      topTags,
      avgImportance,
      avgConfidence,
      avgAccessCount,
      recentlyAccessed,
    };
  }, [memories]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  const getHealthStatus = (score: number) => {
    if (score >= 80) return { label: 'Excellent', color: 'text-green-600', bg: 'bg-green-50', icon: CheckCircle };
    if (score >= 60) return { label: 'Good', color: 'text-blue-600', bg: 'bg-blue-50', icon: CheckCircle };
    if (score >= 40) return { label: 'Fair', color: 'text-yellow-600', bg: 'bg-yellow-50', icon: AlertCircle };
    return { label: 'Needs Attention', color: 'text-red-600', bg: 'bg-red-50', icon: AlertCircle };
  };

  const healthStatus = healthScore ? getHealthStatus(healthScore.overall_score) : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <HeartPulse className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Health Monitoring</h1>
            <p className="text-gray-600">Memory health scores, stats, and metrics</p>
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

      {/* Loading state */}
      {isLoading && (
        <div className="card text-center py-12">
          <RefreshCw className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading health data...</p>
        </div>
      )}

      {!isLoading && (
        <>
          {/* Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Total Memories */}
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Total Memories</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.total}</p>
                </div>
                <Brain className="w-12 h-12 text-blue-500 opacity-20" />
              </div>
            </div>

            {/* Average Importance */}
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Avg Importance</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.avgImportance.toFixed(1)}/10</p>
                </div>
                <TrendingUp className="w-12 h-12 text-orange-500 opacity-20" />
              </div>
            </div>

            {/* Average Confidence */}
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Avg Confidence</p>
                  <p className="text-3xl font-bold text-gray-900">{(stats.avgConfidence * 100).toFixed(0)}%</p>
                </div>
                <Activity className="w-12 h-12 text-green-500 opacity-20" />
              </div>
            </div>

            {/* Average Access Count */}
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Avg Access Count</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.avgAccessCount.toFixed(1)}</p>
                </div>
                <BarChart3 className="w-12 h-12 text-purple-500 opacity-20" />
              </div>
            </div>
          </div>

          {/* Health Score */}
          {healthScore && healthStatus && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900">Overall Health Score</h2>
                <div className={clsx('flex items-center space-x-2 px-3 py-1 rounded-lg', healthStatus.bg)}>
                  <healthStatus.icon className={clsx('w-5 h-5', healthStatus.color)} />
                  <span className={clsx('font-semibold', healthStatus.color)}>{healthStatus.label}</span>
                </div>
              </div>

              <div className="relative pt-1">
                <div className="flex mb-2 items-center justify-between">
                  <div>
                    <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full bg-gray-200 text-gray-700">
                      Health Score
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-2xl font-bold text-gray-900">
                      {healthScore.overall_score.toFixed(1)}
                    </span>
                    <span className="text-sm text-gray-500">/100</span>
                  </div>
                </div>
                <div className="overflow-hidden h-4 mb-4 text-xs flex rounded-full bg-gray-200">
                  <div
                    style={{ width: `${healthScore.overall_score}%` }}
                    className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-gradient-to-r from-primary-500 to-primary-600"
                  />
                </div>
              </div>

              {healthScore.issues && healthScore.issues.length > 0 && (
                <div className="mt-4 space-y-2">
                  <h3 className="font-semibold text-gray-900">Issues Found:</h3>
                  {healthScore.issues.map((issue, idx) => (
                    <div key={idx} className="flex items-start space-x-2 text-sm">
                      <AlertCircle className="w-4 h-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{issue.message}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Type Distribution */}
            <div className="card">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Memory Type Distribution</h2>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={stats.byType}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {stats.byType.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Importance Distribution */}
            <div className="card">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Importance Distribution</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stats.byImportance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#F59E0B" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Confidence Distribution */}
            <div className="card">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Confidence Distribution</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stats.byConfidence}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#10B981" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Top Tags */}
            <div className="card">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Top 10 Tags</h2>
              {stats.topTags.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={stats.topTags} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={100} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8B5CF6" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-12 text-gray-400">
                  <p>No tags found</p>
                </div>
              )}
            </div>
          </div>

          {/* Recently Accessed Memories */}
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Recently Accessed Memories</h2>
            {stats.recentlyAccessed.length > 0 ? (
              <div className="space-y-3">
                {stats.recentlyAccessed.map((memory) => (
                  <div
                    key={memory.id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900 line-clamp-1">{memory.text}</p>
                      <div className="flex items-center space-x-3 mt-1 text-xs text-gray-500">
                        <span className="px-2 py-0.5 bg-white rounded">{memory.type}</span>
                        <span>⭐ {memory.importance.toFixed(1)}</span>
                        <span>👁 {memory.access_count} views</span>
                      </div>
                    </div>
                    <div className="ml-4 text-xs text-gray-400">
                      {new Date(memory.updated_at).toLocaleDateString()}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <p>No memories found</p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
