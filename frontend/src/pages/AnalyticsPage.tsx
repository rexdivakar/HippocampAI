import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart3,
  TrendingUp,
  Target,
  Zap,
  Clock,
  Calendar,
  RefreshCw,
  AlertTriangle,
  Lightbulb,
  Activity,
} from 'lucide-react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { apiClient } from '../services/api';
import type { Memory } from '../types';
import clsx from 'clsx';
import { format, subDays, differenceInDays } from 'date-fns';

interface AnalyticsPageProps {
  userId: string;
}

const COLORS = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#6B7280'];

export function AnalyticsPage({ userId }: AnalyticsPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [timeRange, setTimeRange] = useState<7 | 30 | 90>(30);

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

  // Calculate analytics
  const analytics = useMemo(() => {
    if (memories.length === 0) {
      return {
        timeline: [],
        typeEvolution: [],
        importanceVsAccess: [],
        topTags: [],
        engagementTrend: [],
        lifecycle: { new: 0, active: 0, stale: 0 },
        insights: [],
        memoryVelocity: 0,
        retentionRate: 0,
        avgLifespan: 0,
      };
    }

    const now = new Date();
    const startDate = subDays(now, timeRange);

    // Timeline: Memory creation over time
    const timelineMap = new Map<string, number>();
    for (let i = 0; i <= timeRange; i++) {
      const date = format(subDays(now, timeRange - i), 'MMM dd');
      timelineMap.set(date, 0);
    }

    memories.forEach((m) => {
      const createdDate = new Date(m.created_at);
      if (createdDate >= startDate) {
        const dateKey = format(createdDate, 'MMM dd');
        timelineMap.set(dateKey, (timelineMap.get(dateKey) || 0) + 1);
      }
    });

    const timeline = Array.from(timelineMap.entries()).map(([date, count]) => ({
      date,
      count,
    }));

    // Type evolution over time
    const typeByDate = new Map<string, Record<string, number>>();
    memories.forEach((m) => {
      const createdDate = new Date(m.created_at);
      if (createdDate >= startDate) {
        const dateKey = format(createdDate, 'MMM dd');
        if (!typeByDate.has(dateKey)) {
          typeByDate.set(dateKey, {});
        }
        const dateTypes = typeByDate.get(dateKey)!;
        dateTypes[m.type] = (dateTypes[m.type] || 0) + 1;
      }
    });

    const typeEvolution = Array.from(timelineMap.keys()).map((date) => {
      const types = typeByDate.get(date) || {};
      return {
        date,
        fact: types.fact || 0,
        preference: types.preference || 0,
        goal: types.goal || 0,
        habit: types.habit || 0,
        event: types.event || 0,
        context: types.context || 0,
      };
    });

    // Importance vs Access correlation
    const importanceVsAccess = memories.map((m) => ({
      importance: m.importance,
      accessCount: m.access_count,
      name: m.type,
    }));

    // Top performing tags
    const tagPerformance = new Map<string, { count: number; avgAccess: number; avgImportance: number }>();
    memories.forEach((m) => {
      m.tags.forEach((tag) => {
        if (!tagPerformance.has(tag)) {
          tagPerformance.set(tag, { count: 0, avgAccess: 0, avgImportance: 0 });
        }
        const perf = tagPerformance.get(tag)!;
        perf.count += 1;
        perf.avgAccess += m.access_count;
        perf.avgImportance += m.importance;
      });
    });

    const topTags = Array.from(tagPerformance.entries())
      .map(([name, data]) => ({
        name,
        count: data.count,
        avgAccess: data.avgAccess / data.count,
        avgImportance: data.avgImportance / data.count,
        score: (data.avgAccess / data.count) * (data.avgImportance / data.count),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    // Engagement trend (access over time)
    const engagementMap = new Map<string, number>();
    for (let i = 0; i <= timeRange; i++) {
      const date = format(subDays(now, timeRange - i), 'MMM dd');
      engagementMap.set(date, 0);
    }

    memories.forEach((m) => {
      const accessDate = new Date(m.updated_at);
      if (accessDate >= startDate) {
        const dateKey = format(accessDate, 'MMM dd');
        engagementMap.set(dateKey, (engagementMap.get(dateKey) || 0) + m.access_count);
      }
    });

    const engagementTrend = Array.from(engagementMap.entries()).map(([date, totalAccess]) => ({
      date,
      totalAccess,
    }));

    // Memory lifecycle
    const daysSinceCreation = (m: Memory) => differenceInDays(now, new Date(m.created_at));
    const daysSinceAccess = (m: Memory) => differenceInDays(now, new Date(m.updated_at));

    const lifecycle = {
      new: memories.filter((m) => daysSinceCreation(m) <= 7).length,
      active: memories.filter((m) => daysSinceAccess(m) <= 7 && daysSinceCreation(m) > 7).length,
      stale: memories.filter((m) => daysSinceAccess(m) > 30).length,
    };

    // Calculate insights
    const insights: string[] = [];

    // Memory velocity
    const recentMemories = memories.filter((m) => daysSinceCreation(m) <= 7).length;
    const memoryVelocity = (recentMemories / 7) * 7; // per week
    if (memoryVelocity > 10) {
      insights.push(`High memory creation rate: ${memoryVelocity.toFixed(1)} memories/week`);
    }

    // Stale memories
    const stalePercent = (lifecycle.stale / memories.length) * 100;
    if (stalePercent > 30) {
      insights.push(`${stalePercent.toFixed(0)}% of memories haven't been accessed in 30+ days`);
    }

    // Importance correlation
    const highImportanceHighAccess = memories.filter((m) => m.importance >= 7 && m.access_count >= 5).length;
    if (highImportanceHighAccess > memories.length * 0.3) {
      insights.push(`${highImportanceHighAccess} high-value memories are frequently accessed`);
    }

    // Tag usage
    const untaggedPercent = (memories.filter((m) => m.tags.length === 0).length / memories.length) * 100;
    if (untaggedPercent > 20) {
      insights.push(`${untaggedPercent.toFixed(0)}% of memories are untagged - consider adding tags for better organization`);
    }

    // Type distribution anomaly
    const typeCount = memories.reduce((acc, m) => {
      acc[m.type] = (acc[m.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    const dominantType = Object.entries(typeCount).sort(([, a], [, b]) => b - a)[0];
    if (dominantType && dominantType[1] > memories.length * 0.5) {
      insights.push(`${dominantType[0]} memories dominate (${((dominantType[1] / memories.length) * 100).toFixed(0)}%)`);
    }

    // Retention rate
    const activeMemories = memories.filter((m) => daysSinceAccess(m) <= 30).length;
    const retentionRate = (activeMemories / memories.length) * 100;

    // Average lifespan
    const avgLifespan = memories.reduce((sum, m) => sum + daysSinceCreation(m), 0) / memories.length;

    return {
      timeline,
      typeEvolution,
      importanceVsAccess,
      topTags,
      engagementTrend,
      lifecycle,
      insights,
      memoryVelocity,
      retentionRate,
      avgLifespan,
    };
  }, [memories, timeRange]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <BarChart3 className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
            <p className="text-gray-600">Advanced insights, patterns, and trends</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* Time range selector */}
          <div className="flex items-center bg-white rounded-lg border border-gray-200 p-1">
            {[7, 30, 90].map((days) => (
              <button
                key={days}
                onClick={() => setTimeRange(days as any)}
                className={clsx(
                  'px-3 py-1.5 rounded text-sm font-medium transition-all',
                  timeRange === days
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                )}
              >
                {days}d
              </button>
            ))}
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
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="card text-center py-12">
          <RefreshCw className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading analytics...</p>
        </div>
      )}

      {!isLoading && memories.length === 0 && (
        <div className="card text-center py-12">
          <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-700 mb-2">No Data Yet</h2>
          <p className="text-gray-500">Create some memories to see analytics</p>
        </div>
      )}

      {!isLoading && memories.length > 0 && (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Memory Velocity</p>
                  <p className="text-2xl font-bold text-gray-900">{analytics.memoryVelocity.toFixed(1)}</p>
                  <p className="text-xs text-gray-500">memories/week</p>
                </div>
                <Zap className="w-10 h-10 text-yellow-500 opacity-20" />
              </div>
            </div>

            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Retention Rate</p>
                  <p className="text-2xl font-bold text-gray-900">{analytics.retentionRate.toFixed(0)}%</p>
                  <p className="text-xs text-gray-500">active in 30 days</p>
                </div>
                <Target className="w-10 h-10 text-green-500 opacity-20" />
              </div>
            </div>

            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Avg Lifespan</p>
                  <p className="text-2xl font-bold text-gray-900">{analytics.avgLifespan.toFixed(0)}</p>
                  <p className="text-xs text-gray-500">days</p>
                </div>
                <Clock className="w-10 h-10 text-blue-500 opacity-20" />
              </div>
            </div>

            <div className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Lifecycle Status</p>
                  <div className="flex items-center space-x-2 text-sm">
                    <span className="text-green-600 font-semibold">{analytics.lifecycle.new}</span>
                    <span className="text-gray-400">/</span>
                    <span className="text-blue-600 font-semibold">{analytics.lifecycle.active}</span>
                    <span className="text-gray-400">/</span>
                    <span className="text-orange-600 font-semibold">{analytics.lifecycle.stale}</span>
                  </div>
                  <p className="text-xs text-gray-500">new / active / stale</p>
                </div>
                <Activity className="w-10 h-10 text-purple-500 opacity-20" />
              </div>
            </div>
          </div>

          {/* Insights */}
          {analytics.insights.length > 0 && (
            <div className="card bg-gradient-to-r from-blue-50 to-purple-50">
              <div className="flex items-start space-x-3">
                <Lightbulb className="w-6 h-6 text-yellow-500 flex-shrink-0 mt-1" />
                <div className="flex-1">
                  <h2 className="text-lg font-bold text-gray-900 mb-3">Key Insights</h2>
                  <div className="space-y-2">
                    {analytics.insights.map((insight, idx) => (
                      <div key={idx} className="flex items-start space-x-2">
                        <AlertTriangle className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <p className="text-sm text-gray-700">{insight}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Memory Creation Timeline */}
            <div className="card">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Memory Creation Timeline</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analytics.timeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#3B82F6" strokeWidth={2} dot={{ fill: '#3B82F6' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Engagement Trend */}
            <div className="card">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Engagement Trend</h2>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={analytics.engagementTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Area
                    type="monotone"
                    dataKey="totalAccess"
                    stroke="#10B981"
                    fill="#10B981"
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Type Evolution */}
            <div className="card lg:col-span-2">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Memory Type Evolution</h2>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={analytics.typeEvolution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="fact" stackId="1" stroke={COLORS[0]} fill={COLORS[0]} />
                  <Area type="monotone" dataKey="preference" stackId="1" stroke={COLORS[1]} fill={COLORS[1]} />
                  <Area type="monotone" dataKey="goal" stackId="1" stroke={COLORS[2]} fill={COLORS[2]} />
                  <Area type="monotone" dataKey="habit" stackId="1" stroke={COLORS[3]} fill={COLORS[3]} />
                  <Area type="monotone" dataKey="event" stackId="1" stroke={COLORS[4]} fill={COLORS[4]} />
                  <Area type="monotone" dataKey="context" stackId="1" stroke={COLORS[5]} fill={COLORS[5]} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Importance vs Access Correlation */}
            <div className="card">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Importance vs Access Pattern</h2>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="importance" name="Importance" label={{ value: 'Importance', position: 'bottom' }} />
                  <YAxis type="number" dataKey="accessCount" name="Access Count" label={{ value: 'Access Count', angle: -90, position: 'left' }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter name="Memories" data={analytics.importanceVsAccess} fill="#8B5CF6" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Top Performing Tags */}
            <div className="card">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Top Performing Tags</h2>
              {analytics.topTags.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analytics.topTags} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                    <Tooltip content={({ payload }) => {
                      if (payload && payload[0]) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
                            <p className="font-semibold">{data.name}</p>
                            <p className="text-sm text-gray-600">Count: {data.count}</p>
                            <p className="text-sm text-gray-600">Avg Access: {data.avgAccess.toFixed(1)}</p>
                            <p className="text-sm text-gray-600">Avg Importance: {data.avgImportance.toFixed(1)}</p>
                          </div>
                        );
                      }
                      return null;
                    }} />
                    <Bar dataKey="score" fill="#F59E0B" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-12 text-gray-400">
                  <p>No tags found</p>
                </div>
              )}
            </div>
          </div>

          {/* Memory Lifecycle Breakdown */}
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Memory Lifecycle Breakdown</h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-green-900">New Memories</h3>
                  <Calendar className="w-5 h-5 text-green-600" />
                </div>
                <p className="text-3xl font-bold text-green-600">{analytics.lifecycle.new}</p>
                <p className="text-sm text-green-700 mt-1">Created in last 7 days</p>
              </div>

              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-blue-900">Active Memories</h3>
                  <TrendingUp className="w-5 h-5 text-blue-600" />
                </div>
                <p className="text-3xl font-bold text-blue-600">{analytics.lifecycle.active}</p>
                <p className="text-sm text-blue-700 mt-1">Accessed in last 7 days</p>
              </div>

              <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-orange-900">Stale Memories</h3>
                  <AlertTriangle className="w-5 h-5 text-orange-600" />
                </div>
                <p className="text-3xl font-bold text-orange-600">{analytics.lifecycle.stale}</p>
                <p className="text-sm text-orange-700 mt-1">Not accessed in 30+ days</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
