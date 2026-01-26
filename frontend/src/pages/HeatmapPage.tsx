import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  TrendingUp,
  RefreshCw,
  Eye,
  Brain,
  Zap,
  BarChart3,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type { Memory } from '../types';
import clsx from 'clsx';
import { format, subDays, eachDayOfInterval, startOfWeek } from 'date-fns';

interface HeatmapPageProps {
  userId: string;
}

interface DayData {
  date: Date;
  created: number;
  retrieved: number;
  intensity: number;
  avgImportance: number;
}

export function HeatmapPage({ userId }: HeatmapPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [timeRange, setTimeRange] = useState<'week' | 'month' | 'year'>('month');

  // Fetch memories
  const { data: memories = [], isLoading } = useQuery({
    queryKey: ['memories', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        filters: {
          session_id: userId, // Pass userId as session_id to match by either field
        },
        limit: 10000,
      });
      return result;
    },
  });

  // Calculate date range
  const days = timeRange === 'week' ? 7 : timeRange === 'month' ? 30 : 365;
  const dateRange = useMemo(() => {
    const end = new Date();
    const start = subDays(end, days - 1);
    return eachDayOfInterval({ start, end });
  }, [days]);

  // Build heatmap data
  const heatmapData = useMemo<DayData[]>(() => {
    const dataMap = new Map<string, DayData>();

    // Initialize all days with zero values
    dateRange.forEach((date) => {
      const key = format(date, 'yyyy-MM-dd');
      dataMap.set(key, {
        date,
        created: 0,
        retrieved: 0,
        intensity: 0,
        avgImportance: 0,
      });
    });

    // Populate with actual data
    memories.forEach((memory: Memory) => {
      // Count created memories
      const createdKey = format(new Date(memory.created_at), 'yyyy-MM-dd');
      if (dataMap.has(createdKey)) {
        const day = dataMap.get(createdKey)!;
        day.created += 1;
        day.avgImportance =
          (day.avgImportance * (day.created - 1) + memory.importance) / day.created;
      }

      // Count retrieved memories (using access_count as proxy)
      if (memory.last_accessed_at) {
        const retrievedKey = format(new Date(memory.last_accessed_at), 'yyyy-MM-dd');
        if (dataMap.has(retrievedKey)) {
          const day = dataMap.get(retrievedKey)!;
          day.retrieved += memory.access_count;
          day.intensity += memory.confidence * memory.access_count;
        }
      }
    });

    return Array.from(dataMap.values());
  }, [memories, dateRange]);

  // Calculate stats
  const stats = useMemo(() => {
    const totalCreated = heatmapData.reduce((sum, day) => sum + day.created, 0);
    const totalRetrieved = heatmapData.reduce((sum, day) => sum + day.retrieved, 0);
    const maxIntensity = Math.max(...heatmapData.map((d) => d.intensity));
    const avgImportance =
      heatmapData.reduce((sum, day) => sum + day.avgImportance, 0) / heatmapData.length;

    return { totalCreated, totalRetrieved, maxIntensity, avgImportance };
  }, [heatmapData]);

  // Get color intensity for heatmap cell
  const getHeatmapColor = (value: number, max: number, type: 'created' | 'retrieved') => {
    if (value === 0) return 'bg-gray-100';

    const intensity = Math.min((value / max) * 100, 100);

    if (type === 'created') {
      if (intensity > 75) return 'bg-green-600';
      if (intensity > 50) return 'bg-green-500';
      if (intensity > 25) return 'bg-green-400';
      return 'bg-green-300';
    } else {
      if (intensity > 75) return 'bg-blue-600';
      if (intensity > 50) return 'bg-blue-500';
      if (intensity > 25) return 'bg-blue-400';
      return 'bg-blue-300';
    }
  };

  // Organize days into weeks
  const weeks = useMemo(() => {
    const weekMap = new Map<string, DayData[]>();
    heatmapData.forEach((day) => {
      const weekStart = format(startOfWeek(day.date, { weekStartsOn: 1 }), 'yyyy-MM-dd');
      if (!weekMap.has(weekStart)) {
        weekMap.set(weekStart, []);
      }
      weekMap.get(weekStart)!.push(day);
    });
    return Array.from(weekMap.values());
  }, [heatmapData]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Memory Heatmap</h1>
            <p className="text-gray-600">Activity visualization over time</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* Time range selector */}
          <div className="flex items-center bg-white rounded-lg border border-gray-200 p-1">
            <button
              onClick={() => setTimeRange('week')}
              className={clsx(
                'px-3 py-1.5 rounded text-sm transition-all',
                timeRange === 'week'
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              )}
            >
              Week
            </button>
            <button
              onClick={() => setTimeRange('month')}
              className={clsx(
                'px-3 py-1.5 rounded text-sm transition-all',
                timeRange === 'month'
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              )}
            >
              Month
            </button>
            <button
              onClick={() => setTimeRange('year')}
              className={clsx(
                'px-3 py-1.5 rounded text-sm transition-all',
                timeRange === 'year'
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              )}
            >
              Year
            </button>
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

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card bg-green-50 border border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 mb-1">Memories Created</p>
              <p className="text-2xl font-bold text-green-900">{stats.totalCreated}</p>
            </div>
            <Brain className="w-10 h-10 text-green-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-blue-50 border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 mb-1">Times Retrieved</p>
              <p className="text-2xl font-bold text-blue-900">{stats.totalRetrieved}</p>
            </div>
            <Eye className="w-10 h-10 text-blue-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-purple-50 border border-purple-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-purple-600 mb-1">Max Intensity</p>
              <p className="text-2xl font-bold text-purple-900">
                {stats.maxIntensity.toFixed(0)}
              </p>
            </div>
            <Zap className="w-10 h-10 text-purple-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-orange-50 border border-orange-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-orange-600 mb-1">Avg Importance</p>
              <p className="text-2xl font-bold text-orange-900">
                {stats.avgImportance.toFixed(1)}/10
              </p>
            </div>
            <TrendingUp className="w-10 h-10 text-orange-500 opacity-20" />
          </div>
        </div>
      </div>

      {/* Memories Created Heatmap */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Brain className="w-6 h-6 text-green-600" />
            <h2 className="text-xl font-bold text-gray-900">Memories Created</h2>
          </div>
          <div className="flex items-center space-x-2 text-xs text-gray-600">
            <span>Less</span>
            <div className="flex space-x-1">
              <div className="w-3 h-3 bg-gray-100 rounded"></div>
              <div className="w-3 h-3 bg-green-300 rounded"></div>
              <div className="w-3 h-3 bg-green-400 rounded"></div>
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <div className="w-3 h-3 bg-green-600 rounded"></div>
            </div>
            <span>More</span>
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-12 h-12 text-primary-600 animate-spin" />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <div className="inline-flex flex-col space-y-1">
              {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, idx) => (
                <div key={day} className="flex items-center space-x-1">
                  <span className="text-xs text-gray-500 w-8">{idx % 2 === 0 ? day : ''}</span>
                  <div className="flex space-x-1">
                    {weeks.map((week, weekIdx) => {
                      const dayData = week[idx];
                      if (!dayData) return <div key={weekIdx} className="w-3 h-3" />;

                      const maxCreated = Math.max(...heatmapData.map((d) => d.created));
                      const color = getHeatmapColor(dayData.created, maxCreated, 'created');

                      return (
                        <div
                          key={weekIdx}
                          className={clsx(
                            'w-3 h-3 rounded',
                            color,
                            'hover:ring-2 hover:ring-green-600 cursor-pointer transition-all'
                          )}
                          title={`${format(dayData.date, 'MMM d, yyyy')}: ${dayData.created} created`}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Memories Retrieved Heatmap */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Eye className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-bold text-gray-900">Retrieval Intensity</h2>
          </div>
          <div className="flex items-center space-x-2 text-xs text-gray-600">
            <span>Less</span>
            <div className="flex space-x-1">
              <div className="w-3 h-3 bg-gray-100 rounded"></div>
              <div className="w-3 h-3 bg-blue-300 rounded"></div>
              <div className="w-3 h-3 bg-blue-400 rounded"></div>
              <div className="w-3 h-3 bg-blue-500 rounded"></div>
              <div className="w-3 h-3 bg-blue-600 rounded"></div>
            </div>
            <span>More</span>
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-12 h-12 text-primary-600 animate-spin" />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <div className="inline-flex flex-col space-y-1">
              {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, idx) => (
                <div key={day} className="flex items-center space-x-1">
                  <span className="text-xs text-gray-500 w-8">{idx % 2 === 0 ? day : ''}</span>
                  <div className="flex space-x-1">
                    {weeks.map((week, weekIdx) => {
                      const dayData = week[idx];
                      if (!dayData) return <div key={weekIdx} className="w-3 h-3" />;

                      const maxRetrieved = Math.max(...heatmapData.map((d) => d.retrieved));
                      const color = getHeatmapColor(dayData.retrieved, maxRetrieved, 'retrieved');

                      return (
                        <div
                          key={weekIdx}
                          className={clsx(
                            'w-3 h-3 rounded',
                            color,
                            'hover:ring-2 hover:ring-blue-600 cursor-pointer transition-all'
                          )}
                          title={`${format(dayData.date, 'MMM d, yyyy')}: ${dayData.retrieved} retrievals`}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Importance Distribution */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-6">
          <BarChart3 className="w-6 h-6 text-orange-600" />
          <h2 className="text-xl font-bold text-gray-900">Importance Distribution Over Time</h2>
        </div>

        <div className="space-y-2">
          {heatmapData
            .filter((d) => d.created > 0)
            .slice(-10)
            .map((day) => (
              <div key={day.date.toISOString()} className="flex items-center space-x-3">
                <span className="text-xs text-gray-600 w-20">
                  {format(day.date, 'MMM d')}
                </span>
                <div className="flex-1 bg-gray-200 rounded-full h-6 relative overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-orange-400 to-orange-600 h-6 rounded-full flex items-center justify-end pr-2"
                    style={{ width: `${(day.avgImportance / 10) * 100}%` }}
                  >
                    <span className="text-xs font-semibold text-white">
                      {day.avgImportance.toFixed(1)}
                    </span>
                  </div>
                </div>
                <span className="text-xs text-gray-600 w-16">
                  {day.created} created
                </span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
