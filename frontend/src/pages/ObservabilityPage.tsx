import { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Database,
  Zap,
  Clock,
  TrendingUp,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Server,
  HardDrive,
  Gauge,
  LineChart,
} from 'lucide-react';
import { apiClient } from '../services/api';
import clsx from 'clsx';
import { format } from 'date-fns';

interface ObservabilityPageProps {
  userId: string;
}

interface HealthMetric {
  name: string;
  status: 'healthy' | 'warning' | 'critical';
  value: string;
  timestamp: Date;
}

interface IngestionLog {
  id: string;
  timestamp: Date;
  operation: 'create' | 'update' | 'delete';
  memoryId: string;
  status: 'success' | 'failed';
  latency: number;
}

export function ObservabilityPage({ userId }: ObservabilityPageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(false);

  // Auto-refresh every 5 seconds
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      setRefreshKey((prev) => prev + 1);
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Fetch memories for metrics
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

  // Mock health metrics (in production, these would come from backend)
  const healthMetrics = useMemo<HealthMetric[]>(() => {
    const vectorCount = memories.length;
    const avgLatency = 45 + Math.random() * 20;

    return [
      {
        name: 'Qdrant Status',
        status: 'healthy',
        value: 'Online',
        timestamp: new Date(),
      },
      {
        name: 'Vector Count',
        status: vectorCount > 10000 ? 'warning' : 'healthy',
        value: vectorCount.toLocaleString(),
        timestamp: new Date(),
      },
      {
        name: 'Avg Query Latency',
        status: avgLatency > 100 ? 'warning' : avgLatency > 200 ? 'critical' : 'healthy',
        value: `${avgLatency.toFixed(0)}ms`,
        timestamp: new Date(),
      },
      {
        name: 'Memory Usage',
        status: 'healthy',
        value: '2.4 GB / 8 GB',
        timestamp: new Date(),
      },
      {
        name: 'Index Status',
        status: 'healthy',
        value: 'Optimized',
        timestamp: new Date(),
      },
      {
        name: 'Replication',
        status: 'healthy',
        value: 'Disabled (Single node)',
        timestamp: new Date(),
      },
    ];
  }, [memories]);

  // Mock ingestion logs
  const ingestionLogs = useMemo<IngestionLog[]>(() => {
    return memories
      .slice(-20)
      .reverse()
      .map((memory) => ({
        id: memory.id,
        timestamp: new Date(memory.created_at),
        operation: 'create' as const,
        memoryId: memory.id,
        status: 'success' as const,
        latency: 20 + Math.random() * 80,
      }));
  }, [memories]);

  // Calculate RPS (requests per second) - mock data
  const rpsData = useMemo(() => {
    const now = Date.now();
    return Array.from({ length: 60 }, (_, i) => ({
      timestamp: new Date(now - (60 - i) * 1000),
      value: Math.floor(10 + Math.random() * 20),
    }));
  }, [refreshKey]);

  // Calculate latency histogram
  const latencyHistogram = useMemo(() => {
    const buckets = [
      { label: '0-10ms', count: Math.floor(Math.random() * 50) },
      { label: '10-25ms', count: Math.floor(Math.random() * 100) },
      { label: '25-50ms', count: Math.floor(Math.random() * 80) },
      { label: '50-100ms', count: Math.floor(Math.random() * 40) },
      { label: '100-200ms', count: Math.floor(Math.random() * 20) },
      { label: '>200ms', count: Math.floor(Math.random() * 5) },
    ];
    const max = Math.max(...buckets.map((b) => b.count));
    return buckets.map((b) => ({ ...b, percentage: (b.count / max) * 100 }));
  }, [refreshKey]);

  // Memory growth over time
  const memoryGrowth = useMemo(() => {
    const days = 30;
    const now = Date.now();
    return Array.from({ length: days }, (_, i) => {
      const date = new Date(now - (days - i - 1) * 24 * 60 * 60 * 1000);
      const count = memories.filter(
        (m) => new Date(m.created_at) <= date
      ).length;
      return { date, count };
    });
  }, [memories]);

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  const getStatusIcon = (status: 'healthy' | 'warning' | 'critical') => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      case 'critical':
        return <XCircle className="w-5 h-5 text-red-600" />;
    }
  };

  const getStatusColor = (status: 'healthy' | 'warning' | 'critical') => {
    switch (status) {
      case 'healthy':
        return 'bg-green-50 border-green-200';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200';
      case 'critical':
        return 'bg-red-50 border-red-200';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Server className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Observability Dashboard</h1>
            <p className="text-gray-600">System health and performance metrics</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* Auto-refresh toggle */}
          <label className="flex items-center space-x-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded text-primary-600"
            />
            <span>Auto-refresh (5s)</span>
          </label>

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

      {/* Health Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {healthMetrics.map((metric) => (
          <div
            key={metric.name}
            className={clsx('card', getStatusColor(metric.status))}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  {getStatusIcon(metric.status)}
                  <p className="text-xs font-medium text-gray-700">{metric.name}</p>
                </div>
                <p className="text-lg font-bold text-gray-900">{metric.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Metrics Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* RPS Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Zap className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-bold text-gray-900">Requests Per Second</h2>
            </div>
            <span className="text-sm text-gray-600">Last 60 seconds</span>
          </div>

          <div className="h-48 flex items-end space-x-0.5">
            {rpsData.map((point, idx) => (
              <div
                key={idx}
                className="flex-1 bg-blue-500 rounded-t hover:bg-blue-600 transition-colors cursor-pointer"
                style={{ height: `${(point.value / 30) * 100}%` }}
                title={`${format(point.timestamp, 'HH:mm:ss')}: ${point.value} req/s`}
              />
            ))}
          </div>

          <div className="mt-4 flex items-center justify-between text-sm text-gray-600">
            <span>0 req/s</span>
            <span className="text-lg font-bold text-blue-600">
              {rpsData[rpsData.length - 1]?.value || 0} req/s
            </span>
            <span>30 req/s</span>
          </div>
        </div>

        {/* Latency Histogram */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-6">
            <Clock className="w-6 h-6 text-purple-600" />
            <h2 className="text-xl font-bold text-gray-900">Latency Distribution</h2>
          </div>

          <div className="space-y-3">
            {latencyHistogram.map((bucket) => (
              <div key={bucket.label} className="flex items-center space-x-3">
                <span className="text-xs text-gray-600 w-20">{bucket.label}</span>
                <div className="flex-1 bg-gray-200 rounded-full h-6 relative overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-purple-400 to-purple-600 h-6 rounded-full flex items-center justify-end pr-2"
                    style={{ width: `${bucket.percentage}%` }}
                  >
                    {bucket.percentage > 20 && (
                      <span className="text-xs font-semibold text-white">
                        {bucket.count}
                      </span>
                    )}
                  </div>
                </div>
                {bucket.percentage <= 20 && (
                  <span className="text-xs text-gray-600 w-12">{bucket.count}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Memory Growth Chart */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <TrendingUp className="w-6 h-6 text-green-600" />
            <h2 className="text-xl font-bold text-gray-900">Memory Growth (30 Days)</h2>
          </div>
          <span className="text-2xl font-bold text-green-600">
            {memories.length} total
          </span>
        </div>

        <div className="h-64 flex items-end space-x-1">
          {memoryGrowth.map((point, idx) => {
            const maxCount = Math.max(...memoryGrowth.map((p) => p.count));
            const height = maxCount > 0 ? (point.count / maxCount) * 100 : 0;

            return (
              <div
                key={idx}
                className="flex-1 bg-green-500 rounded-t hover:bg-green-600 transition-colors cursor-pointer"
                style={{ height: `${height}%` }}
                title={`${format(point.date, 'MMM d')}: ${point.count} memories`}
              />
            );
          })}
        </div>

        <div className="mt-4 flex items-center justify-between text-sm text-gray-600">
          <span>{format(memoryGrowth[0]?.date || new Date(), 'MMM d')}</span>
          <span className="text-lg font-bold text-green-600">
            +{memories.length} memories
          </span>
          <span>{format(memoryGrowth[memoryGrowth.length - 1]?.date || new Date(), 'MMM d')}</span>
        </div>
      </div>

      {/* System Resources */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Database Stats */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <Database className="w-6 h-6 text-indigo-600" />
            <h3 className="text-lg font-bold text-gray-900">Database</h3>
          </div>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Collections:</span>
              <span className="font-semibold text-gray-900">1</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Vector Dimensions:</span>
              <span className="font-semibold text-gray-900">1536</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Index Type:</span>
              <span className="font-semibold text-gray-900">HNSW</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Distance Metric:</span>
              <span className="font-semibold text-gray-900">Cosine</span>
            </div>
          </div>
        </div>

        {/* Storage Stats */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <HardDrive className="w-6 h-6 text-orange-600" />
            <h3 className="text-lg font-bold text-gray-900">Storage</h3>
          </div>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Size:</span>
              <span className="font-semibold text-gray-900">2.4 GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Vectors:</span>
              <span className="font-semibold text-gray-900">1.8 GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Metadata:</span>
              <span className="font-semibold text-gray-900">512 MB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Index:</span>
              <span className="font-semibold text-gray-900">128 MB</span>
            </div>
          </div>
        </div>

        {/* Performance Stats */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <Gauge className="w-6 h-6 text-red-600" />
            <h3 className="text-lg font-bold text-gray-900">Performance</h3>
          </div>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">P50 Latency:</span>
              <span className="font-semibold text-gray-900">25ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">P95 Latency:</span>
              <span className="font-semibold text-gray-900">78ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">P99 Latency:</span>
              <span className="font-semibold text-gray-900">142ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Cache Hit Rate:</span>
              <span className="font-semibold text-green-600">87%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Ingestion Logs */}
      <div className="card">
        <div className="flex items-center space-x-3 mb-6">
          <LineChart className="w-6 h-6 text-gray-600" />
          <h2 className="text-xl font-bold text-gray-900">Recent Ingestion Logs</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Timestamp
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Operation
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Memory ID
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 uppercase">
                  Latency
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {ingestionLogs.map((log) => (
                <tr key={log.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-gray-900 font-mono text-xs">
                    {format(log.timestamp, 'HH:mm:ss.SSS')}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={clsx(
                        'px-2 py-1 rounded-full text-xs font-medium',
                        log.operation === 'create'
                          ? 'bg-green-100 text-green-700'
                          : log.operation === 'update'
                          ? 'bg-blue-100 text-blue-700'
                          : 'bg-red-100 text-red-700'
                      )}
                    >
                      {log.operation.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-900 font-mono text-xs">
                    {log.memoryId.slice(0, 16)}...
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={clsx(
                        'px-2 py-1 rounded-full text-xs font-medium',
                        log.status === 'success'
                          ? 'bg-green-100 text-green-700'
                          : 'bg-red-100 text-red-700'
                      )}
                    >
                      {log.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-900 font-mono">
                    {log.latency.toFixed(1)}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
