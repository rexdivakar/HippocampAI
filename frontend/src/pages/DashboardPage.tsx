import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import {
  Brain,
  Sparkles,
  Moon,
  Network,
  Tag,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Plus,
  Play,
  ArrowRight,
  Activity,
  Clock,
  RefreshCw,
} from 'lucide-react';
import clsx from 'clsx';

interface DashboardPageProps {
  userId: string;
}

export function DashboardPage({ userId }: DashboardPageProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Fetch dashboard stats with error handling
  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError,
    refetch: refetchStats,
  } = useQuery({
    queryKey: ['dashboard-stats', userId],
    queryFn: async () => {
      const res = await fetch(`/api/dashboard/stats?user_id=${userId}`);
      if (!res.ok) throw new Error('Stats API not available');
      return res.json();
    },
    retry: 2,
    staleTime: 30000, // 30 seconds
  });

  // Fetch sleep phase status with error handling
  const {
    data: sleepStatus,
    isLoading: sleepLoading,
    error: sleepError,
  } = useQuery({
    queryKey: ['consolidation-status', userId],
    queryFn: async () => {
      const res = await fetch(`/api/consolidation/status?user_id=${userId}`);
      if (!res.ok) throw new Error('Status API not available');
      return res.json();
    },
    retry: 2,
  });

  // Fetch recent activity with error handling
  const {
    data: recentActivity,
    isLoading: activityLoading,
    error: activityError,
  } = useQuery({
    queryKey: ['recent-activity', userId],
    queryFn: async () => {
      const res = await fetch(`/api/dashboard/recent-activity?user_id=${userId}`);
      if (!res.ok) throw new Error('Activity API not available');
      const data = await res.json();
      return data?.activities || [];
    },
    retry: 2,
  });

  // Trigger manual consolidation
  const triggerMutation = useMutation({
    mutationFn: async (dryRun: boolean) => {
      const res = await fetch(`/api/consolidation/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dry_run: dryRun, lookback_hours: 24 }),
      });
      if (!res.ok) throw new Error('Failed to trigger consolidation');
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats', userId] });
      queryClient.invalidateQueries({ queryKey: ['consolidation-status', userId] });
    },
  });

  const isLoading = statsLoading || sleepLoading || activityLoading;
  const hasError = statsError || sleepError || activityError;

  // Safe defaults
  const safeStats = stats || {
    total_memories: 0,
    total_entities: 0,
    total_concepts: 0,
    total_tags: 0,
    total_sleep_runs: 0,
    health_score: 0,
    top_tags: [],
    recent_memories: [],
    top_clusters: [],
    total_connections: 0,
    potential_duplicates: 0,
    uncategorized_memories: 0,
    archived_memories: 0,
  };

  const safeSleepStatus = sleepStatus || {
    enabled: false,
    dry_run: false,
    schedule_hour: 3,
  };

  const safeActivity = recentActivity || [];

  return (
    <div className="w-full min-h-screen">
      {/* Content Container */}
      <div className="w-full">
        {/* Page Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-3">
            <Brain className="w-8 h-8 text-primary-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
              <p className="text-sm text-gray-500">Overview of your memories, insights, and consolidation</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => refetchStats()}
              disabled={isLoading}
              className="px-3 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 flex items-center space-x-2"
              title="Refresh"
            >
              <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
            <button
              onClick={() => navigate('/memories')}
              className="px-4 py-2 bg-primary-600 text-white hover:bg-primary-700 rounded-lg text-sm font-medium transition-colors flex items-center space-x-2"
            >
              <Plus className="w-4 h-4" />
              <span>New Memory</span>
            </button>
          </div>
        </div>

        {/* Error State */}
        {hasError && (
          <div className="mb-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-sm font-medium text-yellow-900">Unable to load some data</h3>
                <p className="text-sm text-yellow-700 mt-1">
                  Some dashboard data couldn't be loaded. Showing available information.
                </p>
              </div>
              <button
                onClick={() => refetchStats()}
                className="text-sm text-yellow-700 hover:text-yellow-900 font-medium"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && !stats && (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="w-8 h-8 text-primary-600 animate-spin" />
          </div>
        )}

        {/* Content */}
        {!isLoading || stats ? (
          <div className="space-y-6">
            {/* Global KPIs Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
              <StatCard
                icon={<Brain className="w-5 h-5 text-indigo-600" />}
                label="Total Memories"
                value={safeStats.total_memories}
                subtitle="all time"
                color="indigo"
              />
              <StatCard
                icon={<Network className="w-5 h-5 text-blue-600" />}
                label="Entities"
                value={safeStats.total_entities}
                subtitle="people, places, things"
                color="blue"
              />
              <StatCard
                icon={<Sparkles className="w-5 h-5 text-purple-600" />}
                label="Concepts"
                value={safeStats.total_concepts}
                subtitle="ideas extracted"
                color="purple"
              />
              <StatCard
                icon={<Tag className="w-5 h-5 text-green-600" />}
                label="Tags"
                value={safeStats.total_tags}
                subtitle="unique labels"
                color="green"
              />
              <StatCard
                icon={<Moon className="w-5 h-5 text-indigo-600" />}
                label="Sleep Cycles"
                value={safeStats.total_sleep_runs}
                subtitle="completed"
                color="indigo"
              />
              <StatCard
                icon={<Activity className="w-5 h-5 text-orange-600" />}
                label="Memory Health"
                value={safeStats.health_score}
                subtitle="out of 100"
                color="orange"
              />
            </div>

            {/* Memory Activity & Trends */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                <TrendingUp className="w-5 h-5 text-indigo-600" />
                <span>Memory Activity</span>
              </h3>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Chart Placeholder */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Memories Created (Last 7 Days)</h4>
                  <div className="h-48 bg-gray-50 rounded-lg border border-gray-200 flex items-center justify-center">
                    <span className="text-sm text-gray-500">Chart visualization coming soon</span>
                  </div>
                </div>

                {/* Top Topics & Recent Memories */}
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-3">Most Active Topics</h4>
                    <div className="space-y-2">
                      {safeStats.top_tags && safeStats.top_tags.length > 0 ? (
                        safeStats.top_tags.slice(0, 5).map((tag: any) => (
                          <div key={tag?.name || Math.random()} className="flex items-center justify-between text-sm">
                            <span className="text-gray-700">#{tag?.name || 'Unknown'}</span>
                            <span className="text-gray-500 font-medium">{tag?.count || 0}</span>
                          </div>
                        ))
                      ) : (
                        <p className="text-sm text-gray-500">No tags yet</p>
                      )}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-3">Recently Added</h4>
                    <div className="space-y-2">
                      {safeStats.recent_memories && safeStats.recent_memories.length > 0 ? (
                        safeStats.recent_memories.slice(0, 3).map((memory: any) => (
                          <div key={memory?.id || Math.random()} className="text-sm">
                            <p className="text-gray-700 truncate">{memory?.text || 'No content'}</p>
                            <p className="text-gray-500 text-xs capitalize">{memory?.type || 'unknown'}</p>
                          </div>
                        ))
                      ) : (
                        <p className="text-sm text-gray-500">No recent memories</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Sleep Phase Snapshot */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <Moon className="w-5 h-5 text-indigo-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Sleep Phase</h3>
                </div>
                <div
                  className={clsx(
                    'px-3 py-1 rounded-full text-xs font-medium',
                    safeSleepStatus.enabled
                      ? 'bg-green-100 text-green-700 border border-green-200'
                      : 'bg-gray-100 text-gray-600 border border-gray-200'
                  )}
                >
                  {safeSleepStatus.enabled ? 'Active' : 'Inactive'}
                </div>
              </div>

              {safeStats.last_sleep_run && (
                <p className="text-sm text-gray-600 mb-4">
                  Last run: {new Date(safeStats.last_sleep_run).toLocaleString()}
                </p>
              )}

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                <MiniStat label="Reviewed" value={safeStats.last_sleep_reviewed} />
                <MiniStat label="Promoted" value={safeStats.last_sleep_promoted} />
                <MiniStat label="Archived" value={safeStats.last_sleep_archived} />
                <MiniStat label="Synthesized" value={safeStats.last_sleep_synthesized} />
              </div>

              <div className="flex items-center space-x-2 pt-4 border-t border-gray-200">
                <button
                  onClick={() => triggerMutation.mutate(true)}
                  disabled={triggerMutation.isPending}
                  className="px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
                >
                  Dry Run
                </button>
                <button
                  onClick={() => triggerMutation.mutate(false)}
                  disabled={triggerMutation.isPending}
                  className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors flex items-center space-x-1 disabled:opacity-50"
                >
                  <Play className="w-4 h-4" />
                  <span>Run Now</span>
                </button>
                <button
                  onClick={() => navigate('/sleep-phase')}
                  className="ml-auto px-3 py-2 text-sm font-medium text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 rounded-lg transition-colors flex items-center space-x-1"
                >
                  <span>View Details</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Clusters & Knowledge Graph Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Clusters Snapshot */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                  <Sparkles className="w-5 h-5 text-purple-600" />
                  <span>Memory Islands</span>
                </h3>

                <div className="space-y-3 mb-4">
                  {safeStats.top_clusters && safeStats.top_clusters.length > 0 ? (
                    safeStats.top_clusters.slice(0, 3).map((cluster: any, idx: number) => (
                      <div
                        key={cluster?.id || idx}
                        className="flex items-center justify-between p-3 bg-purple-50 border border-purple-200 rounded-lg"
                      >
                        <div>
                          <p className="text-sm font-medium text-gray-900">{cluster?.label || `Cluster ${idx + 1}`}</p>
                          <p className="text-xs text-gray-600">{cluster?.size || 0} memories</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-semibold text-purple-700">
                            {cluster?.coherence?.toFixed(1) || 'N/A'}
                          </p>
                          <p className="text-xs text-gray-500">coherence</p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 py-4">No clusters available yet</p>
                  )}
                </div>

                <button
                  onClick={() => navigate('/cluster')}
                  className="w-full px-3 py-2 text-sm font-medium text-purple-600 hover:text-purple-700 hover:bg-purple-50 rounded-lg transition-colors flex items-center justify-center space-x-1"
                >
                  <span>View All Clusters</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>

              {/* Knowledge Graph Snapshot */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                  <Network className="w-5 h-5 text-blue-600" />
                  <span>Knowledge Graph</span>
                </h3>

                <div className="space-y-3 mb-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">Entities</span>
                    <span className="text-sm font-semibold text-gray-900">{safeStats.total_entities || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">Concepts</span>
                    <span className="text-sm font-semibold text-gray-900">{safeStats.total_concepts || 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">Connections</span>
                    <span className="text-sm font-semibold text-gray-900">{safeStats.total_connections || 0}</span>
                  </div>
                </div>

                <div className="h-32 bg-blue-50 rounded-lg border border-blue-200 flex items-center justify-center mb-4">
                  <span className="text-sm text-blue-600">Graph visualization coming soon</span>
                </div>

                <button
                  onClick={() => navigate('/graph')}
                  className="w-full px-3 py-2 text-sm font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors flex items-center justify-center space-x-1"
                >
                  <span>Open Knowledge Graph</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Quality & Hygiene Section */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                <AlertCircle className="w-5 h-5 text-orange-600" />
                <span>Memory Hygiene</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <HygieneCard
                  label="Potential Duplicates"
                  value={safeStats.potential_duplicates}
                  status={safeStats.potential_duplicates > 0 ? 'warning' : 'ok'}
                />
                <HygieneCard
                  label="Uncategorized"
                  value={safeStats.uncategorized_memories}
                  status={safeStats.uncategorized_memories > 10 ? 'warning' : 'ok'}
                />
                <HygieneCard
                  label="Archived"
                  value={safeStats.archived_memories}
                  status="ok"
                />
              </div>

              <button
                onClick={() => navigate('/memories')}
                className="mt-4 px-3 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
              >
                View Details
              </button>
            </div>

            {/* Recent Activity Feed */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                <Clock className="w-5 h-5 text-gray-600" />
                <span>Recent Activity</span>
              </h3>

              <div className="space-y-3">
                {safeActivity.length === 0 ? (
                  <p className="text-sm text-gray-500">No recent activity</p>
                ) : (
                  safeActivity.slice(0, 10).map((activity: any, idx: number) => (
                    <div
                      key={activity?.id || idx}
                      className="flex items-start space-x-3 py-2 border-b border-gray-100 last:border-0"
                    >
                      <div className="flex-shrink-0 mt-0.5">
                        {activity?.type === 'memory_created' && <Brain className="w-4 h-4 text-indigo-600" />}
                        {activity?.type === 'sleep_cycle' && <Moon className="w-4 h-4 text-purple-600" />}
                        {activity?.type === 'archived' && <AlertCircle className="w-4 h-4 text-orange-600" />}
                        {!['memory_created', 'sleep_cycle', 'archived'].includes(activity?.type) && (
                          <Activity className="w-4 h-4 text-gray-600" />
                        )}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-gray-900">{activity?.description || 'Activity'}</p>
                        <p className="text-xs text-gray-500">
                          {activity?.timestamp ? new Date(activity.timestamp).toLocaleString() : 'Unknown time'}
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ============================================
// HELPER COMPONENTS
// ============================================

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: number | undefined | null;
  subtitle: string;
  color: 'indigo' | 'blue' | 'purple' | 'green' | 'orange';
}

function StatCard({ icon, label, value, subtitle, color }: StatCardProps) {
  const colorClasses = {
    indigo: 'bg-indigo-50 border-indigo-200',
    blue: 'bg-blue-50 border-blue-200',
    purple: 'bg-purple-50 border-purple-200',
    green: 'bg-green-50 border-green-200',
    orange: 'bg-orange-50 border-orange-200',
  };

  // Safe value handling
  const safeValue = typeof value === 'number' && !isNaN(value) ? value : 0;

  return (
    <div className={clsx('bg-white rounded-xl shadow-sm border p-4', colorClasses[color])}>
      <div className="flex items-center space-x-2 mb-2">
        <div className="p-1.5 rounded-lg">{icon}</div>
        <div className="text-xs font-medium text-gray-600">{label}</div>
      </div>
      <div className="text-2xl font-bold text-gray-900">{safeValue.toLocaleString()}</div>
      <div className="text-xs text-gray-500 mt-1">{subtitle}</div>
    </div>
  );
}

interface MiniStatProps {
  label: string;
  value: number | undefined | null;
}

function MiniStat({ label, value }: MiniStatProps) {
  const safeValue = typeof value === 'number' && !isNaN(value) ? value : 0;

  return (
    <div className="text-center">
      <div className="text-xl font-bold text-gray-900">{safeValue}</div>
      <div className="text-xs text-gray-500 mt-1">{label}</div>
    </div>
  );
}

interface HygieneCardProps {
  label: string;
  value: number | undefined | null;
  status: 'ok' | 'warning';
}

function HygieneCard({ label, value, status }: HygieneCardProps) {
  const safeValue = typeof value === 'number' && !isNaN(value) ? value : 0;

  return (
    <div
      className={clsx(
        'p-4 rounded-lg border',
        status === 'ok' ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        {status === 'ok' ? (
          <CheckCircle className="w-4 h-4 text-green-600" />
        ) : (
          <AlertCircle className="w-4 h-4 text-yellow-600" />
        )}
      </div>
      <div className="text-2xl font-bold text-gray-900">{safeValue}</div>
      <div className={clsx('text-xs mt-1', status === 'ok' ? 'text-green-600' : 'text-yellow-600')}>
        {status === 'ok' ? 'All good' : 'Needs attention'}
      </div>
    </div>
  );
}
