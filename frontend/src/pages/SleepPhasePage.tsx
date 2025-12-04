import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Brain,
  Moon,
  Sparkles,
  TrendingUp,
  TrendingDown,
  Archive,
  Trash2,
  Clock,
  Play,
  Settings as SettingsIcon,
  BarChart3,
  Calendar,
  Zap,
} from 'lucide-react';
import { apiClient } from '../services/api';
import clsx from 'clsx';

interface SleepPhasePageProps {
  userId: string;
}

export function SleepPhasePage({ userId }: SleepPhasePageProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'history' | 'settings'>('overview');
  const queryClient = useQueryClient();

  // Fetch consolidation status
  const { data: status } = useQuery({
    queryKey: ['consolidation-status', userId],
    queryFn: async () => {
      const res = await fetch(`/api/consolidation/status?user_id=${userId}`);
      return res.json();
    },
  });

  // Fetch consolidation stats
  const { data: stats } = useQuery({
    queryKey: ['consolidation-stats', userId],
    queryFn: async () => {
      const res = await fetch(`/api/consolidation/stats?user_id=${userId}`);
      return res.json();
    },
  });

  // Fetch consolidation runs
  const { data: runs = [] } = useQuery({
    queryKey: ['consolidation-runs', userId],
    queryFn: async () => {
      const res = await fetch(`/api/consolidation/runs?user_id=${userId}`);
      return res.json();
    },
  });

  // Trigger manual consolidation
  const triggerMutation = useMutation({
    mutationFn: async (dryRun: boolean) => {
      const res = await fetch(`/api/consolidation/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dry_run: dryRun, lookback_hours: 24 }),
      });
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['consolidation-runs', userId] });
      queryClient.invalidateQueries({ queryKey: ['consolidation-stats', userId] });
    },
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-8 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center space-x-4">
            <div className="bg-white/20 p-4 rounded-2xl backdrop-blur-sm">
              <Moon className="w-10 h-10" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Sleep Phase</h1>
              <p className="text-indigo-100 mt-1">
                Automated memory consolidation • Mimicking your brain's hippocampus during sleep
              </p>
            </div>
          </div>

          {/* Status Badge */}
          <div className="mt-6 flex items-center space-x-4">
            <div
              className={clsx(
                'px-4 py-2 rounded-full text-sm font-medium backdrop-blur-sm',
                status?.enabled
                  ? 'bg-green-400/30 text-green-100 border border-green-300/50'
                  : 'bg-gray-400/30 text-gray-100 border border-gray-300/50'
              )}
            >
              {status?.enabled ? '🌙 Active' : '💤 Inactive'}
            </div>

            {status?.dry_run && (
              <div className="px-4 py-2 rounded-full text-sm font-medium bg-yellow-400/30 text-yellow-100 border border-yellow-300/50 backdrop-blur-sm">
                🧪 Dry Run Mode
              </div>
            )}

            <div className="px-4 py-2 rounded-full text-sm font-medium bg-white/20 text-white border border-white/30 backdrop-blur-sm">
              Next run: {status?.enabled ? `Tonight at ${status?.schedule_hour || 3}:00 AM` : 'Disabled'}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="max-w-7xl mx-auto px-6 mt-6">
        <div className="flex space-x-1 bg-white rounded-lg p-1 shadow-sm border border-gray-200">
          <button
            onClick={() => setActiveTab('overview')}
            className={clsx(
              'flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center justify-center space-x-2',
              activeTab === 'overview'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-gray-600 hover:bg-gray-100'
            )}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Overview</span>
          </button>

          <button
            onClick={() => setActiveTab('history')}
            className={clsx(
              'flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center justify-center space-x-2',
              activeTab === 'history'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-gray-600 hover:bg-gray-100'
            )}
          >
            <Calendar className="w-4 h-4" />
            <span>Dream History</span>
          </button>

          <button
            onClick={() => setActiveTab('settings')}
            className={clsx(
              'flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center justify-center space-x-2',
              activeTab === 'settings'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-gray-600 hover:bg-gray-100'
            )}
          >
            <SettingsIcon className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'overview' && (
          <OverviewTab stats={stats} lastRun={runs[0]} onTrigger={(dryRun) => triggerMutation.mutate(dryRun)} />
        )}

        {activeTab === 'history' && <HistoryTab runs={runs} />}

        {activeTab === 'settings' && <SettingsTab status={status} />}
      </div>
    </div>
  );
}

// ============================================
// OVERVIEW TAB
// ============================================

function OverviewTab({ stats, lastRun, onTrigger }: any) {
  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<Brain className="w-6 h-6 text-indigo-600" />}
          label="Total Sleep Cycles"
          value={stats?.total_runs || 0}
          subtitle="completed runs"
          color="indigo"
        />

        <StatCard
          icon={<Sparkles className="w-6 h-6 text-purple-600" />}
          label="Memories Synthesized"
          value={stats?.total_memories_synthesized || 0}
          subtitle="new insights created"
          color="purple"
        />

        <StatCard
          icon={<TrendingUp className="w-6 h-6 text-green-600" />}
          label="Memories Promoted"
          value={stats?.total_memories_promoted || 0}
          subtitle="boosted importance"
          color="green"
        />

        <StatCard
          icon={<Archive className="w-6 h-6 text-orange-600" />}
          label="Memories Archived"
          value={stats?.total_memories_deleted || 0}
          subtitle="low-value cleaned"
          color="orange"
        />
      </div>

      {/* Last Run Card */}
      {lastRun && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                <Moon className="w-5 h-5 text-indigo-600" />
                <span>Last Sleep Cycle</span>
              </h3>
              <p className="text-sm text-gray-500 mt-1">
                {new Date(lastRun.started_at).toLocaleString()}
              </p>
            </div>

            <div
              className={clsx(
                'px-3 py-1 rounded-full text-xs font-medium',
                lastRun.status === 'completed'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-red-100 text-red-700'
              )}
            >
              {lastRun.status}
            </div>
          </div>

          {/* Dream Report */}
          {lastRun.dream_report && (
            <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 mb-4 border border-indigo-100">
              <p className="text-sm text-gray-700 italic">"{lastRun.dream_report}"</p>
            </div>
          )}

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <MiniStat label="Reviewed" value={lastRun.memories_reviewed} icon="📚" />
            <MiniStat label="Promoted" value={lastRun.memories_promoted} icon="⬆️" />
            <MiniStat label="Updated" value={lastRun.memories_updated} icon="✏️" />
            <MiniStat label="Synthesized" value={lastRun.memories_synthesized} icon="🔄" />
            <MiniStat label="Archived" value={lastRun.memories_archived} icon="📦" />
            <MiniStat label="Deleted" value={lastRun.memories_deleted} icon="🗑️" />
          </div>

          <div className="mt-4 pt-4 border-t border-gray-200 flex items-center justify-between text-sm text-gray-500">
            <span>Duration: {lastRun.duration_seconds?.toFixed(1)}s</span>
            <span>{lastRun.clusters_created} clusters • {lastRun.llm_calls_made} LLM calls</span>
          </div>
        </div>
      )}

      {/* Manual Trigger Section */}
      <div className="bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl shadow-lg p-6 text-white">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold flex items-center space-x-2">
              <Zap className="w-5 h-5" />
              <span>Manual Sleep Cycle</span>
            </h3>
            <p className="text-indigo-100 mt-1 text-sm">
              Trigger consolidation on-demand to clean up and organize your memories now
            </p>
          </div>
        </div>

        <div className="mt-4 flex space-x-3">
          <button
            onClick={() => onTrigger(true)}
            className="px-4 py-2 bg-white/20 hover:bg-white/30 backdrop-blur-sm rounded-lg text-sm font-medium transition-colors border border-white/30"
          >
            🧪 Dry Run (Preview Only)
          </button>

          <button
            onClick={() => onTrigger(false)}
            className="px-4 py-2 bg-white text-indigo-600 hover:bg-indigo-50 rounded-lg text-sm font-medium transition-colors shadow-sm"
          >
            <Play className="w-4 h-4 inline mr-2" />
            Run Now (Live)
          </button>
        </div>
      </div>

      {/* How It Works */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
          <Brain className="w-5 h-5 text-indigo-600" />
          <span>How Sleep Phase Works</span>
        </h3>

        <div className="space-y-4">
          <ProcessStep
            number="1"
            title="Review Recent Memories"
            description="Every night at 3 AM, the system reviews all memories from the last 24 hours"
            icon="📚"
          />

          <ProcessStep
            number="2"
            title="Cluster & Analyze"
            description="Related memories are grouped together and analyzed by AI to identify patterns and insights"
            icon="🧠"
          />

          <ProcessStep
            number="3"
            title="Promote Important Facts"
            description="Strategic decisions, key learnings, and important relationships get boosted in importance"
            icon="⬆️"
          />

          <ProcessStep
            number="4"
            title="Archive Low-Value Events"
            description="Transient, routine activities (coffee breaks, routine tasks) are archived or deleted"
            icon="🗑️"
          />

          <ProcessStep
            number="5"
            title="Synthesize Insights"
            description="AI creates higher-level summaries capturing the essence of related events"
            icon="✨"
          />

          <ProcessStep
            number="6"
            title="Generate Dream Report"
            description="A summary of what changed is logged for observability and debugging"
            icon="📊"
          />
        </div>
      </div>
    </div>
  );
}

// ============================================
// HISTORY TAB
// ============================================

function HistoryTab({ runs }: any) {
  const [selectedRun, setSelectedRun] = useState<any>(null);

  if (runs.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
        <Moon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900">No Sleep Cycles Yet</h3>
        <p className="text-gray-500 mt-2">
          Your first consolidation will run tonight, or you can trigger one manually from the Overview tab
        </p>
      </div>
    );
  }

  if (selectedRun) {
    return (
      <div className="space-y-6">
        <button
          onClick={() => setSelectedRun(null)}
          className="text-indigo-600 hover:text-indigo-700 text-sm font-medium"
        >
          ← Back to history
        </button>

        <DreamRunDetail run={selectedRun} />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold text-gray-900">Sleep Cycle History</h2>

      <div className="space-y-3">
        {runs.map((run: any) => (
          <DreamRunCard key={run.run_id} run={run} onClick={() => setSelectedRun(run)} />
        ))}
      </div>
    </div>
  );
}

// ============================================
// SETTINGS TAB
// ============================================

function SettingsTab({ status }: any) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h3>

        <div className="space-y-4">
          <SettingRow
            label="Status"
            value={status?.enabled ? 'Enabled ✅' : 'Disabled ❌'}
            description="Sleep Phase is currently running"
          />

          <SettingRow
            label="Schedule"
            value={`Daily at ${status?.schedule_hour || 3}:00 AM UTC`}
            description="When the consolidation runs automatically"
          />

          <SettingRow
            label="Lookback Period"
            value="24 hours"
            description="How far back to review memories"
          />

          <SettingRow label="Dry Run Mode" value={status?.dry_run ? 'ON' : 'OFF'} description="Preview mode (no changes made)" />
        </div>
      </div>

      <div className="bg-yellow-50 rounded-xl border border-yellow-200 p-6">
        <h4 className="font-semibold text-yellow-900 mb-2">⚙️ Advanced Configuration</h4>
        <p className="text-sm text-yellow-800">
          To modify Sleep Phase settings, update environment variables in your <code className="bg-yellow-100 px-2 py-1 rounded">.env</code> file:
        </p>
        <ul className="mt-3 text-sm text-yellow-800 space-y-1 list-disc list-inside">
          <li><code className="bg-yellow-100 px-2 py-1 rounded text-xs">ACTIVE_CONSOLIDATION_ENABLED</code></li>
          <li><code className="bg-yellow-100 px-2 py-1 rounded text-xs">CONSOLIDATION_SCHEDULE_HOUR</code></li>
          <li><code className="bg-yellow-100 px-2 py-1 rounded text-xs">CONSOLIDATION_MIN_IMPORTANCE</code></li>
          <li><code className="bg-yellow-100 px-2 py-1 rounded text-xs">CONSOLIDATION_DRY_RUN</code></li>
        </ul>
      </div>
    </div>
  );
}

// ============================================
// HELPER COMPONENTS
// ============================================

function StatCard({ icon, label, value, subtitle, color }: any) {
  const colorClasses = {
    indigo: 'from-indigo-500 to-indigo-600',
    purple: 'from-purple-500 to-purple-600',
    green: 'from-green-500 to-green-600',
    orange: 'from-orange-500 to-orange-600',
  };

  return (
    <div className={clsx('bg-gradient-to-br rounded-xl shadow-sm p-6 text-white', colorClasses[color as keyof typeof colorClasses])}>
      <div className="flex items-center space-x-3 mb-3">
        <div className="bg-white/20 p-2 rounded-lg backdrop-blur-sm">{icon}</div>
        <div className="text-sm font-medium opacity-90">{label}</div>
      </div>
      <div className="text-3xl font-bold">{value.toLocaleString()}</div>
      <div className="text-sm opacity-75 mt-1">{subtitle}</div>
    </div>
  );
}

function MiniStat({ label, value, icon }: any) {
  return (
    <div className="text-center">
      <div className="text-2xl mb-1">{icon}</div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

function ProcessStep({ number, title, description, icon }: any) {
  return (
    <div className="flex items-start space-x-4">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center text-sm font-bold">
        {number}
      </div>
      <div className="flex-1">
        <h4 className="font-semibold text-gray-900 flex items-center space-x-2">
          <span>{icon}</span>
          <span>{title}</span>
        </h4>
        <p className="text-sm text-gray-600 mt-1">{description}</p>
      </div>
    </div>
  );
}

function DreamRunCard({ run, onClick }: any) {
  return (
    <button
      onClick={onClick}
      className="w-full bg-white rounded-lg shadow-sm border border-gray-200 p-4 hover:border-indigo-300 hover:shadow-md transition-all text-left"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-3">
          <Moon className="w-5 h-5 text-indigo-600" />
          <span className="font-semibold text-gray-900">
            {new Date(run.started_at).toLocaleDateString()} at {new Date(run.started_at).toLocaleTimeString()}
          </span>
        </div>
        <div
          className={clsx(
            'px-2 py-1 rounded text-xs font-medium',
            run.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          )}
        >
          {run.status}
        </div>
      </div>

      <div className="flex items-center space-x-4 text-sm text-gray-600">
        <span>📚 {run.memories_reviewed} reviewed</span>
        <span>⬆️ {run.memories_promoted} promoted</span>
        <span>🔄 {run.memories_synthesized} synthesized</span>
        <span>🗑️ {run.memories_deleted} deleted</span>
      </div>
    </button>
  );
}

function DreamRunDetail({ run }: any) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center space-x-3">
          <Moon className="w-7 h-7 text-indigo-600" />
          <span>Sleep Cycle Detail</span>
        </h2>
        <p className="text-gray-500">{new Date(run.started_at).toLocaleString()}</p>

        {run.dream_report && (
          <div className="mt-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-4 border border-indigo-100">
            <p className="text-sm text-gray-700 italic">"{run.dream_report}"</p>
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Reviewed" value={run.memories_reviewed} icon="📚" color="blue" />
        <MetricCard label="Promoted" value={run.memories_promoted} icon="⬆️" color="green" />
        <MetricCard label="Synthesized" value={run.memories_synthesized} icon="🔄" color="purple" />
        <MetricCard label="Deleted" value={run.memories_deleted} icon="🗑️" color="red" />
      </div>

      {/* Metadata */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Metadata</h3>
        <dl className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-gray-500">Duration</dt>
            <dd className="font-medium text-gray-900">{run.duration_seconds?.toFixed(1)}s</dd>
          </div>
          <div>
            <dt className="text-gray-500">Clusters Created</dt>
            <dd className="font-medium text-gray-900">{run.clusters_created}</dd>
          </div>
          <div>
            <dt className="text-gray-500">LLM Calls</dt>
            <dd className="font-medium text-gray-900">{run.llm_calls_made}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Status</dt>
            <dd className="font-medium text-gray-900">{run.status}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}

function MetricCard({ label, value, icon, color }: any) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    green: 'bg-green-50 text-green-700 border-green-200',
    purple: 'bg-purple-50 text-purple-700 border-purple-200',
    red: 'bg-red-50 text-red-700 border-red-200',
  };

  return (
    <div className={clsx('rounded-lg border p-4', colorClasses[color as keyof typeof colorClasses])}>
      <div className="text-2xl mb-2">{icon}</div>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-sm opacity-75">{label}</div>
    </div>
  );
}

function SettingRow({ label, value, description }: any) {
  return (
    <div className="flex items-start justify-between py-3 border-b border-gray-200 last:border-0">
      <div>
        <div className="font-medium text-gray-900">{label}</div>
        <div className="text-sm text-gray-500 mt-1">{description}</div>
      </div>
      <div className="text-sm font-semibold text-gray-900 bg-gray-100 px-3 py-1 rounded">{value}</div>
    </div>
  );
}
