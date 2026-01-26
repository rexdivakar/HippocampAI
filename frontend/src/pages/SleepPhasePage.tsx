import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Moon,
  Play,
  Settings,
  History,
  Trash2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Eye,
  Archive,
  TrendingUp,
  Brain,
  Sparkles,
  Clock,
} from 'lucide-react';
import clsx from 'clsx';

interface SleepPhasePageProps {
  userId: string;
}

interface ConsolidationResult {
  id: string;
  user_id: string;
  status: string;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
  memories_reviewed: number;
  memories_deleted: number;
  memories_archived: number;
  memories_promoted: number;
  memories_updated: number;
  memories_synthesized: number;
  dream_report: string | null;
  dry_run?: boolean;
}

export function SleepPhasePage({ userId }: SleepPhasePageProps) {
  const [activeTab, setActiveTab] = useState<'run' | 'history' | 'settings' | 'danger'>('run');
  const [notification, setNotification] = useState<{ type: 'success' | 'error' | 'warning'; message: string } | null>(null);
  const [lastResult, setLastResult] = useState<ConsolidationResult | null>(null);
  const [showWipeConfirm, setShowWipeConfirm] = useState(false);
  const [wipeReason, setWipeReason] = useState('');
  const queryClient = useQueryClient();

  const showNotification = (type: 'success' | 'error' | 'warning', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  // Fetch consolidation stats
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['consolidation-stats', userId],
    queryFn: async () => {
      const res = await fetch(`/api/consolidation/stats?user_id=${userId}`);
      return res.json();
    },
  });

  // Fetch consolidation runs
  const { data: runs = [], isLoading: runsLoading } = useQuery({
    queryKey: ['consolidation-runs', userId],
    queryFn: async () => {
      const res = await fetch(`/api/consolidation/runs?user_id=${userId}`);
      return res.json();
    },
  });

  // Fetch session stats
  const { data: sessionStats } = useQuery({
    queryKey: ['session-stats'],
    queryFn: async () => {
      const res = await fetch('/api/sessions/stats');
      return res.json();
    },
  });

  // Trigger consolidation mutation
  const triggerMutation = useMutation({
    mutationFn: async (dryRun: boolean) => {
      const res = await fetch('/api/consolidation/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dry_run: dryRun, lookback_hours: 24, user_id: userId }),
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Failed to trigger consolidation');
      }
      const result = await res.json();
      return { ...result, dry_run: dryRun };
    },
    onSuccess: (data) => {
      setLastResult(data);
      showNotification('success', data.dry_run
        ? `Dry run complete! Reviewed ${data.memories_reviewed} memories.`
        : `Consolidation complete! ${data.memories_promoted} promoted, ${data.memories_synthesized} synthesized.`
      );
      queryClient.invalidateQueries({ queryKey: ['consolidation-runs', userId] });
      queryClient.invalidateQueries({ queryKey: ['consolidation-stats', userId] });
    },
    onError: (error: Error) => {
      showNotification('error', error.message);
    },
  });

  // Wipe user data mutation
  const wipeMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch('/api/sessions/wipe-user-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          admin_user_id: userId,
          reason: wipeReason || 'User requested data wipe',
          confirm: true,
        }),
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Failed to wipe data');
      }
      return res.json();
    },
    onSuccess: (data) => {
      setShowWipeConfirm(false);
      setWipeReason('');
      showNotification('warning', `Soft-deleted ${data.sessions_deleted} sessions. Data can be restored by admin.`);
      queryClient.invalidateQueries({ queryKey: ['session-stats'] });
    },
    onError: (error: Error) => {
      showNotification('error', error.message);
    },
  });

  const tabs = [
    { id: 'run', label: 'Run', icon: Play },
    { id: 'history', label: 'History', icon: History },
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'danger', label: 'Danger Zone', icon: AlertTriangle },
  ] as const;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Notification */}
      {notification && (
        <div className="fixed top-4 right-4 z-50 animate-in slide-in-from-top-2">
          <div className={clsx(
            'rounded-lg shadow-lg px-4 py-3 flex items-center gap-3 min-w-[320px]',
            notification.type === 'success' && 'bg-green-50 border border-green-200 text-green-800',
            notification.type === 'error' && 'bg-red-50 border border-red-200 text-red-800',
            notification.type === 'warning' && 'bg-yellow-50 border border-yellow-200 text-yellow-800'
          )}>
            {notification.type === 'success' && <CheckCircle className="w-5 h-5 text-green-600" />}
            {notification.type === 'error' && <XCircle className="w-5 h-5 text-red-600" />}
            {notification.type === 'warning' && <AlertTriangle className="w-5 h-5 text-yellow-600" />}
            <p className="text-sm font-medium flex-1">{notification.message}</p>
            <button onClick={() => setNotification(null)} className="text-gray-400 hover:text-gray-600">×</button>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="bg-white border-b">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-100 rounded-lg">
                <Moon className="w-6 h-6 text-indigo-600" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Sleep Phase</h1>
                <p className="text-sm text-gray-500">Memory consolidation & optimization</p>
              </div>
            </div>
            
            {/* Tab Navigation */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={clsx(
                    'px-4 py-2 text-sm font-medium rounded-md flex items-center gap-2 transition-all',
                    activeTab === tab.id
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900',
                    tab.id === 'danger' && activeTab === tab.id && 'text-red-600'
                  )}
                >
                  <tab.icon className={clsx('w-4 h-4', tab.id === 'danger' && 'text-red-500')} />
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-6 py-6">
        {activeTab === 'run' && (
          <RunTab
            stats={stats}
            statsLoading={statsLoading}
            lastResult={lastResult}
            isRunning={triggerMutation.isPending}
            onTrigger={(dryRun) => triggerMutation.mutate(dryRun)}
          />
        )}
        {activeTab === 'history' && (
          <HistoryTab runs={runs} isLoading={runsLoading} />
        )}
        {activeTab === 'settings' && (
          <SettingsTab stats={stats} sessionStats={sessionStats} />
        )}
        {activeTab === 'danger' && (
          <DangerTab
            userId={userId}
            showWipeConfirm={showWipeConfirm}
            setShowWipeConfirm={setShowWipeConfirm}
            wipeReason={wipeReason}
            setWipeReason={setWipeReason}
            isWiping={wipeMutation.isPending}
            onWipe={() => wipeMutation.mutate()}
          />
        )}
      </div>
    </div>
  );
}


// ============================================
// RUN TAB - Main consolidation interface
// ============================================

function RunTab({ stats, statsLoading, lastResult, isRunning, onTrigger }: {
  stats: any;
  statsLoading: boolean;
  lastResult: ConsolidationResult | null;
  isRunning: boolean;
  onTrigger: (dryRun: boolean) => void;
}) {
  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Brain className="w-5 h-5 text-indigo-600" />}
          label="Total Runs"
          value={statsLoading ? '...' : stats?.total_runs || 0}
          color="indigo"
        />
        <StatCard
          icon={<TrendingUp className="w-5 h-5 text-green-600" />}
          label="Promoted"
          value={statsLoading ? '...' : stats?.total_memories_promoted || 0}
          color="green"
        />
        <StatCard
          icon={<Sparkles className="w-5 h-5 text-purple-600" />}
          label="Synthesized"
          value={statsLoading ? '...' : stats?.total_memories_synthesized || 0}
          color="purple"
        />
        <StatCard
          icon={<Archive className="w-5 h-5 text-orange-600" />}
          label="Archived"
          value={statsLoading ? '...' : stats?.total_memories_deleted || 0}
          color="orange"
        />
      </div>

      {/* Running Indicator */}
      {isRunning && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-6">
          <div className="flex items-center gap-4">
            <div className="relative">
              <Moon className="w-10 h-10 text-indigo-600 animate-pulse" />
              <span className="absolute -top-1 -right-1 flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
              </span>
            </div>
            <div>
              <h3 className="font-semibold text-indigo-900">Sleep Phase Running...</h3>
              <p className="text-sm text-indigo-700">Analyzing and consolidating memories</p>
            </div>
          </div>
          <div className="mt-4 flex gap-6 text-xs text-indigo-600">
            <span className="flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> Collecting</span>
            <span className="flex items-center gap-1"><Brain className="w-3 h-3" /> Clustering</span>
            <span className="flex items-center gap-1"><Sparkles className="w-3 h-3" /> Analyzing</span>
          </div>
        </div>
      )}

      {/* Last Result */}
      {lastResult && !isRunning && (
        <ResultCard result={lastResult} />
      )}

      {/* Action Buttons */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="font-semibold text-gray-900 mb-2">Run Consolidation</h3>
        <p className="text-sm text-gray-600 mb-4">
          Analyze recent memories, promote important ones, and clean up noise.
        </p>
        <div className="flex gap-3">
          <button
            onClick={() => onTrigger(true)}
            disabled={isRunning}
            className={clsx(
              'flex-1 px-4 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-all',
              isRunning
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700 border border-gray-300'
            )}
          >
            <Eye className="w-4 h-4" />
            Dry Run (Preview)
          </button>
          <button
            onClick={() => onTrigger(false)}
            disabled={isRunning}
            className={clsx(
              'flex-1 px-4 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-all',
              isRunning
                ? 'bg-indigo-400 text-white cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white'
            )}
          >
            {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {isRunning ? 'Running...' : 'Run Now'}
          </button>
        </div>
      </div>

      {/* How It Works */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="font-semibold text-gray-900 mb-4">How It Works</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <ProcessCard step="1" title="Collect" desc="Gather recent memories from last 24 hours" />
          <ProcessCard step="2" title="Analyze" desc="AI clusters and evaluates importance" />
          <ProcessCard step="3" title="Optimize" desc="Promote, synthesize, or archive memories" />
        </div>
      </div>
    </div>
  );
}


// ============================================
// HISTORY TAB
// ============================================

function HistoryTab({ runs, isLoading }: { runs: ConsolidationResult[]; isLoading: boolean }) {
  const [selectedRun, setSelectedRun] = useState<ConsolidationResult | null>(null);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 text-indigo-600 animate-spin" />
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="bg-white rounded-xl border p-12 text-center">
        <Moon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
        <h3 className="font-semibold text-gray-900">No Sleep Cycles Yet</h3>
        <p className="text-sm text-gray-500 mt-2">Run your first consolidation from the Run tab</p>
      </div>
    );
  }

  if (selectedRun) {
    return (
      <div className="space-y-4">
        <button
          onClick={() => setSelectedRun(null)}
          className="text-indigo-600 hover:text-indigo-700 text-sm font-medium"
        >
          ← Back to history
        </button>
        <ResultCard result={selectedRun} expanded />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {runs.map((run) => (
        <button
          key={run.id}
          onClick={() => setSelectedRun(run)}
          className="w-full bg-white rounded-lg border p-4 hover:border-indigo-300 hover:shadow-sm transition-all text-left"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <Moon className="w-5 h-5 text-indigo-600" />
              <span className="font-medium text-gray-900">
                {new Date(run.started_at).toLocaleDateString()} at {new Date(run.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            <span className={clsx(
              'px-2 py-1 rounded text-xs font-medium',
              run.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            )}>
              {run.status}
            </span>
          </div>
          <div className="flex gap-4 text-sm text-gray-600">
            <span>{run.memories_reviewed} reviewed</span>
            <span className="text-green-600">+{run.memories_promoted} promoted</span>
            <span className="text-purple-600">+{run.memories_synthesized} synthesized</span>
            <span className="text-orange-600">-{run.memories_deleted} archived</span>
          </div>
        </button>
      ))}
    </div>
  );
}

// ============================================
// SETTINGS TAB
// ============================================

function SettingsTab({ sessionStats }: { stats?: any; sessionStats: any }) {
  return (
    <div className="space-y-6">
      {/* Current Config */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Current Configuration</h3>
        <div className="grid md:grid-cols-2 gap-4">
          <ConfigItem label="Schedule" value="Daily at 3:00 AM UTC" />
          <ConfigItem label="Lookback Period" value="24 hours" />
          <ConfigItem label="Min Importance" value="3.0 / 10" />
          <ConfigItem label="Mode" value="Live (changes applied)" />
        </div>
      </div>

      {/* Session Stats */}
      {sessionStats && (
        <div className="bg-white rounded-xl border p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Storage Stats</h3>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{sessionStats.total_users || 0}</div>
              <div className="text-sm text-gray-500">Users</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{sessionStats.total_sessions || 0}</div>
              <div className="text-sm text-gray-500">Sessions</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{sessionStats.total_memories || 0}</div>
              <div className="text-sm text-gray-500">Memories</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-red-600">{sessionStats.deleted_sessions || 0}</div>
              <div className="text-sm text-gray-500">Deleted</div>
            </div>
          </div>
        </div>
      )}

      {/* Environment Variables */}
      <div className="bg-gray-50 rounded-xl border p-6">
        <h4 className="font-medium text-gray-700 mb-3">Environment Variables</h4>
        <div className="grid md:grid-cols-2 gap-2 text-xs font-mono">
          <code className="bg-white px-3 py-2 rounded border">ACTIVE_CONSOLIDATION_ENABLED=true</code>
          <code className="bg-white px-3 py-2 rounded border">CONSOLIDATION_SCHEDULE_HOUR=3</code>
          <code className="bg-white px-3 py-2 rounded border">CONSOLIDATION_LOOKBACK_HOURS=24</code>
          <code className="bg-white px-3 py-2 rounded border">CONSOLIDATION_MIN_IMPORTANCE=3.0</code>
        </div>
      </div>
    </div>
  );
}


// ============================================
// DANGER TAB - Data wipe functionality
// ============================================

function DangerTab({
  userId,
  showWipeConfirm,
  setShowWipeConfirm,
  wipeReason,
  setWipeReason,
  isWiping,
  onWipe,
}: {
  userId: string;
  showWipeConfirm: boolean;
  setShowWipeConfirm: (show: boolean) => void;
  wipeReason: string;
  setWipeReason: (reason: string) => void;
  isWiping: boolean;
  onWipe: () => void;
}) {
  return (
    <div className="space-y-6">
      {/* Warning Banner */}
      <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
        <div>
          <h4 className="font-medium text-red-800">Danger Zone</h4>
          <p className="text-sm text-red-700 mt-1">
            Actions here can affect your data. Soft-deleted data can be restored by an admin.
          </p>
        </div>
      </div>

      {/* Wipe Data Card */}
      <div className="bg-white rounded-xl border border-red-200 p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-red-100 rounded-lg">
            <Trash2 className="w-6 h-6 text-red-600" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900">Wipe All Session Data</h3>
            <p className="text-sm text-gray-600 mt-1">
              Soft-delete all memories and sessions for user <code className="bg-gray-100 px-1 rounded">{userId}</code>.
              Data will be hidden from agents and dashboard but can be restored by admin.
            </p>

            {!showWipeConfirm ? (
              <button
                onClick={() => setShowWipeConfirm(true)}
                className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
              >
                Wipe My Data
              </button>
            ) : (
              <div className="mt-4 p-4 bg-red-50 rounded-lg border border-red-200">
                <h4 className="font-medium text-red-800 mb-3">⚠️ Confirm Data Wipe</h4>
                
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Reason (optional)
                  </label>
                  <input
                    type="text"
                    value={wipeReason}
                    onChange={(e) => setWipeReason(e.target.value)}
                    placeholder="Why are you wiping this data?"
                    className="w-full px-3 py-2 border rounded-lg text-sm"
                  />
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
                  <p className="text-sm text-yellow-800">
                    <strong>What will happen:</strong>
                  </p>
                  <ul className="text-sm text-yellow-700 mt-2 space-y-1">
                    <li>• All sessions will be marked as deleted</li>
                    <li>• Memories will be hidden from agents</li>
                    <li>• Dashboard will no longer show this data</li>
                    <li>• Admin can restore data if needed</li>
                  </ul>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={() => {
                      setShowWipeConfirm(false);
                      setWipeReason('');
                    }}
                    className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={onWipe}
                    disabled={isWiping}
                    className={clsx(
                      'flex-1 px-4 py-2 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors',
                      isWiping
                        ? 'bg-red-400 text-white cursor-not-allowed'
                        : 'bg-red-600 text-white hover:bg-red-700'
                    )}
                  >
                    {isWiping ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Wiping...
                      </>
                    ) : (
                      <>
                        <Trash2 className="w-4 h-4" />
                        Confirm Wipe
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Info about soft delete */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <h4 className="font-medium text-blue-800 mb-2">About Soft Delete</h4>
        <p className="text-sm text-blue-700">
          Soft delete preserves your data in a hidden state. Unlike permanent deletion, 
          soft-deleted data can be recovered by an administrator. This is useful for 
          compliance, debugging, or if you change your mind.
        </p>
      </div>
    </div>
  );
}


// ============================================
// HELPER COMPONENTS
// ============================================

function StatCard({ icon, label, value, color }: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  color: 'indigo' | 'green' | 'purple' | 'orange';
}) {
  const colors = {
    indigo: 'bg-indigo-50 border-indigo-200',
    green: 'bg-green-50 border-green-200',
    purple: 'bg-purple-50 border-purple-200',
    orange: 'bg-orange-50 border-orange-200',
  };

  return (
    <div className={clsx('rounded-xl border p-4', colors[color])}>
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-sm text-gray-600">{label}</span>
      </div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
    </div>
  );
}

function ResultCard({ result }: { result: ConsolidationResult; expanded?: boolean }) {
  const isDryRun = result.dry_run;

  return (
    <div className={clsx(
      'rounded-xl border p-6',
      isDryRun ? 'bg-yellow-50 border-yellow-200' : 'bg-green-50 border-green-200'
    )}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{isDryRun ? '🔍' : '✨'}</span>
          <div>
            <h3 className="font-semibold text-gray-900">
              {isDryRun ? 'Dry Run Complete' : 'Consolidation Complete'}
            </h3>
            <p className="text-sm text-gray-600">
              {isDryRun ? 'Preview only - no changes made' : 'Memories have been optimized'}
            </p>
          </div>
        </div>
        <span className={clsx(
          'px-3 py-1 rounded-full text-xs font-semibold',
          isDryRun ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'
        )}>
          {isDryRun ? 'DRY RUN' : 'LIVE'}
        </span>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-4">
        <MiniStat label="Reviewed" value={result.memories_reviewed} icon="📋" />
        <MiniStat label="Promoted" value={result.memories_promoted} icon="⬆️" highlight={result.memories_promoted > 0} />
        <MiniStat label="Updated" value={result.memories_updated} icon="✏️" />
        <MiniStat label="Synthesized" value={result.memories_synthesized} icon="🧠" highlight={result.memories_synthesized > 0} />
        <MiniStat label="Archived" value={result.memories_archived} icon="📦" />
        <MiniStat label="Deleted" value={result.memories_deleted} icon="🗑️" />
      </div>

      {/* Dream Report */}
      {result.dream_report && (
        <div className="bg-white rounded-lg p-4 border mb-4">
          <div className="flex items-center gap-2 mb-2">
            <Moon className="w-4 h-4 text-indigo-600" />
            <span className="text-sm font-medium text-gray-700">Dream Report</span>
          </div>
          <p className="text-sm text-gray-600 whitespace-pre-line">{result.dream_report}</p>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between text-xs text-gray-500 pt-3 border-t">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {result.duration_seconds?.toFixed(2)}s
        </span>
        <span>ID: {result.id?.slice(0, 8)}</span>
      </div>
    </div>
  );
}

function MiniStat({ label, value, icon, highlight = false }: {
  label: string;
  value: number;
  icon: string;
  highlight?: boolean;
}) {
  return (
    <div className={clsx(
      'text-center p-3 rounded-lg bg-white border',
      highlight && value > 0 && 'ring-2 ring-indigo-400 ring-offset-1'
    )}>
      <div className="text-lg mb-1">{icon}</div>
      <div className={clsx('text-xl font-bold', highlight && value > 0 ? 'text-indigo-600' : 'text-gray-900')}>
        {value}
      </div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

function ProcessCard({ step, title, desc }: { step: string; title: string; desc: string }) {
  return (
    <div className="flex items-start gap-3 p-4 bg-gray-50 rounded-lg">
      <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center text-sm font-bold flex-shrink-0">
        {step}
      </div>
      <div>
        <h4 className="font-medium text-gray-900">{title}</h4>
        <p className="text-sm text-gray-600 mt-1">{desc}</p>
      </div>
    </div>
  );
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
      <span className="text-sm text-gray-600">{label}</span>
      <span className="text-sm font-medium text-gray-900">{value}</span>
    </div>
  );
}
