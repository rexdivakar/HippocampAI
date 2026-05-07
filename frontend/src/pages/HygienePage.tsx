import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Sparkles,
  Trash2,
  Copy,
  GitMerge,
  Shield,
  Tag,
  Star,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  Loader2,
  XCircle,
  TrendingUp,
  Search,
  Settings,
  Zap,
} from 'lucide-react';
import { apiClient } from '../services/api';
import type {
  Memory,
  HealingConfig,
  StaleMemory,
  DuplicateCluster,
  KnowledgeGap,
  FullHealthCheckResult,
} from '../types';
import clsx from 'clsx';

interface HygienePageProps {
  userId: string;
}

export function HygienePage({ userId }: HygienePageProps) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [notification, setNotification] = useState<{
    type: 'success' | 'error' | 'warning';
    message: string;
  } | null>(null);

  const queryClient = useQueryClient();

  const showNotification = useCallback(
    (type: 'success' | 'error' | 'warning', message: string): void => {
      setNotification({ type, message });
      setTimeout(() => setNotification(null), 5000);
    },
    []
  );

  // ============================================================================
  // DATA QUERIES — ALL REAL BACKEND DATA
  // ============================================================================

  // Fetch memories
  const { data: memories = [], isLoading } = useQuery<Memory[]>({
    queryKey: ['memories', userId, refreshKey],
    queryFn: async () => {
      const result = await apiClient.getMemories({
        user_id: userId,
        filters: { session_id: userId },
        limit: 1000,
      });
      return result;
    },
  });

  // Healing config from backend
  const { data: healingConfig } = useQuery<HealingConfig>({
    queryKey: ['healingConfig', userId],
    queryFn: () => apiClient.getHealingConfig(userId),
  });

  // Local state initialized from backend config
  const [configForm, setConfigForm] = useState<Partial<HealingConfig>>({});
  const [configDirty, setConfigDirty] = useState(false);

  useEffect(() => {
    if (healingConfig) {
      setConfigForm({
        enabled: healingConfig.enabled,
        auto_cleanup_enabled: healingConfig.auto_cleanup_enabled,
        auto_dedup_enabled: healingConfig.auto_dedup_enabled,
        cleanup_threshold_days: healingConfig.cleanup_threshold_days,
        dedup_similarity_threshold: healingConfig.dedup_similarity_threshold,
        require_user_approval: healingConfig.require_user_approval,
        max_actions_per_run: healingConfig.max_actions_per_run,
      });
      setConfigDirty(false);
    }
  }, [healingConfig]);

  // Detect duplicates from backend
  const { data: duplicatesData } = useQuery<{
    clusters: DuplicateCluster[];
    total_duplicates: number;
  }>({
    queryKey: ['duplicates', userId, configForm.dedup_similarity_threshold, refreshKey],
    queryFn: () =>
      apiClient.detectDuplicates(userId, configForm.dedup_similarity_threshold ?? 0.9),
    enabled: memories.length > 0,
  });

  // Detect stale memories from backend
  const { data: staleData } = useQuery<{
    stale_memories: StaleMemory[];
    count: number;
  }>({
    queryKey: ['stale', userId, configForm.cleanup_threshold_days, refreshKey],
    queryFn: () =>
      apiClient.detectStaleMemories(userId, configForm.cleanup_threshold_days ?? 90),
    enabled: memories.length > 0,
  });

  // Detect knowledge gaps from backend
  const { data: gapsData } = useQuery<{
    gaps: KnowledgeGap[];
    count: number;
  }>({
    queryKey: ['knowledgeGaps', userId, refreshKey],
    queryFn: () => apiClient.detectKnowledgeGapsDetailed(userId),
    enabled: memories.length > 0,
  });

  const duplicateClusters = duplicatesData?.clusters ?? [];
  const totalDuplicates = duplicatesData?.total_duplicates ?? 0;
  const staleMemories = staleData?.stale_memories ?? [];
  const knowledgeGaps = gapsData?.gaps ?? [];

  // ============================================================================
  // MUTATIONS — ALL REAL BACKEND OPERATIONS
  // ============================================================================

  const invalidateAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['memories', userId] });
    queryClient.invalidateQueries({ queryKey: ['duplicates', userId] });
    queryClient.invalidateQueries({ queryKey: ['stale', userId] });
    queryClient.invalidateQueries({ queryKey: ['knowledgeGaps', userId] });
  }, [queryClient, userId]);

  // Deduplicate
  const deduplicateMutation = useMutation({
    mutationFn: () => apiClient.runDeduplication(userId, false),
    onSuccess: (data) => {
      showNotification(
        'success',
        `Deduplication complete: ${data.details?.duplicates_removed ?? data.affected_count} duplicate(s) removed.`
      );
      invalidateAll();
    },
    onError: (error: Error) => {
      showNotification('error', `Deduplication failed: ${error.message}`);
    },
  });

  // Consolidate (merge)
  const consolidateMutation = useMutation({
    mutationFn: () => apiClient.consolidateMemories(userId, false),
    onSuccess: (data) => {
      showNotification(
        'success',
        `Consolidation complete: ${data.affected_count} memories merged.`
      );
      invalidateAll();
    },
    onError: (error: Error) => {
      showNotification('error', `Consolidation failed: ${error.message}`);
    },
  });

  // Auto cleanup
  const cleanupMutation = useMutation({
    mutationFn: () => apiClient.runAutoCleanup(userId, false),
    onSuccess: () => {
      showNotification('success', 'Auto-cleanup complete. Stale and low-quality memories removed.');
      invalidateAll();
    },
    onError: (error: Error) => {
      showNotification('error', `Cleanup failed: ${error.message}`);
    },
  });

  // Auto-tag
  const autoTagMutation = useMutation({
    mutationFn: () => apiClient.autoTagMemories(userId, false),
    onSuccess: (data) => {
      showNotification(
        'success',
        `Auto-tagging complete: ${data.affected_count} memories tagged.`
      );
      invalidateAll();
    },
    onError: (error: Error) => {
      showNotification('error', `Auto-tagging failed: ${error.message}`);
    },
  });

  // Adjust importance
  const importanceMutation = useMutation({
    mutationFn: () => apiClient.adjustImportance(userId, false),
    onSuccess: (data) => {
      showNotification(
        'success',
        `Importance adjusted for ${data.affected_count} memories.`
      );
      invalidateAll();
    },
    onError: (error: Error) => {
      showNotification('error', `Importance adjustment failed: ${error.message}`);
    },
  });

  // Full health check
  const fullCheckMutation = useMutation<FullHealthCheckResult>({
    mutationFn: () => apiClient.runFullHealthCheck(userId, false),
    onSuccess: (data) => {
      const improvement = data.health_improvement > 0 ? `+${data.health_improvement.toFixed(1)}%` : 'no change';
      showNotification(
        'success',
        `Full health check done. ${data.total_actions_applied} actions applied (${improvement} health). ${data.summary}`
      );
      invalidateAll();
    },
    onError: (error: Error) => {
      showNotification('error', `Full health check failed: ${error.message}`);
    },
  });

  // Save config
  const saveConfigMutation = useMutation({
    mutationFn: () => apiClient.updateHealingConfig(userId, configForm),
    onSuccess: () => {
      showNotification('success', 'Healing configuration saved.');
      setConfigDirty(false);
      queryClient.invalidateQueries({ queryKey: ['healingConfig', userId] });
    },
    onError: (error: Error) => {
      showNotification('error', `Failed to save config: ${error.message}`);
    },
  });

  const handleRefresh = (): void => {
    setRefreshKey((prev) => prev + 1);
  };

  const updateConfig = (patch: Partial<HealingConfig>): void => {
    setConfigForm((prev) => ({ ...prev, ...patch }));
    setConfigDirty(true);
  };

  return (
    <div className="w-full space-y-6">
      {/* Notification banner */}
      {notification && (
        <div className="fixed top-4 right-4 z-50 animate-in slide-in-from-top-2">
          <div
            className={clsx(
              'rounded-lg shadow-lg px-4 py-3 flex items-center gap-3 min-w-[320px]',
              notification.type === 'success' &&
                'bg-green-50 border border-green-200 text-green-800',
              notification.type === 'error' &&
                'bg-red-50 border border-red-200 text-red-800',
              notification.type === 'warning' &&
                'bg-yellow-50 border border-yellow-200 text-yellow-800'
            )}
          >
            {notification.type === 'success' && (
              <CheckCircle className="w-5 h-5 text-green-600 shrink-0" />
            )}
            {notification.type === 'error' && (
              <XCircle className="w-5 h-5 text-red-600 shrink-0" />
            )}
            {notification.type === 'warning' && (
              <AlertTriangle className="w-5 h-5 text-yellow-600 shrink-0" />
            )}
            <span className="text-sm font-medium">{notification.message}</span>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center space-x-3">
          <Sparkles className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Memory Hygiene Tools</h1>
            <p className="text-gray-600">Maintain a healthy and clean knowledge base</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => fullCheckMutation.mutate()}
            disabled={fullCheckMutation.isPending}
            className="btn-primary flex items-center space-x-2"
          >
            {fullCheckMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Zap className="w-4 h-4" />
            )}
            <span>Full Health Check</span>
          </button>
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

      {/* Overview Stats — all from real backend queries */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="card bg-blue-50 border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 mb-1">Total Memories</p>
              <p className="text-2xl font-bold text-blue-900">{memories.length}</p>
            </div>
            <CheckCircle className="w-10 h-10 text-blue-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-yellow-50 border border-yellow-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-yellow-600 mb-1">Duplicates</p>
              <p className="text-2xl font-bold text-yellow-900">{totalDuplicates}</p>
            </div>
            <Copy className="w-10 h-10 text-yellow-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-orange-50 border border-orange-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-orange-600 mb-1">Stale Memories</p>
              <p className="text-2xl font-bold text-orange-900">{staleMemories.length}</p>
            </div>
            <Trash2 className="w-10 h-10 text-orange-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-red-50 border border-red-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-red-600 mb-1">Knowledge Gaps</p>
              <p className="text-2xl font-bold text-red-900">{knowledgeGaps.length}</p>
            </div>
            <Search className="w-10 h-10 text-red-500 opacity-20" />
          </div>
        </div>

        <div className="card bg-green-50 border border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 mb-1">Untagged</p>
              <p className="text-2xl font-bold text-green-900">
                {memories.filter((m) => m.tags.length === 0).length}
              </p>
            </div>
            <Tag className="w-10 h-10 text-green-500 opacity-20" />
          </div>
        </div>
      </div>

      {/* Healing Actions Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Deduplicate Memories */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
              <Copy className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900">Deduplicate</h2>
              <p className="text-sm text-gray-600">Remove similar or duplicate memories</p>
            </div>
          </div>

          <div className="bg-yellow-50 rounded-lg p-3 mb-4">
            <p className="text-sm text-gray-700">
              Found <span className="font-bold text-yellow-700">{duplicateClusters.length}</span> duplicate
              clusters ({totalDuplicates} total duplicates)
            </p>
          </div>

          <button
            onClick={() => deduplicateMutation.mutate()}
            disabled={duplicateClusters.length === 0 || deduplicateMutation.isPending}
            className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {deduplicateMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Deduplicating...</span>
              </>
            ) : (
              <span>Deduplicate Now</span>
            )}
          </button>
        </div>

        {/* Merge/Consolidate */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
              <GitMerge className="w-6 h-6 text-purple-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900">Consolidate</h2>
              <p className="text-sm text-gray-600">Merge related memories into insights</p>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-3 mb-4">
            <p className="text-sm text-gray-700">
              Intelligently merge similar memories while preserving unique information.
            </p>
          </div>

          <button
            onClick={() => consolidateMutation.mutate()}
            disabled={consolidateMutation.isPending}
            className="btn-primary w-full bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {consolidateMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Merging...</span>
              </>
            ) : (
              <span>Consolidate Memories</span>
            )}
          </button>
        </div>

        {/* Auto Cleanup */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
              <Trash2 className="w-6 h-6 text-red-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900">Auto Cleanup</h2>
              <p className="text-sm text-gray-600">Remove stale and low-quality memories</p>
            </div>
          </div>

          <div className="bg-red-50 rounded-lg p-3 mb-4">
            <p className="text-sm text-gray-700">
              Found <span className="font-bold text-red-700">{staleMemories.length}</span> stale
              memories older than {configForm.cleanup_threshold_days ?? 90} days
            </p>
          </div>

          <button
            onClick={() => cleanupMutation.mutate()}
            disabled={staleMemories.length === 0 || cleanupMutation.isPending}
            className="btn-primary w-full bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {cleanupMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Cleaning up...</span>
              </>
            ) : (
              <span>Run Auto Cleanup</span>
            )}
          </button>
        </div>

        {/* Auto-Tag */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <Tag className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900">Auto-Tag</h2>
              <p className="text-sm text-gray-600">Automatically tag untagged memories</p>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-3 mb-4">
            <p className="text-sm text-gray-700">
              <span className="font-bold text-green-700">
                {memories.filter((m) => m.tags.length === 0).length}
              </span>{' '}
              memories have no tags
            </p>
          </div>

          <button
            onClick={() => autoTagMutation.mutate()}
            disabled={autoTagMutation.isPending}
            className="btn-primary w-full bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {autoTagMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Tagging...</span>
              </>
            ) : (
              <span>Auto-Tag Memories</span>
            )}
          </button>
        </div>

        {/* Adjust Importance */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-indigo-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900">Adjust Importance</h2>
              <p className="text-sm text-gray-600">Recalculate scores from usage patterns</p>
            </div>
          </div>

          <div className="bg-indigo-50 rounded-lg p-3 mb-4">
            <p className="text-sm text-gray-700">
              Automatically adjust importance scores based on access frequency and recency.
            </p>
          </div>

          <button
            onClick={() => importanceMutation.mutate()}
            disabled={importanceMutation.isPending}
            className="btn-primary w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {importanceMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Adjusting...</span>
              </>
            ) : (
              <span>Adjust Importance</span>
            )}
          </button>
        </div>

        {/* Privacy / PII Scan */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-pink-100 rounded-lg flex items-center justify-center">
              <Shield className="w-6 h-6 text-pink-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-gray-900">Privacy Scan</h2>
              <p className="text-sm text-gray-600">Detect and remove sensitive data</p>
            </div>
          </div>

          <div className="bg-pink-50 rounded-lg p-3 mb-4">
            <p className="text-sm text-gray-700">
              Scan for PII, IP addresses, and sensitive information using the full health check engine.
            </p>
          </div>

          <button
            onClick={() => fullCheckMutation.mutate()}
            disabled={fullCheckMutation.isPending}
            className="btn-primary w-full bg-pink-600 hover:bg-pink-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {fullCheckMutation.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Scanning...</span>
              </>
            ) : (
              <span>Run Privacy Scan</span>
            )}
          </button>
        </div>
      </div>

      {/* Duplicate Preview — from backend detection */}
      {duplicateClusters.length > 0 && (
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
            <Copy className="w-6 h-6 text-yellow-600" />
            <span>Detected Duplicate Clusters</span>
          </h2>
          <div className="space-y-4">
            {duplicateClusters.slice(0, 5).map((cluster, idx) => (
              <div key={idx} className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-gray-900">
                    Cluster #{idx + 1} — {cluster.duplicate_count} duplicates
                  </span>
                  <span className="text-xs px-2 py-0.5 bg-yellow-200 text-yellow-800 rounded-full">
                    {(cluster.average_similarity * 100).toFixed(0)}% similar
                  </span>
                </div>
                <p className="text-sm text-gray-700 mb-2">
                  Representative: {cluster.representative_text}
                </p>
                <div className="flex items-center space-x-2 text-xs text-gray-500">
                  <span>Action: {cluster.suggested_action}</span>
                  <span>|</span>
                  <span>IDs: {cluster.duplicate_ids.slice(0, 3).map((id) => id.slice(0, 8)).join(', ')}
                    {cluster.duplicate_ids.length > 3 && ` +${cluster.duplicate_ids.length - 3} more`}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stale Memories Preview — from backend detection */}
      {staleMemories.length > 0 && (
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
            <Trash2 className="w-6 h-6 text-orange-600" />
            <span>Stale Memories</span>
          </h2>
          <div className="space-y-3">
            {staleMemories.slice(0, 6).map((sm) => (
              <div
                key={sm.id}
                className={clsx(
                  'border rounded-lg p-4',
                  sm.should_delete
                    ? 'bg-red-50 border-red-200'
                    : sm.should_archive
                      ? 'bg-orange-50 border-orange-200'
                      : 'bg-yellow-50 border-yellow-200'
                )}
              >
                <p className="text-sm text-gray-900 mb-2">{sm.text}</p>
                <div className="flex items-center space-x-3 text-xs text-gray-500">
                  <span>ID: {sm.id.slice(0, 8)}</span>
                  <span>{sm.days_since_access} days since last access</span>
                  <span>Staleness: {(sm.staleness_score * 100).toFixed(0)}%</span>
                  {sm.should_delete && (
                    <span className="px-2 py-0.5 bg-red-200 text-red-800 rounded-full">
                      Should delete
                    </span>
                  )}
                  {sm.should_archive && !sm.should_delete && (
                    <span className="px-2 py-0.5 bg-orange-200 text-orange-800 rounded-full">
                      Should archive
                    </span>
                  )}
                </div>
              </div>
            ))}
            {staleMemories.length > 6 && (
              <p className="text-sm text-gray-500 text-center">
                ... and {staleMemories.length - 6} more stale memories
              </p>
            )}
          </div>
        </div>
      )}

      {/* Knowledge Gaps — from backend analysis */}
      {knowledgeGaps.length > 0 && (
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center space-x-2">
            <Search className="w-6 h-6 text-red-600" />
            <span>Knowledge Gaps</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {knowledgeGaps.map((gap, idx) => (
              <div key={idx} className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-gray-900">{gap.topic}</span>
                  <span
                    className={clsx(
                      'text-xs px-2 py-0.5 rounded-full',
                      gap.coverage_level === 'missing' && 'bg-red-200 text-red-800',
                      gap.coverage_level === 'minimal' && 'bg-orange-200 text-orange-800',
                      gap.coverage_level === 'sparse' && 'bg-yellow-200 text-yellow-800'
                    )}
                  >
                    {gap.coverage_level}
                  </span>
                </div>
                <div className="text-xs text-gray-600 space-y-1">
                  <p>{gap.memory_count} memories | Quality: {(gap.quality_score * 100).toFixed(0)}%</p>
                  {gap.gaps.length > 0 && (
                    <ul className="list-disc list-inside">
                      {gap.gaps.slice(0, 3).map((g, i) => (
                        <li key={i}>{g}</li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Healing Configuration — loaded from and saved to backend */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900 flex items-center space-x-2">
            <Settings className="w-6 h-6 text-gray-600" />
            <span>Healing Configuration</span>
          </h2>
          {configDirty && (
            <button
              onClick={() => saveConfigMutation.mutate()}
              disabled={saveConfigMutation.isPending}
              className="btn-primary flex items-center space-x-2"
            >
              {saveConfigMutation.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Star className="w-4 h-4" />
              )}
              <span>Save Config</span>
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Toggles */}
          <div className="space-y-4">
            <label className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Auto-healing enabled</span>
              <input
                type="checkbox"
                checked={configForm.enabled ?? true}
                onChange={(e) => updateConfig({ enabled: e.target.checked })}
                className="rounded text-primary-600"
              />
            </label>
            <label className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Auto-cleanup enabled</span>
              <input
                type="checkbox"
                checked={configForm.auto_cleanup_enabled ?? true}
                onChange={(e) => updateConfig({ auto_cleanup_enabled: e.target.checked })}
                className="rounded text-primary-600"
              />
            </label>
            <label className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Auto-dedup enabled</span>
              <input
                type="checkbox"
                checked={configForm.auto_dedup_enabled ?? true}
                onChange={(e) => updateConfig({ auto_dedup_enabled: e.target.checked })}
                className="rounded text-primary-600"
              />
            </label>
            <label className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Require user approval</span>
              <input
                type="checkbox"
                checked={configForm.require_user_approval ?? true}
                onChange={(e) => updateConfig({ require_user_approval: e.target.checked })}
                className="rounded text-primary-600"
              />
            </label>
          </div>

          {/* Sliders */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Cleanup threshold: {configForm.cleanup_threshold_days ?? 90} days
              </label>
              <input
                type="range"
                min="30"
                max="365"
                step="15"
                value={configForm.cleanup_threshold_days ?? 90}
                onChange={(e) => updateConfig({ cleanup_threshold_days: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dedup similarity: {((configForm.dedup_similarity_threshold ?? 0.9) * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.5"
                max="1"
                step="0.05"
                value={configForm.dedup_similarity_threshold ?? 0.9}
                onChange={(e) =>
                  updateConfig({ dedup_similarity_threshold: parseFloat(e.target.value) })
                }
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max actions per run: {configForm.max_actions_per_run ?? 50}
              </label>
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={configForm.max_actions_per_run ?? 50}
                onChange={(e) => updateConfig({ max_actions_per_run: parseInt(e.target.value) })}
                className="w-full"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
